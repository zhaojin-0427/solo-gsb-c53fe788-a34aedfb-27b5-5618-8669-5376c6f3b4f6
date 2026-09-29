"""Regression harness: tenant logical destruction of a custody lineage (v1.6).

Run (needs KEYVAULT_FAULT_INJECTION=1 for the lock-waiter scenarios; the docker
compose ``demo`` profile sets it automatically):

    python -m scripts.race_destruction

It boots the *real* FastAPI app in-process (ASGI transport, real PostgreSQL,
real row locks) and verifies the destruction contract end to end:

  D1  basic lineage: root custody + one completed-migration descendant are
      both turned into secret-free tombstones in one transaction (online
      encrypted shard rows deleted, secret digest/length nulled), a receipt is
      stored and the hash-chain audit stays valid and plaintext-free.
  D2  idempotency: retrying with the same Idempotency-Key returns the same
      receipt (200, no second execution, no extra audit); a different root
      under the same key is IDEMPOTENCY_CONFLICT; cross-tenant access is
      rejected; destroying an unknown id 404s.
  D3  multi-hop lineage: a grandchild reached through two completed migration
      edges is destroyed too; destroying starting from the child covers the
      child subtree only (and reports the already-destroyed root correctly).
  D4  post-commit hard stop: after commit, no old recovery/migration session
      can claim a shard, submit, complete a threshold recovery/wrap, issue a
      nonce, or start a new recovery/migration — every channel returns the
      deterministic destroyed error and never emits shard bytes/ciphertext.
  D5  in-flight recovery blocked: an open recovery session is marked
      ``destroyed`` in the same transaction and its in-process pending share is
      removed only after commit; migration sessions sourced from the lineage
      are blocked and counted as well.
  D6  concurrent destroy requests (same key and different keys) never produce
      two receipts/executions or half-destroyed lineages.
  D7  migration-completion-before-destroy ordering: a threshold submit parked
      holding the migration row lock completes (creates a grandchild) and
      commits while destroy is queued on that lock; destroy must then include
      the just-created grandchild — no descendant is ever left behind.
  D8  destroy-before-migration-completion ordering: destroy parks holding the
      open migration session row lock; a submit queued behind it can only fail
      with MIGRATION_DESTROYED once destroy commits, and never creates a child.
  D9  recovery-completion vs destroy ordering: a threshold recovery submit
      parked on the session row lock either completes before destroy (destroy
      still succeeds on the tombstoned lineage and reports the blocked count)
      or is rejected — the wrap ciphertext is delivered at most once and the
      lineage is always fully destroyed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

# The gates must be armed before app construction reads settings.
os.environ.setdefault("KEYVAULT_FAULT_INJECTION", "1")

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from app.config import get_settings
from app.db import Database, wait_for_database
from app.faults import FaultBoard
from app.main import app
from app.pending import PendingShareStore
from app.seed import seed, demo_credentials

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []

EPOCH = {"valid_from": "2026-01-01T00:00:00Z",
         "valid_until": "2035-01-01T00:00:00Z", "rotation_interval": 0}


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail and not ok else ""))


def body_of(r) -> dict:
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return {}


def sign(priv_hex: str, message: str) -> str:
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).sign(
        message.encode()).hex()


@contextlib.asynccontextmanager
async def server():
    settings = get_settings()
    await wait_for_database(settings.database_url)
    await seed()
    database = Database(settings.database_url)
    await database.connect()
    app.state.db = database
    app.state.pending = PendingShareStore()
    app.state.fault_enabled = True
    board = FaultBoard()
    app.state.fault_board = board
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        try:
            yield http, database.pool, board
        finally:
            await database.close()


def auth(key: str) -> dict:
    return {"X-API-Key": key}


class Harness:
    def __init__(self, http, pool, board, creds):
        self.http = http
        self.pool = pool
        self.board = board
        self.tenant_key = creds["tenant"]["api_key"]
        self.tenant_id = creds["tenant"]["tenant_id"]
        self.admin_key = creds["admin_api_key"]
        self.cust = {c["name"]: c for c in creds["custodians"]}

    def h(self, name: str) -> dict:
        return auth(self.cust[name]["api_key"])

    async def policy(self, pid: str, threshold: int, custodians=("alice", "bob", "carol")):
        cids = [f"cus_demo_{n}" for n in custodians]
        r = await self.http.post("/policies", headers=auth(self.tenant_key), json={
            "policy_id": pid, "threshold": threshold, "custodian_ids": cids, "epoch": EPOCH})
        assert r.status_code == 201, r.text

    async def custody(self, policy: str, secret: bytes, ver: int = 1) -> str:
        r = await self.http.post("/custodies", headers=auth(self.tenant_key), json={
            "policy_id": policy, "expected_version": ver, "secret_hex": secret.hex()})
        assert r.status_code == 201, r.text
        return r.json()["custody_id"]

    async def start_mig(self, source: str, target: str, idem: str):
        return await self.http.post(
            "/migrations", headers={**auth(self.tenant_key), "Idempotency-Key": idem},
            json={"source_custody_id": source, "target_policy_id": target,
                  "target_version": 1})

    async def mig_prep(self, name: str, mid: str) -> dict:
        h = self.h(name)
        r = await self.http.post("/migrations/nonces", headers=h, json={"migration_id": mid})
        assert r.status_code == 200, r.text
        nonce = r.json()["nonce"]
        r = await self.http.post("/migrations/shards/claim", headers=h,
                                 json={"migration_id": mid, "nonce": nonce})
        assert r.status_code == 200, r.text
        c = r.json()
        return {"headers": dict(h),
                "json": {"migration_id": mid, "nonce": nonce, "point_x": c["point_x"],
                         "shard_hex": c["shard_hex"],
                         "signature": sign(self.cust[name]["signing_private_key"],
                                           c["signing_message"])}}

    async def mig_submit(self, prepped: dict, extra_headers: dict | None = None):
        return await self.http.post("/migrations/shares",
                                    headers={**prepped["headers"], **(extra_headers or {})},
                                    json=prepped["json"])

    async def complete_migration(self, mid: str, names):
        prepped = [await self.mig_prep(n, mid) for n in names]
        last = None
        for p in prepped:
            last = await self.mig_submit(p)
        assert last.status_code == 200 and body_of(last).get("completed"), last.text
        return body_of(last)["new_custody_id"]

    async def destroy(self, custody_id: str, idem: str, extra_headers: dict | None = None):
        return await self.http.post("/destructions",
                                    headers={**auth(self.tenant_key),
                                             "Idempotency-Key": idem, **(extra_headers or {})},
                                    json={"custody_id": custody_id})

    async def wrap_keypair(self) -> tuple[str, str]:
        wp = X25519PrivateKey.generate()
        priv = wp.private_bytes(serialization.Encoding.Raw,
                                serialization.PrivateFormat.Raw,
                                serialization.NoEncryption()).hex()
        pub = wp.public_key().public_bytes(serialization.Encoding.Raw,
                                           serialization.PublicFormat.Raw).hex()
        return priv, pub

    async def start_recovery(self, cid: str, pub: str, digest: str = "11" * 32):
        return await self.http.post("/recoveries", headers=auth(self.tenant_key), json={
            "custody_id": cid, "request_digest": digest, "recovery_public_key": pub})

    async def rec_prep(self, name: str, sid: str) -> dict:
        h = self.h(name)
        r = await self.http.post("/recoveries/nonces", headers=h, json={"session_id": sid})
        assert r.status_code == 200, r.text
        nonce = r.json()["nonce"]
        r = await self.http.post("/recoveries/shards/claim", headers=h,
                                 json={"session_id": sid, "nonce": nonce})
        assert r.status_code == 200, r.text
        c = r.json()
        return {"headers": dict(h),
                "json": {"session_id": sid, "nonce": nonce, "point_x": c["point_x"],
                         "shard_hex": c["shard_hex"],
                         "signature": sign(self.cust[name]["signing_private_key"],
                                           c["signing_message"])}}

    async def rec_submit(self, p: dict, extra_headers: dict | None = None):
        return await self.http.post("/recoveries/shares",
                                    headers={**p["headers"], **(extra_headers or {})},
                                    json=p["json"])

    async def custody_row(self, cid: str):
        return await self.pool.fetchrow(
            "SELECT id, status, secret_digest, secret_length, destroyed_at "
            "FROM key_custodies WHERE id=$1", cid)

    async def shard_count(self, cid: str) -> int:
        return await self.pool.fetchval(
            "SELECT count(*) FROM shards WHERE custody_id=$1", cid)

    async def wait_lock_waiter(self, need_sql: str, timeout: float = 10.0) -> bool:
        """Poll until another backend is blocked waiting for a row lock."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            n = await self.pool.fetchval(need_sql)
            if n and n > 0:
                return True
            await asyncio.sleep(0.05)
        return False


# 等待“任意未授予事务级行锁”的其他后端（销毁/提交互等场景）。
LOCK_WAIT_SQL = """
    SELECT count(*)
      FROM pg_locks l
      JOIN pg_stat_activity a ON a.pid = l.pid
     WHERE l.locktype = 'transactionid' AND NOT l.granted
       AND a.state = 'active' AND a.pid <> pg_backend_pid()
"""


async def scenario_d1_d2(hs: Harness, tag: str) -> None:
    secret = b"destruction-lineage-secret-D1-D2-0123456789"
    await hs.policy(f"pol_d1s_{tag}", 2)
    await hs.policy(f"pol_d1t_{tag}", 1, ("alice",))
    root = await hs.custody(f"pol_d1s_{tag}", secret)
    r = await hs.start_mig(root, f"pol_d1t_{tag}", f"d1mig-{tag}")
    assert r.status_code == 201, r.text
    child = await hs.complete_migration(r.json()["migration_id"], ("alice", "bob"))

    r = await hs.destroy(root, f"d1-{tag}")
    rc = body_of(r)
    check("D1: destroy 201 receipt with full lineage",
          r.status_code == 201 and set(rc["lineage_custody_ids"]) == {root, child}
          and rc["destroyed_count"] == 2 and rc["lineage_size"] == 2
          and rc["shards_deleted"] == 4, r.text)
    check("D1: scope explicitly limited to online data",
          rc["scope"] == "online_service_data_only_external_backups_not_erased")
    rid = rc["receipt_id"]

    # 墓碑无秘密：摘要/长度 NULL，分片行删除；身份与时间戳保留。
    for cid, was in ((root, "root"), (child, "child")):
        row = await hs.custody_row(cid)
        check(f"D1: {was} is secret-free tombstone",
              row["status"] == "destroyed" and row["secret_digest"] is None
              and row["secret_length"] is None and row["destroyed_at"] is not None,
              str(dict(row)))
        check(f"D1: {was} online shards erased", await hs.shard_count(cid) == 0)

    # GET 回执
    r = await hs.http.get(f"/destructions/{rid}", headers=auth(hs.tenant_key))
    check("D2: GET receipt returns same receipt",
          r.status_code == 200 and body_of(r)["receipt_id"] == rid, r.text)

    # 幂等重试：200 同一回执，不再次执行（shards_deleted 保持 4）。
    r = await hs.destroy(root, f"d1-{tag}")
    check("D2: same idempotency key replay -> 200 same receipt, no re-execution",
          r.status_code == 200 and body_of(r)["receipt_id"] == rid
          and body_of(r)["shards_deleted"] == 4
          and body_of(r)["destroyed_count"] == 2, r.text)

    # 同键不同根 -> IDEMPOTENCY_CONFLICT
    other = await hs.custody(f"pol_d1s_{tag}", b"another-secret-for-key-conflict-xy")
    r = await hs.destroy(other, f"d1-{tag}")
    check("D2: same key different root -> IDEMPOTENCY_CONFLICT",
          r.status_code == 409 and body_of(r).get("error", {}).get("code")
          == "IDEMPOTENCY_CONFLICT", r.text)

    # 缺失/超长幂等键
    r = await hs.http.post("/destructions", headers=auth(hs.tenant_key),
                           json={"custody_id": root})
    check("D2: missing idempotency key -> BAD_REQUEST",
          r.status_code == 400 and body_of(r)["error"]["code"] == "BAD_REQUEST", r.text)

    # 未知 custody
    r = await hs.destroy("cus_secret_does_not_exist", f"d1x-{tag}")
    check("D2: unknown custody -> NOT_FOUND",
          r.status_code == 404 and body_of(r)["error"]["code"] == "NOT_FOUND", r.text)

    # 审计：record.destroy x2 + destruction.complete，链完整无明文
    r = await hs.http.get("/audit/replay", headers=auth(hs.tenant_key))
    chain = r.json()
    actions = [e["action"] for e in chain["entries"]]
    check("D1: audit chain valid, records destroy entries + summary",
          chain.get("valid") is True
          and actions.count("record.destroy") >= 2 and "destruction.complete" in actions)
    raw = json.dumps(chain)
    check("D1: audit carries no plaintext secret/shards",
          secret.hex() not in raw)


async def scenario_d2_cross_tenant(hs: Harness, tag: str) -> None:
    secret = b"cross-tenant-destruction-secret-0123456789"
    await hs.policy(f"pol_d2s_{tag}", 1)
    root = await hs.custody(f"pol_d2s_{tag}", secret)
    r = await hs.destroy(root, f"d2root-{tag}")
    rid = body_of(r)["receipt_id"]
    assert r.status_code == 201, r.text

    r = await hs.http.post("/admin/tenants", headers=auth(hs.admin_key),
                           json={"name": f"destroy-other-{tag}"})
    assert r.status_code == 201, r.text
    other = auth(r.json()["api_key"])
    r = await hs.http.get(f"/destructions/{rid}", headers=other)
    check("D2: cross-tenant receipt GET -> FORBIDDEN",
          r.status_code == 403 and body_of(r)["error"]["code"] == "FORBIDDEN", r.text)
    r = await hs.http.post("/destructions", headers={**other, "Idempotency-Key": "k"},
                           json={"custody_id": root})
    check("D2: cross-tenant destroy -> FORBIDDEN",
          r.status_code == 403 and body_of(r)["error"]["code"] == "FORBIDDEN", r.text)


async def scenario_d3(hs: Harness, tag: str) -> str:
    secret = b"multi-hop-lineage-secret-D3-0123456789-abcd"
    await hs.policy(f"pol_d3a_{tag}", 2)
    await hs.policy(f"pol_d3b_{tag}", 1, ("alice",))
    await hs.policy(f"pol_d3c_{tag}", 1, ("alice",))
    root = await hs.custody(f"pol_d3a_{tag}", secret)
    r = await hs.start_mig(root, f"pol_d3b_{tag}", f"d3m1-{tag}")
    child = await hs.complete_migration(r.json()["migration_id"], ("alice", "bob"))
    r = await hs.start_mig(child, f"pol_d3c_{tag}", f"d3m2-{tag}")
    grand = await hs.complete_migration(r.json()["migration_id"], ("alice",))

    r = await hs.destroy(root, f"d3-{tag}")
    rc = body_of(r)
    check("D3: multi-hop lineage destroys root+child+grandchild",
          r.status_code == 201 and set(rc["lineage_custody_ids"]) == {root, child, grand}
          and rc["destroyed_count"] == 3 and rc["shards_deleted"] == 5, r.text)

    # 从已销毁的 child 再发起销毁：谱系仍可递归到 grand（已销毁），
    # destroyed_custody_ids 为空集、回执 201（不同幂等键）。
    r = await hs.destroy(child, f"d3b-{tag}")
    rc2 = body_of(r)
    check("D3: re-destroy from child reports empty new-tombstone set",
          r.status_code == 201 and set(rc2["lineage_custody_ids"]) == {child, grand}
          and rc2["destroyed_count"] == 0 and rc2["shards_deleted"] == 0, r.text)
    return root


async def scenario_d4(hs: Harness, tag: str) -> None:
    secret = b"post-commit-hard-stop-secret-D4-0123456789"
    await hs.policy(f"pol_d4s_{tag}", 2)
    await hs.policy(f"pol_d4t_{tag}", 1, ("alice",))
    await hs.policy(f"pol_d4u_{tag}", 1, ("alice",))
    root = await hs.custody(f"pol_d4s_{tag}", secret)
    r = await hs.start_mig(root, f"pol_d4t_{tag}", f"d4m-{tag}")
    mid = body_of(r)["migration_id"]
    child = await hs.complete_migration(mid, ("alice", "bob"))

    # 销毁前先准备一个 child 上的 open 恢复会话（1-of-1 alice）。
    _, pub = await hs.wrap_keypair()
    r = await hs.start_recovery(child, pub)
    sid = body_of(r)["session_id"]
    p = await hs.rec_prep("alice", sid)
    # 销毁前先留一个 child 上的 open 迁移会话（root 已完成迁移，不能再开）。
    r = await hs.start_mig(child, f"pol_d4u_{tag}", f"d4m-open-{tag}")
    assert r.status_code == 201, r.text
    open_mid = body_of(r)["migration_id"]

    r = await hs.destroy(root, f"d4-{tag}")
    assert r.status_code == 201, r.text
    rc = body_of(r)
    check("D4: receipt counts blocked recovery + migration",
          rc["recoveries_blocked"] == 1 and rc["migrations_blocked"] == 1, r.text)

    # 旧恢复会话：提交 -> RECOVERY_DESTROYED（绝不下发密文）。
    r = await hs.rec_submit(p)
    check("D4: pending recovery submit -> RECOVERY_DESTROYED, no wrap",
          r.status_code == 410 and body_of(r).get("error", {}).get("code")
          == "RECOVERY_DESTROYED" and body_of(r).get("wrap") is None, r.text[:300])
    # 旧会话 nonce / claim 同样拒绝（不再下发分片）。
    r = await hs.http.post("/recoveries/nonces", headers=hs.h("bob"),
                           json={"session_id": sid})
    check("D4: nonce issue on destroyed recovery -> RECOVERY_DESTROYED",
          r.status_code == 410 and body_of(r)["error"]["code"] == "RECOVERY_DESTROYED",
          r.text[:200])
    r = await hs.http.post("/recoveries/shards/claim", headers=hs.h("alice"),
                           json={"session_id": sid, "nonce": p["json"]["nonce"]})
    check("D4: claim on destroyed recovery -> 410, no shard bytes",
          r.status_code == 410 and "shard_hex" not in r.text, r.text[:200])
    # GET 旧会话可见 destroyed 终态（排障可查）。
    r = await hs.http.get(f"/recoveries/{sid}", headers=auth(hs.tenant_key))
    check("D4: GET recovery reports destroyed",
          r.status_code == 200 and body_of(r)["status"] == "destroyed", r.text)

    # 已完成迁移会话的旧通道（nonce/claim/submit）-> MIGRATION_DESTROYED 由
    # 于其源记录被销毁；未完成 open 迁移则直接是 destroyed 终态。
    r = await hs.http.post("/migrations/nonces", headers=hs.h("alice"),
                           json={"migration_id": open_mid})
    check("D4: nonce on blocked open migration -> MIGRATION_DESTROYED",
          r.status_code == 410 and body_of(r)["error"]["code"] == "MIGRATION_DESTROYED",
          r.text[:200])
    # 已完成迁移（mid）再走通道：源记录已是墓碑（completed 检查通常先生效）。
    r = await hs.http.post("/migrations/nonces", headers=hs.h("alice"),
                           json={"migration_id": mid})
    check("D4: nonce on completed migration after lineage destroy -> 409/410",
          r.status_code in (409, 410) and body_of(r)["error"]["code"] in
          ("MIGRATION_DESTROYED", "MIGRATION_COMPLETED", "CUSTODY_DESTROYED"),
          r.text[:200])

    # 新恢复 / 新迁移一律 CUSTODY_DESTROYED。
    r = await hs.start_recovery(child, pub)
    check("D4: new recovery on destroyed custody -> CUSTODY_DESTROYED",
          r.status_code == 410 and body_of(r)["error"]["code"] == "CUSTODY_DESTROYED",
          r.text[:200])
    r = await hs.start_mig(root, f"pol_d4t_{tag}", f"d4-after-{tag}")
    check("D4: new migration from destroyed custody -> CUSTODY_DESTROYED",
          r.status_code == 410 and body_of(r)["error"]["code"] == "CUSTODY_DESTROYED",
          r.text[:200])

    # 内存待累计分片已被清理（销毁提交后 clear）。
    check("D4: pending pool drained for blocked session",
          app.state.pending.count(sid) == 0)


async def scenario_d5(hs: Harness, tag: str) -> None:
    secret = b"inflight-recovery-blocked-secret-D5-0123456789"
    await hs.policy(f"pol_d5s_{tag}", 2)
    root = await hs.custody(f"pol_d5s_{tag}", secret)
    _, pub = await hs.wrap_keypair()
    r = await hs.start_recovery(root, pub)
    sid = body_of(r)["session_id"]
    p1 = await hs.rec_prep("alice", sid)
    r = await hs.rec_submit(p1)
    assert r.status_code == 200 and body_of(r)["shares_received"] == 1, r.text
    check("D5: verified share held in process memory before destroy",
          app.state.pending.count(sid) == 1)

    r = await hs.destroy(root, f"d5-{tag}")
    rc = body_of(r)
    check("D5: destroy commits with one blocked recovery",
          r.status_code == 201 and rc["recoveries_blocked"] == 1, r.text)
    # 会话行同事务置 destroyed；内存分片在提交后被清理。
    row = await hs.pool.fetchrow(
        "SELECT status FROM recovery_sessions WHERE id=$1", sid)
    check("D5: recovery row committed as destroyed", row["status"] == "destroyed")
    check("D5: in-memory pending share removed after commit",
          app.state.pending.count(sid) == 0)
    # 旧托管人补足门限也拿不到密文（nonce 通道即被终态拒绝）。
    r = await hs.http.post("/recoveries/nonces", headers=hs.h("bob"),
                           json={"session_id": sid})
    check("D5: second custodian nonce after destroy -> RECOVERY_DESTROYED",
          r.status_code == 410 and body_of(r)["error"]["code"] == "RECOVERY_DESTROYED",
          r.text[:200])
    # 已领取但未提交的 alice 再提交同样被拒（无密文）。
    r = await hs.rec_submit(p1)
    check("D5: already-held share cannot complete after destroy (no wrap ever)",
          r.status_code == 410 and body_of(r).get("wrap") is None
          and body_of(r)["error"]["code"] == "RECOVERY_DESTROYED", r.text[:300])


async def scenario_d6(hs: Harness, tag: str) -> None:
    secret = b"concurrent-destroy-secret-D6-0123456789"
    await hs.policy(f"pol_d6s_{tag}", 2)
    root = await hs.custody(f"pol_d6s_{tag}", secret)

    # 同键并发：恰一次 201，其余 200 同一回执。
    rs = await asyncio.gather(
        hs.destroy(root, f"d6same-{tag}"),
        hs.destroy(root, f"d6same-{tag}"),
        hs.destroy(root, f"d6same-{tag}"),
    )
    codes = sorted(r.status_code for r in rs)
    ids = {body_of(r)["receipt_id"] for r in rs}
    check("D6: same-key concurrent destroy -> one 201 + two 200, one receipt",
          codes == [200, 200, 201] and len(ids) == 1, f"{codes} {ids}")

    # 不同键并发：只有一次真正执行；另一个看到根记录已销毁——
    # 幂等重放不适用，第二次返回同一份（已销毁谱系的）回执。
    secret2 = b"concurrent-destroy-secret-D6b-0123456789"
    root2 = await hs.custody(f"pol_d6s_{tag}", secret2)
    rs = await asyncio.gather(
        hs.destroy(root2, f"d6a-{tag}"),
        hs.destroy(root2, f"d6b-{tag}"),
    )
    receipts = [body_of(r) for r in rs]
    # 两者都应成功（串行执行；第二个幂等地面对已销毁谱系），且谱系一致、
    # 分片只被真正删除一次（shards_deleted 之和 == 原始分片数）。
    check("D6: different-key concurrent destroys both succeed deterministically",
          all(r.status_code in (200, 201) for r in rs)
          and all(rc["lineage_custody_ids"] == [root2] for rc in receipts),
      str([(r.status_code, receipts[0]) for r in rs]))
    total_deleted = sum(rc["shards_deleted"] for rc in receipts)
    check("D6: shards physically deleted exactly once across serial destroys",
          total_deleted == 3 and await hs.shard_count(root2) == 0,
          f"total_deleted={total_deleted}")

    # 同键并发但根记录不同（根行锁不互斥）：唯一索引兜底，恰一次成功、
    # 另一次确定性 IDEMPOTENCY_CONFLICT；两个根记录不得都被销毁。
    secret3 = b"same-key-different-root-A-0123456789"
    secret4 = b"same-key-different-root-B-0123456789"
    root3 = await hs.custody(f"pol_d6s_{tag}", secret3)
    root4 = await hs.custody(f"pol_d6s_{tag}", secret4)
    rs = await asyncio.gather(
        hs.destroy(root3, f"d6conf-{tag}"),
        hs.destroy(root4, f"d6conf-{tag}"),
    )
    codes = sorted(r.status_code for r in rs)
    codes_json = [body_of(r).get("error", {}).get("code", "OK") for r in rs]
    check("D6: same-key different-root concurrent -> one 201, one IDEMPOTENCY_CONFLICT",
          codes == [201, 409] and codes_json.count("OK") == 1
          and "IDEMPOTENCY_CONFLICT" in codes_json, f"{codes} {codes_json}")
    st3 = await hs.custody_row(root3)
    st4 = await hs.custody_row(root4)
    destroyed_roots = {cid for cid, st in ((root3, st3), (root4, st4))
                       if st["status"] == "destroyed"}
    check("D6: exactly one of the two roots destroyed",
          len(destroyed_roots) == 1, f"{st3['status']} {st4['status']}")


async def scenario_d7(hs: Harness, tag: str) -> None:
    label = f"D7:{tag}"
    secret = b"migration-completes-before-destroy-D7-0123456789"
    await hs.policy(f"pol_d7s_{tag}", 2)
    await hs.policy(f"pol_d7t_{tag}", 1, ("alice",))
    await hs.policy(f"pol_d7u_{tag}", 1, ("alice",))
    root = await hs.custody(f"pol_d7s_{tag}", secret)
    r = await hs.start_mig(root, f"pol_d7t_{tag}", f"d7m1-{tag}")
    child = await hs.complete_migration(r.json()["migration_id"], ("alice", "bob"))
    # child 上的 open 迁移（将产生 grandchild）
    r = await hs.start_mig(child, f"pol_d7u_{tag}", f"d7m2-{tag}")
    mid2 = body_of(r)["migration_id"]
    p = await hs.mig_prep("alice", mid2)

    # 提交持有迁移行锁时 park（在守卫之后、resplit 之前）。
    gate = {"X-Fault-Key": label, "X-Fault-Gate": "migration.after_lock"}
    submit_task = asyncio.create_task(hs.mig_submit(p, gate))
    await hs.board.wait_reached(f"migration.after_lock:{label}")

    # 销毁与提交并发：从根出发（谱系含 child 与其 open 迁移）。
    destroy_task = asyncio.create_task(hs.destroy(root, f"d7-{tag}"))
    # 销毁事务必在 mid2 行锁上排队等待（提交方持锁 park）。
    waited = await hs.wait_lock_waiter(LOCK_WAIT_SQL)
    check("D7: destroy waits on the migration row lock held by completing submit",
          waited, "no lock waiter seen")

    await hs.board.release_gate(f"migration.after_lock:{label}")
    rs = await submit_task
    rd = await destroy_task
    b = body_of(rs)
    check("D7: queued migration submit completes first (grandchild created)",
          rs.status_code == 200 and b.get("completed") and b.get("new_custody_id"),
          rs.text[:300])
    grand = b.get("new_custody_id")
    rc = body_of(rd)
    check("D7: destroy then covers grandchild — no descendant left behind",
          rd.status_code == 201 and set(rc["lineage_custody_ids"]) == {root, child, grand}
          and rc["destroyed_count"] == 3, rd.text[:400])
    for cid in (root, child, grand):
        row = await hs.custody_row(cid)
        n = await hs.shard_count(cid)
        check(f"D7: {cid[:20]} destroyed and shards erased",
              row["status"] == "destroyed" and n == 0, f"{row['status']} shards={n}")


async def scenario_d8(hs: Harness, tag: str) -> None:
    label = f"D8:{tag}"
    secret = b"destroy-before-migration-completes-D8-0123456789"
    await hs.policy(f"pol_d8s_{tag}", 2)
    await hs.policy(f"pol_d8t_{tag}", 1, ("alice",))
    root = await hs.custody(f"pol_d8s_{tag}", secret)
    r = await hs.start_mig(root, f"pol_d8t_{tag}", f"d8m-{tag}")
    mid = body_of(r)["migration_id"]
    p = await hs.mig_prep("alice", mid)

    # 销毁在“谱系与全部在途迁移行锁已持有、阻断之前”park。
    gate = {"X-Fault-Key": label, "X-Fault-Gate": "destruction.after_lineage_locked"}
    destroy_task = asyncio.create_task(hs.destroy(root, f"d8-{tag}", gate))
    await hs.board.wait_reached(f"destruction.after_lineage_locked:{label}")

    # 另一托管人提交排队等待同一迁移行锁。
    waiter_task = asyncio.create_task(hs.mig_submit(p))
    waited = await hs.wait_lock_waiter(LOCK_WAIT_SQL)
    check("D8: submit waits on migration row lock held by destroy", waited, "no lock waiter seen")

    await hs.board.release_gate(f"destruction.after_lineage_locked:{label}")
    rd = await destroy_task
    rw = await waiter_task
    check("D8: destroy commits, lineage destroyed",
          rd.status_code == 201 and body_of(rd)["destroyed_count"] == 1
          and body_of(rd)["migrations_blocked"] == 1, rd.text[:300])
    check("D8: queued submit -> MIGRATION_DESTROYED, no child created",
          rw.status_code == 410 and body_of(rw).get("error", {}).get("code")
          == "MIGRATION_DESTROYED" and body_of(rw).get("new_custody_id") is None,
          rw.text[:300])
    # 迁移会话终态可查询；没有新托管记录。
    row = await hs.pool.fetchrow(
        "SELECT status, new_custody_id FROM migration_sessions WHERE id=$1", mid)
    check("D8: migration row committed destroyed, no new custody",
          row["status"] == "destroyed" and row["new_custody_id"] is None, str(dict(row)))
    # 源记录在线分片已删；内存无残留。
    check("D8: source shards erased, pending pool empty",
          await hs.shard_count(root) == 0 and app.state.pending.count(mid) == 0)


async def scenario_d9(hs: Harness, tag: str) -> None:
    label = f"D9:{tag}"
    secret = b"recovery-completion-vs-destroy-D9-0123456789"
    await hs.policy(f"pol_d9s_{tag}", 2)
    root = await hs.custody(f"pol_d9s_{tag}", secret)
    _, pub = await hs.wrap_keypair()
    r = await hs.start_recovery(root, pub)
    sid = body_of(r)["session_id"]
    p = await hs.rec_prep("alice", sid)

    # 2-of-2：alice 先正常提交计数 1/2；bob 的门限（最后一份）提交在拿到
    # 会话行锁后 park，与并发销毁严格排定先后。
    r = await hs.rec_submit(p)
    assert r.status_code == 200 and body_of(r)["shares_received"] == 1, r.text
    p2 = await hs.rec_prep("bob", sid)

    gate = {"X-Fault-Key": label, "X-Fault-Gate": "submit.after_lock"}
    bob_task = asyncio.create_task(hs.rec_submit(p2, gate))
    await hs.board.wait_reached(f"submit.after_lock:{label}")

    # 销毁与门限提交并发：它会阻塞在 recovery 行锁上，等提交事务出结果。
    destroy_task = asyncio.create_task(hs.destroy(root, f"d9-{tag}"))
    waited = await hs.wait_lock_waiter(LOCK_WAIT_SQL)
    check("D9: destroy waits on recovery row lock held by threshold submit",
          waited, "no lock waiter seen")

    await hs.board.release_gate(f"submit.after_lock:{label}")
    rb = await bob_task
    rd = await destroy_task
    wraps = 1 if body_of(rb).get("wrap") else 0
    # bob 先提交 -> completed（密文恰好下发一次），随后销毁仍成功（谱系已墓碑化，
    # 恢复会话不再 open，故 blocked=0）；或销毁先生效 -> RECOVERY_DESTROYED。
    if rb.status_code == 200 and body_of(rb).get("completed"):
        check("D9: recovery completed before destroy; wrap delivered exactly once",
              wraps == 1, rb.text[:200])
        check("D9: destroy still succeeds after concurrent completion",
              rd.status_code == 201 and body_of(rd)["lineage_custody_ids"] == [root],
              rd.text[:300])
    else:
        check("D9: recovery lost the race -> RECOVERY_DESTROYED, no wrap",
              rb.status_code == 410 and body_of(rb).get("wrap") is None
              and body_of(rb)["error"]["code"] == "RECOVERY_DESTROYED", rb.text[:200])
        check("D9: destroy wins and blocks the recovery",
              rd.status_code == 201 and body_of(rd)["recoveries_blocked"] == 1,
              rd.text[:300])
    row = await hs.custody_row(root)
    check("D9: lineage destroyed regardless of ordering",
          row["status"] == "destroyed" and await hs.shard_count(root) == 0)


async def amain() -> int:
    async with server() as (http, pool, board):
        creds = demo_credentials()
        creds.setdefault("admin_api_key", get_settings().admin_api_key)
        hs = Harness(http, pool, board, creds)
        suffix = os.urandom(4).hex()
        await scenario_d1_d2(hs, f"{suffix}a")
        await scenario_d2_cross_tenant(hs, f"{suffix}b")
        await scenario_d3(hs, f"{suffix}c")
        await scenario_d4(hs, f"{suffix}d")
        await scenario_d5(hs, f"{suffix}e")
        await scenario_d6(hs, f"{suffix}f")
        await scenario_d7(hs, f"{suffix}g")
        await scenario_d8(hs, f"{suffix}h")
        await scenario_d9(hs, f"{suffix}i")

        r = await http.get("/audit/replay", headers=auth(hs.tenant_key))
        check("final: audit hash chain valid across all destruction scenarios",
              r.json().get("valid") is True)

    failed = [n for n, ok, _ in results if not ok]
    print("\n" + "=" * 70)
    print(f"{len(results) - len(failed)}/{len(results)} destruction checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(" -", n)
        return 1
    print("DESTRUCTION REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
