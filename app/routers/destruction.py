"""Tenant logical destruction of a managed-record lineage (v1.6).

A tenant destroys a custody record and **every descendant** reached along the
completed-migration edges (``migration_sessions.source_custody_id ->
new_custody_id``, recursively). All of the following happens in **one
database transaction** while holding row locks that make the ordering versus
in-flight recovery/migration deterministic:

1. the root custody row is locked ``FOR UPDATE`` and the lineage is re-derived
   to a fixpoint (every completed edge on the frontier is locked as it is
   discovered, so a migration that commits concurrently — whether before or
   after — is either seen and its new child destroyed too, or blocks until this
   transaction commits and is then rejected; a child can never be left behind);
2. in-flight (open) recovery sessions on any lineage record are marked
   ``destroyed`` and open migration sessions sourced from any lineage record
   are marked ``destroyed`` — they can no longer issue claims/ciphertext or
   complete;
3. every still-live lineage record is turned into a **secret-free tombstone**:
   the custody row keeps its identity / policy / audit linkage but its secret
   digest and secret length are nulled, ``status='destroyed'``, and all online
   encrypted shard rows are deleted;
4. a replayable, plaintext-free ``record.destroy`` audit entry is appended per
   tombstoned record plus a ``destruction.complete`` summary;
5. a destruction receipt (ids/counts + idempotency-key digest only) is stored.

Only **after** the transaction commits do we remove the corresponding in-flight
sessions' in-process pending shard blobs — those blobs never outlive the
committed destruction on any serialization.

Retries carrying the same ``Idempotency-Key`` return the same receipt.
Cross-tenant requests are rejected. Destruction covers service online data
only — external backups are never claimed to be erased.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from .. import audit, crypto, deps, faults, repo
from ..errors import (
    ApiError,
    E_BAD_REQUEST,
    E_FORBIDDEN,
    E_IDEMPOTENCY_CONFLICT,
    E_NOT_FOUND,
)
from ..schemas import DestructionReceiptOut, DestructionRequestIn

router = APIRouter(tags=["destruction"])


def _receipt_out(row) -> DestructionReceiptOut:
    lineage = list(row["lineage_custody_ids"])
    destroyed = list(row["destroyed_custody_ids"])
    return DestructionReceiptOut(
        receipt_id=row["id"],
        root_custody_id=row["root_custody_id"],
        status=row["status"],
        destroyed_custody_ids=destroyed,
        lineage_custody_ids=lineage,
        lineage_size=len(lineage),
        destroyed_count=len(destroyed),
        shards_deleted=row["shards_deleted"],
        recoveries_blocked=row["recoveries_blocked"],
        migrations_blocked=row["migrations_blocked"],
        created_at=row["created_at"],
    )


@router.post("/destructions", response_model=DestructionReceiptOut, status_code=201)
async def destroy_lineage(
    body: DestructionRequestIn,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    tenant=deps.Tenant,
) -> DestructionReceiptOut:
    if not idempotency_key or not 1 <= len(idempotency_key) <= 200:
        raise ApiError(
            E_BAD_REQUEST,
            basis={"reason": "missing_or_malformed_idempotency_key",
                   "header": "Idempotency-Key", "max_length": 200},
        )
    idem_hash = crypto.sha256_hex(idempotency_key.encode())
    pool = request.app.state.db.pool
    pending = request.app.state.pending

    replayed = False
    receipt_row = None
    # 需要在提交后清理进程内待累计分片的在途会话（恢复 + 迁移）。
    blocked_session_ids: list[str] = []

    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                # 根记录行锁：所有“销毁 / 迁移创建（同样锁源 custody 行）”严格串行。
                root = await conn.fetchrow(
                    "SELECT * FROM key_custodies WHERE id = $1 FOR UPDATE", body.custody_id
                )
                await faults.maybe_park(request, "destruction.after_root_lock")
                if root is None:
                    raise ApiError(E_NOT_FOUND, basis={"custody_id": body.custody_id})
                if root["tenant_id"] != tenant["id"]:
                    raise ApiError(E_FORBIDDEN,
                                   basis={"reason": "custody_belongs_to_other_tenant"})

                # 幂等优先：同键（无论谱系当前状态）永远返回同一份回执。
                existing = await conn.fetchrow(
                    "SELECT * FROM destruction_receipts "
                    "WHERE tenant_id=$1 AND idempotency_key_hash=$2",
                    tenant["id"], idem_hash,
                )
                if existing is not None:
                    if existing["root_custody_id"] != body.custody_id:
                        raise ApiError(
                            E_IDEMPOTENCY_CONFLICT,
                            basis={"reason": "key_bound_to_different_request",
                                   "idempotency_key_digest": idem_hash},
                        )
                    receipt_row = existing
                    replayed = True
                else:
                    # 谱系递归到不动点，且**每轮都锁住与当前谱系相关的在途
                    # 会话行**后再重新递归——这是“销毁与迁移完成/分片领取或
                    # 提交并发时须确定先后”的关键：
                    #   * 迁移完成（resplit 同事务插入新后代）全程持有迁移行
                    #     锁；本事务在下面的 FOR UPDATE 处排队等待，其提交后
                    #     新边对本事务可见，下一轮递归必然把新后代纳入销毁；
                    #   * 反之本事务先拿到锁：在途迁移/恢复会话随后被同事务置
                    #     destroyed，旧会话拿锁后只能得到确定性拒绝，绝不可能
                    #     再完成（不会在销毁提交后漏删任何后代记录）。
                    lineage: list[str] = [root["id"]]
                    for _ in range(10_000):
                        # 锁住“源在当前谱系内”的全部迁移会话行（任意状态）：
                        # 含 open（在途）与已完成边（遍历需要）。
                        await conn.fetch(
                            "SELECT id FROM migration_sessions "
                            "WHERE source_custody_id = ANY($1::text[]) FOR UPDATE",
                            lineage,
                        )
                        # 同样锁住谱系内所有 open 恢复会话：门限完成路径持有
                        # 该行锁直至提交，故销毁与其恢复完成严格排序。
                        await conn.fetch(
                            "SELECT id FROM recovery_sessions "
                            "WHERE custody_id = ANY($1::text[]) AND status='open' FOR UPDATE",
                            lineage,
                        )
                        # 锁住已完成迁移边指向的子记录行（随递归逐层扩大）。
                        await conn.fetch(
                            """
                            SELECT kc.id FROM key_custodies kc
                            JOIN migration_sessions ms ON ms.new_custody_id = kc.id
                            WHERE ms.new_custody_id IS NOT NULL
                              AND ms.source_custody_id = ANY($1::text[])
                            FOR UPDATE OF kc
                            """,
                            lineage,
                        )
                        # 重新递归全集：等锁期间提交的迁移完成边此刻可见，
                        # 其新后代必被纳入。
                        rows = await conn.fetch(
                            """
                            WITH RECURSIVE closure AS (
                                SELECT id FROM key_custodies WHERE id = $1
                                UNION
                                SELECT kc.id
                                  FROM key_custodies kc
                                  JOIN migration_sessions ms ON ms.new_custody_id = kc.id
                                  JOIN closure l ON ms.source_custody_id = l.id
                                 WHERE ms.new_custody_id IS NOT NULL
                            )
                            SELECT kc.id
                              FROM key_custodies kc
                              JOIN closure cl ON cl.id = kc.id
                             ORDER BY kc.created_at NULLS LAST, kc.id
                            """,
                            root["id"],
                        )
                        nxt = [r["id"] for r in rows]
                        if len(nxt) == len(lineage):
                            lineage = nxt
                            break
                        lineage = nxt
                    else:  # pragma: no cover - 防御性上限
                        raise ApiError(E_BAD_REQUEST,
                                       basis={"reason": "lineage_too_large"})
                    # 回归闸门：谱系与全部在途会话行锁均已持有、阻断/墓碑化之前。
                    await faults.maybe_park(request, "destruction.after_lineage_locked")

                    # ① 阻断在途恢复：谱系内所有 open 恢复会话置 destroyed。
                    rec_rows = await conn.fetch(
                        "SELECT id FROM recovery_sessions "
                        "WHERE custody_id = ANY($1::text[]) AND status='open' FOR UPDATE",
                        lineage,
                    )
                    blocked_sessions_ids = [r["id"] for r in rec_rows]
                    await conn.execute(
                        "UPDATE recovery_sessions SET status='destroyed' "
                        "WHERE custody_id = ANY($1::text[]) AND status='open'",
                        lineage,
                    )

                    # ② 阻断在途迁移：源在谱系内的 open 迁移置 destroyed。
                    mig_rows = await conn.fetch(
                        "SELECT id FROM migration_sessions "
                        "WHERE source_custody_id = ANY($1::text[]) AND status='open' FOR UPDATE",
                        lineage,
                    )
                    blocked_mig_ids = [r["id"] for r in mig_rows]
                    await conn.execute(
                        "UPDATE migration_sessions SET status='destroyed' "
                        "WHERE source_custody_id = ANY($1::text[]) AND status='open'",
                        lineage,
                    )
                    blocked_session_ids = blocked_sessions_ids + blocked_mig_ids

                    # ③ 删除在线库里的加密分片（墓碑不再持有任何秘密载体）。
                    shards_deleted = await conn.fetchval(
                        "SELECT count(*) FROM shards WHERE custody_id = ANY($1::text[])",
                        lineage,
                    )
                    await conn.execute(
                        "DELETE FROM shards WHERE custody_id = ANY($1::text[])", lineage
                    )

                    # ④ 记录置墓碑：去秘密化（摘要/长度置 NULL），保留身份与
                    # 策略/审计关联；只翻转此前未销毁者（可能部分后代此前已销毁）。
                    destroyed_rows = await conn.fetch(
                        """
                        UPDATE key_custodies
                           SET status='destroyed', secret_digest=NULL,
                               secret_length=NULL, destroyed_at=now()
                         WHERE id = ANY($1::text[]) AND status <> 'destroyed'
                        RETURNING id, policy_id, policy_version
                        """,
                        lineage,
                    )
                    destroyed_ids = [r["id"] for r in destroyed_rows]

                    # ⑤ 可重放、无秘密的审计：每条墓碑一录 + 一条汇总。
                    for r in destroyed_rows:
                        await audit.append_audit(
                            conn, tenant_id=tenant["id"], actor_type="tenant",
                            actor_id=tenant["id"],
                            action="record.destroy", resource=r["id"], result="ok",
                            basis={
                                "policy_id": r["policy_id"],
                                "policy_version": r["policy_version"],
                                "root_custody_id": root["id"],
                            },
                        )
                    receipt_id = deps.new_id("dst")
                    # savepoint：同键并发（根记录不同、根行锁不互斥）时由
                    # uq_destruction_idem 唯一索引兜底，转确定性 409。
                    try:
                        async with conn.transaction():
                            await conn.execute(
                                """
                                INSERT INTO destruction_receipts
                                  (id, tenant_id, root_custody_id, idempotency_key_hash,
                                   destroyed_custody_ids, lineage_custody_ids, shards_deleted,
                                   recoveries_blocked, migrations_blocked, status)
                                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'completed')
                                """,
                                receipt_id, tenant["id"], root["id"], idem_hash,
                                destroyed_ids, lineage, shards_deleted,
                                len(blocked_sessions_ids), len(blocked_mig_ids),
                            )
                    except Exception as exc:
                        # 注意：savepoint 回滚后 asyncpg 重建的异常不再带 pgcode，
                        # 须按 sqlstate/异常类型判定（普通 execute 路径有 pgcode）。
                        code = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
                        if code != "23505":
                            raise
                        other = await conn.fetchrow(
                            "SELECT * FROM destruction_receipts "
                            "WHERE tenant_id=$1 AND idempotency_key_hash=$2",
                            tenant["id"], idem_hash,
                        )
                        if other is not None and other["root_custody_id"] != root["id"]:
                            raise ApiError(
                                E_IDEMPOTENCY_CONFLICT,
                                basis={"reason": "concurrent_destroy_with_same_idempotency_key",
                                       "idempotency_key_digest": idem_hash},
                            )
                        raise ApiError(
                            E_IDEMPOTENCY_CONFLICT,
                            basis={"reason": "idempotency_key_concurrent_reuse",
                                   "idempotency_key_digest": idem_hash},
                        )
                    await audit.append_audit(
                        conn, tenant_id=tenant["id"], actor_type="tenant",
                        actor_id=tenant["id"],
                        action="destruction.complete", resource=receipt_id, result="ok",
                        basis={
                            "root_custody_id": root["id"],
                            "lineage_size": len(lineage),
                            "destroyed_count": len(destroyed_ids),
                            "shards_deleted": shards_deleted,
                            "recoveries_blocked": len(blocked_sessions_ids),
                            "migrations_blocked": len(blocked_mig_ids),
                            "scope": "online_service_data_only",
                            "idempotency_key_digest": idem_hash,
                        },
                    )
                    receipt_row = await conn.fetchrow(
                        "SELECT * FROM destruction_receipts WHERE id=$1", receipt_id
                    )
        except ApiError as exc:
            # 拒绝同样留痕（独立事务），形成可重放审计；不含任何秘密材料。
            await repo.audit_denied(
                pool, tenant_id=tenant["id"], actor_type="tenant", actor_id=tenant["id"],
                action="destruction.start", resource=body.custody_id,
                error_code=exc.code, basis=exc.basis,
            )
            raise

    # 事务提交成功后再清理对应在途会话的进程内待累计分片：销毁已落盘，
    # 任何旧会话拿锁后只能读到 destroyed 终态，内存明文随之立即消失。
    for sid in blocked_session_ids:
        pending.clear(sid)

    if replayed:
        # 幂等重放返回同一回执（200，不重复执行、不重复留痕）。
        response.status_code = 200
    return _receipt_out(receipt_row)


@router.get("/destructions/{receipt_id}", response_model=DestructionReceiptOut)
async def get_receipt(receipt_id: str, request: Request, tenant=deps.Tenant) -> DestructionReceiptOut:
    pool = request.app.state.db.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM destruction_receipts WHERE id=$1", receipt_id
        )
        if row is None:
            raise ApiError(E_NOT_FOUND, basis={"receipt_id": receipt_id})
        if row["tenant_id"] != tenant["id"]:
            raise ApiError(E_FORBIDDEN, basis={"reason": "receipt_belongs_to_other_tenant"})
    return _receipt_out(row)
