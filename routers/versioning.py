"""
KG Version Control — Phase 3c

三個版本查詢端點：
- GET /kg/{kg_id}/changelog  — 近期 SVO 變更（updated_at 降序）
- GET /kg/{kg_id}/diff       — 指定時間點之後的所有變更
- GET /kg/{kg_id}/snapshot   — 某時間點的知識快照（created_at <= at）
"""
from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from core.database import get_driver
from repositories.knowledge_graph_repo import KnowledgeGraphRepository

router = APIRouter(tags=["versioning"])
logger = logging.getLogger(__name__)

_ALL_REL = (
    "IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
    "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
    "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
    "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO"
)


async def _get_kg_or_404(kg_id: str):
    repo = KnowledgeGraphRepository(get_driver())
    kg = await repo.get_by_id(UUID(kg_id))
    if not kg:
        raise HTTPException(status_code=404, detail="KG 不存在")
    return kg


def _fact_row(r) -> dict:
    updated = r.get("updated_at") or ""
    created = r.get("created_at") or ""
    return {
        "subject": r["subject"],
        "subject_type": r.get("subject_type") or "",
        "rel_type": r["rel_type"],
        "verb": r.get("verb") or r["rel_type"],
        "object": r["object"],
        "object_type": r.get("object_type") or "",
        "confidence": r.get("confidence") or 1,
        "source_doc_id": r.get("source_doc_id") or "",
        "created_at": created,
        "updated_at": updated,
        "change_type": "created" if not updated else "updated",
    }


# ── GET /kg/{kg_id}/changelog ─────────────────────────────────────────────────

@router.get("/kg/{kg_id}/changelog", summary="KG 近期 SVO 變更記錄（Phase 3c）")
async def kg_changelog(
    kg_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """列出 KG 中近期新增或修改的 SVO 事實，依 updated_at / created_at 降序。"""
    kg = await _get_kg_or_404(kg_id)
    driver = get_driver()

    base_return = """
        s.name AS subject, s.type AS subject_type,
        type(r) AS rel_type, r.verb AS verb,
        o.name AS object, o.type AS object_type,
        r.confidence AS confidence, r.source_doc_id AS source_doc_id,
        toString(r.created_at) AS created_at,
        toString(r.updated_at) AS updated_at
    """

    if kg.db_name:
        cypher = f"""
        MATCH (s:Entity)-[r:{_ALL_REL}]->(o:Entity)
        WITH s, r, o,
             CASE WHEN r.updated_at IS NOT NULL THEN r.updated_at ELSE r.created_at END AS ts
        RETURN {base_return}
        ORDER BY ts DESC
        SKIP $offset LIMIT $limit
        """
        result = await driver.execute_query(
            cypher, offset=offset, limit=limit, database_=kg.db_name
        )
    else:
        cypher = f"""
        MATCH (s:Entity {{kg_id: $kg_id}})-[r:{_ALL_REL}]->(o:Entity)
        WITH s, r, o,
             CASE WHEN r.updated_at IS NOT NULL THEN r.updated_at ELSE r.created_at END AS ts
        RETURN {base_return}
        ORDER BY ts DESC
        SKIP $offset LIMIT $limit
        """
        result = await driver.execute_query(cypher, kg_id=kg_id, offset=offset, limit=limit)

    return {
        "kg_id": kg_id,
        "kg_name": kg.name,
        "offset": offset,
        "limit": limit,
        "facts": [_fact_row(r) for r in result.records],
    }


# ── GET /kg/{kg_id}/diff ──────────────────────────────────────────────────────

@router.get("/kg/{kg_id}/diff", summary="KG 時間點差異（Phase 3c）")
async def kg_diff(
    kg_id: str,
    since: str = Query(..., description="ISO 8601，如 2026-06-21T00:00:00"),
    limit: int = Query(200, ge=1, le=1000),
):
    """回傳 since 之後新增或修改的所有 SVO 事實。"""
    kg = await _get_kg_or_404(kg_id)
    driver = get_driver()

    base_return = """
        s.name AS subject, s.type AS subject_type,
        type(r) AS rel_type, r.verb AS verb,
        o.name AS object, o.type AS object_type,
        r.confidence AS confidence, r.source_doc_id AS source_doc_id,
        toString(r.created_at) AS created_at,
        toString(r.updated_at) AS updated_at
    """
    order = "ORDER BY CASE WHEN r.updated_at IS NOT NULL THEN r.updated_at ELSE r.created_at END DESC"

    if kg.db_name:
        cypher = f"""
        MATCH (s:Entity)-[r:{_ALL_REL}]->(o:Entity)
        WHERE r.created_at >= datetime($since) OR r.updated_at >= datetime($since)
        RETURN {base_return}
        {order} LIMIT $limit
        """
        result = await driver.execute_query(
            cypher, since=since, limit=limit, database_=kg.db_name
        )
    else:
        cypher = f"""
        MATCH (s:Entity {{kg_id: $kg_id}})-[r:{_ALL_REL}]->(o:Entity)
        WHERE r.created_at >= datetime($since) OR r.updated_at >= datetime($since)
        RETURN {base_return}
        {order} LIMIT $limit
        """
        result = await driver.execute_query(cypher, kg_id=kg_id, since=since, limit=limit)

    facts = [_fact_row(r) for r in result.records]
    return {
        "kg_id": kg_id,
        "kg_name": kg.name,
        "since": since,
        "fact_count": len(facts),
        "created_count": sum(1 for f in facts if f["change_type"] == "created"),
        "updated_count": sum(1 for f in facts if f["change_type"] == "updated"),
        "facts": facts,
    }


# ── GET /kg/{kg_id}/snapshot ──────────────────────────────────────────────────

@router.get("/kg/{kg_id}/snapshot", summary="KG 歷史快照（Phase 3c）")
async def kg_snapshot(
    kg_id: str,
    at: str = Query(..., description="ISO 8601，如 2026-06-21T12:00:00"),
    limit: int = Query(500, ge=1, le=2000),
):
    """回傳指定時間點時 KG 中存在的所有 SVO 事實（created_at ≤ at）。"""
    kg = await _get_kg_or_404(kg_id)
    driver = get_driver()

    base_return = """
        s.name AS subject, s.type AS subject_type,
        type(r) AS rel_type, r.verb AS verb,
        o.name AS object, o.type AS object_type,
        r.confidence AS confidence, r.source_doc_id AS source_doc_id,
        toString(r.created_at) AS created_at,
        toString(r.updated_at) AS updated_at
    """

    if kg.db_name:
        cypher = f"""
        MATCH (s:Entity)-[r:{_ALL_REL}]->(o:Entity)
        WHERE r.created_at <= datetime($at)
        RETURN {base_return}
        ORDER BY r.created_at ASC LIMIT $limit
        """
        result = await driver.execute_query(
            cypher, at=at, limit=limit, database_=kg.db_name
        )
    else:
        cypher = f"""
        MATCH (s:Entity {{kg_id: $kg_id}})-[r:{_ALL_REL}]->(o:Entity)
        WHERE r.created_at <= datetime($at)
        RETURN {base_return}
        ORDER BY r.created_at ASC LIMIT $limit
        """
        result = await driver.execute_query(cypher, kg_id=kg_id, at=at, limit=limit)

    return {
        "kg_id": kg_id,
        "kg_name": kg.name,
        "snapshot_at": at,
        "fact_count": len(result.records),
        "facts": [_fact_row(r) for r in result.records],
    }
