"""
World Knowledge Router — 公開知識庫聚合層

提供以下能力：
- GET  /world/knowledge-graphs      列出所有公開 KG
- POST /world/chat                  World Agent 問答（只查公開 KG，SSE）
- GET  /world/explore/entities      跨公開 KG 實體搜尋
- GET  /world/explore/neighbors     實體鄰居查詢（圖探索）
- GET  /world/stats                 公開知識庫整體統計
"""
from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from core.constants import KG_ROUTE_THRESHOLD, MAX_KG_PER_QUERY
from core.database import get_driver
from core.providers.factory import get_llm_provider
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from services.concept_engine import build_query_concepts, compute_match_score
from services.svo_service import query_svo_facts

router = APIRouter(prefix="/world", tags=["world"])
logger = logging.getLogger(__name__)

# ── GET /world/registry ───────────────────────────────────────────────────────

@router.get("/registry", summary="取得本機 KB Registry（供 world-hub 讀取）")
async def get_registry():
    from services.kb_skill_service import load_registry
    registry = load_registry()
    return registry.model_dump(exclude_none=True)


# ── POST /world/sync ──────────────────────────────────────────────────────────

@router.post("/sync", summary="同步所有公開 KG 到 registry.json")
async def sync_registry():
    from services.kb_skill_service import sync_public_kgs
    result = await sync_public_kgs(get_driver())
    return {"status": "ok", **result}

_ALL_REL = (
    "IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
    "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
    "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
    "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO"
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── GET /world/knowledge-graphs ───────────────────────────────────────────────

@router.get("/knowledge-graphs", summary="列出所有公開知識庫")
async def list_public_kgs():
    kg_repo = KnowledgeGraphRepository(get_driver())
    kgs = await kg_repo.list_all(include_private=False)
    return [
        {
            "id": str(kg.id),
            "name": kg.name,
            "description": kg.description,
            "doc_count": kg.doc_count,
            "entity_count": kg.entity_count,
            "relation_count": kg.relation_count,
            "is_public": kg.is_public,
        }
        for kg in kgs
        if kg.is_public
    ]


# ── GET /world/stats ──────────────────────────────────────────────────────────

@router.get("/stats", summary="公開知識庫整體統計")
async def world_stats():
    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)
    public_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    total_entities = sum(kg.entity_count for kg in public_kgs)
    total_relations = sum(kg.relation_count for kg in public_kgs)
    total_docs = sum(kg.doc_count for kg in public_kgs)

    return {
        "public_kg_count": len(public_kgs),
        "total_entities": total_entities,
        "total_relations": total_relations,
        "total_docs": total_docs,
        "public_kgs": [
            {"id": str(kg.id), "name": kg.name, "entity_count": kg.entity_count}
            for kg in public_kgs
        ],
    }


# ── POST /world/chat ──────────────────────────────────────────────────────────

@router.post("/chat", summary="World Agent 問答（只查公開 KG，SSE 串流）")
async def world_chat(req: dict):
    """
    與 /agent/chat 邏輯相同，但：
    - 路由層只考慮 is_public=true 的 KG
    - Prompt 聲明「世界知識助手」角色
    """
    question = req.get("question", "").strip()
    top_k = int(req.get("top_k", 3))
    max_chars = int(req.get("max_chars_per_doc", 1500))
    use_svo = req.get("use_svo", True)
    svo_hops = int(req.get("svo_hops", 1))

    async def generate():
        try:
            if not question:
                yield _sse({"error": "請輸入問題"})
                return

            yield _sse({"status": "searching"})

            query_concepts = await build_query_concepts(question)
            if not query_concepts:
                yield _sse({"error": "無法理解問題，請換個說法"})
                return

            driver = get_driver()
            concept_repo = ConceptRepository(driver)
            doc_repo = DocumentRepository(driver)
            kg_repo = KnowledgeGraphRepository(driver)

            # 只取公開 KG 的概念
            public_kg_concepts = await concept_repo.get_public_kgs_concepts()

            kg_scores: list[tuple[UUID, float, list[str]]] = []
            for kg_id, kg_concepts in public_kg_concepts.items():
                score, matched = compute_match_score(query_concepts, kg_concepts)
                if score >= KG_ROUTE_THRESHOLD:
                    kg_scores.append((kg_id, score, matched))

            kg_scores.sort(key=lambda x: x[1], reverse=True)
            selected_kgs = kg_scores[:MAX_KG_PER_QUERY]

            kg_route_info = []
            selected_kg_objects: dict[UUID, object] = {}
            for kg_id, score, matched in selected_kgs:
                kg = await kg_repo.get_by_id(kg_id)
                if kg and kg.is_public:
                    selected_kg_objects[kg_id] = kg
                    kg_route_info.append({
                        "id": str(kg_id),
                        "name": kg.name,
                        "score": round(score, 3),
                        "matched_concepts": matched[:5],
                        "is_public": True,
                    })
            yield _sse({"kg_route": kg_route_info})

            svo_facts: list[str] = []
            graph_doc_ids: list[str] = []

            if use_svo and selected_kgs:
                terms = [c["name"] for c in query_concepts]
                seen_facts: set[str] = set()
                seen_doc_ids: set[str] = set()
                for kg_id, _, _ in selected_kgs:
                    kg_obj = selected_kg_objects.get(kg_id)
                    db_name = getattr(kg_obj, "db_name", "") if kg_obj else ""
                    facts, src_ids = await query_svo_facts(
                        kg_id, terms, hops=svo_hops, limit=50, db_name=db_name
                    )
                    for f in facts:
                        if f not in seen_facts:
                            seen_facts.add(f)
                            svo_facts.append(f)
                    for doc_id in src_ids:
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            graph_doc_ids.append(doc_id)

            if svo_facts:
                yield _sse({"svo_facts": svo_facts})

            contexts: list[dict] = []
            sources: list[dict] = []
            seen_doc_ids_ctx: set[str] = set()

            for doc_id_str in graph_doc_ids:
                try:
                    doc = await doc_repo.get_by_id(UUID(doc_id_str))
                except Exception:
                    continue
                if not doc:
                    continue
                contexts.append({"title": doc.title, "content": (doc.content or "")[:max_chars]})
                sources.append({"title": doc.title, "source": "graph"})
                seen_doc_ids_ctx.add(doc_id_str)

            all_doc_concepts = await concept_repo.get_all_documents_concepts()
            allowed: set[str] = set()
            for kg_id, _, _ in selected_kgs:
                for d in await kg_repo.get_documents(kg_id):
                    allowed.add(str(d["id"]))

            scored_docs = []
            for doc_id, dc in all_doc_concepts.items():
                if allowed and str(doc_id) not in allowed:
                    continue
                score, matched = compute_match_score(query_concepts, dc)
                if score > 0:
                    scored_docs.append((doc_id, score, matched))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            sim_added = 0
            for doc_id, score, matched in scored_docs:
                if sim_added >= top_k:
                    break
                if str(doc_id) in seen_doc_ids_ctx:
                    continue
                doc = await doc_repo.get_by_id(doc_id)
                if not doc:
                    continue
                contexts.append({"title": doc.title, "content": (doc.content or "")[:max_chars]})
                sources.append({"title": doc.title, "score": round(score, 3), "source": "similarity"})
                seen_doc_ids_ctx.add(str(doc_id))
                sim_added += 1

            yield _sse({"sources": sources})

            if not svo_facts and not contexts:
                yield _sse({"error": "公開知識庫中沒有找到相關資訊"})
                return

            prompt = _build_world_prompt(question, svo_facts, contexts)
            yield _sse({"status": "generating"})

            async for token in get_llm_provider().stream(prompt):
                yield _sse({"token": token})

            yield _sse({"done": True})

        except Exception as e:
            logger.exception("world/chat 發生錯誤")
            yield _sse({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_world_prompt(question: str, svo_facts: list[str], contexts: list[dict]) -> str:
    parts = [
        "你是世界知識助手，能根據多個公開知識圖譜的事實與文件，提供客觀、全面的回答。\n"
        "【重要】無論問題使用何種語言，你的所有回覆必須使用正體（繁體）中文，不得使用簡體字。\n"
    ]
    if svo_facts:
        facts_text = "\n".join(f"• {f}" for f in svo_facts[:40])
        parts.append(f"\n【知識圖譜事實（來自公開知識庫）】\n{facts_text}\n")
    if contexts:
        docs = "".join(
            f"\n=== 文件 {i}：{c['title']} ===\n{c['content']}\n"
            for i, c in enumerate(contexts, 1)
        )
        parts.append(f"\n【相關文件】{docs}")
    parts.append(
        "\n---\n"
        f"請根據以上公開知識，用繁體中文回答：\n{question}\n\n"
        "回答要求：\n"
        "- 優先引用知識圖譜事實中的具體資訊\n"
        "- 若多份知識庫都有相關資訊，請整合說明\n"
        "- 若資訊不足，請誠實說明\n"
        "- 回答要清晰有條理"
    )
    return "".join(parts)


# ── GET /world/explore/entities ───────────────────────────────────────────────

@router.get("/explore/entities", summary="跨公開 KG 實體搜尋")
async def explore_entities(
    q: str = Query("", description="實體名稱關鍵字（空白回傳高頻實體）"),
    limit: int = Query(30, ge=1, le=100),
):
    """
    在所有公開 KG 中搜尋實體。
    每個 KG 可能使用獨立的 Neo4j 資料庫（Enterprise mode），逐一查詢後合併。
    """
    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)
    public_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    per_kg = max(1, limit // max(len(public_kgs), 1))

    async def _query_kg(kg):
        try:
            db = kg.db_name or None
            if q.strip():
                cypher = (
                    "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) "
                    "RETURN e.name AS name, e.type AS type, count{(e)-[]->()} AS deg "
                    "ORDER BY deg DESC LIMIT $lim"
                )
                params = {"q": q.strip(), "lim": per_kg * 2}
            else:
                cypher = (
                    "MATCH (e:Entity) WITH e, count{(e)-[]->()} AS deg "
                    "ORDER BY deg DESC LIMIT $lim "
                    "RETURN e.name AS name, e.type AS type, deg"
                )
                params = {"lim": per_kg * 2}

            if db:
                result = await driver.execute_query(cypher, **params, database_=db)
            else:
                # Community fallback: filter by kg_id
                if q.strip():
                    cypher_c = (
                        "MATCH (e:Entity {kg_id: $kg_id}) WHERE toLower(e.name) CONTAINS toLower($q) "
                        "RETURN e.name AS name, e.type AS type, 0 AS deg LIMIT $lim"
                    )
                else:
                    cypher_c = (
                        "MATCH (e:Entity {kg_id: $kg_id}) RETURN e.name AS name, e.type AS type, 0 AS deg LIMIT $lim"
                    )
                result = await driver.execute_query(cypher_c, kg_id=str(kg.id), **params)

            return [
                {"name": r["name"], "type": r["type"] or "Entity",
                 "kg_id": str(kg.id), "kg_name": kg.name, "degree": r["deg"]}
                for r in result.records
            ]
        except Exception as e:
            logger.warning(f"探索實體失敗 [{kg.name}]: {e}")
            return []

    results_nested = await asyncio.gather(*[_query_kg(kg) for kg in public_kgs])

    # 合併、去重（同名實體保留最高 degree 的那筆），排序
    seen: dict[str, dict] = {}
    for batch in results_nested:
        for item in batch:
            key = item["name"].lower()
            if key not in seen or item["degree"] > seen[key]["degree"]:
                seen[key] = item

    merged = sorted(seen.values(), key=lambda x: x["degree"], reverse=True)[:limit]
    return {"entities": merged, "total": len(merged), "query": q}


# ── GET /world/explore/neighbors ──────────────────────────────────────────────

@router.get("/explore/neighbors", summary="實體鄰居查詢（圖探索）")
async def explore_neighbors(
    entity: str = Query(..., description="實體名稱"),
    kg_id: str = Query(..., description="所屬 KG 的 UUID"),
    limit: int = Query(30, ge=1, le=100),
):
    """
    取得指定實體在其 KG 中的直接鄰居（1 跳），供互動式圖探索使用。
    回傳：{ nodes: [...], edges: [...] }
    """
    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)
    kg = await kg_repo.get_by_id(UUID(kg_id))
    if not kg or not kg.is_public:
        return {"nodes": [], "edges": [], "error": "KG 不存在或非公開"}

    db = kg.db_name or None

    cypher = (
        f"MATCH (e:Entity {{name: $name}})-[r:{_ALL_REL}]-(n:Entity) "
        "RETURN e.name AS src, e.type AS src_type, "
        "type(r) AS rel_type, properties(r).verb AS verb, "
        "n.name AS dst, n.type AS dst_type "
        f"LIMIT $lim"
    )

    try:
        if db:
            result = await driver.execute_query(cypher, name=entity, lim=limit, database_=db)
        else:
            cypher_c = (
                f"MATCH (e:Entity {{name: $name, kg_id: $kg_id}})-[r:{_ALL_REL}]-(n:Entity) "
                "RETURN e.name AS src, e.type AS src_type, "
                "type(r) AS rel_type, properties(r).verb AS verb, "
                "n.name AS dst, n.type AS dst_type "
                f"LIMIT $lim"
            )
            result = await driver.execute_query(cypher_c, name=entity, kg_id=kg_id, lim=limit)
    except Exception as e:
        logger.warning(f"探索鄰居失敗 [{entity}]: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}

    node_set: dict[str, dict] = {}
    edges: list[dict] = []

    for r in result.records:
        node_set[r["src"]] = {"id": r["src"], "type": r["src_type"] or "Entity", "is_seed": True}
        node_set[r["dst"]] = {"id": r["dst"], "type": r["dst_type"] or "Entity", "is_seed": False}
        edges.append({
            "src": r["src"], "dst": r["dst"],
            "rel_type": r["rel_type"],
            "verb": r["verb"] or r["rel_type"],
        })

    return {
        "nodes": list(node_set.values()),
        "edges": edges,
        "kg_name": kg.name,
        "seed_entity": entity,
    }
