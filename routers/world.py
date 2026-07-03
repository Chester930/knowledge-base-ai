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
from services.concept_engine import build_query_concepts, compute_match_score, route_via_two_stage

router = APIRouter(prefix="/world", tags=["world"])
logger = logging.getLogger(__name__)

# ── GET /world/federation/status ──────────────────────────────────────────────

@router.get("/federation/status", summary="聯邦分片狀態（Phase 2b）")
async def federation_status():
    from services.federation_service import get_federation_cache
    return get_federation_cache().status()


# ── GET /world/federation/registry ───────────────────────────────────────────

@router.get("/federation/registry", summary="合併本機 + GitHub 遠端 registry（Phase 2b）")
async def federation_registry():
    from services.federation_service import get_federation_cache
    merged = await get_federation_cache().merged_registry()
    return merged.model_dump(exclude_none=True)


# ── POST /world/federation/refresh ───────────────────────────────────────────

@router.post("/federation/refresh", summary="強制重新下載 GitHub registry（Phase 2b）")
async def federation_refresh():
    from services.federation_service import get_federation_cache
    cache = get_federation_cache()
    cache._fetched_at = 0.0   # 使快取過期
    await cache.get_remote_registry()
    return cache.status()


# ── GET /world/provenance/facts ── Phase 3a ───────────────────────────────────

@router.get("/provenance/facts", summary="查詢事實的完整溯源（Phase 3a）")
async def provenance_facts(
    q: str = Query(..., description="查詢詞（逗號分隔多詞）"),
    kg_id: str = Query("", description="限定 KG 的 UUID（空白 = 所有公開 KG）"),
    hops: int = Query(1, ge=1, le=2),
    limit: int = Query(30, ge=1, le=100),
):
    """
    回傳與查詢詞相關的 SVO 事實，每條事實帶有：
    - 來源文件標題
    - 信心分數（同一對實體在幾份文件中被提及）
    - 建立時間
    """
    from services.svo_service import query_svo_facts_with_provenance
    from services.entity_alignment import expand_terms
    from models.provenance import ProvenanceReport

    terms = [t.strip() for t in q.split(",") if t.strip()]
    terms = await expand_terms(terms)  # Phase 2d 同義詞展開

    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)

    if kg_id:
        target_kgs = [kg for kg in [await kg_repo.get_by_id(UUID(kg_id))] if kg and kg.is_public]
    else:
        target_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    all_sourced = []
    for kg in target_kgs:
        sourced = await query_svo_facts_with_provenance(
            kg.id, terms, hops=hops, limit=limit,
            db_name=kg.db_name or None,
        )
        all_sourced.extend(sourced)

    # 組裝文件引用摘要
    from collections import Counter
    doc_counter: Counter = Counter()
    doc_titles: dict[str, str] = {}
    for f in all_sourced:
        if f.source_doc_id:
            doc_counter[f.source_doc_id] += 1
            doc_titles[f.source_doc_id] = f.source_doc_title

    citations = [
        {"doc_id": did, "title": doc_titles.get(did, ""), "fact_count": cnt}
        for did, cnt in doc_counter.most_common()
    ]

    report = ProvenanceReport(
        query_terms=terms,
        facts=sorted(all_sourced, key=lambda x: x.confidence, reverse=True)[:limit],
        doc_citations=citations,
    )
    return report.to_dict()


# ── GET /world/instance ───────────────────────────────────────────────────────

@router.get("/instance", summary="本實例的連線拓樸與子層資訊")
async def world_instance():
    """
    回傳 World Agent 查詢的資料來源拓樸：
    - 實例識別（instance_id）
    - Neo4j 連線類型與顯示名稱（本機 / AuraDB / 遠端）
    - 管道說明（資料如何進入這個 Neo4j）
    - 唯讀模式（World Agent 永遠只讀）
    - 雲端同步設定（是否已設定同步目標）
    - 子層（公開 KG 清單，含實體 / 關係數，不含文件數）
    """
    from core.config import settings as _s

    uri = _s.neo4j_uri
    if "aura" in uri or uri.startswith("neo4j+s://") or uri.startswith("neo4j+ssc://"):
        neo4j_type = "aura"
        neo4j_display = "Neo4j AuraDB（雲端）"
        pipeline = "雲端 AuraDB 直連（唯讀）"
    elif "localhost" in uri or "127.0.0.1" in uri or uri.startswith("bolt://neo4j"):
        neo4j_type = "local"
        neo4j_display = "本機 Neo4j"
        pipeline = "本地文件建圖 → 本機 Neo4j"
    else:
        neo4j_type = "remote"
        neo4j_display = "遠端 Neo4j"
        pipeline = f"遠端 Neo4j（{uri.split('@')[-1].split('/')[0]}）"

    sync_enabled = bool(_s.github_registry_url)
    sync_target = _s.github_registry_url if sync_enabled else None

    kg_repo = KnowledgeGraphRepository(get_driver())
    public_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    return {
        "instance_id": _s.instance_id,
        "neo4j_type": neo4j_type,
        "neo4j_display": neo4j_display,
        "pipeline": pipeline,
        "mode": "read-only",
        "sync": {
            "enabled": sync_enabled,
            "target": sync_target,
        },
        "sub_layers": [
            {
                "id": str(kg.id),
                "name": kg.name,
                "description": kg.description,
                "entity_count": kg.entity_count,
                "relation_count": kg.relation_count,
                "db_name": kg.db_name or "主資料庫",
            }
            for kg in public_kgs
        ],
    }


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
            public_kg_concepts = await route_via_two_stage(
                query_concepts, concept_repo.get_public_kgs_concepts,
            )

            kg_scores: list[tuple[UUID, float, list[str]]] = []
            for kg_id, kg_concepts in public_kg_concepts.items():
                score, matched = compute_match_score(query_concepts, kg_concepts)
                if score >= KG_ROUTE_THRESHOLD:
                    kg_scores.append((kg_id, score, matched))

            kg_scores.sort(key=lambda x: x[1], reverse=True)
            selected_kgs = kg_scores[:MAX_KG_PER_QUERY]

            # ── 本機路由結果整理 ──────────────────────────────────────────────
            selected_kg_objects: dict[UUID, object] = {}
            selected_local_ids: set[str] = set()
            for kg_id, score, matched in selected_kgs:
                kg = await kg_repo.get_by_id(kg_id)
                if kg and kg.is_public:
                    selected_kg_objects[kg_id] = kg
                    selected_local_ids.add(str(kg_id))

            # ── 組裝 KBSkill 列表（本機 + 遠端）────────────────────────────
            from services.federation_service import get_federation_cache
            from services.shard_query import query_shards_parallel, route_skill_score
            from models.kb_skill import KBSkill

            merged_reg = await get_federation_cache().merged_registry()
            query_terms = [c["name"] for c in query_concepts]
            local_score_map = {str(kg_id): sc for kg_id, sc, _ in selected_kgs}

            skills_to_query: list[KBSkill] = []
            kg_route_info = []

            for skill in merged_reg.skills:
                kb_str = skill.kb_id
                if skill.is_local:
                    if kb_str not in selected_local_ids:
                        continue
                    sc = local_score_map.get(kb_str, 0.0)
                    skills_to_query.append(skill)
                    kg_route_info.append({
                        "id": kb_str, "name": skill.name,
                        "score": round(sc, 3),
                        "instance_id": skill.instance_id,
                        "is_local": True,
                    })
                else:
                    # 遠端分片：用 top_concepts 關鍵字路由
                    remote_sc = route_skill_score(skill, query_terms)
                    if remote_sc >= KG_ROUTE_THRESHOLD:
                        skills_to_query.append(skill)
                        kg_route_info.append({
                            "id": kb_str, "name": skill.name,
                            "score": round(remote_sc, 3),
                            "instance_id": skill.instance_id,
                            "is_local": False,
                        })

            # 本機 KG 若不在 registry 中，補上臨時 KBSkill
            registry_local_ids = {s.kb_id for s in merged_reg.skills if s.is_local}
            for kg_id, sc, matched in selected_kgs:
                kg_str = str(kg_id)
                if kg_str not in registry_local_ids:
                    kg_obj = selected_kg_objects.get(kg_id)
                    tmp_skill = KBSkill(
                        instance_id="local",
                        kb_id=kg_str,
                        name=getattr(kg_obj, "name", kg_str),
                        last_sync="",
                        is_local=True,
                        db_name=getattr(kg_obj, "db_name", None),
                    )
                    skills_to_query.append(tmp_skill)
                    kg_route_info.append({
                        "id": kg_str, "name": tmp_skill.name,
                        "score": round(sc, 3),
                        "instance_id": "local",
                        "is_local": True,
                    })

            yield _sse({"kg_route": kg_route_info})

            # ── 並行查詢所有分片（Phase 2c）──────────────────────────────────
            svo_facts: list[str] = []
            graph_doc_ids: list[str] = []

            sourced_facts_all = []
            if use_svo and skills_to_query:
                merged_facts, merged_docs, shard_results, sourced_facts_all = \
                    await query_shards_parallel(
                        skills=skills_to_query,
                        terms=query_terms,
                        hops=svo_hops,
                        limit_per_shard=50,
                    )
                svo_facts = merged_facts
                graph_doc_ids = merged_docs

                shard_meta = [
                    {
                        "name": sr.shard_name,
                        "instance_id": sr.instance_id,
                        "status": sr.status,
                        "facts": len(sr.facts),
                        "elapsed_ms": sr.elapsed_ms,
                    }
                    for sr in shard_results
                ]
                shards_offline = [sr.shard_name for sr in shard_results if sr.status in ("timeout", "offline")]
                yield _sse({
                    "shard_meta": shard_meta,
                    "shards_queried": len(shard_results),
                    "shards_offline": shards_offline,
                })

                # Phase 3a：溯源事件（文件引用清單）
                if sourced_facts_all:
                    from collections import Counter
                    doc_cnt: Counter = Counter()
                    doc_ttl: dict[str, str] = {}
                    for sf in sourced_facts_all:
                        if sf.source_doc_id:
                            doc_cnt[sf.source_doc_id] += 1
                            doc_ttl[sf.source_doc_id] = sf.source_doc_title
                    citations = [
                        {"doc_id": did, "title": doc_ttl.get(did, ""), "fact_count": cnt}
                        for did, cnt in doc_cnt.most_common(10)
                    ]
                    if citations:
                        yield _sse({"provenance": citations})

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

            all_doc_concepts = await route_via_two_stage(
                query_concepts,
                lambda ids: concept_repo.get_all_documents_concepts(concept_ids=ids),
            )
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

            prompt = _build_world_prompt(question, svo_facts, contexts, sourced_facts_all)
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


def _build_world_prompt(
    question: str,
    svo_facts: list[str],
    contexts: list[dict],
    sourced_facts: list | None = None,
) -> str:
    parts = [
        "你是世界知識助手，能根據多個公開知識圖譜的事實與文件，提供客觀、全面的回答。\n"
        "【重要】無論問題使用何種語言，你的所有回覆必須使用正體（繁體）中文，不得使用簡體字。\n"
    ]

    # Phase 3a：優先使用帶引用格式的事實（cite_str）
    if sourced_facts:
        cited = [sf.cite_str() for sf in sourced_facts[:40]]
        facts_text = "\n".join(f"• {c}" for c in cited)
        parts.append(f"\n【知識圖譜事實（附來源引用）】\n{facts_text}\n")
    elif svo_facts:
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
        "- 若事實有標注來源（如「[來源：《文件名》]」），請在回答中提及出處\n"
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
    expand_synonyms: bool = Query(False, description="是否展開同義詞（Phase 2d）"),
):
    """
    在所有公開 KG 中搜尋實體。
    每個 KG 可能使用獨立的 Neo4j 資料庫（Enterprise mode），逐一查詢後合併。
    結果包含 instance_id（Phase 2d）。
    """
    from services.federation_service import get_federation_cache
    from core.config import settings as _settings

    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)
    public_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    # Phase 2d：建立 kg_id → instance_id 對照
    merged_reg = await get_federation_cache().merged_registry()
    kg_instance_map = {s.kb_id: s.instance_id for s in merged_reg.skills if s.is_local}

    per_kg = max(1, limit // max(len(public_kgs), 1))

    # Phase 2d：同義詞展開
    search_q = q.strip()
    search_terms: list[str] = []
    if search_q and expand_synonyms:
        from services.entity_alignment import expand_terms
        search_terms = await expand_terms([search_q])
    elif search_q:
        search_terms = [search_q]

    async def _query_kg(kg):
        try:
            db = kg.db_name or None
            instance_id = kg_instance_map.get(str(kg.id), _settings.instance_id)

            if search_terms:
                where = " OR ".join(
                    f"toLower(e.name) CONTAINS toLower($t{i})"
                    for i in range(len(search_terms))
                )
                params: dict = {f"t{i}": t for i, t in enumerate(search_terms)}
                params["lim"] = per_kg * 2
                if db:
                    cypher = (
                        f"MATCH (e:Entity) WHERE {where} "
                        "RETURN e.name AS name, e.type AS type, count{{(e)-[]->()}} AS deg "
                        "ORDER BY deg DESC LIMIT $lim"
                    )
                    result = await driver.execute_query(cypher, **params, database_=db)
                else:
                    params["kg_id"] = str(kg.id)
                    cypher = (
                        f"MATCH (e:Entity {{kg_id: $kg_id}}) WHERE {where} "
                        "RETURN e.name AS name, e.type AS type, 0 AS deg LIMIT $lim"
                    )
                    result = await driver.execute_query(cypher, **params)
            else:
                params = {"lim": per_kg * 2}
                if db:
                    cypher = (
                        "MATCH (e:Entity) WITH e, count{(e)-[]->()} AS deg "
                        "ORDER BY deg DESC LIMIT $lim "
                        "RETURN e.name AS name, e.type AS type, deg"
                    )
                    result = await driver.execute_query(cypher, **params, database_=db)
                else:
                    params["kg_id"] = str(kg.id)
                    cypher = (
                        "MATCH (e:Entity {kg_id: $kg_id}) "
                        "RETURN e.name AS name, e.type AS type, 0 AS deg LIMIT $lim"
                    )
                    result = await driver.execute_query(cypher, **params)

            return [
                {
                    "name": r["name"], "type": r["type"] or "Entity",
                    "kg_id": str(kg.id), "kg_name": kg.name,
                    "degree": r["deg"], "instance_id": instance_id,
                }
                for r in result.records
            ]
        except Exception as e:
            logger.warning(f"探索實體失敗 [{kg.name}]: {e}")
            return []

    results_nested = await asyncio.gather(*[_query_kg(kg) for kg in public_kgs])

    # 合併：同名實體保留最高 degree 的那筆
    seen: dict[str, dict] = {}
    for batch in results_nested:
        for item in batch:
            key = item["name"].lower()
            if key not in seen or item["degree"] > seen[key]["degree"]:
                seen[key] = item

    merged = sorted(seen.values(), key=lambda x: x["degree"], reverse=True)[:limit]
    return {
        "entities": merged,
        "total": len(merged),
        "query": q,
        "search_terms": search_terms or ([q] if q else []),
    }


# ── GET /world/align/synonyms ─────────────────────────────────────────────────

@router.get("/align/synonyms", summary="查詢術語的同義詞組（Phase 2d）")
async def get_synonyms(term: str = Query(..., description="術語（zh 或 en）")):
    """回傳 term 所屬的同義詞組，以及展開後的查詢詞清單。"""
    from services.entity_alignment import get_synonym_group, expand_terms
    group = get_synonym_group(term)
    expanded = await expand_terms([term])
    return {
        "term": term,
        "synonym_group": group,
        "expanded_query": expanded,
        "found": bool(group),
    }


# ── GET /world/align/entities ─────────────────────────────────────────────────

@router.get("/align/entities", summary="跨 instance 實體對齊查詢（Phase 2d）")
async def align_entities(
    name: str = Query(..., description="實體名稱（支援同義詞自動展開）"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    跨所有本機公開 KG 搜尋同名／同義詞實體，
    回傳 AlignedEntity 清單（每個實體保留所有 instance 來源）。
    """
    from services.entity_alignment import expand_terms, align_entity_results
    from services.federation_service import get_federation_cache
    from core.config import settings as _settings

    search_terms = await expand_terms([name])

    driver = get_driver()
    kg_repo = KnowledgeGraphRepository(driver)
    public_kgs = [kg for kg in await kg_repo.list_all(include_private=True) if kg.is_public]

    merged_reg = await get_federation_cache().merged_registry()
    kg_instance_map = {s.kb_id: s.instance_id for s in merged_reg.skills if s.is_local}

    per_kg = max(1, limit // max(len(public_kgs), 1))

    async def _search_kg(kg):
        try:
            db = kg.db_name or None
            instance_id = kg_instance_map.get(str(kg.id), _settings.instance_id)
            where = " OR ".join(
                f"toLower(e.name) CONTAINS toLower($t{i})"
                for i in range(len(search_terms))
            )
            params: dict = {f"t{i}": t for i, t in enumerate(search_terms)}
            params["lim"] = per_kg * 3

            if db:
                cypher = (
                    f"MATCH (e:Entity) WHERE {where} "
                    "RETURN e.name AS name, e.type AS type, count{(e)-[]->()} AS deg "
                    "ORDER BY deg DESC LIMIT $lim"
                )
                result = await driver.execute_query(cypher, **params, database_=db)
            else:
                params["kg_id"] = str(kg.id)
                cypher = (
                    f"MATCH (e:Entity {{kg_id: $kg_id}}) WHERE {where} "
                    "RETURN e.name AS name, e.type AS type, 0 AS deg LIMIT $lim"
                )
                result = await driver.execute_query(cypher, **params)

            return [
                {
                    "name": r["name"], "type": r["type"] or "Entity",
                    "kg_id": str(kg.id), "kg_name": kg.name,
                    "degree": r["deg"], "instance_id": instance_id,
                }
                for r in result.records
            ]
        except Exception as e:
            logger.warning(f"對齊搜尋失敗 [{kg.name}]: {e}")
            return []

    results_nested = await asyncio.gather(*[_search_kg(kg) for kg in public_kgs])
    all_entities = [item for batch in results_nested for item in batch]

    aligned = align_entity_results(all_entities)
    return {
        "query": name,
        "search_terms": search_terms,
        "aligned": [a.to_dict() for a in aligned[:limit]],
        "total": len(aligned),
        "cross_instance_count": sum(1 for a in aligned if a.instance_count > 1),
    }


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
        src_name = r["src"]
        dst_name = r["dst"]
        node_set[src_name] = {
            "id": src_name,
            "type": r["src_type"] or "Entity",
            "is_seed": (src_name == entity),
        }
        if dst_name not in node_set or dst_name == entity:
            node_set[dst_name] = {
                "id": dst_name,
                "type": r["dst_type"] or "Entity",
                "is_seed": (dst_name == entity),
            }
        edges.append({
            "src": src_name,
            "dst": dst_name,
            "rel_type": r["rel_type"],
            "verb": r["verb"] or r["rel_type"],
        })

    return {
        "nodes": list(node_set.values()),
        "edges": edges,
        "kg_name": kg.name,
        "seed_entity": entity,
    }
