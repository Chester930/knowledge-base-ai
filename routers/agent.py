from __future__ import annotations
import json
import logging
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.constants import KG_ROUTE_THRESHOLD, MAX_KG_PER_QUERY
from core.database import get_driver
from core.config import settings
from core.providers.factory import get_llm_provider
from models.document import AgentQueryRequest, AgentQueryResponse, AgentContext, ChatRequest
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from services.concept_engine import build_query_concepts, compute_match_score
from services.svo_service import query_svo_facts

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_rag_prompt(
    question: str,
    svo_facts: list[str],
    contexts: list[dict],
) -> str:
    parts: list[str] = [
        "你是一個知識庫助手，能根據提供的知識圖譜事實與文件內容精確回答問題。\n"
    ]

    if svo_facts:
        facts_text = "\n".join(f"• {f}" for f in svo_facts[:40])
        parts.append(f"\n【知識圖譜事實】\n{facts_text}\n")

    if contexts:
        docs = "".join(
            f"\n=== 文件 {i}：{c['title']} ===\n{c['content']}\n"
            for i, c in enumerate(contexts, 1)
        )
        parts.append(f"\n【相關文件】{docs}")

    parts.append(
        "\n---\n"
        f"請根據以上資訊，用繁體中文回答這個問題：\n{question}\n\n"
        "回答要求：\n"
        "- 優先引用知識圖譜事實與文件中的具體資訊\n"
        "- 若多份資訊都相關，請整合說明\n"
        "- 若資訊不足，請誠實說明並補充你的知識\n"
        "- 回答要清晰有條理"
    )
    return "".join(parts)


# ── /agent/chat（雙層路由）────────────────────────────────────────────────────

@router.post("/chat", summary="雙層路由 RAG 問答（SSE）")
async def chat(req: ChatRequest):
    """
    雙層路由 RAG 問答：
    1. ConceptNode 路由層 → 選出相關 KG
    2. SVO 知識層 → BFS 圖遍歷取得知識事實
    3. 文件層 → 取得相關文件原文片段
    4. 統合成 RAG prompt → LLM 串流回答

    SSE 事件序列：
      status:searching → kg_route → svo_facts → sources → status:generating → token... → done
    """

    async def generate():
        try:
            yield _sse({"status": "searching"})

            # ── Step 1：提取問題概念 ──────────────────────────────────────────
            query_concepts = await build_query_concepts(req.question)
            if not query_concepts:
                yield _sse({"error": "無法理解問題，請換個說法"})
                return

            concept_repo = ConceptRepository(get_driver())
            doc_repo = DocumentRepository(get_driver())
            kg_repo = KnowledgeGraphRepository(get_driver())

            # ── Step 2：KG 路由層 ─────────────────────────────────────────────
            all_kg_concepts = await concept_repo.get_all_kgs_concepts()

            kg_scores: list[tuple[UUID, float, list[str]]] = []
            for kg_id, kg_concepts in all_kg_concepts.items():
                score, matched = compute_match_score(query_concepts, kg_concepts)
                if score >= KG_ROUTE_THRESHOLD:
                    kg_scores.append((kg_id, score, matched))

            kg_scores.sort(key=lambda x: x[1], reverse=True)
            selected_kgs = kg_scores[:MAX_KG_PER_QUERY]

            # 推送 KG 路由結果，同時快取 KG 物件（含 db_name）
            kg_route_info = []
            selected_kg_objects: dict[UUID, object] = {}
            for kg_id, score, matched in selected_kgs:
                kg = await kg_repo.get_by_id(kg_id)
                if kg:
                    selected_kg_objects[kg_id] = kg
                    kg_route_info.append({
                        "id": str(kg_id),
                        "name": kg.name,
                        "score": round(score, 3),
                        "matched_concepts": matched[:5],
                    })
            yield _sse({"kg_route": kg_route_info})

            # ── Step 3：SVO 知識層（BFS 圖遍歷，同時收集來源文件）────────────
            svo_facts: list[str] = []
            graph_doc_ids: list[str] = []   # 圖譜指向的文件 ID

            if req.use_svo and selected_kgs:
                terms = [c["name"] for c in query_concepts]
                seen_facts: set[str] = set()
                seen_doc_ids: set[str] = set()
                for kg_id, _, _ in selected_kgs:
                    kg_obj = selected_kg_objects.get(kg_id)
                    db_name = getattr(kg_obj, "db_name", "") if kg_obj else ""
                    facts, src_ids = await query_svo_facts(
                        kg_id, terms, hops=req.svo_hops, limit=50, db_name=db_name
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

            # ── Step 4：文件層（圖譜驅動優先，無圖譜時 fallback 相似度搜尋）──
            from uuid import UUID as _UUID
            contexts: list[dict] = []
            sources: list[dict] = []

            if graph_doc_ids:
                # 圖譜驅動：只讀 SVO 指向的文件，不限數量
                for doc_id_str in graph_doc_ids:
                    try:
                        doc = await doc_repo.get_by_id(_UUID(doc_id_str))
                    except Exception:
                        continue
                    if not doc:
                        continue
                    snippet = (doc.content or "")[: req.max_chars_per_doc]
                    contexts.append({"title": doc.title, "content": snippet})
                    sources.append({"title": doc.title, "source": "graph"})
            else:
                # Fallback：SVO 圖譜尚未建立，改用概念相似度搜尋
                all_doc_concepts = await concept_repo.get_all_documents_concepts()
                allowed: set[str] = set()
                if selected_kgs:
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
                for doc_id, score, matched in scored_docs[:req.top_k]:
                    doc = await doc_repo.get_by_id(doc_id)
                    if not doc:
                        continue
                    snippet = (doc.content or "")[: req.max_chars_per_doc]
                    contexts.append({"title": doc.title, "content": snippet})
                    sources.append({"title": doc.title, "score": round(score, 3),
                                    "matched": matched, "source": "similarity"})

            yield _sse({"sources": sources})

            # ── Step 5：組 RAG prompt → LLM 串流 ─────────────────────────────
            if not svo_facts and not contexts:
                yield _sse({"error": "知識庫中沒有找到相關資訊，請先建立知識圖譜或匯入文件"})
                return

            prompt = _build_rag_prompt(req.question, svo_facts, contexts)
            yield _sse({"status": "generating"})

            async for token in get_llm_provider().stream(prompt):
                yield _sse({"token": token})

            yield _sse({"done": True})

        except Exception as e:
            logger.exception("chat 發生錯誤")
            yield _sse({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /agent/query（非串流，向下相容）──────────────────────────────────────────

@router.post("/query", response_model=AgentQueryResponse, summary="RAG 查詢（非串流）")
async def agent_query(req: AgentQueryRequest):
    """非串流版本，回傳最相關文件片段，供程式整合使用。"""
    query_concepts = await build_query_concepts(req.question)
    if not query_concepts:
        return AgentQueryResponse(question=req.question, context=[], sources=[])

    concept_repo = ConceptRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())
    all_doc_concepts = await concept_repo.get_all_documents_concepts()

    scored = []
    for doc_id, doc_concepts in all_doc_concepts.items():
        score, matched = compute_match_score(query_concepts, doc_concepts)
        if score >= settings.score_threshold * 0.5:
            scored.append((doc_id, score, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: req.top_k]

    context_list, sources = [], []
    for doc_id, score, matched in top:
        doc = await doc_repo.get_by_id(doc_id)
        if not doc:
            continue
        snippet = doc.content[: req.max_content_chars] if req.include_content else ""
        context_list.append(
            AgentContext(
                title=doc.title, content_snippet=snippet,
                score=score, file_path=doc.file_path,
            )
        )
        sources.append(doc.title)

    return AgentQueryResponse(
        question=req.question, context=context_list, sources=sources
    )


# ── /agent/health ─────────────────────────────────────────────────────────────

@router.get("/health", summary="Agent 健康狀態")
async def agent_health():
    doc_count = await DocumentRepository(get_driver()).get_count()
    kg_count_result = await get_driver().execute_query(
        "MATCH (kg:KnowledgeGraph) RETURN count(kg) AS cnt"
    )
    entity_count_result = await get_driver().execute_query(
        "MATCH (e:Entity) RETURN count(e) AS cnt"
    )
    return {
        "status": "ok",
        "document_count": doc_count,
        "kg_count": kg_count_result.records[0]["cnt"],
        "entity_count": entity_count_result.records[0]["cnt"],
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "dual_layer_routing": True,
    }
