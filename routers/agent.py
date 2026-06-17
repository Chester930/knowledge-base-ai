from fastapi import APIRouter
from models.document import AgentQueryRequest, AgentQueryResponse, AgentContext
from services.concept_engine import build_query_concepts, compute_match_score
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from core.database import get_driver
from core.config import settings

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(req: AgentQueryRequest):
    """
    Agent / RAG 查詢端點。

    輸入自然語言問題，回傳最相關的文件片段供 Agent 生成回答使用。
    Agent 應將 context 注入自身 prompt 後再生成最終回答。
    """
    query_concepts = await build_query_concepts(req.question)
    if not query_concepts:
        return AgentQueryResponse(question=req.question, context=[], sources=[])

    concept_repo = ConceptRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())
    all_doc_concepts = await concept_repo.get_all_documents_concepts()

    scored = []
    for doc_id, doc_concepts in all_doc_concepts.items():
        score, matched = compute_match_score(query_concepts, doc_concepts)
        if score >= settings.score_threshold * 0.5:  # agent 使用較寬鬆的閾值
            scored.append((doc_id, score, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:req.top_k]

    context_list = []
    sources = []

    for doc_id, score, matched in top:
        doc = await doc_repo.get_by_id(doc_id)
        if not doc:
            continue
        snippet = doc.content[:req.max_content_chars] if req.include_content else ""
        context_list.append(AgentContext(
            title=doc.title,
            content_snippet=snippet,
            score=score,
            file_path=doc.file_path,
        ))
        sources.append(doc.title)

    return AgentQueryResponse(
        question=req.question,
        context=context_list,
        sources=sources,
    )


@router.get("/health")
async def agent_health():
    """確認知識庫是否可用（供 Agent 在呼叫前 ping）。"""
    count = await DocumentRepository(get_driver()).get_count()
    return {
        "status": "ok",
        "document_count": count,
        "query_endpoint": "POST /agent/query",
        "usage_example": {
            "question": "你的問題",
            "top_k": 5,
            "include_content": True,
            "max_content_chars": 2000,
        },
    }
