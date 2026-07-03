from fastapi import APIRouter
from models.document import SearchRequest, SearchResult
from services.concept_engine import build_query_concepts, compute_match_score, route_via_two_stage
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from core.database import get_driver

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=list[SearchResult])
async def search(req: SearchRequest):
    query_concepts = await build_query_concepts(req.text)
    if not query_concepts:
        return []

    concept_repo = ConceptRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())

    all_doc_concepts = await route_via_two_stage(
        query_concepts,
        lambda ids: concept_repo.get_all_documents_concepts(concept_ids=ids),
    )
    results = []

    for doc_id, doc_concepts in all_doc_concepts.items():
        score, matched = compute_match_score(query_concepts, doc_concepts)
        if score < req.min_score:
            continue
        doc = await doc_repo.get_by_id(doc_id)
        if doc:
            results.append(SearchResult(document=doc, score=score, matched_concepts=matched))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:req.top_k]


@router.get("/concepts")
async def list_concepts():
    return await ConceptRepository(get_driver()).get_all_concepts()
