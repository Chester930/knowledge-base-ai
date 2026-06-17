from __future__ import annotations
import logging
import httpx
from uuid import UUID
from core.config import settings
from core.constants import INTEREST_INIT, PROFESSIONAL_INIT
from repositories.concept_repo import ConceptRepository
from services.embedding_service import get_embedding_service
from core.database import get_driver

logger = logging.getLogger(__name__)


# ── LLM concept extraction ────────────────────────────────────────────────────

async def extract_concepts(text: str, domain: str = "general") -> list[str]:
    prompt = (
        f"請從以下文字中，提取最多 {settings.concept_extraction_max} 個核心概念。\n"
        f"每個概念用 2-6 個字表達，只回傳概念名稱，每行一個，不加序號或標點。\n\n"
        f"文字：\n{text[:3000]}"
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": settings.llm_model, "prompt": prompt, "stream": False},
            )
            res.raise_for_status()
            raw = res.json().get("response", "")
        concepts = [line.strip() for line in raw.splitlines() if line.strip()]
        return concepts[:settings.concept_extraction_max]
    except Exception as e:
        logger.warning(f"概念提取失敗：{e}")
        return []


# ── Match score ───────────────────────────────────────────────────────────────

def _alignment(i_a: float, p_a: float, i_b: float, p_b: float) -> float:
    return max(0.0, 1.0 - (abs(i_a - i_b) + abs(p_a - p_b)) / 2.0)


def _magnitude(i_a: float, p_a: float, i_b: float, p_b: float) -> float:
    return (i_a + p_a + i_b + p_b) / 4.0


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def compute_match_score(
    query_concepts: list[dict],
    doc_concepts: list[dict],
) -> tuple[float, list[str]]:
    if not query_concepts or not doc_concepts:
        return 0.0, []

    total_weight = 0.0
    weighted_score = 0.0
    matched: dict[str, float] = {}

    for qc in query_concepts:
        for dc in doc_concepts:
            cos = _cosine(qc["q_vector"], dc["q_vector"])
            if cos < 0.01:
                continue
            align = _alignment(
                qc["interest_score"], qc["professional_score"],
                dc["interest_score"], dc["professional_score"],
            )
            mag = _magnitude(
                qc["interest_score"], qc["professional_score"],
                dc["interest_score"], dc["professional_score"],
            )
            contribution = cos * align * mag
            total_weight += mag
            weighted_score += contribution
            if cos > 0.7:
                matched[dc["name"]] = max(matched.get(dc["name"], 0.0), cos)

    if total_weight < 1e-9:
        return 0.0, []

    score = weighted_score / total_weight
    top_concepts = sorted(matched, key=matched.get, reverse=True)[:5]
    return round(score, 4), top_concepts


# ── Document concept initialization ──────────────────────────────────────────

async def extract_and_init_document_concepts(
    doc_id: UUID, text: str, domain: str = "general"
) -> None:
    svc = get_embedding_service()
    repo = ConceptRepository(get_driver())

    concepts = await extract_concepts(text, domain)
    if not concepts:
        logger.warning(f"文件 {doc_id} 無法提取概念")
        return

    for name in concepts:
        try:
            vec = svc.encode(name)
            await repo.get_or_create(name, domain, vec)
            await repo.init_document_concept(doc_id, name, INTEREST_INIT, PROFESSIONAL_INIT)
        except Exception as e:
            logger.warning(f"概念初始化失敗 [{name}]: {e}")

    await repo.sync_document_effective(doc_id)
    logger.info(f"文件 {doc_id} 概念初始化完成，共 {len(concepts)} 個")


# ── Query concept extraction (temp, not stored) ───────────────────────────────

async def build_query_concepts(text: str) -> list[dict]:
    svc = get_embedding_service()
    names = await extract_concepts(text)
    if not names:
        names = [text[:50]]

    result = []
    for name in names:
        vec = svc.encode(name)
        result.append({
            "name": name,
            "q_vector": vec,
            "interest_score": 0.8,
            "professional_score": 0.8,
        })
    return result
