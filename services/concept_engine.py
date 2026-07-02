from __future__ import annotations
import hashlib
import logging
from collections import OrderedDict
from uuid import UUID

from core.config import settings
from core.constants import CONCEPT_COARSE_TOP_K, INTEREST_INIT, PROFESSIONAL_INIT
from core.providers.factory import get_embedding_provider, get_llm_provider
from repositories.concept_repo import ConceptRepository
from core.database import get_driver

logger = logging.getLogger(__name__)

# 概念提取 LRU 快取（☆5 優化）—— 避免相同文字重複呼叫 LLM
_CONCEPT_CACHE_MAX = 256
_concept_cache: OrderedDict[str, list[str]] = OrderedDict()


def _concept_cache_key(text: str, domain: str) -> str:
    return hashlib.md5(f"{domain}:{text[:3000]}".encode()).hexdigest()


def _concept_cache_get(key: str) -> list[str] | None:
    if key in _concept_cache:
        _concept_cache.move_to_end(key)
        return _concept_cache[key]
    return None


def _concept_cache_set(key: str, value: list[str]) -> None:
    if key in _concept_cache:
        _concept_cache.move_to_end(key)
    else:
        if len(_concept_cache) >= _CONCEPT_CACHE_MAX:
            _concept_cache.popitem(last=False)
    _concept_cache[key] = value


# ── LLM concept extraction ────────────────────────────────────────────────────

async def extract_concepts(text: str, domain: str = "general") -> list[str]:
    cache_key = _concept_cache_key(text, domain)
    cached = _concept_cache_get(cache_key)
    if cached is not None:
        return cached

    prompt = (
        f"請從以下文字中，提取最多 {settings.concept_extraction_max} 個核心概念。\n"
        f"每個概念用 2-6 個字表達，只回傳概念名稱，每行一個，不加序號或標點。\n\n"
        f"文字：\n{text[:3000]}"
    )
    import asyncio as _asyncio
    _MAX_RETRIES = 2
    for attempt in range(1 + _MAX_RETRIES):
        try:
            raw = await get_llm_provider().generate(prompt)
            import re as _re
            concepts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 去除常見序號，如: "1. ", "1、", "- ", "* ", "(1) " 等
                line = _re.sub(r'^(?:(?:\d+[\.、\s\)])|[-*•·])\s*', '', line).strip()
                # 去除前後可能的引號
                line = line.strip('\'"`“”‘’')
                if line:
                    concepts.append(line)
            result = concepts[:settings.concept_extraction_max]
            _concept_cache_set(cache_key, result)
            return result
        except Exception as e:
            if attempt < _MAX_RETRIES:
                wait_s = 2 ** attempt
                logger.warning(f"概念提取失敗，第 {attempt+1} 次重試 (等 {wait_s} 秒)：{e}")
                await _asyncio.sleep(wait_s)
            else:
                logger.exception(f"概念提取最終失敗：{e}")
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


# ── 兩階段路由查詢（Vector Index 粗篩 + Python 精篩，索引不可用時自動 fallback）───

async def route_kgs(
    concept_repo: ConceptRepository,
    query_concepts: list[dict],
    top_k: int = CONCEPT_COARSE_TOP_K,
    public_only: bool = False,
) -> dict[UUID, list[dict]]:
    vectors = [qc["q_vector"] for qc in query_concepts]
    if public_only:
        result = await concept_repo.get_public_kgs_concepts_for_query(vectors, top_k)
        if result is None:
            result = await concept_repo.get_public_kgs_concepts()
    else:
        result = await concept_repo.get_kgs_concepts_for_query(vectors, top_k)
        if result is None:
            result = await concept_repo.get_all_kgs_concepts()
    return result


async def route_documents(
    concept_repo: ConceptRepository,
    query_concepts: list[dict],
    top_k: int = CONCEPT_COARSE_TOP_K,
    exclude_doc_ids: list[UUID] | None = None,
) -> dict[UUID, list[dict]]:
    vectors = [qc["q_vector"] for qc in query_concepts]
    result = await concept_repo.get_documents_concepts_for_query(vectors, top_k, exclude_doc_ids)
    if result is None:
        result = await concept_repo.get_all_documents_concepts(exclude_doc_ids)
    return result


# ── Document concept initialization ──────────────────────────────────────────

async def extract_and_init_document_concepts(
    doc_id: UUID, text: str, domain: str = "general"
) -> None:
    embedding = get_embedding_provider()
    repo = ConceptRepository(get_driver())

    concepts = await extract_concepts(text, domain)
    if not concepts:
        logger.warning(f"文件 {doc_id} 無法提取概念")
        return

    for name in concepts:
        try:
            vec = embedding.encode(name)
            await repo.get_or_create(name, domain, vec)
            await repo.init_document_concept(doc_id, name, INTEREST_INIT, PROFESSIONAL_INIT)
        except Exception as e:
            logger.warning(f"概念初始化失敗 [{name}]: {e}")

    await repo.sync_document_effective(doc_id)
    logger.info(f"文件 {doc_id} 概念初始化完成，共 {len(concepts)} 個")


# ── Query concept extraction (temp, not stored) ───────────────────────────────

async def build_query_concepts(text: str) -> list[dict]:
    embedding = get_embedding_provider()
    names = await extract_concepts(text)
    if not names:
        names = [text[:50]]

    result = []
    for name in names:
        vec = embedding.encode(name)
        result.append({
            "name": name,
            "q_vector": vec,
            "interest_score": 0.8,
            "professional_score": 0.8,
        })
    return result
