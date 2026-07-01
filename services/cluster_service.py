"""
cluster_service.py
分析 _staging/ 中的 unmatched 文件，找出有明顯關聯的群組，
並建議新的 KG 分類。
"""
from __future__ import annotations
import logging
from pathlib import Path

from core.config import settings
from core.constants import CLASSIFY_MIN_THRESHOLD
from core.providers.factory import get_llm_provider

logger = logging.getLogger(__name__)

# 群內平均相似度門檻：高於此值才視為「有明顯關聯」
CLUSTER_INTRA_THRESHOLD = 0.35
# 群最小文件數
CLUSTER_MIN_SIZE = 2
# 文件向量計算用的前幾個 concept
MAX_CONCEPTS_PER_DOC = 15


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def _avg_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    avg = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            avg[i] += x
    n = len(vectors)
    return [x / n for x in avg]


async def _get_doc_embedding(txt_path: Path) -> tuple[list[float], list[str]]:
    """讀取文件，提取概念，回傳平均向量 + 概念名稱清單。"""
    from services.concept_engine import build_query_concepts
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    concepts = await build_query_concepts(text[:4000])
    concepts = concepts[:MAX_CONCEPTS_PER_DOC]
    if not concepts:
        return [], []
    vectors = [c["q_vector"] for c in concepts]
    names = [c["name"] for c in concepts]
    return _avg_vector(vectors), names


def _connected_components(
    nodes: list[str],
    sim_matrix: dict[tuple[str, str], float],
    threshold: float,
) -> list[list[str]]:
    """閾值以上視為同群，回傳 connected components。"""
    visited = set()
    components = []

    def dfs(node: str, group: list[str]):
        visited.add(node)
        group.append(node)
        for other in nodes:
            if other not in visited:
                key = (min(node, other), max(node, other))
                if sim_matrix.get(key, 0.0) >= threshold:
                    dfs(other, group)

    for n in nodes:
        if n not in visited:
            group: list[str] = []
            dfs(n, group)
            components.append(group)

    return components


def _intra_cluster_avg(
    members: list[str],
    sim_matrix: dict[tuple[str, str], float],
) -> float:
    if len(members) < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            key = (min(members[i], members[j]), max(members[i], members[j]))
            total += sim_matrix.get(key, 0.0)
            count += 1
    return total / count if count else 0.0


async def _suggest_kg_name(filenames: list[str], concepts: list[str]) -> tuple[str, str]:
    """讓 LLM 根據文件名稱和關鍵概念，建議 KG 名稱和描述。"""
    files_str = "\n".join(f"- {f}" for f in filenames[:10])
    concepts_str = "、".join(concepts[:20])
    prompt = (
        "以下是一批主題相近的文件，請為它們建議一個知識圖譜（KG）的分類名稱和簡短描述。\n\n"
        f"文件列表：\n{files_str}\n\n"
        f"共同關鍵概念：{concepts_str}\n\n"
        "請用以下格式回答（只回這兩行，不加其他文字）：\n"
        "名稱：<2-8個字的中文名稱>\n"
        "描述：<一句話說明這個知識庫的主題範圍>"
    )
    try:
        raw = await get_llm_provider().generate(prompt)
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        name, desc = "", ""
        for line in lines:
            if line.startswith("名稱：") or line.startswith("名称："):
                name = line.split("：", 1)[-1].strip()
            elif line.startswith("描述："):
                desc = line.split("：", 1)[-1].strip()
        if not name:
            name = "新知識庫"
        return name, desc
    except Exception as e:
        logger.warning(f"LLM 建議 KG 名稱失敗：{e}")
        return "新知識庫", ""


async def cluster_staging_files() -> list[dict]:
    """
    分析 _staging/ 的所有 .txt，找出有明顯關聯的群組，回傳建議清單。

    回傳格式：
    [
      {
        "suggested_name": "空間智慧",
        "suggested_description": "...",
        "files": ["a.txt", "b.txt"],
        "top_concepts": ["空間計算", "AR", ...],
        "intra_similarity": 0.61,
      },
      ...
    ]
    """
    staging = Path(settings.workspace_dir) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    txts = sorted(staging.glob("*.txt"))

    if not txts:
        return []

    logger.info(f"分析 {len(txts)} 個暫存文件")

    # 先做初步 KG 分類，把 score < CLASSIFY_MIN_THRESHOLD 的篩出來視為 unmatched
    from services.classify_service import classify_document
    unmatched_files: list[Path] = []
    for txt in txts:
        try:
            result = await classify_document(txt.name, auto_assign=False)
            if result.status == "unmatched" or result.score < CLASSIFY_MIN_THRESHOLD:
                unmatched_files.append(txt)
        except Exception as e:
            logger.warning(f"分類檢查失敗 [{txt.name}]: {e}")
            unmatched_files.append(txt)  # 失敗的也納入分析

    if not unmatched_files:
        logger.info("所有文件都已可分類，無需建立新 KG")
        return []

    logger.info(f"找到 {len(unmatched_files)} 個 unmatched 文件，進行分群分析")

    # 並行提取每個文件的向量和概念（並行優化）
    import asyncio as _asyncio

    async def _process_one_doc(txt: Path):
        try:
            vec, concepts = await _get_doc_embedding(txt)
            return txt.name, vec, concepts
        except Exception as e:
            logger.warning(f"提取文件嵌入失敗 [{txt.name}]: {e}")
            return txt.name, [], []

    tasks = [_process_one_doc(txt) for txt in unmatched_files]
    results = await _asyncio.gather(*tasks)

    doc_embeddings: dict[str, list[float]] = {}
    doc_concepts: dict[str, list[str]] = {}
    for name, vec, concepts in results:
        if vec:
            doc_embeddings[name] = vec
            doc_concepts[name] = concepts

    names = list(doc_embeddings.keys())
    if len(names) < CLUSTER_MIN_SIZE:
        return []

    # 計算 pairwise 相似度
    sim_matrix: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = (min(a, b), max(a, b))
            sim_matrix[key] = _cosine(doc_embeddings[a], doc_embeddings[b])

    # 分群
    components = _connected_components(names, sim_matrix, CLUSTER_INTRA_THRESHOLD)

    # 過濾：群太小 or 群內相似度不夠的跳過，並收集各群組的概念頻率
    candidate_groups = []
    for members in components:
        if len(members) < CLUSTER_MIN_SIZE:
            continue
        intra_sim = _intra_cluster_avg(members, sim_matrix)
        if intra_sim < CLUSTER_INTRA_THRESHOLD:
            continue

        # 收集這群的 top concepts（出現次數最多）
        concept_freq: dict[str, int] = {}
        for fname in members:
            for c in doc_concepts.get(fname, []):
                concept_freq[c] = concept_freq.get(c, 0) + 1
        top_concepts = sorted(concept_freq, key=concept_freq.get, reverse=True)[:15]
        candidate_groups.append((members, top_concepts, intra_sim))

    # 並行調用 LLM 建議名稱（並行優化）
    async def _suggest_one_group(members, top_concepts, intra_sim):
        suggested_name, suggested_desc = await _suggest_kg_name(members, top_concepts)
        return {
            "suggested_name": suggested_name,
            "suggested_description": suggested_desc,
            "files": members,
            "top_concepts": top_concepts,
            "intra_similarity": round(intra_sim, 3),
        }

    llm_tasks = [_suggest_one_group(m, tc, sim) for m, tc, sim in candidate_groups]
    suggestions = await _asyncio.gather(*llm_tasks)

    for s in suggestions:
        logger.info(
            f"群組建議：{s['suggested_name']}（{len(s['files'])} 份文件，"
            f"avg_sim={s['intra_similarity']:.2f}）"
        )

    suggestions.sort(key=lambda s: s["intra_similarity"], reverse=True)
    return suggestions
