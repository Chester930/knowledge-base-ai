from __future__ import annotations
import asyncio
import logging
import shutil
from pathlib import Path
from uuid import UUID

from core.config import settings
from core.constants import CLASSIFY_AUTO_THRESHOLD, CLASSIFY_MIN_THRESHOLD
from core.database import get_driver
from models.knowledge_graph import ClassifyResult, KGCandidate
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository

logger = logging.getLogger(__name__)


async def classify_document(
    txt_filename: str,
    threshold: float = CLASSIFY_AUTO_THRESHOLD,
    auto_assign: bool = False,
    owner_id: str = "default",
) -> ClassifyResult:
    """
    對 _staging/ 中的單一 .txt 進行 KG 分配：
    1. 提取文件概念
    2. 與所有 KG 的路由層概念做 compute_match_score
    3. auto_assign=True 且 top_score ≥ threshold → 自動呼叫 assign_document_to_kg
    回傳 ClassifyResult（含完整候選排名）。
    """
    from services.concept_engine import build_query_concepts, compute_match_score

    staging = Path(settings.workspace_dir) / "_staging"
    txt_path = staging / txt_filename
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到暫存檔案：{txt_filename}")

    text = txt_path.read_text(encoding="utf-8")
    doc_concepts = await build_query_concepts(text[:4000])

    concept_repo = ConceptRepository(get_driver())
    kg_repo = KnowledgeGraphRepository(get_driver())
    all_kg_concepts = await concept_repo.get_all_kgs_concepts()

    candidates: list[KGCandidate] = []
    for kg_id, kg_concepts in all_kg_concepts.items():
        score, matched = compute_match_score(doc_concepts, kg_concepts)
        if score < CLASSIFY_MIN_THRESHOLD:
            continue
        kg = await kg_repo.get_by_id(kg_id)
        if kg is None:
            continue
        candidates.append(KGCandidate(
            kg_id=kg_id,
            kg_name=kg.name,
            score=round(score, 4),
            top_matched_concepts=matched,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)

    result = ClassifyResult(
        txt_filename=txt_filename,
        candidates=candidates,
        status="pending",
    )

    if not candidates:
        result.status = "unmatched"
        return result

    top = candidates[0]
    result.matched_kg_id = top.kg_id
    result.matched_kg_name = top.kg_name
    result.score = top.score

    if auto_assign and top.score >= threshold:
        await assign_document_to_kg(txt_filename, top.kg_id)
        result.auto_assigned = True
        result.status = "assigned"
    else:
        result.status = "pending"

    return result


async def assign_document_to_kg(txt_filename: str, kg_id: UUID) -> None:
    """
    手動或自動分配：
    1. 移動 .txt 檔案：_staging/{name}.txt → kg_{name}/_text/{name}.txt
    2. 建立 Document 節點（content 存入 Neo4j）
    3. 提取概念 → 路由層 EFFECTIVE 邊（Document 層）
    4. 更新 KG 文件關聯 + 刷新 KG 路由層概念
    """
    from services.concept_engine import extract_and_init_document_concepts
    from services.knowledge_graph_service import refresh_kg_concepts

    staging = Path(settings.workspace_dir) / "_staging"
    txt_path = staging / txt_filename
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到暫存檔案：{txt_filename}")

    kg_repo = KnowledgeGraphRepository(get_driver())
    kg = await kg_repo.get_by_id(kg_id)
    if kg is None:
        raise ValueError(f"KG 不存在：{kg_id}")

    # 移動檔案到 KG _text/ 目錄
    text_dir = Path(kg.folder_path) / "_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    dest = text_dir / txt_filename
    counter = 1
    while dest.exists():
        dest = text_dir / f"{txt_path.stem}_{counter}.txt"
        counter += 1
    shutil.move(str(txt_path), str(dest))
    logger.info(f"檔案移動：{txt_filename} → {dest}")

    # 建立 Document 節點
    text = dest.read_text(encoding="utf-8")
    doc_repo = DocumentRepository(get_driver())
    doc = await doc_repo.create(
        title=txt_path.stem,
        content=text,
        file_path=str(dest),
        file_type="txt",
    )

    # 提取概念，建立路由層
    await extract_and_init_document_concepts(doc.id, text)

    # 建立 KG ↔ Document 關聯
    await kg_repo.add_document(kg_id, doc.id)

    # 刷新 KG 路由層概念
    await refresh_kg_concepts(kg_id)
    logger.info(f"分配完成：{txt_filename} → KG {kg.name}（doc_id={doc.id}）")

    # 自動觸發增量 SVO 提取（背景執行，不阻塞回應）
    async def _auto_svo():
        try:
            from services.svo_service import build_graph_for_kg, apply_type_labels
            async for _ in build_graph_for_kg(kg_id, doc_ids=[doc.id], force_rebuild=False):
                pass
            await apply_type_labels(kg_id, db_name=kg.db_name)
            logger.info(f"自動 SVO 提取完成：{txt_filename}")
        except Exception as e:
            logger.warning(f"自動 SVO 提取失敗：{e}")

    asyncio.create_task(_auto_svo())


async def classify_all(
    threshold: float = CLASSIFY_AUTO_THRESHOLD,
    auto_assign: bool = False,
    owner_id: str = "default",
) -> list[ClassifyResult]:
    """對 _staging/ 中所有 .txt 批次執行分配。"""
    staging = Path(settings.workspace_dir) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    files = sorted(staging.glob("*.txt"))
    if not files:
        return []

    results: list[ClassifyResult] = []
    for f in files:
        try:
            r = await classify_document(
                f.name,
                threshold=threshold,
                auto_assign=auto_assign,
                owner_id=owner_id,
            )
            results.append(r)
        except Exception as e:
            logger.warning(f"分配失敗 [{f.name}]: {e}")
            results.append(ClassifyResult(txt_filename=f.name, status="error"))

    ok = sum(1 for r in results if r.status == "assigned")
    logger.info(f"批次分配完成：{ok}/{len(results)} 個已分配")
    return results
