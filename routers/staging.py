from __future__ import annotations
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.config import settings
from core.constants import CLASSIFY_AUTO_THRESHOLD
from models.knowledge_graph import (
    ApproveSuggestionRequest,
    AssignRequest,
    ClassifyRequest,
    ClassifyResult,
    ClusterSuggestion,
)
from services.classify_service import assign_document_to_kg, classify_all, classify_document

router = APIRouter(prefix="/staging", tags=["staging"])
logger = logging.getLogger(__name__)


def _validate_staging_filename(filename: str) -> None:
    r"""
    拒絕含路徑分隔符或 `..` 的檔名，避免 `staging / filename` 拼接時逃出
    `_staging/` 目錄（尤其 Windows 原生部署下 `\` 會被當作真正的路徑分隔符，
    FastAPI 路由層只對 `/` 有防護）。
    """
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail=f"檔名不合法：{filename}")


@router.get("", summary="列出暫存區所有 .txt")
async def list_staging():
    staging = Path(settings.workspace_dir) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    files = sorted(staging.glob("*.txt"))
    return {
        "staging_dir": str(staging),
        "count": len(files),
        "files": [
            {
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "size_chars": len(f.read_text(encoding="utf-8", errors="replace")),
            }
            for f in files
        ],
    }


@router.post("/{filename}/classify", response_model=ClassifyResult, summary="分析單一文件適合哪個 KG")
async def classify_one(
    filename: str,
    body: ClassifyRequest = ClassifyRequest(),
):
    """
    對 `_staging/{filename}` 執行概念比對，回傳 KG 候選清單與分數。
    `auto_assign=true` 且 top_score ≥ threshold 時自動分配（移動檔案 + 建 Document）。
    """
    _validate_staging_filename(filename)
    try:
        return await classify_document(
            filename,
            threshold=body.threshold,
            auto_assign=body.auto_assign,
            owner_id=body.owner_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-all", response_model=list[ClassifyResult], summary="批次自動分配所有暫存文件")
async def classify_all_endpoint(
    threshold: float = Query(default=CLASSIFY_AUTO_THRESHOLD, ge=0.0, le=1.0),
    auto_assign: bool = Query(default=False),
):
    """
    對 `_staging/` 中所有 .txt 執行批次分配。
    `auto_assign=true` 才會實際移動檔案；否則只回傳候選清單。
    """
    return await classify_all(threshold=threshold, auto_assign=auto_assign)


@router.post("/{filename}/assign", status_code=200, summary="手動指定文件到特定 KG")
async def assign_one(filename: str, body: AssignRequest):
    """
    手動將 `_staging/{filename}` 分配給指定的 KG，
    移動檔案 + 建立 Document + 刷新 KG 路由層概念。
    """
    _validate_staging_filename(filename)
    try:
        await assign_document_to_kg(filename, body.kg_id)
        return {"status": "assigned", "filename": filename, "kg_id": str(body.kg_id)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename}", status_code=204, summary="刪除暫存區文件")
async def delete_staging_file(filename: str):
    _validate_staging_filename(filename)
    staging = Path(settings.workspace_dir) / "_staging"
    target = staging / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"找不到：{filename}")
    target.unlink()


@router.post(
    "/suggest-kgs",
    response_model=list[ClusterSuggestion],
    summary="分析 unmatched 文件，建議新 KG 分類",
)
async def suggest_kgs():
    """
    掃描 _staging/ 中無法分配到現有 KG 的文件，
    找出主題相近的群組，並建議新的 KG 名稱與描述。
    分析時間依文件數量而定（每份約 2-5 秒）。
    """
    try:
        from services.cluster_service import cluster_staging_files
        suggestions = await cluster_staging_files()
        return suggestions
    except Exception as e:
        logger.exception("分群分析失敗")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve-suggestion", status_code=200, summary="核准建議並建立新 KG")
async def approve_suggestion(body: ApproveSuggestionRequest, background_tasks: BackgroundTasks):
    """
    核准一個分群建議：
    1. 建立新的 KnowledgeGraph（名稱 = suggested_name）
    2. 將指定的 .txt 檔案從 _staging/ 分配到新 KG
    3. 自動觸發 SVO 知識圖譜建構（背景執行）
    """
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from core.database import get_driver
    from models.knowledge_graph import KnowledgeGraphCreate

    try:
        kg_repo = KnowledgeGraphRepository(get_driver())

        # 建立新 KG
        new_kg = await kg_repo.create(KnowledgeGraphCreate(
            name=body.suggested_name,
            description=body.suggested_description,
            is_public=True,
        ))
        logger.info(f"建立新 KG：{new_kg.name}（{new_kg.id}）")

        # 逐一分配文件
        assigned, failed = [], []
        for filename in body.files:
            if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
                failed.append({"file": filename, "reason": "檔名不合法"})
                continue
            staging = Path(settings.workspace_dir) / "_staging" / filename
            if not staging.exists():
                failed.append({"file": filename, "reason": "不在 staging"})
                continue
            try:
                await assign_document_to_kg(filename, new_kg.id)
                assigned.append(filename)
            except Exception as e:
                failed.append({"file": filename, "reason": str(e)})

        # 有文件成功分配才觸發建圖
        if assigned:
            async def _auto_build(kg_id):
                try:
                    from services.svo_service import build_graph_for_kg
                    async for _ in build_graph_for_kg(kg_id):
                        pass
                    logger.info(f"approve-suggestion 自動建圖完成：{kg_id}")
                except Exception as e:
                    logger.warning(f"approve-suggestion 自動建圖失敗：{e}")

            import asyncio
            background_tasks.add_task(asyncio.ensure_future, _auto_build(new_kg.id))

        return {
            "kg_id": str(new_kg.id),
            "kg_name": new_kg.name,
            "assigned": assigned,
            "failed": failed,
            "build_graph": "triggered" if assigned else "skipped",
        }
    except Exception as e:
        logger.exception("核准建議失敗")
        raise HTTPException(status_code=500, detail=str(e))
