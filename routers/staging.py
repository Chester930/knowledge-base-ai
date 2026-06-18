from __future__ import annotations
import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from core.config import settings
from core.constants import CLASSIFY_AUTO_THRESHOLD
from models.knowledge_graph import AssignRequest, ClassifyRequest, ClassifyResult
from services.classify_service import assign_document_to_kg, classify_all, classify_document

router = APIRouter(prefix="/staging", tags=["staging"])
logger = logging.getLogger(__name__)


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
    staging = Path(settings.workspace_dir) / "_staging"
    target = staging / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"找不到：{filename}")
    target.unlink()
