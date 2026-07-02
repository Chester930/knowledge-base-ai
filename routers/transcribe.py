from __future__ import annotations
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import settings
from core.upload_guard import write_upload_with_guard
from services.transcribe_service import (
    SUPPORTED_EXTENSIONS,
    TranscribeResult,
    transcribe_file,
    transcribe_folder,
)
from services.file_watcher_service import get_status as watcher_status

router = APIRouter(prefix="/transcribe", tags=["transcribe"])
logger = logging.getLogger(__name__)


# ── 回應模型 ──────────────────────────────────────────────────────────────────

class TranscribeResultOut(BaseModel):
    src_path: str
    txt_path: str | None
    success: bool
    error: str | None
    char_count: int
    elapsed_seconds: float

    @classmethod
    def from_result(cls, r: TranscribeResult) -> "TranscribeResultOut":
        return cls(
            src_path=r.src_path,
            txt_path=r.txt_path,
            success=r.success,
            error=r.error,
            char_count=r.char_count,
            elapsed_seconds=r.elapsed_seconds,
        )


class FolderTranscribeRequest(BaseModel):
    folder_path: str
    recursive: bool = False
    overwrite: bool = False


class FolderTranscribeResponse(BaseModel):
    total: int
    success: int
    failed: int
    results: list[TranscribeResultOut]


# ── 端點 ──────────────────────────────────────────────────────────────────────

@router.post("/file", response_model=TranscribeResultOut, summary="上傳單一檔案並轉譯")
async def transcribe_upload(
    file: UploadFile = File(...),
    staging_subdir: str = Form(default=""),
    overwrite: bool = Form(default=False),
):
    """
    上傳原始檔案（PDF / PPTX / DOCX / TXT / MD 等），
    轉譯後將 .txt 存入 `workspace/_staging/`（或 staging_subdir 子目錄）。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支援的格式：{suffix}（支援：{', '.join(sorted(SUPPORTED_EXTENSIONS))}）",
        )

    # 暫存上傳的原始檔
    staging = Path(settings.workspace_dir) / "_staging"
    if staging_subdir:
        staging = staging / staging_subdir
    staging.mkdir(parents=True, exist_ok=True)

    upload_tmp = staging / f"__upload__{file.filename}"
    await write_upload_with_guard(file, suffix, upload_tmp)
    try:
        result = transcribe_file(upload_tmp, staging_dir=staging, overwrite=overwrite)
    finally:
        # 刪除上傳的原始檔（保留轉譯後的 .txt）
        if upload_tmp.exists() and not upload_tmp.suffix == ".txt":
            upload_tmp.unlink(missing_ok=True)

    if not result.success:
        raise HTTPException(status_code=422, detail=result.error)

    return TranscribeResultOut.from_result(result)


@router.post("/folder", response_model=FolderTranscribeResponse, summary="批次轉譯資料夾")
async def transcribe_dir(req: FolderTranscribeRequest):
    """
    批次轉譯本地資料夾內所有支援格式的檔案，
    轉譯結果的 .txt 存入 `workspace/_staging/`。
    """
    folder = Path(req.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"資料夾不存在：{req.folder_path}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"不是資料夾：{req.folder_path}")

    staging = Path(settings.workspace_dir) / "_staging"
    results = transcribe_folder(
        folder,
        staging_dir=staging,
        recursive=req.recursive,
        overwrite=req.overwrite,
    )

    ok = sum(1 for r in results if r.success)
    return FolderTranscribeResponse(
        total=len(results),
        success=ok,
        failed=len(results) - ok,
        results=[TranscribeResultOut.from_result(r) for r in results],
    )


@router.get("/staging", summary="列出暫存區 .txt 清單")
async def list_staging():
    """回傳 workspace/_staging/ 下所有 .txt 檔案的基本資訊。"""
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
                "modified": f.stat().st_mtime,
            }
            for f in files
        ],
    }


@router.get("/watcher/status", summary="File Watcher 狀態")
async def watcher_status_endpoint():
    """查詢 File Watcher 目前監控的資料夾與運行狀態。"""
    return watcher_status()


@router.get("/supported-formats", summary="支援的原始格式清單")
async def supported_formats():
    return {"formats": sorted(SUPPORTED_EXTENSIONS)}
