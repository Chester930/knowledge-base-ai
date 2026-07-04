"""
共用的上傳檔案防護：串流寫入時檢查大小上限，並用 magic bytes 校驗副檔名是否被偽造。
供 routers/documents.py、routers/transcribe.py 共用。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from core.config import settings

_UPLOAD_CHUNK_SIZE = 1024 * 1024

# 僅對有固定簽章的二進位格式檢查；純文字（.md/.txt）與音訊/影片格式不校驗
MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    ".ppt": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
}


async def write_upload_with_guard(file: UploadFile, suffix: str, dest_path: Path) -> None:
    """
    串流寫入上傳檔案至 dest_path。
    超過 settings.max_upload_size_mb 或 magic bytes 與宣稱的副檔名不符時，
    中止寫入、刪除已寫入的部分檔案，並拋出 HTTPException（413 / 400）。
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total = 0
    header = b""
    try:
        with dest_path.open("wb") as f:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        413, f"檔案超過大小上限 {settings.max_upload_size_mb}MB"
                    )
                if len(header) < 16:
                    header += chunk[: 16 - len(header)]
                f.write(chunk)

        signatures = MAGIC_SIGNATURES.get(suffix)
        if signatures and not any(header.startswith(sig) for sig in signatures):
            raise HTTPException(400, f"檔案內容與副檔名 {suffix} 不符")
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
