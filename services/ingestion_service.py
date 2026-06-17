from __future__ import annotations
import logging
from pathlib import Path

from core.database import get_driver
from models.document import Document
from repositories.document_repo import DocumentRepository
from services.concept_engine import extract_and_init_document_concepts

logger = logging.getLogger(__name__)


async def ingest_file(file_path: str) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content = _read_pdf(path)
        file_type = "pdf"
    elif suffix == ".md":
        content = path.read_text(encoding="utf-8")
        file_type = "md"
    elif suffix == ".txt":
        content = path.read_text(encoding="utf-8")
        file_type = "txt"
    else:
        raise ValueError(f"不支援的檔案格式：{suffix}")

    title = path.stem
    repo = DocumentRepository(get_driver())
    doc = await repo.create(
        title=title, content=content, file_path=str(path), file_type=file_type
    )
    await extract_and_init_document_concepts(doc.id, content)
    return doc


async def ingest_directory(dir_path: str) -> list[Document]:
    path = Path(dir_path)
    results = []
    for f in path.rglob("*"):
        if f.suffix.lower() in (".md", ".txt", ".pdf"):
            try:
                doc = await ingest_file(str(f))
                results.append(doc)
                logger.info(f"匯入成功：{f.name}")
            except Exception as e:
                logger.warning(f"匯入失敗 [{f.name}]: {e}")
    return results


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
