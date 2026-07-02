from uuid import UUID
from pathlib import Path
import tempfile

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel

from core.config import settings
from core.database import get_driver
from models.document import Document, DocumentCreate, DocumentConcept
from repositories.document_repo import DocumentRepository
from repositories.concept_repo import ConceptRepository
from services.concept_engine import extract_and_init_document_concepts
from services.ingestion_service import (
    ingest_file, ingest_directory, move_and_ingest, SUPPORTED_EXTENSIONS
)

router = APIRouter(prefix="/documents", tags=["documents"])

# 上傳檔案 magic bytes 校驗（僅對有固定簽章的二進位格式檢查；.md/.txt 純文字不校驗）
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    ".ppt": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
}
_UPLOAD_CHUNK_SIZE = 1024 * 1024


class MoveIngestRequest(BaseModel):
    source_dir: str
    target_dir: str


@router.post("", response_model=Document, status_code=201)
async def create_document(data: DocumentCreate, background_tasks: BackgroundTasks):
    repo = DocumentRepository(get_driver())
    doc = await repo.create(data.title, data.content, data.file_path, data.file_type)
    background_tasks.add_task(extract_and_init_document_concepts, doc.id, data.content)
    if data.kg_id:
        from repositories.knowledge_graph_repo import KnowledgeGraphRepository
        await KnowledgeGraphRepository(get_driver()).add_document(data.kg_id, doc.id)
    return doc


@router.post("/upload", response_model=Document, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支援的格式 {suffix}，支援：{', '.join(SUPPORTED_EXTENSIONS)}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total = 0
    header = b""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
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
                tmp.write(chunk)

        signatures = _MAGIC_SIGNATURES.get(suffix)
        if signatures and not any(header.startswith(sig) for sig in signatures):
            raise HTTPException(400, f"檔案內容與副檔名 {suffix} 不符")

        doc = await ingest_file(tmp_path)
        return doc
    except HTTPException:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise
    except Exception:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise


@router.post("/ingest-dir")
async def ingest_dir(dir_path: str):
    """批次匯入指定目錄下所有支援格式的文件。"""
    success, errors = await ingest_directory(dir_path)
    return {
        "ingested": len(success),
        "failed": len(errors),
        "titles": [d.title for d in success],
        "errors": errors,
    }


@router.post("/move-and-ingest")
async def move_and_ingest_endpoint(req: MoveIngestRequest):
    """
    將 source_dir 的文件搬移到 target_dir，再批次匯入知識庫。
    匯入失敗的檔案會自動搬回來源，不遺失。
    """
    success, errors = await move_and_ingest(
        source_dir=req.source_dir,
        target_dir=req.target_dir,
        delete_on_success=True,
    )
    return {
        "ingested": len(success),
        "failed": len(errors),
        "titles": [d.title for d in success],
        "errors": errors,
    }


@router.get("", response_model=list[Document])
async def list_documents(limit: int = 50, offset: int = 0):
    return await DocumentRepository(get_driver()).list_all(limit, offset)


@router.get("/{doc_id}", response_model=Document)
async def get_document(doc_id: UUID):
    doc = await DocumentRepository(get_driver()).get_by_id(doc_id)
    if not doc:
        raise HTTPException(404, "文件不存在")
    return doc


@router.get("/{doc_id}/concepts", response_model=list[DocumentConcept])
async def get_document_concepts(doc_id: UUID):
    concepts = await ConceptRepository(get_driver()).get_document_concepts(doc_id)
    return [
        DocumentConcept(
            concept_id=UUID(c["id"]),
            name=c["name"],
            domain=c["domain"],
            interest_score=c["interest_score"],
            professional_score=c["professional_score"],
        )
        for c in concepts
    ]


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: UUID):
    deleted = await DocumentRepository(get_driver()).delete(doc_id)
    if not deleted:
        raise HTTPException(404, "文件不存在")
