from uuid import UUID
from pathlib import Path
import tempfile

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from core.database import get_driver
from models.document import Document, DocumentCreate, DocumentConcept
from repositories.document_repo import DocumentRepository
from repositories.concept_repo import ConceptRepository
from services.concept_engine import extract_and_init_document_concepts
from services.ingestion_service import ingest_file, ingest_directory

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=Document, status_code=201)
async def create_document(data: DocumentCreate, background_tasks: BackgroundTasks):
    repo = DocumentRepository(get_driver())
    doc = await repo.create(data.title, data.content, data.file_path, data.file_type)
    background_tasks.add_task(extract_and_init_document_concepts, doc.id, data.content)
    return doc


@router.post("/upload", response_model=Document, status_code=201)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".md", ".txt", ".pdf"):
        raise HTTPException(400, "僅支援 .md / .txt / .pdf 格式")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    doc = await ingest_file(tmp_path)
    return doc


@router.post("/ingest-dir", status_code=200)
async def ingest_dir(dir_path: str):
    docs = await ingest_directory(dir_path)
    return {"ingested": len(docs), "titles": [d.title for d in docs]}


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
