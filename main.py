import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from core.config import settings
from core.database import connect, disconnect, get_driver
from core.providers.factory import init_providers
from repositories.concept_repo import ConceptRepository
from routers import documents, search, agent, transcribe, knowledge_graph, staging
from services.file_watcher_service import start_watcher, stop_watcher
from services.svo_service import create_entity_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()
    start_watcher()
    logger.info(
        f"智慧知識庫 API 啟動完成 "
        f"[LLM={settings.llm_provider}, Embedding={settings.embedding_provider}]"
    )
    yield
    stop_watcher()
    await disconnect()


app = FastAPI(
    title="智慧知識庫",
    description="個人文件知識庫 API，支援概念媒合搜尋與 Agent RAG 查詢",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(agent.router)
app.include_router(transcribe.router)
app.include_router(knowledge_graph.router)
app.include_router(staging.router)

templates = Jinja2Templates(directory="ui/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
