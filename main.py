import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from core.config import settings
from core.database import connect, disconnect, get_driver
from repositories.concept_repo import ConceptRepository
from services.embedding_service import init_embedding_service
from routers import documents, search, agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    svc = init_embedding_service(settings.embedding_model)
    await ConceptRepository(get_driver()).create_vector_index(svc.dim)
    logger.info("智慧知識庫 API 啟動完成")
    yield
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

templates = Jinja2Templates(directory="ui/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
