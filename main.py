import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from core.config import settings
from core.database import connect, disconnect, get_driver
from core.providers.factory import init_providers
from repositories.concept_repo import ConceptRepository
from routers import documents, search, agent, transcribe, knowledge_graph, staging, world
from routers import versioning, subscription
from services.classify_service import classify_all
from services.federation_service import startup_prefetch, shutdown_cleanup
from services.file_watcher_service import start_watcher, stop_watcher
from services.svo_service import create_entity_index
from services.subscription_service import sync_all_subscriptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()
    start_watcher()
    await startup_prefetch()
    # Phase 3d：每 6 小時自動同步訂閱（首次啟動後 10 分鐘觸發）
    _scheduler.add_job(
        sync_all_subscriptions,
        trigger="interval",
        hours=6,
        id="sync_subscriptions",
        replace_existing=True,
        misfire_grace_time=300,
    )
    # ☆7：暫存區自動分類 Cron（staging_cron_interval > 0 才啟用）
    if settings.staging_cron_interval > 0:
        async def _staging_auto_classify():
            try:
                from core.constants import CLASSIFY_AUTO_THRESHOLD
                results = await classify_all(threshold=CLASSIFY_AUTO_THRESHOLD, auto_assign=True)
                if results:
                    logger.info(f"[StagingCron] 自動分配 {len(results)} 份暫存文件")
            except Exception as e:
                logger.warning(f"[StagingCron] 例外：{e}")
        _scheduler.add_job(
            _staging_auto_classify,
            trigger="interval",
            minutes=settings.staging_cron_interval,
            id="staging_auto_classify",
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info(f"[StagingCron] 自動分類排程已啟動（每 {settings.staging_cron_interval} 分鐘）")
    _scheduler.start()
    logger.info(
        f"智慧知識庫 API 啟動完成 "
        f"[LLM={settings.llm_provider}, Embedding={settings.embedding_provider}]"
    )
    yield
    _scheduler.shutdown(wait=False)
    stop_watcher()
    await shutdown_cleanup()
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
app.include_router(versioning.router)
app.include_router(staging.router)
app.include_router(world.router)
app.include_router(subscription.router)

templates = Jinja2Templates(directory="ui/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
