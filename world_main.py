"""
world_main.py — 獨立 World Agent 服務（port 8001）
只掛 /world 路由，供對外或嵌入使用，不含管理功能。
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import connect, disconnect
from core.providers.factory import init_providers
from repositories.concept_repo import ConceptRepository
from core.database import get_driver
from services.svo_service import create_entity_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()
    yield
    await disconnect()


app = FastAPI(
    title="World Knowledge Agent",
    description="公開知識庫問答服務",
    lifespan=lifespan,
)

_cors_origins = (
    ["*"]
    if settings.world_cors_origins.strip() == "*"
    else [o.strip() for o in settings.world_cors_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import world
app.include_router(world.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "ui" / "templates" / "world_public.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# 需明確加 HEAD：docker-compose healthcheck 用 `wget --spider` 送 HEAD 請求，
# 純 GET route 不會自動接受 HEAD，會回 405 讓 healthcheck 永遠判定 unhealthy。
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "service": "world-agent"}
