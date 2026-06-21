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

from core.database import connect, disconnect
from core.providers.factory import init_providers
from repositories.concept_repo import ConceptRepository
from core.database import get_driver


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    yield
    await disconnect()


app = FastAPI(
    title="World Knowledge Agent",
    description="公開知識庫問答服務",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import world
app.include_router(world.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "ui" / "templates" / "world_public.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "world-agent"}
