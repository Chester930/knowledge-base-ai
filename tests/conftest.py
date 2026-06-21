from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI

from models.document import Document


def make_doc(**kwargs) -> Document:
    defaults = dict(
        id=uuid4(),
        title="測試文件",
        content="這是一份測試內容，足夠長到不會被視為空白。",
        file_type="manual",
        file_path=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Document(**defaults)


@pytest.fixture
def sample_doc():
    return make_doc()


@pytest.fixture
def sample_vector():
    return [1.0] + [0.0] * 383


@pytest.fixture
def sample_concept(sample_vector):
    return {
        "name": "機器學習",
        "q_vector": sample_vector,
        "interest_score": 0.8,
        "professional_score": 0.7,
    }


@pytest.fixture
def mock_driver():
    return MagicMock()


@pytest.fixture
def test_app():
    """最小化 FastAPI app，不啟動 DB lifespan，供 router 測試使用。"""
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    from routers import documents, search, agent, versioning, subscription, knowledge_graph, world

    app = FastAPI(lifespan=noop_lifespan)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(agent.router)
    app.include_router(versioning.router)
    app.include_router(subscription.router)
    app.include_router(knowledge_graph.router)
    app.include_router(world.router)
    return app
