from __future__ import annotations
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from models.knowledge_graph import KnowledgeGraph
from services.knowledge_graph_service import delete_kg


def _kg(**kwargs) -> KnowledgeGraph:
    defaults = dict(
        id=uuid4(), name="測試KG", description="", folder_path="workspace/kg_test",
        owner_id="default", is_public=False, db_name="",
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return KnowledgeGraph(**defaults)


class TestDeleteKg:
    async def test_nonexistent_kg_returns_false(self):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = None
        with patch("services.knowledge_graph_service.KnowledgeGraphRepository",
                   return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.get_driver"):
            result = await delete_kg(uuid4())
        assert result is False

    async def test_successful_delete_cleans_up_chunk_store(self):
        kg = _kg()
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg
        mock_kg_repo.delete.return_value = True

        mock_chunk_store = MagicMock()

        with patch("services.knowledge_graph_service.KnowledgeGraphRepository",
                   return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.drop_kg_database", new=AsyncMock()), \
             patch("services.chunk_store.get_chunk_store", return_value=mock_chunk_store):
            result = await delete_kg(kg.id)

        assert result is True
        mock_chunk_store.delete_kg.assert_called_once_with(kg.id)

    async def test_failed_neo4j_delete_does_not_touch_chunk_store(self):
        kg = _kg()
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg
        mock_kg_repo.delete.return_value = False

        mock_chunk_store = MagicMock()

        with patch("services.knowledge_graph_service.KnowledgeGraphRepository",
                   return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.drop_kg_database", new=AsyncMock()), \
             patch("services.chunk_store.get_chunk_store", return_value=mock_chunk_store):
            result = await delete_kg(kg.id)

        assert result is False
        mock_chunk_store.delete_kg.assert_not_called()
