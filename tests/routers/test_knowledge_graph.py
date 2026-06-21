"""
Knowledge Graph Router 測試

POST   /knowledge-graphs               — 建立 KG
GET    /knowledge-graphs               — 列出 KG
GET    /knowledge-graphs/{kg_id}       — 取得 KG 詳情
PUT    /knowledge-graphs/{kg_id}       — 更新 KG
DELETE /knowledge-graphs/{kg_id}       — 刪除 KG
GET    /knowledge-graphs/{kg_id}/documents    — 列出 KG 文件
POST   /knowledge-graphs/{kg_id}/refresh      — 刷新路由層
GET    /knowledge-graphs/{kg_id}/graph        — 取得圖結構
DELETE /knowledge-graphs/{kg_id}/graph        — 清除圖結構
GET    /knowledge-graphs/auto-cluster/preview — 預覽分群
POST   /knowledge-graphs/auto-cluster/confirm — 確認分群
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from models.knowledge_graph import KnowledgeGraph, KnowledgeGraphDetail


# ── helpers ───────────────────────────────────────────────────────────────────

def _kg(kg_id=None, name="測試KG", is_public=True, owner_id="default") -> KnowledgeGraph:
    now = datetime.now()
    return KnowledgeGraph(
        id=kg_id or uuid4(),
        name=name,
        description="",
        folder_path=f"workspace/{name}",
        owner_id=owner_id,
        is_public=is_public,
        doc_count=0,
        entity_count=0,
        relation_count=0,
        created_at=now,
        updated_at=now,
    )


def _kg_detail(kg_id=None, name="測試KG") -> KnowledgeGraphDetail:
    base = _kg(kg_id=kg_id, name=name)
    return KnowledgeGraphDetail(**base.model_dump(), top_concepts=["概念A"], top_entities=["實體B"])


def _mock_repo(kg=None, kgs=None, detail=None, docs=None):
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=kg)
    repo.get_detail = AsyncMock(return_value=detail)
    repo.list_all = AsyncMock(return_value=kgs or [])
    repo.update = AsyncMock(return_value=kg)
    repo.get_documents = AsyncMock(return_value=docs or [])
    repo.refresh_counts = AsyncMock()
    return repo


# ── POST /knowledge-graphs ────────────────────────────────────────────────────

class TestCreateKg:
    async def test_create_returns_201(self, test_app):
        kg = _kg(name="新KG")
        with patch("routers.knowledge_graph.create_kg", new=AsyncMock(return_value=kg)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs", json={"name": "新KG"})

        assert res.status_code == 201
        data = res.json()
        assert data["name"] == "新KG"
        assert "id" in data

    async def test_create_missing_name_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/knowledge-graphs", json={})
        assert res.status_code == 422

    async def test_create_empty_name_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/knowledge-graphs", json={"name": ""})
        assert res.status_code == 422

    async def test_create_duplicate_name_returns_409(self, test_app):
        with patch("routers.knowledge_graph.create_kg",
                   new=AsyncMock(side_effect=ValueError("已存在"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs", json={"name": "重複KG"})
        assert res.status_code == 409

    async def test_create_with_all_fields(self, test_app):
        kg = _kg(name="完整KG", owner_id="alice", is_public=False)
        with patch("routers.knowledge_graph.create_kg", new=AsyncMock(return_value=kg)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs", json={
                    "name": "完整KG", "description": "說明", "owner_id": "alice", "is_public": False,
                })
        assert res.status_code == 201
        assert res.json()["owner_id"] == "alice"

    async def test_create_name_too_long_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/knowledge-graphs", json={"name": "A" * 101})
        assert res.status_code == 422


# ── GET /knowledge-graphs ─────────────────────────────────────────────────────

class TestListKgs:
    async def test_list_returns_array(self, test_app):
        kgs = [_kg(name="KG1"), _kg(name="KG2")]
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository",
                   return_value=_mock_repo(kgs=kgs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/knowledge-graphs")

        assert res.status_code == 200
        assert len(res.json()) == 2

    async def test_list_empty(self, test_app):
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository",
                   return_value=_mock_repo(kgs=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/knowledge-graphs")

        assert res.status_code == 200
        assert res.json() == []

    async def test_list_passes_owner_id(self, test_app):
        mock_repo = _mock_repo(kgs=[])
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.get("/knowledge-graphs?owner_id=alice")

        mock_repo.list_all.assert_called_once_with(owner_id="alice", include_private=False)

    async def test_list_include_private(self, test_app):
        mock_repo = _mock_repo(kgs=[])
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.get("/knowledge-graphs?include_private=true")

        mock_repo.list_all.assert_called_once_with(owner_id=None, include_private=True)


# ── GET /knowledge-graphs/{kg_id} ────────────────────────────────────────────

class TestGetKgDetail:
    async def test_returns_200_with_detail(self, test_app):
        kg_id = uuid4()
        detail = _kg_detail(kg_id=kg_id)
        mock_repo = _mock_repo(detail=detail)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}")

        assert res.status_code == 200
        data = res.json()
        assert data["id"] == str(kg_id)
        assert "top_concepts" in data
        assert "top_entities" in data

    async def test_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository",
                   return_value=_mock_repo(detail=None)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}")

        assert res.status_code == 404

    async def test_invalid_uuid_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/knowledge-graphs/not-a-uuid")
        assert res.status_code == 422


# ── PUT /knowledge-graphs/{kg_id} ────────────────────────────────────────────

class TestUpdateKg:
    async def test_update_returns_200(self, test_app):
        kg_id = uuid4()
        updated = _kg(kg_id=kg_id, name="新名稱")
        mock_repo = _mock_repo(kg=updated)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.put(f"/knowledge-graphs/{kg_id}", json={"name": "新名稱"})

        assert res.status_code == 200
        assert res.json()["name"] == "新名稱"

    async def test_update_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        mock_repo.update = AsyncMock(return_value=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.put(f"/knowledge-graphs/{kg_id}", json={"name": "X"})

        assert res.status_code == 404

    async def test_update_is_public_triggers_skill_sync(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id, is_public=True)
        mock_repo = _mock_repo(kg=kg)
        mock_skill = MagicMock()
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("services.kb_skill_service.generate_skill", new=AsyncMock(return_value=mock_skill)), \
             patch("services.kb_skill_service.upsert_skill") as mock_upsert:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.put(f"/knowledge-graphs/{kg_id}", json={"is_public": True})

        assert res.status_code == 200

    async def test_update_set_private_calls_remove_skill(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id, is_public=False)
        mock_repo = _mock_repo(kg=kg)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("services.kb_skill_service.remove_skill") as mock_remove:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.put(f"/knowledge-graphs/{kg_id}", json={"is_public": False})

        assert res.status_code == 200
        mock_remove.assert_called_once_with(str(kg_id))

    async def test_update_empty_name_returns_422(self, test_app):
        kg_id = uuid4()
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.put(f"/knowledge-graphs/{kg_id}", json={"name": ""})
        assert res.status_code == 422


# ── DELETE /knowledge-graphs/{kg_id} ─────────────────────────────────────────

class TestDeleteKg:
    async def test_delete_returns_204(self, test_app):
        kg_id = uuid4()
        with patch("routers.knowledge_graph.delete_kg", new=AsyncMock(return_value=True)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/knowledge-graphs/{kg_id}")

        assert res.status_code == 204

    async def test_delete_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        with patch("routers.knowledge_graph.delete_kg", new=AsyncMock(return_value=False)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/knowledge-graphs/{kg_id}")

        assert res.status_code == 404

    async def test_delete_with_files_flag(self, test_app):
        kg_id = uuid4()
        with patch("routers.knowledge_graph.delete_kg", new=AsyncMock(return_value=True)) as mock_del:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.delete(f"/knowledge-graphs/{kg_id}?delete_files=true")

        mock_del.assert_called_once_with(kg_id, delete_files=True)


# ── GET /knowledge-graphs/{kg_id}/documents ───────────────────────────────────

class TestListKgDocuments:
    async def test_returns_doc_list(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id)
        docs = [{"id": str(uuid4()), "title": "文件A"}, {"id": str(uuid4()), "title": "文件B"}]
        mock_repo = _mock_repo(kg=kg, docs=docs)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}/documents")

        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 2
        assert data["kg_id"] == str(kg_id)
        assert len(data["documents"]) == 2

    async def test_kg_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}/documents")

        assert res.status_code == 404


# ── POST /knowledge-graphs/{kg_id}/refresh ────────────────────────────────────

class TestRefreshKg:
    async def test_returns_ok(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=_kg(kg_id=kg_id))
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("routers.knowledge_graph.refresh_kg_concepts", new=AsyncMock()):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/knowledge-graphs/{kg_id}/refresh")

        assert res.status_code == 200
        assert res.json()["status"] == "ok"
        assert res.json()["kg_id"] == str(kg_id)

    async def test_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/knowledge-graphs/{kg_id}/refresh")

        assert res.status_code == 404


# ── GET /knowledge-graphs/{kg_id}/graph ──────────────────────────────────────

class TestGetGraph:
    async def test_returns_graph_data(self, test_app):
        kg_id = uuid4()
        graph_data = {"nodes": [{"id": "A"}], "edges": [{"source": "A", "target": "B"}]}
        mock_repo = _mock_repo(kg=_kg(kg_id=kg_id))
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("routers.knowledge_graph.get_kg_graph", new=AsyncMock(return_value=graph_data)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}/graph")

        assert res.status_code == 200
        assert "nodes" in res.json()
        assert "edges" in res.json()

    async def test_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/knowledge-graphs/{kg_id}/graph")

        assert res.status_code == 404

    async def test_limit_param_passed(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=_kg(kg_id=kg_id))
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("routers.knowledge_graph.get_kg_graph",
                   new=AsyncMock(return_value={})) as mock_graph:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.get(f"/knowledge-graphs/{kg_id}/graph?limit=500&min_confidence=2")

        mock_graph.assert_called_once_with(kg_id, limit=500, min_confidence=2)

    async def test_invalid_limit_returns_422(self, test_app):
        kg_id = uuid4()
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get(f"/knowledge-graphs/{kg_id}/graph?limit=0")
        assert res.status_code == 422


# ── DELETE /knowledge-graphs/{kg_id}/graph ────────────────────────────────────

class TestClearGraph:
    async def test_clear_returns_204(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id)
        mock_repo = _mock_repo(kg=kg)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("services.svo_service._clear_kg_entities", new=AsyncMock()):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/knowledge-graphs/{kg_id}/graph")

        assert res.status_code == 204

    async def test_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/knowledge-graphs/{kg_id}/graph")

        assert res.status_code == 404


# ── POST /knowledge-graphs/{kg_id}/build-graph (SSE) ─────────────────────────

class TestBuildGraph:
    async def test_returns_streaming_response(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id)
        mock_repo = _mock_repo(kg=kg)

        async def _fake_build(kg_id, doc_ids=None, force_rebuild=False):
            progress = MagicMock()
            progress.event = "done"
            progress.chunk_idx = 1
            progress.total_chunks = 1
            progress.triples_extracted = 5
            progress.triples_merged = 5
            progress.message = "完成"
            yield progress

        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("routers.knowledge_graph.build_graph_for_kg", side_effect=_fake_build), \
             patch("routers.knowledge_graph.apply_type_labels",
                   new=AsyncMock(return_value={"Concept": 3})):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/knowledge-graphs/{kg_id}/build-graph")

        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        lines = [l for l in res.text.split("\n\n") if l.strip()]
        payloads = [json.loads(l.removeprefix("data: ")) for l in lines]
        assert any(p["event"] == "done" for p in payloads)
        assert any(p["event"] == "labels_done" for p in payloads)

    async def test_build_graph_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        mock_repo = _mock_repo(kg=None)
        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/knowledge-graphs/{kg_id}/build-graph")

        assert res.status_code == 404

    async def test_build_graph_force_rebuild_param(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id)
        mock_repo = _mock_repo(kg=kg)
        calls = []

        async def _spy_build(kg_id, doc_ids=None, force_rebuild=False):
            calls.append({"force_rebuild": force_rebuild})
            yield MagicMock(event="done", chunk_idx=1, total_chunks=1,
                            triples_extracted=0, triples_merged=0, message="")

        with patch("routers.knowledge_graph.get_driver"), \
             patch("routers.knowledge_graph.KnowledgeGraphRepository", return_value=mock_repo), \
             patch("routers.knowledge_graph.build_graph_for_kg", side_effect=_spy_build), \
             patch("routers.knowledge_graph.apply_type_labels", new=AsyncMock(return_value={})):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post(f"/knowledge-graphs/{kg_id}/build-graph",
                             json={"force_rebuild": True})

        assert calls[0]["force_rebuild"] is True


# ── GET /knowledge-graphs/auto-cluster/preview ────────────────────────────────

class TestAutoClusterPreview:
    async def test_returns_cluster_list(self, test_app):
        clusters = [
            {"name": "KG-A", "description": "", "files": ["a.txt", "b.txt"], "doc_ids": []},
            {"name": "KG-B", "description": "", "files": ["c.txt"], "doc_ids": []},
        ]
        with patch("routers.knowledge_graph.auto_cluster_kgs",
                   new=AsyncMock(return_value=clusters)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/knowledge-graphs/auto-cluster/preview")

        assert res.status_code == 200
        data = res.json()
        assert len(data["clusters"]) == 2
        assert data["total_files"] == 3

    async def test_too_few_docs_returns_400(self, test_app):
        with patch("routers.knowledge_graph.auto_cluster_kgs",
                   new=AsyncMock(side_effect=ValueError("文件不足"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/knowledge-graphs/auto-cluster/preview")

        assert res.status_code == 400

    async def test_min_docs_param_passed(self, test_app):
        with patch("routers.knowledge_graph.auto_cluster_kgs",
                   new=AsyncMock(return_value=[])) as mock_cluster:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.get("/knowledge-graphs/auto-cluster/preview?min_docs=5")

        mock_cluster.assert_called_once_with(min_docs=5)


# ── POST /knowledge-graphs/auto-cluster/confirm ───────────────────────────────

class TestAutoClusterConfirm:
    async def test_confirm_returns_created_count(self, test_app):
        results = [
            {"kg_id": str(uuid4()), "name": "KG-A", "assigned": 2},
            {"kg_id": str(uuid4()), "name": "KG-B", "assigned": 1},
        ]
        with patch("routers.knowledge_graph.confirm_auto_cluster",
                   new=AsyncMock(return_value=results)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs/auto-cluster/confirm", json=[
                    {"name": "KG-A", "files": ["a.txt"]},
                    {"name": "KG-B", "files": ["b.txt"]},
                ])

        assert res.status_code == 200
        data = res.json()
        assert data["created"] == 2
        assert len(data["results"]) == 2

    async def test_empty_cluster_list_returns_zero(self, test_app):
        with patch("routers.knowledge_graph.confirm_auto_cluster",
                   new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs/auto-cluster/confirm", json=[])

        assert res.status_code == 200
        assert res.json()["created"] == 0

    async def test_cluster_without_kg_id_not_counted(self, test_app):
        results = [
            {"kg_id": str(uuid4()), "name": "成功"},
            {"error": "失敗", "name": "失敗的"},
        ]
        with patch("routers.knowledge_graph.confirm_auto_cluster",
                   new=AsyncMock(return_value=results)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/knowledge-graphs/auto-cluster/confirm",
                                   json=[{"name": "X"}, {"name": "Y"}])

        assert res.json()["created"] == 1
