"""
World Router 測試

GET  /world/knowledge-graphs         — 列出公開 KG
GET  /world/stats                    — 整體統計
POST /world/chat                     — World Agent SSE 問答
GET  /world/explore/entities         — 跨 KG 實體搜尋
GET  /world/explore/neighbors        — 實體鄰居查詢
GET  /world/provenance/facts         — 事實溯源
GET  /world/align/synonyms           — 同義詞查詢
GET  /world/align/entities           — 跨 instance 實體對齊
GET  /world/federation/status        — 聯邦狀態
GET  /world/registry                 — 本機 registry
POST /world/sync                     — 同步 registry
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from models.knowledge_graph import KnowledgeGraph


# ── helpers ───────────────────────────────────────────────────────────────────

def _kg(kg_id=None, name="公開KG", is_public=True, entity_count=10, relation_count=20, doc_count=3):
    now = datetime.now()
    return KnowledgeGraph(
        id=kg_id or uuid4(),
        name=name,
        description="",
        folder_path=f"workspace/{name}",
        owner_id="default",
        is_public=is_public,
        doc_count=doc_count,
        entity_count=entity_count,
        relation_count=relation_count,
        created_at=now,
        updated_at=now,
    )


def _mock_kg_repo(kgs=None, kg=None, docs=None):
    repo = MagicMock()
    repo.list_all = AsyncMock(return_value=kgs or [])
    repo.get_by_id = AsyncMock(return_value=kg)
    repo.get_documents = AsyncMock(return_value=docs or [])
    return repo


def _mock_driver(records=None):
    result = MagicMock()
    result.records = records or []
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=result)
    return driver


def _mock_federation(skills=None):
    reg = MagicMock()
    reg.skills = skills or []
    cache = MagicMock()
    cache.merged_registry = AsyncMock(return_value=reg)
    cache.status = MagicMock(return_value={"total": 0, "local": 0, "remote": 0})
    cache.get_remote_registry = AsyncMock()
    cache._fetched_at = 999.0
    return cache


def _parse_sse(text: str) -> list[dict]:
    events = []
    for chunk in text.split("\n\n"):
        line = chunk.strip()
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ── GET /world/knowledge-graphs ───────────────────────────────────────────────

class TestListPublicKgs:
    async def test_returns_only_public(self, test_app):
        kgs = [_kg(name="公開A", is_public=True), _kg(name="私有B", is_public=False)]
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=kgs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/knowledge-graphs")

        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["name"] == "公開A"

    async def test_empty_returns_empty_list(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/knowledge-graphs")

        assert res.status_code == 200
        assert res.json() == []

    async def test_response_fields(self, test_app):
        kgs = [_kg(entity_count=5, relation_count=10, doc_count=2)]
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=kgs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/knowledge-graphs")

        item = res.json()[0]
        for field in ("id", "name", "description", "doc_count", "entity_count", "relation_count", "is_public"):
            assert field in item


# ── GET /world/stats ──────────────────────────────────────────────────────────

class TestWorldStats:
    async def test_returns_aggregated_counts(self, test_app):
        kgs = [
            _kg(is_public=True, entity_count=10, relation_count=20, doc_count=3),
            _kg(is_public=True, entity_count=5, relation_count=8, doc_count=1),
            _kg(is_public=False, entity_count=100, relation_count=200, doc_count=50),
        ]
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=kgs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/stats")

        assert res.status_code == 200
        data = res.json()
        assert data["public_kg_count"] == 2
        assert data["total_entities"] == 15
        assert data["total_relations"] == 28
        assert data["total_docs"] == 4

    async def test_stats_fields_present(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/stats")

        data = res.json()
        for field in ("public_kg_count", "total_entities", "total_relations", "total_docs", "public_kgs"):
            assert field in data

    async def test_empty_returns_zeros(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/stats")

        data = res.json()
        assert data["public_kg_count"] == 0
        assert data["total_entities"] == 0


# ── POST /world/chat (SSE) ────────────────────────────────────────────────────

class TestWorldChat:
    def _base_patches(self, kgs=None, query_concepts=None, match_score=0.8, shard_result=None):
        """回傳所有 world/chat 所需 patch 的 context list。"""
        kg_list = kgs or [_kg()]
        concepts = query_concepts or [{"name": "深度學習", "vector": [0.1] * 384}]

        mock_repo = _mock_kg_repo(kgs=kg_list, kg=kg_list[0] if kg_list else None)
        mock_concept_repo = MagicMock()
        mock_concept_repo.get_public_kgs_concepts = AsyncMock(
            return_value={kg_list[0].id: concepts} if kg_list else {}
        )
        mock_concept_repo.get_all_documents_concepts = AsyncMock(return_value={})
        mock_doc_repo = MagicMock()
        mock_doc_repo.get_by_id = AsyncMock(return_value=None)

        from models.kb_skill import KBSkill
        mock_skill = KBSkill(
            instance_id="local", kb_id=str(kg_list[0].id) if kg_list else str(uuid4()),
            name="TestKG", last_sync="", is_local=True,
        )
        mock_fed = _mock_federation(skills=[mock_skill])

        merged = shard_result or (["深度學習 IS_A 機器學習"], [], [], [])
        mock_shards = AsyncMock(return_value=merged)

        mock_llm = MagicMock()
        async def _stream(prompt):
            for t in ["答案", "在此"]:
                yield t
        mock_llm.stream = _stream

        return {
            "kg_repo": mock_repo,
            "concept_repo": mock_concept_repo,
            "doc_repo": mock_doc_repo,
            "fed": mock_fed,
            "shards": mock_shards,
            "llm": mock_llm,
            "concepts": concepts,
        }

    async def test_empty_question_returns_error_event(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("routers.world.build_query_concepts", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": ""})

        assert res.status_code == 200
        events = _parse_sse(res.text)
        assert any("error" in e for e in events)

    async def test_no_matching_kgs_returns_error(self, test_app):
        concepts = [{"name": "量子", "vector": [0.0] * 384}]
        mock_concept_repo = MagicMock()
        mock_concept_repo.get_public_kgs_concepts = AsyncMock(return_value={})
        mock_concept_repo.get_all_documents_concepts = AsyncMock(return_value={})
        mock_doc_repo = MagicMock()
        mock_doc_repo.get_by_id = AsyncMock(return_value=None)
        mock_fed = _mock_federation(skills=[])

        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("routers.world.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.world.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.world.build_query_concepts", new=AsyncMock(return_value=concepts)), \
             patch("routers.world.compute_match_score", return_value=(0.0, [])), \
             patch("services.federation_service.get_federation_cache", return_value=mock_fed):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": "量子電腦"})

        assert res.status_code == 200
        events = _parse_sse(res.text)
        assert any("error" in e for e in events)

    async def test_happy_path_contains_required_events(self, test_app):
        p = self._base_patches()
        with patch("routers.world.get_driver", return_value=_mock_driver()), \
             patch("routers.world.KnowledgeGraphRepository", return_value=p["kg_repo"]), \
             patch("routers.world.ConceptRepository", return_value=p["concept_repo"]), \
             patch("routers.world.DocumentRepository", return_value=p["doc_repo"]), \
             patch("routers.world.build_query_concepts",
                   new=AsyncMock(return_value=p["concepts"])), \
             patch("routers.world.compute_match_score", return_value=(0.9, ["深度學習"])), \
             patch("routers.world.get_llm_provider", return_value=p["llm"]), \
             patch("services.federation_service.get_federation_cache", return_value=p["fed"]), \
             patch("services.shard_query.query_shards_parallel", new=p["shards"]):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": "深度學習是什麼？"})

        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        events = _parse_sse(res.text)
        event_keys = {k for e in events for k in e}
        assert "status" in event_keys
        assert "kg_route" in event_keys
        assert "done" in event_keys

    async def test_sse_includes_token_events(self, test_app):
        p = self._base_patches()
        with patch("routers.world.get_driver", return_value=_mock_driver()), \
             patch("routers.world.KnowledgeGraphRepository", return_value=p["kg_repo"]), \
             patch("routers.world.ConceptRepository", return_value=p["concept_repo"]), \
             patch("routers.world.DocumentRepository", return_value=p["doc_repo"]), \
             patch("routers.world.build_query_concepts",
                   new=AsyncMock(return_value=p["concepts"])), \
             patch("routers.world.compute_match_score", return_value=(0.9, ["深度學習"])), \
             patch("routers.world.get_llm_provider", return_value=p["llm"]), \
             patch("services.federation_service.get_federation_cache", return_value=p["fed"]), \
             patch("services.shard_query.query_shards_parallel", new=p["shards"]):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": "深度學習是什麼？"})

        events = _parse_sse(res.text)
        token_events = [e for e in events if "token" in e]
        assert len(token_events) >= 1

    async def test_svo_facts_event_present_when_svo(self, test_app):
        p = self._base_patches(shard_result=(["A IS_A B"], [], [], []))
        with patch("routers.world.get_driver", return_value=_mock_driver()), \
             patch("routers.world.KnowledgeGraphRepository", return_value=p["kg_repo"]), \
             patch("routers.world.ConceptRepository", return_value=p["concept_repo"]), \
             patch("routers.world.DocumentRepository", return_value=p["doc_repo"]), \
             patch("routers.world.build_query_concepts",
                   new=AsyncMock(return_value=p["concepts"])), \
             patch("routers.world.compute_match_score", return_value=(0.9, ["A"])), \
             patch("routers.world.get_llm_provider", return_value=p["llm"]), \
             patch("services.federation_service.get_federation_cache", return_value=p["fed"]), \
             patch("services.shard_query.query_shards_parallel", new=p["shards"]):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": "問題", "use_svo": True})

        events = _parse_sse(res.text)
        assert any("svo_facts" in e for e in events)

    async def test_build_query_concepts_returns_empty_returns_error(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("routers.world.build_query_concepts", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/chat", json={"question": "abcxyz"})

        events = _parse_sse(res.text)
        assert any("error" in e for e in events)


# ── GET /world/explore/entities ───────────────────────────────────────────────

class TestExploreEntities:
    def _entity_record(self, name="深度學習", type_="Algorithm", deg=5):
        r = MagicMock()
        r.__getitem__ = lambda self, k: {"name": name, "type": type_, "deg": deg}.get(k)
        r.get = lambda k, d=None: {"name": name, "type": type_, "deg": deg}.get(k, d)
        return r

    async def test_returns_entities_list(self, test_app):
        kg = _kg()
        records = [self._entity_record("深度學習"), self._entity_record("機器學習")]
        driver = _mock_driver(records=records)
        mock_fed = _mock_federation()

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[kg])), \
             patch("services.federation_service.get_federation_cache", return_value=mock_fed):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/explore/entities?q=學習")

        assert res.status_code == 200
        data = res.json()
        assert "entities" in data
        assert "total" in data
        assert "query" in data

    async def test_empty_query_returns_high_freq(self, test_app):
        kg = _kg()
        driver = _mock_driver(records=[self._entity_record("高頻詞", deg=100)])
        mock_fed = _mock_federation()

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[kg])), \
             patch("services.federation_service.get_federation_cache", return_value=mock_fed):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/explore/entities")

        assert res.status_code == 200

    async def test_no_public_kgs_returns_empty(self, test_app):
        driver = _mock_driver()
        mock_fed = _mock_federation()

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("services.federation_service.get_federation_cache", return_value=mock_fed):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/explore/entities?q=測試")

        assert res.status_code == 200
        assert res.json()["total"] == 0

    async def test_invalid_limit_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/world/explore/entities?limit=0")
        assert res.status_code == 422

    async def test_duplicate_names_deduped_by_highest_degree(self, test_app):
        kg = _kg()
        records = [
            self._entity_record("AI", deg=10),
            self._entity_record("AI", deg=5),
        ]
        driver = _mock_driver(records=records)
        mock_fed = _mock_federation()

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[kg])), \
             patch("services.federation_service.get_federation_cache", return_value=mock_fed):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/explore/entities?q=AI")

        entities = res.json()["entities"]
        ai_entities = [e for e in entities if e["name"] == "AI"]
        assert len(ai_entities) == 1
        assert ai_entities[0]["degree"] == 10


# ── GET /world/explore/neighbors ─────────────────────────────────────────────

class TestExploreNeighbors:
    def _neighbor_record(self, src="AI", dst="機器學習", rel="IS_A"):
        r = MagicMock()
        r.__getitem__ = lambda self, k: {
            "src": src, "src_type": "概念", "dst": dst, "dst_type": "概念",
            "rel_type": rel, "verb": rel,
        }.get(k)
        return r

    async def test_returns_nodes_and_edges(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id)
        records = [self._neighbor_record()]
        driver = _mock_driver(records=records)

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kg=kg)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/world/explore/neighbors?entity=AI&kg_id={kg_id}")

        assert res.status_code == 200
        data = res.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    async def test_nonexistent_kg_returns_empty(self, test_app):
        kg_id = uuid4()
        driver = _mock_driver()
        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kg=None)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/world/explore/neighbors?entity=AI&kg_id={kg_id}")

        assert res.status_code == 200
        data = res.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    async def test_private_kg_returns_empty(self, test_app):
        kg_id = uuid4()
        private_kg = _kg(kg_id=kg_id, is_public=False)
        driver = _mock_driver()

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kg=private_kg)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/world/explore/neighbors?entity=AI&kg_id={kg_id}")

        assert res.status_code == 200
        assert res.json()["nodes"] == []

    async def test_missing_entity_param_returns_422(self, test_app):
        kg_id = uuid4()
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get(f"/world/explore/neighbors?kg_id={kg_id}")
        assert res.status_code == 422

    async def test_response_includes_kg_name_and_seed(self, test_app):
        kg_id = uuid4()
        kg = _kg(kg_id=kg_id, name="AI知識庫")
        driver = _mock_driver(records=[self._neighbor_record()])

        with patch("routers.world.get_driver", return_value=driver), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kg=kg)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/world/explore/neighbors?entity=AI&kg_id={kg_id}")

        data = res.json()
        assert data["kg_name"] == "AI知識庫"
        assert data["seed_entity"] == "AI"


# ── GET /world/provenance/facts ───────────────────────────────────────────────

class TestProvenanceFacts:
    async def test_returns_provenance_report_structure(self, test_app):
        from models.provenance import SourcedFact
        sf = SourcedFact(
            fact_str="A IS_A B", subject="A", subject_type="概念",
            rel_type="IS_A", verb="是", object="B", object_type="概念",
            source_doc_id="doc-1", source_doc_title="文件A", confidence=2,
        )
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[_kg()])), \
             patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[sf])), \
             patch("services.entity_alignment.expand_terms", side_effect=lambda t, **kw: t):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/provenance/facts?q=A")

        assert res.status_code == 200
        data = res.json()
        assert "facts" in data
        assert "query_terms" in data
        assert "doc_citations" in data
        assert data["fact_count"] >= 1

    async def test_empty_query_requires_q_param(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/world/provenance/facts")
        assert res.status_code == 422

    async def test_no_facts_returns_empty(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[_kg()])), \
             patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[])), \
             patch("services.entity_alignment.expand_terms", side_effect=lambda t, **kw: t):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/provenance/facts?q=不存在術語")

        assert res.status_code == 200
        assert res.json()["fact_count"] == 0

    async def test_hops_limit_params_accepted(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("services.entity_alignment.expand_terms", side_effect=lambda t, **kw: t):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/provenance/facts?q=AI&hops=2&limit=50")

        assert res.status_code == 200


# ── GET /world/align/synonyms ─────────────────────────────────────────────────

class TestAlignSynonyms:
    async def test_found_synonym_group(self, test_app):
        with patch("services.entity_alignment.get_synonym_group",
                   return_value=["機器學習", "machine learning", "ML"]), \
             patch("services.entity_alignment.expand_terms",
                   return_value=["機器學習", "machine learning", "ML"]):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/align/synonyms?term=機器學習")

        assert res.status_code == 200
        data = res.json()
        assert data["term"] == "機器學習"
        assert data["found"] is True
        assert len(data["synonym_group"]) > 0
        assert "expanded_query" in data

    async def test_unknown_term_returns_not_found(self, test_app):
        with patch("services.entity_alignment.get_synonym_group", return_value=[]), \
             patch("services.entity_alignment.expand_terms", return_value=["未知術語"]):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/align/synonyms?term=未知術語")

        assert res.status_code == 200
        assert res.json()["found"] is False

    async def test_missing_term_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/world/align/synonyms")
        assert res.status_code == 422


# ── GET /world/align/entities ─────────────────────────────────────────────────

class TestAlignEntities:
    async def test_returns_aligned_structure(self, test_app):
        from services.entity_alignment import AlignedEntity, InstanceRef
        aligned = [AlignedEntity(
            canonical_name="AI",
            instances=[InstanceRef(name="AI", entity_type="概念", kg_id="kg-1",
                                   kg_name="KG1", instance_id="local", degree=5)],
        )]
        with patch("routers.world.get_driver"), \
             patch("routers.world.KnowledgeGraphRepository",
                   return_value=_mock_kg_repo(kgs=[])), \
             patch("services.entity_alignment.expand_terms", return_value=["AI"]), \
             patch("services.entity_alignment.align_entity_results", return_value=aligned), \
             patch("services.federation_service.get_federation_cache",
                   return_value=_mock_federation()):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/align/entities?name=AI")

        assert res.status_code == 200
        data = res.json()
        assert "aligned" in data
        assert "search_terms" in data
        assert "total" in data
        assert "cross_instance_count" in data

    async def test_missing_name_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/world/align/entities")
        assert res.status_code == 422


# ── GET /world/federation/status ─────────────────────────────────────────────

class TestFederationStatus:
    async def test_returns_status_dict(self, test_app):
        cache = _mock_federation()
        cache.status.return_value = {"total": 3, "local": 2, "remote": 1}
        with patch("services.federation_service.get_federation_cache", return_value=cache):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/federation/status")

        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 3


# ── GET /world/registry ───────────────────────────────────────────────────────

class TestGetRegistry:
    async def test_returns_registry(self, test_app):
        mock_registry = MagicMock()
        mock_registry.model_dump.return_value = {"instance_id": "local", "skills": []}
        with patch("services.kb_skill_service.load_registry", return_value=mock_registry):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/registry")

        assert res.status_code == 200
        assert "instance_id" in res.json()


# ── POST /world/sync ──────────────────────────────────────────────────────────

class TestSyncRegistry:
    async def test_returns_ok_status(self, test_app):
        with patch("routers.world.get_driver"), \
             patch("services.kb_skill_service.sync_public_kgs",
                   new=AsyncMock(return_value={"synced": 2, "skipped": 0})):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/sync")

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "synced" in data
