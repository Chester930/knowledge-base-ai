from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

from core.constants import INTEREST_INIT, PROFESSIONAL_INIT, VECTOR_DIM
from repositories.concept_repo import ConceptRepository, _fuse_graph_vector


class _FakeResult:
    def __init__(self, records):
        self.records = records


def _repo_with_driver():
    driver = AsyncMock()
    return ConceptRepository(driver), driver


class TestCreateVectorIndex:
    async def test_uses_default_vector_dim(self):
        repo, driver = _repo_with_driver()
        await repo.create_vector_index()
        _, kwargs = driver.execute_query.call_args
        assert kwargs["dim"] == VECTOR_DIM

    async def test_custom_dim_passed_through(self):
        repo, driver = _repo_with_driver()
        await repo.create_vector_index(dim=768)
        _, kwargs = driver.execute_query.call_args
        assert kwargs["dim"] == 768


class TestGetOrCreate:
    async def test_plain_list_vector_passed_as_is(self):
        repo, driver = _repo_with_driver()
        concept_id = uuid4()
        driver.execute_query.return_value = _FakeResult([{"id": str(concept_id)}])

        result = await repo.get_or_create("機器學習", "AI", [0.1, 0.2, 0.3])

        assert result == concept_id
        _, kwargs = driver.execute_query.call_args
        assert kwargs["q_vector"] == [0.1, 0.2, 0.3]
        assert kwargs["name"] == "機器學習"
        assert kwargs["domain"] == "AI"

    async def test_numpy_like_vector_converted_via_tolist(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"id": str(uuid4())}])

        fake_vector = MagicMock()
        fake_vector.tolist.return_value = [0.5, 0.6]
        await repo.get_or_create("深度學習", "AI", fake_vector)

        _, kwargs = driver.execute_query.call_args
        assert kwargs["q_vector"] == [0.5, 0.6]

    async def test_tuple_vector_converted_to_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"id": str(uuid4())}])

        await repo.get_or_create("X", "domain", (1.0, 2.0))

        _, kwargs = driver.execute_query.call_args
        assert kwargs["q_vector"] == [1.0, 2.0]


class TestInitDocumentConcept:
    async def test_uses_default_scores(self):
        repo, driver = _repo_with_driver()
        doc_id = uuid4()
        await repo.init_document_concept(doc_id, "概念A")
        _, kwargs = driver.execute_query.call_args
        assert kwargs["i"] == INTEREST_INIT
        assert kwargs["p"] == PROFESSIONAL_INIT
        assert kwargs["doc_id"] == str(doc_id)

    async def test_custom_scores_passed_through(self):
        repo, driver = _repo_with_driver()
        await repo.init_document_concept(uuid4(), "概念B", interest=0.9, professional=0.1)
        _, kwargs = driver.execute_query.call_args
        assert kwargs["i"] == 0.9
        assert kwargs["p"] == 0.1


class TestSyncDocumentEffective:
    async def test_issues_two_queries_for_doc_id(self):
        repo, driver = _repo_with_driver()
        doc_id = uuid4()
        await repo.sync_document_effective(doc_id)
        assert driver.execute_query.call_count == 2
        for call in driver.execute_query.call_args_list:
            assert call.kwargs["doc_id"] == str(doc_id)


class TestGetDocumentConcepts:
    async def test_returns_list_of_dicts(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "c1", "name": "AI", "domain": "tech", "q_vector": [1.0],
             "interest_score": 0.8, "professional_score": 0.6},
        ])
        result = await repo.get_document_concepts(uuid4())
        assert result == [
            {"id": "c1", "name": "AI", "domain": "tech", "q_vector": [1.0],
             "interest_score": 0.8, "professional_score": 0.6}
        ]

    async def test_empty_result_returns_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        assert await repo.get_document_concepts(uuid4()) == []


class TestGetAllDocumentsConcepts:
    async def test_groups_concepts_by_doc_id(self):
        repo, driver = _repo_with_driver()
        doc1, doc2 = uuid4(), uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"doc_id": str(doc1), "concept_id": "c1", "name": "AI", "q_vector": [], "interest_score": 0.5, "professional_score": 0.5},
            {"doc_id": str(doc1), "concept_id": "c2", "name": "ML", "q_vector": [], "interest_score": 0.4, "professional_score": 0.4},
            {"doc_id": str(doc2), "concept_id": "c3", "name": "NLP", "q_vector": [], "interest_score": 0.3, "professional_score": 0.3},
        ])

        result = await repo.get_all_documents_concepts()

        assert len(result[doc1]) == 2
        assert len(result[doc2]) == 1

    async def test_exclude_doc_ids_serialized_as_strings(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        excluded = uuid4()

        await repo.get_all_documents_concepts(exclude_doc_ids=[excluded])

        _, kwargs = driver.execute_query.call_args
        assert kwargs["exclude"] == [str(excluded)]

    async def test_none_exclude_doc_ids_becomes_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.get_all_documents_concepts(exclude_doc_ids=None)

        _, kwargs = driver.execute_query.call_args
        assert kwargs["exclude"] == []

    async def test_empty_result_returns_empty_dict(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        assert await repo.get_all_documents_concepts() == {}


class TestKgConceptMethods:
    async def test_init_kg_concept_passes_kg_id_and_scores(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        await repo.init_kg_concept(kg_id, "概念")
        _, kwargs = driver.execute_query.call_args
        assert kwargs["kg_id"] == str(kg_id)
        assert kwargs["i"] == INTEREST_INIT
        assert kwargs["p"] == PROFESSIONAL_INIT

    async def test_sync_kg_effective_issues_two_queries(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        await repo.sync_kg_effective(kg_id)
        assert driver.execute_query.call_count == 2
        for call in driver.execute_query.call_args_list:
            assert call.kwargs["kg_id"] == str(kg_id)

    async def test_get_kg_concepts_returns_records(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "c1", "name": "AI", "domain": "tech", "q_vector": [],
             "interest_score": 0.5, "professional_score": 0.5}
        ])
        result = await repo.get_kg_concepts(uuid4())
        assert len(result) == 1
        assert result[0]["name"] == "AI"

    async def test_get_all_kgs_concepts_groups_by_kg_id(self):
        repo, driver = _repo_with_driver()
        kg1, kg2 = uuid4(), uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"kg_id": str(kg1), "concept_id": "c1", "name": "AI", "q_vector": [], "interest_score": 0.5, "professional_score": 0.5},
            {"kg_id": str(kg2), "concept_id": "c2", "name": "ML", "q_vector": [], "interest_score": 0.5, "professional_score": 0.5},
        ])
        result = await repo.get_all_kgs_concepts()
        assert set(result.keys()) == {kg1, kg2}

    async def test_get_public_kgs_concepts_groups_by_kg_id(self):
        repo, driver = _repo_with_driver()
        kg1 = uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"kg_id": str(kg1), "concept_id": "c1", "name": "AI", "q_vector": [], "interest_score": 0.5, "professional_score": 0.5},
        ])
        result = await repo.get_public_kgs_concepts()
        assert list(result.keys()) == [kg1]


class TestGetAllConcepts:
    async def test_returns_concepts_with_doc_count(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "c1", "name": "AI", "domain": "tech", "doc_count": 5},
        ])
        result = await repo.get_all_concepts()
        assert result[0]["doc_count"] == 5

    async def test_empty_result_returns_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        assert await repo.get_all_concepts() == []


# ── vector_search_concept_ids（兩階段檢索 Stage-1）────────────────────────────

class TestVectorSearchConceptIds:
    async def test_returns_ids_from_index_query(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"id": "c1"}, {"id": "c2"}])

        result = await repo.vector_search_concept_ids([0.1, 0.2], top_k=50)

        assert result == ["c1", "c2"]
        _, kwargs = driver.execute_query.call_args
        assert kwargs["top_k"] == 50
        assert kwargs["vector"] == [0.1, 0.2]

    async def test_numpy_like_vector_converted_via_tolist(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        fake_vector = MagicMock()
        fake_vector.tolist.return_value = [0.5, 0.6]

        await repo.vector_search_concept_ids(fake_vector, top_k=10)

        _, kwargs = driver.execute_query.call_args
        assert kwargs["vector"] == [0.5, 0.6]

    async def test_empty_index_returns_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        assert await repo.vector_search_concept_ids([0.1], top_k=10) == []


# ── set_concept_graph_vectors（圖拓撲共嵌入寫入）──────────────────────────────

class TestSetConceptGraphVectors:
    async def test_skips_query_when_empty(self):
        repo, driver = _repo_with_driver()
        await repo.set_concept_graph_vectors({})
        driver.execute_query.assert_not_called()

    async def test_sends_unwind_with_items(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.set_concept_graph_vectors({
            "機器學習": [0.1, 0.2], "深度學習": [0.3, 0.4],
        })

        call = driver.execute_query.call_args
        assert "SET c.q_vector_graph" in call.args[0]
        items = call.kwargs["items"]
        assert {"name": "機器學習", "vector": [0.1, 0.2]} in items
        assert {"name": "深度學習", "vector": [0.3, 0.4]} in items


# ── _fuse_graph_vector（圖拓撲共嵌入融合公式，第9節①）─────────────────────────

class TestFuseGraphVector:
    def test_missing_graph_vector_returns_unchanged(self):
        record = {"q_vector": [1.0, 0.0], "name": "x"}
        result = _fuse_graph_vector(dict(record))
        assert result["q_vector"] == [1.0, 0.0]

    def test_none_graph_vector_returns_unchanged(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": None}
        result = _fuse_graph_vector(record)
        assert result["q_vector"] == [1.0, 0.0]
        assert "q_vector_graph" not in result

    def test_mismatched_dimensions_returns_unchanged(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [1.0, 0.0, 0.0]}
        result = _fuse_graph_vector(record)
        assert result["q_vector"] == [1.0, 0.0]

    def test_fuses_with_alpha_weighting(self):
        # 兩個已正規化的正交向量，alpha=0.85 應主要偏向文字向量
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0]}
        result = _fuse_graph_vector(record, alpha=0.85)
        fused = result["q_vector"]
        assert fused[0] == pytest.approx(0.85)
        assert fused[1] == pytest.approx(0.15)

    def test_normalizes_before_fusing(self):
        # 未正規化的向量（長度非1）應先正規化再加權
        record = {"q_vector": [2.0, 0.0], "q_vector_graph": [0.0, 4.0]}
        result = _fuse_graph_vector(record, alpha=0.5)
        fused = result["q_vector"]
        assert fused[0] == pytest.approx(0.5)
        assert fused[1] == pytest.approx(0.5)

    def test_pops_graph_vector_key_from_output(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0]}
        result = _fuse_graph_vector(record)
        assert "q_vector_graph" not in result

    def test_alpha_one_equals_pure_text_vector(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0]}
        result = _fuse_graph_vector(record, alpha=1.0)
        assert result["q_vector"][0] == pytest.approx(1.0)
        assert result["q_vector"][1] == pytest.approx(0.0)


# ── 融合套用於 KG 路由查詢 + concept_ids 兩階段過濾 ────────────────────────────

class TestFusionAndConceptIdsFilterAppliedToKgQueries:
    async def test_get_all_kgs_concepts_applies_fusion(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"kg_id": str(kg_id), "concept_id": "c1", "name": "x",
             "q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0],
             "interest_score": 0.5, "professional_score": 0.5},
        ])

        result = await repo.get_all_kgs_concepts()

        concept = result[kg_id][0]
        assert concept["q_vector"] != [1.0, 0.0]  # 已被融合，不再是純文字向量
        assert "q_vector_graph" not in concept

    async def test_get_public_kgs_concepts_applies_fusion(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"kg_id": str(kg_id), "concept_id": "c1", "name": "x",
             "q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0],
             "interest_score": 0.5, "professional_score": 0.5},
        ])

        result = await repo.get_public_kgs_concepts()

        concept = result[kg_id][0]
        assert concept["q_vector"] != [1.0, 0.0]

    async def test_no_graph_vector_leaves_text_vector_untouched(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.return_value = _FakeResult([
            {"kg_id": str(kg_id), "concept_id": "c1", "name": "x",
             "q_vector": [1.0, 0.0], "q_vector_graph": None,
             "interest_score": 0.5, "professional_score": 0.5},
        ])

        result = await repo.get_all_kgs_concepts()

        assert result[kg_id][0]["q_vector"] == [1.0, 0.0]

    async def test_get_all_kgs_concepts_passes_concept_ids_filter(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.get_all_kgs_concepts(concept_ids=["c1", "c2"])

        _, kwargs = driver.execute_query.call_args
        assert kwargs["concept_ids"] == ["c1", "c2"]

    async def test_get_all_kgs_concepts_none_concept_ids_means_no_filter(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.get_all_kgs_concepts()

        _, kwargs = driver.execute_query.call_args
        assert kwargs["concept_ids"] is None

    async def test_get_all_documents_concepts_passes_concept_ids_filter(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.get_all_documents_concepts(concept_ids=["c1"])

        _, kwargs = driver.execute_query.call_args
        assert kwargs["concept_ids"] == ["c1"]
