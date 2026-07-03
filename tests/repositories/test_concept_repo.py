"""
ConceptRepository 測試 — 圖拓撲共嵌入融合邏輯（THEORETICAL_ARCHITECTURE.md 第9節①）

測試：
- _fuse_graph_vector       : 加權融合公式、正規化、缺失欄位的向後相容
- set_concept_graph_vectors: 批次寫入 q_vector_graph 的 Cypher 組裝
- get_all_kgs_concepts / get_public_kgs_concepts: 融合邏輯確實被套用
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from repositories.concept_repo import ConceptRepository, _fuse_graph_vector


def _rec(**kwargs):
    r = MagicMock()
    r.__getitem__ = lambda self, k: kwargs.get(k)
    r.keys = lambda: kwargs.keys()
    return r


def _result(records):
    res = MagicMock()
    res.records = records
    return res


# ── _fuse_graph_vector ───────────────────────────────────────────────────────

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
        assert fused[0] == pytest_approx(0.85)
        assert fused[1] == pytest_approx(0.15)

    def test_normalizes_before_fusing(self):
        # 未正規化的向量（長度非1）應先正規化再加權
        record = {"q_vector": [2.0, 0.0], "q_vector_graph": [0.0, 4.0]}
        result = _fuse_graph_vector(record, alpha=0.5)
        fused = result["q_vector"]
        assert fused[0] == pytest_approx(0.5)
        assert fused[1] == pytest_approx(0.5)

    def test_pops_graph_vector_key_from_output(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0]}
        result = _fuse_graph_vector(record)
        assert "q_vector_graph" not in result

    def test_alpha_one_equals_pure_text_vector(self):
        record = {"q_vector": [1.0, 0.0], "q_vector_graph": [0.0, 1.0]}
        result = _fuse_graph_vector(record, alpha=1.0)
        assert result["q_vector"] == pytest_approx_list([1.0, 0.0])


def pytest_approx(val, tol=1e-6):
    class _Approx(float):
        def __eq__(self, other):
            return abs(other - val) < tol
    return _Approx(val)


def pytest_approx_list(vals, tol=1e-6):
    return [pytest_approx(v, tol) for v in vals]


# ── set_concept_graph_vectors ────────────────────────────────────────────────

class TestSetConceptGraphVectors:
    async def test_skips_query_when_empty(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock()
        await ConceptRepository(driver).set_concept_graph_vectors({})
        driver.execute_query.assert_not_called()

    async def test_sends_unwind_with_items(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))
        await ConceptRepository(driver).set_concept_graph_vectors({
            "機器學習": [0.1, 0.2], "深度學習": [0.3, 0.4],
        })
        call = driver.execute_query.call_args
        assert "SET c.q_vector_graph" in call.args[0]
        items = call.kwargs["items"]
        assert {"name": "機器學習", "vector": [0.1, 0.2]} in items
        assert {"name": "深度學習", "vector": [0.3, 0.4]} in items


# ── 融合套用於 KG 路由查詢 ────────────────────────────────────────────────────

class TestFusionAppliedToKgQueries:
    async def test_get_all_kgs_concepts_applies_fusion(self):
        kg_id = uuid4()
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([
            _rec(kg_id=str(kg_id), concept_id="c1", name="x",
                 q_vector=[1.0, 0.0], q_vector_graph=[0.0, 1.0],
                 interest_score=0.5, professional_score=0.5),
        ]))

        result = await ConceptRepository(driver).get_all_kgs_concepts()

        concept = result[kg_id][0]
        assert concept["q_vector"] != [1.0, 0.0]  # 已被融合，不再是純文字向量
        assert "q_vector_graph" not in concept

    async def test_get_public_kgs_concepts_applies_fusion(self):
        kg_id = uuid4()
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([
            _rec(kg_id=str(kg_id), concept_id="c1", name="x",
                 q_vector=[1.0, 0.0], q_vector_graph=[0.0, 1.0],
                 interest_score=0.5, professional_score=0.5),
        ]))

        result = await ConceptRepository(driver).get_public_kgs_concepts()

        concept = result[kg_id][0]
        assert concept["q_vector"] != [1.0, 0.0]

    async def test_no_graph_vector_leaves_text_vector_untouched(self):
        kg_id = uuid4()
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([
            _rec(kg_id=str(kg_id), concept_id="c1", name="x",
                 q_vector=[1.0, 0.0], q_vector_graph=None,
                 interest_score=0.5, professional_score=0.5),
        ]))

        result = await ConceptRepository(driver).get_all_kgs_concepts()

        assert result[kg_id][0]["q_vector"] == [1.0, 0.0]
