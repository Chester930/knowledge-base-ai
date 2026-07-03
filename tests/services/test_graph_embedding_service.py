"""
Graph Embedding Service 測試（THEORETICAL_ARCHITECTURE.md 第9節①：圖拓撲感知共嵌入空間）

測試：
- build_bipartite_graph      : Document/KG ↔ ConceptNode 二分圖建構
- generate_node2vec_walks    : biased random walk 的基本性質（長度、節點合法性）
- train_concept_vectors      : skip-gram 訓練（真實跑 gensim，非 mock）+ 前綴過濾
- build_graph_embeddings     : 端到端流程整合
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from services.graph_embedding_service import (
    build_bipartite_graph,
    build_graph_embeddings,
    generate_node2vec_walks,
    train_concept_vectors,
)


def _rec(**kwargs):
    r = MagicMock()
    r.__getitem__ = lambda self, k: kwargs.get(k)
    return r


def _result(records):
    res = MagicMock()
    res.records = records
    return res


# ── build_bipartite_graph ────────────────────────────────────────────────────

class TestBuildBipartiteGraph:
    async def test_builds_edges_from_document_and_kg(self):
        doc_edges = [_rec(doc_id="d1", concept_name="機器學習")]
        kg_edges = [_rec(kg_id="k1", concept_name="機器學習"), _rec(kg_id="k1", concept_name="深度學習")]
        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=[_result(doc_edges), _result(kg_edges)])

        graph = await build_bipartite_graph(driver)

        assert graph.has_edge("doc:d1", "concept:機器學習")
        assert graph.has_edge("kg:k1", "concept:機器學習")
        assert graph.has_edge("kg:k1", "concept:深度學習")
        assert graph.number_of_nodes() == 4

    async def test_empty_when_no_effective_edges(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))
        graph = await build_bipartite_graph(driver)
        assert graph.number_of_nodes() == 0


# ── generate_node2vec_walks ──────────────────────────────────────────────────

class TestGenerateNode2vecWalks:
    def _make_graph(self):
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
        return g

    def test_produces_correct_walk_count(self):
        g = self._make_graph()
        walks = generate_node2vec_walks(g, num_walks=3, walk_length=5, seed=42)
        assert len(walks) == 3 * g.number_of_nodes()

    def test_walk_length_bounded_by_param(self):
        g = self._make_graph()
        walks = generate_node2vec_walks(g, num_walks=1, walk_length=6, seed=1)
        assert all(len(w) <= 6 for w in walks)

    def test_all_walk_nodes_exist_in_graph(self):
        g = self._make_graph()
        walks = generate_node2vec_walks(g, num_walks=2, walk_length=8, seed=7)
        for w in walks:
            assert all(n in g.nodes for n in w)

    def test_isolated_node_produces_single_node_walk(self):
        g = nx.Graph()
        g.add_node("lonely")
        walks = generate_node2vec_walks(g, num_walks=1, walk_length=10, seed=1)
        assert walks == [["lonely"]]

    def test_deterministic_with_seed(self):
        g = self._make_graph()
        w1 = generate_node2vec_walks(g, num_walks=2, walk_length=5, seed=99)
        w2 = generate_node2vec_walks(g, num_walks=2, walk_length=5, seed=99)
        assert w1 == w2


# ── train_concept_vectors ────────────────────────────────────────────────────

class TestTrainConceptVectors:
    def test_extracts_only_concept_prefixed_vectors(self):
        walks = [
            ["doc:d1", "concept:機器學習", "concept:深度學習", "doc:d1"],
            ["kg:k1", "concept:機器學習", "concept:深度學習"],
        ] * 5  # 重複幾次確保 min_count=1 下有足夠 context 可訓練
        vectors = train_concept_vectors(walks, vector_size=16, window=2, epochs=2)

        assert "機器學習" in vectors
        assert "深度學習" in vectors
        assert all(not k.startswith("doc:") and not k.startswith("kg:") for k in vectors)
        assert len(vectors["機器學習"]) == 16

    def test_empty_walks_produce_empty_result(self):
        # gensim 對空語料會直接回傳空詞彙表，不應拋例外
        vectors = train_concept_vectors([[]], vector_size=8, epochs=1)
        assert vectors == {}


# ── build_graph_embeddings（整合）───────────────────────────────────────────

class TestBuildGraphEmbeddings:
    async def test_end_to_end_persists_vectors(self):
        doc_edges = [
            _rec(doc_id="d1", concept_name="機器學習"),
            _rec(doc_id="d1", concept_name="深度學習"),
            _rec(doc_id="d2", concept_name="深度學習"),
            _rec(doc_id="d2", concept_name="神經網路"),
        ] * 3
        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=[
            _result(doc_edges), _result([]),  # build_bipartite_graph 的兩次查詢
            _result([]),                       # set_concept_graph_vectors 的寫入
        ])

        n = await build_graph_embeddings(driver, num_walks=3, walk_length=6, epochs=2, vector_size=8)

        assert n > 0
        # 第三次呼叫應為 set_concept_graph_vectors 的 UNWIND 寫入
        write_call = driver.execute_query.call_args_list[2]
        assert "SET c.q_vector_graph" in write_call.args[0]

    async def test_empty_graph_skips_training(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))

        n = await build_graph_embeddings(driver)

        assert n == 0
        # 只呼叫了 build_bipartite_graph 的兩次查詢，沒有進到寫入階段
        assert driver.execute_query.call_count == 2
