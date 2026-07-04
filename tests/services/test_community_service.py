"""
Community Service 測試（THEORETICAL_ARCHITECTURE.md 第9節⑤：多層次社群摘要檢索）

測試：
- build_communities_for_kg : 社群偵測（networkx Louvain）+ LLM 摘要 + 持久化
- _summarize_community     : LLM 摘要 prompt 組裝與容錯
- get_community_summaries  : 讀取已建立的社群摘要
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.community_service import (
    _summarize_community,
    build_communities_for_kg,
    get_community_summaries,
)


def _rec(**kwargs):
    r = MagicMock()
    r.__getitem__ = lambda self, k: kwargs.get(k)
    r.get = lambda k, d=None: kwargs.get(k, d)
    r.keys = lambda: kwargs.keys()
    return r


def _result(records):
    res = MagicMock()
    res.records = records
    return res


def _two_cluster_edges():
    """兩個緊密子群（A/B）+ 一條橋接邊，Louvain 應穩定切成 2 群。"""
    cluster_a = [("a1", "a2"), ("a1", "a3"), ("a1", "a4"), ("a2", "a3"), ("a2", "a4"), ("a3", "a4")]
    cluster_b = [("b1", "b2"), ("b1", "b3"), ("b1", "b4"), ("b2", "b3"), ("b2", "b4"), ("b3", "b4")]
    bridge = [("a1", "b1")]
    return [_rec(a=a, b=b) for a, b in cluster_a + cluster_b + bridge]


# ── build_communities_for_kg ────────────────────────────────────────────────

class TestBuildCommunitiesForKg:
    async def test_no_edges_returns_zero(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))
        with patch("services.community_service.get_driver", return_value=driver):
            n = await build_communities_for_kg(uuid4())
        assert n == 0

    async def test_graph_smaller_than_min_size_returns_zero(self):
        driver = MagicMock()
        # 只有一條邊（2 個節點），min_size 預設 3
        driver.execute_query = AsyncMock(return_value=_result([_rec(a="x", b="y")]))
        with patch("services.community_service.get_driver", return_value=driver):
            n = await build_communities_for_kg(uuid4(), min_size=3)
        assert n == 0

    async def test_detects_two_communities_and_persists(self):
        driver = MagicMock()

        async def _execute(query, **kwargs):
            if "WHERE s.name <> o.name" in query:
                return _result(_two_cluster_edges())
            if "DETACH DELETE" in query:
                return _result([])
            if "CREATE (c:Community" in query:
                return _result([])
            # _sample_community_facts
            return _result([_rec(s="a1", rel_type="USES", verb="使用", o="a2")])

        driver.execute_query = AsyncMock(side_effect=_execute)

        with patch("services.community_service.get_driver", return_value=driver), \
             patch("services.community_service._summarize_community",
                   new=AsyncMock(return_value="這是一個摘要。")):
            n = await build_communities_for_kg(uuid4(), min_size=3)

        assert n == 2  # 兩個緊密子群各自成一個社群

    async def test_communities_below_min_size_filtered_out(self):
        driver = MagicMock()

        async def _execute(query, **kwargs):
            if "WHERE s.name <> o.name" in query:
                # 一個大群（4節點全連通）+ 一對孤立小群（2節點），min_size=3 應濾掉小群
                return _result([
                    _rec(a="a1", b="a2"), _rec(a="a1", b="a3"), _rec(a="a1", b="a4"),
                    _rec(a="a2", b="a3"), _rec(a="a2", b="a4"), _rec(a="a3", b="a4"),
                    _rec(a="x1", b="x2"),
                ])
            if "DETACH DELETE" in query:
                return _result([])
            if "CREATE (c:Community" in query:
                return _result([])
            return _result([])

        driver.execute_query = AsyncMock(side_effect=_execute)

        with patch("services.community_service.get_driver", return_value=driver), \
             patch("services.community_service._summarize_community",
                   new=AsyncMock(return_value="摘要")):
            n = await build_communities_for_kg(uuid4(), min_size=3)

        assert n == 1  # 只有 4 節點的群符合 min_size=3

    async def test_empty_summary_skips_persistence(self):
        driver = MagicMock()
        create_calls = []

        async def _execute(query, **kwargs):
            if "WHERE s.name <> o.name" in query:
                return _result(_two_cluster_edges())
            if "DETACH DELETE" in query:
                return _result([])
            if "CREATE (c:Community" in query:
                create_calls.append(query)
                return _result([])
            return _result([])

        driver.execute_query = AsyncMock(side_effect=_execute)

        with patch("services.community_service.get_driver", return_value=driver), \
             patch("services.community_service._summarize_community",
                   new=AsyncMock(return_value="")):  # LLM 摘要失敗回傳空字串
            n = await build_communities_for_kg(uuid4(), min_size=3)

        assert n == 0
        assert create_calls == []


# ── _summarize_community ────────────────────────────────────────────────────

class TestSummarizeCommunity:
    async def test_uses_facts_when_available(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=" 這是關於 A 的摘要。 ")
        with patch("services.community_service.get_llm_provider", return_value=mock_llm):
            result = await _summarize_community(["a", "b"], ["a -[使用]→ b"])
        assert result == "這是關於 A 的摘要。"
        prompt = mock_llm.generate.call_args[0][0]
        assert "a -[使用]→ b" in prompt

    async def test_falls_back_to_names_when_no_facts(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="摘要")
        with patch("services.community_service.get_llm_provider", return_value=mock_llm):
            await _summarize_community(["概念A", "概念B"], [])
        prompt = mock_llm.generate.call_args[0][0]
        assert "概念A" in prompt and "概念B" in prompt

    async def test_llm_error_returns_empty_string(self):
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM 掛了"))
        with patch("services.community_service.get_llm_provider", return_value=mock_llm):
            result = await _summarize_community(["a"], [])
        assert result == ""


# ── get_community_summaries ─────────────────────────────────────────────────

class TestGetCommunitySummaries:
    async def test_returns_summaries_sorted_by_member_count(self):
        driver = MagicMock()
        records = [
            _rec(summary="摘要1", member_count=10, top_entities=["a", "b"]),
            _rec(summary="摘要2", member_count=5, top_entities=["c"]),
        ]
        driver.execute_query = AsyncMock(return_value=_result(records))
        with patch("services.community_service.get_driver", return_value=driver):
            result = await get_community_summaries(uuid4())
        assert len(result) == 2
        assert result[0]["summary"] == "摘要1"
        assert result[0]["member_count"] == 10

    async def test_empty_when_no_communities(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))
        with patch("services.community_service.get_driver", return_value=driver):
            result = await get_community_summaries(uuid4())
        assert result == []
