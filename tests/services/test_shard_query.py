"""
Shard Query Engine 測試 — Phase 2c

測試：
- ShardResult    : dataclass 預設值與欄位
- route_skill_score : top_concepts 關鍵字路由分數
- query_shards_parallel : 並行查詢、超時、去重、同義詞展開
"""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from models.kb_skill import ConceptScore, KBSkill
from services.shard_query import ShardResult, query_shards_parallel, route_skill_score


# ── 輔助 ──────────────────────────────────────────────────────────────────────

def _skill(
    name: str = "TestKG",
    is_local: bool = True,
    kb_id: str | None = None,
    instance_id: str = "local",
    top_concepts: list[ConceptScore] | None = None,
    db_name: str | None = None,
) -> KBSkill:
    return KBSkill(
        instance_id=instance_id,
        kb_id=kb_id or str(uuid4()),
        name=name,
        is_local=is_local,
        db_name=db_name,
        last_sync="2026-06-21T00:00:00Z",
        top_concepts=top_concepts or [],
    )


def _shard_result(shard_id: str = "s1", status: str = "ok",
                  facts: list[str] | None = None) -> ShardResult:
    return ShardResult(
        shard_id=shard_id,
        shard_name="Test",
        instance_id="local",
        status=status,
        facts=facts or [],
    )


# ── ShardResult ───────────────────────────────────────────────────────────────

class TestShardResult:
    def test_default_values(self):
        sr = ShardResult(shard_id="x", shard_name="X", instance_id="i", status="ok")
        assert sr.facts == []
        assert sr.source_docs == []
        assert sr.elapsed_ms == 0
        assert sr.sourced_facts == []

    def test_status_values(self):
        for status in ("ok", "timeout", "offline", "error"):
            sr = _shard_result(status=status)
            assert sr.status == status

    def test_facts_stored(self):
        sr = _shard_result(facts=["A→B", "C→D"])
        assert len(sr.facts) == 2


# ── route_skill_score ─────────────────────────────────────────────────────────

class TestRouteSkillScore:
    def _cs(self, name: str) -> ConceptScore:
        return ConceptScore(name=name, score=0.9)

    def test_empty_top_concepts_returns_zero(self):
        skill = _skill(top_concepts=[])
        assert route_skill_score(skill, ["機器學習"]) == 0.0

    def test_empty_query_terms_returns_zero(self):
        skill = _skill(top_concepts=[self._cs("機器學習")])
        assert route_skill_score(skill, []) == 0.0

    def test_exact_match_returns_one(self):
        skill = _skill(top_concepts=[self._cs("機器學習")])
        score = route_skill_score(skill, ["機器學習"])
        assert score == pytest.approx(1.0)

    def test_partial_match_returns_fraction(self):
        skill = _skill(top_concepts=[self._cs("機器學習"), self._cs("深度學習")])
        score = route_skill_score(skill, ["機器學習", "transformer"])
        assert 0.0 < score < 1.0

    def test_no_match_returns_zero(self):
        skill = _skill(top_concepts=[self._cs("量子運算")])
        score = route_skill_score(skill, ["機器學習", "深度學習"])
        assert score == pytest.approx(0.0)

    def test_substring_match(self):
        # "機器" in "機器學習" → should match
        skill = _skill(top_concepts=[self._cs("機器學習")])
        score = route_skill_score(skill, ["機器"])
        assert score > 0.0

    def test_case_insensitive(self):
        skill = _skill(top_concepts=[self._cs("Deep Learning")])
        score = route_skill_score(skill, ["deep learning"])
        assert score == pytest.approx(1.0)


# ── query_shards_parallel ─────────────────────────────────────────────────────

class TestQueryShardsParallel:
    async def test_empty_skills_returns_empty(self):
        facts, docs, shards, sourced = await query_shards_parallel([], ["機器學習"])
        assert facts == [] and docs == [] and shards == [] and sourced == []

    async def test_empty_terms_returns_empty(self):
        skill = _skill(is_local=True)
        facts, docs, shards, sourced = await query_shards_parallel([skill], [])
        assert facts == [] and docs == [] and shards == []

    async def test_local_shard_queried(self):
        skill = _skill(is_local=True)
        mock_sf = MagicMock()
        mock_sf.fact_str = "A→B"
        mock_sf.source_doc_id = "doc1"

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[mock_sf])):
            facts, docs, shards, sourced = await query_shards_parallel(
                [skill], ["機器學習"], expand_synonyms=False
            )

        assert len(facts) == 1
        assert "A→B" in facts
        assert shards[0].status == "ok"

    async def test_facts_deduplicated(self):
        skill1 = _skill(name="KG1", kb_id=str(uuid4()), is_local=True)
        skill2 = _skill(name="KG2", kb_id=str(uuid4()), is_local=True)
        mock_sf = MagicMock()
        mock_sf.fact_str = "SAME_FACT"
        mock_sf.source_doc_id = "doc1"

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[mock_sf])):
            facts, docs, shards, sourced = await query_shards_parallel(
                [skill1, skill2], ["query"], expand_synonyms=False
            )

        assert facts.count("SAME_FACT") == 1

    async def test_timeout_shard_marked_offline(self):
        skill = _skill(is_local=True)

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)
            return []

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=_slow):
            facts, docs, shards, sourced = await query_shards_parallel(
                [skill], ["query"], timeout=0.05, expand_synonyms=False
            )

        assert shards[0].status == "timeout"
        assert facts == []

    async def test_error_shard_does_not_raise(self):
        skill = _skill(is_local=True)

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(side_effect=RuntimeError("DB crash"))):
            facts, docs, shards, sourced = await query_shards_parallel(
                [skill], ["query"], expand_synonyms=False
            )

        assert shards[0].status == "error"
        assert facts == []

    async def test_expand_synonyms_calls_expand_terms(self):
        skill = _skill(is_local=True)

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[])), \
             patch("services.entity_alignment.expand_terms",
                   wraps=lambda t, **kw: t) as mock_expand:
            await query_shards_parallel([skill], ["機器學習"], expand_synonyms=True)

        mock_expand.assert_called_once()

    async def test_returns_four_tuple(self):
        result = await query_shards_parallel([], ["x"])
        assert len(result) == 4

    async def test_multiple_shards_all_queried(self):
        skills = [_skill(name=f"KG{i}", kb_id=str(uuid4()), is_local=True) for i in range(3)]
        mock_sf = MagicMock()
        mock_sf.fact_str = "fact"
        mock_sf.source_doc_id = ""

        with patch("services.svo_service.query_svo_facts_with_provenance",
                   new=AsyncMock(return_value=[mock_sf])):
            facts, docs, shards, sourced = await query_shards_parallel(
                skills, ["query"], expand_synonyms=False
            )

        assert len(shards) == 3
        assert all(s.status == "ok" for s in shards)
