from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.concept_engine import (
    _alignment,
    _cosine,
    _magnitude,
    compute_match_score,
    extract_concepts,
    build_query_concepts,
)


# ── _cosine ─────────────────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero_vectors_return_zero(self):
        assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_result_clamped_to_one(self):
        # floating-point accumulation should never exceed 1
        v = [0.577350] * 3
        result = _cosine(v, v)
        assert result <= 1.0
        assert result >= -1.0

    def test_similar_vectors_high_score(self):
        v1 = [1.0, 1.0, 0.0]
        v2 = [1.0, 0.9, 0.1]
        assert _cosine(v1, v2) > 0.98


# ── _alignment ──────────────────────────────────────────────────────────────

class TestAlignment:
    def test_identical_scores_return_one(self):
        assert _alignment(0.5, 0.5, 0.5, 0.5) == pytest.approx(1.0)

    def test_max_gap_returns_zero(self):
        # |1-0| + |1-0| = 2, (2/2) = 1, 1-1 = 0
        assert _alignment(1.0, 1.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_never_negative(self):
        assert _alignment(0.0, 0.0, 1.0, 1.0) >= 0.0

    def test_partial_mismatch(self):
        result = _alignment(0.8, 0.6, 0.6, 0.4)
        assert 0.0 < result < 1.0


# ── _magnitude ──────────────────────────────────────────────────────────────

class TestMagnitude:
    def test_all_ones(self):
        assert _magnitude(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert _magnitude(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_average_of_four(self):
        assert _magnitude(0.4, 0.6, 0.8, 0.2) == pytest.approx(0.5)

    def test_mixed_values(self):
        assert _magnitude(1.0, 0.0, 1.0, 0.0) == pytest.approx(0.5)


# ── compute_match_score ─────────────────────────────────────────────────────

class TestComputeMatchScore:
    def _vec(self, *vals, dim=10):
        return list(vals) + [0.0] * (dim - len(vals))

    def test_empty_query_returns_zero(self):
        dc = {"name": "x", "q_vector": self._vec(1.0), "interest_score": 0.5, "professional_score": 0.5}
        score, matched = compute_match_score([], [dc])
        assert score == 0.0
        assert matched == []

    def test_empty_docs_returns_zero(self):
        qc = {"name": "x", "q_vector": self._vec(1.0), "interest_score": 0.5, "professional_score": 0.5}
        score, matched = compute_match_score([qc], [])
        assert score == 0.0
        assert matched == []

    def test_identical_concepts_produce_high_score(self):
        c = {"name": "機器學習", "q_vector": self._vec(1.0), "interest_score": 0.8, "professional_score": 0.8}
        score, matched = compute_match_score([c], [c])
        assert score > 0.5
        assert "機器學習" in matched

    def test_orthogonal_vectors_score_zero(self):
        qc = {"name": "a", "q_vector": self._vec(1.0, 0.0), "interest_score": 0.8, "professional_score": 0.8}
        dc = {"name": "b", "q_vector": self._vec(0.0, 1.0), "interest_score": 0.8, "professional_score": 0.8}
        score, matched = compute_match_score([qc], [dc])
        assert score == pytest.approx(0.0)
        assert matched == []

    def test_score_rounded_to_four_decimals(self):
        c = {"name": "a", "q_vector": self._vec(1.0), "interest_score": 0.8, "professional_score": 0.8}
        score, _ = compute_match_score([c], [c])
        assert score == round(score, 4)

    def test_matched_concepts_capped_at_five(self):
        v = self._vec(1.0)
        concepts = [
            {"name": f"c{i}", "q_vector": v, "interest_score": 0.9, "professional_score": 0.9}
            for i in range(8)
        ]
        _, matched = compute_match_score(concepts, concepts)
        assert len(matched) <= 5

    def test_low_cosine_pair_not_in_matched(self):
        # cos < 0.01 threshold means near-orthogonal concepts are ignored
        qc = {"name": "a", "q_vector": [1.0, 0.001] + [0.0]*8, "interest_score": 0.9, "professional_score": 0.9}
        dc = {"name": "b", "q_vector": [0.001, 1.0] + [0.0]*8, "interest_score": 0.9, "professional_score": 0.9}
        score, matched = compute_match_score([qc], [dc])
        assert "b" not in matched


# ── extract_concepts ────────────────────────────────────────────────────────

class TestExtractConcepts:
    def _make_http_mock(self, response_text: str):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": response_text}
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        return mock_client

    async def test_parses_newline_separated_concepts(self):
        mock_client = self._make_http_mock("機器學習\n深度學習\n神經網路")
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await extract_concepts("人工智慧文件")

        assert "機器學習" in result
        assert "深度學習" in result
        assert "神經網路" in result

    async def test_empty_lines_are_filtered(self):
        mock_client = self._make_http_mock("概念A\n\n\n概念B\n")
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await extract_concepts("text")

        assert "" not in result
        assert all(c.strip() for c in result)

    async def test_respects_max_concept_count(self):
        many = "\n".join(f"概念{i}" for i in range(20))
        mock_client = self._make_http_mock(many)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await extract_concepts("text")

        assert len(result) <= 8  # concept_extraction_max default

    async def test_http_error_returns_empty_list(self):
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("連線失敗")
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await extract_concepts("text")

        assert result == []

    async def test_text_truncated_to_3000_chars(self):
        # The 3001st character onward is a unique marker — should not appear in the prompt
        long_text = "a" * 3000 + "TRUNCATION_SENTINEL_XYZ" + "b" * 1000
        mock_client = self._make_http_mock("概念A")
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            await extract_concepts(long_text)

        sent_prompt = mock_client.post.call_args[1]["json"]["prompt"]
        assert "TRUNCATION_SENTINEL_XYZ" not in sent_prompt


# ── build_query_concepts ────────────────────────────────────────────────────

class TestBuildQueryConcepts:
    async def test_returns_list_of_dicts_with_required_keys(self):
        mock_svc = MagicMock()
        mock_svc.encode.return_value = [0.1] * 384

        with patch("services.concept_engine.get_embedding_service", return_value=mock_svc), \
             patch("services.concept_engine.extract_concepts", new=AsyncMock(return_value=["概念A", "概念B"])):
            result = await build_query_concepts("test query")

        assert len(result) == 2
        for item in result:
            assert "name" in item
            assert "q_vector" in item
            assert "interest_score" in item
            assert "professional_score" in item

    async def test_fallback_when_no_concepts_extracted(self):
        mock_svc = MagicMock()
        mock_svc.encode.return_value = [0.1] * 384

        with patch("services.concept_engine.get_embedding_service", return_value=mock_svc), \
             patch("services.concept_engine.extract_concepts", new=AsyncMock(return_value=[])):
            result = await build_query_concepts("fallback text")

        # Should fall back to using the raw text
        assert len(result) == 1
        assert result[0]["name"] == "fallback text"[:50]
