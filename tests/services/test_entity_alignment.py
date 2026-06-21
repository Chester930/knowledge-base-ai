"""
Entity Alignment 測試 — Phase 2d

測試：
- expand_terms       : 同義詞展開（zh↔en 術語）
- get_synonym_group  : 同義詞組查找
- align_entity_results : 跨 instance 實體對齊
- AlignedEntity      : max_degree / instance_count / to_dict
"""
from __future__ import annotations
import pytest

from services.entity_alignment import (
    AlignedEntity,
    InstanceRef,
    align_entity_results,
    expand_terms,
    get_synonym_group,
)


# ── expand_terms ──────────────────────────────────────────────────────────────

class TestExpandTerms:
    def test_known_zh_term_expands(self):
        result = expand_terms(["機器學習"])
        assert "機器學習" in result
        assert len(result) > 1

    def test_known_en_term_expands(self):
        result = expand_terms(["machine learning"])
        assert "machine learning" in result
        assert len(result) > 1

    def test_zh_en_cross_expansion(self):
        result = expand_terms(["機器學習"])
        joined = " ".join(result).lower()
        assert "machine learning" in joined or "ml" in joined

    def test_unknown_term_returned_unchanged(self):
        result = expand_terms(["完全未知的術語XYZ"])
        assert result == ["完全未知的術語XYZ"]

    def test_original_term_is_first(self):
        result = expand_terms(["深度學習"])
        assert result[0] == "深度學習"

    def test_max_expansion_respected(self):
        # 強化學習同義詞組有 3 個成員，max_expansion=1 只展開 1 個
        result = expand_terms(["強化學習"], max_expansion=1)
        # 原詞 + 最多 1 個展開
        assert len(result) <= 2

    def test_deduplication_when_two_terms_same_group(self):
        # ML 和 machine learning 同組，不應重複
        result = expand_terms(["機器學習", "machine learning"])
        seen = set()
        for t in result:
            assert t not in seen, f"Duplicate: {t}"
            seen.add(t)

    def test_empty_list_returns_empty(self):
        assert expand_terms([]) == []

    def test_multiple_terms_all_expanded(self):
        result = expand_terms(["機器學習", "知識圖譜"])
        assert "機器學習" in result
        assert "知識圖譜" in result
        assert len(result) > 2

    def test_case_insensitive_lookup(self):
        result_lower = expand_terms(["machine learning"])
        result_upper = expand_terms(["Machine Learning"])
        # 兩者都應觸發展開
        assert len(result_lower) > 1
        assert len(result_upper) > 1


# ── get_synonym_group ─────────────────────────────────────────────────────────

class TestGetSynonymGroup:
    def test_known_term_returns_group(self):
        group = get_synonym_group("機器學習")
        assert len(group) > 1
        assert "機器學習" in group

    def test_english_term_returns_same_group(self):
        group = get_synonym_group("machine learning")
        assert "機器學習" in group

    def test_abbreviation_term(self):
        group = get_synonym_group("ml")
        assert "機器學習" in group

    def test_unknown_term_returns_empty(self):
        assert get_synonym_group("不存在的術語") == []

    def test_result_is_sorted(self):
        group = get_synonym_group("深度學習")
        assert group == sorted(group)

    def test_case_insensitive(self):
        group1 = get_synonym_group("ML")
        group2 = get_synonym_group("ml")
        assert group1 == group2


# ── AlignedEntity ─────────────────────────────────────────────────────────────

class TestAlignedEntity:
    def _ref(self, name="A", instance_id="inst1", degree=5) -> InstanceRef:
        return InstanceRef(name=name, instance_id=instance_id, degree=degree)

    def test_max_degree_from_instances(self):
        entity = AlignedEntity(
            canonical_name="A",
            instances=[self._ref(degree=3), self._ref(degree=7), self._ref(degree=1)],
        )
        assert entity.max_degree == 7

    def test_max_degree_empty_instances(self):
        entity = AlignedEntity(canonical_name="A")
        assert entity.max_degree == 0

    def test_instance_count_unique_instance_ids(self):
        entity = AlignedEntity(
            canonical_name="A",
            instances=[
                self._ref(instance_id="inst1"),
                self._ref(instance_id="inst1"),  # same instance
                self._ref(instance_id="inst2"),
            ],
        )
        assert entity.instance_count == 2

    def test_to_dict_structure(self):
        ref = self._ref(name="機器學習", instance_id="chester", degree=10)
        ref.kg_name = "AI KG"
        ref.kg_id = "kg-001"
        ref.entity_type = "算法"
        entity = AlignedEntity(
            canonical_name="機器學習",
            synonym_group=["機器學習", "machine learning", "ml"],
            instances=[ref],
        )
        d = entity.to_dict()
        assert d["canonical_name"] == "機器學習"
        assert d["instance_count"] == 1
        assert d["max_degree"] == 10
        assert len(d["instances"]) == 1
        assert d["instances"][0]["name"] == "機器學習"
        assert d["instances"][0]["type"] == "算法"


# ── align_entity_results ─────────────────────────────────────────────────────

class TestAlignEntityResults:
    def _raw(self, name: str, instance_id: str = "local",
             kg_id: str = "kg1", kg_name: str = "KG", degree: int = 1, type_: str = "概念"):
        return {"name": name, "instance_id": instance_id,
                "kg_id": kg_id, "kg_name": kg_name, "degree": degree, "type": type_}

    def test_same_name_different_instances_merged(self):
        raw = [
            self._raw("機器學習", instance_id="inst1", degree=5),
            self._raw("機器學習", instance_id="inst2", degree=3),
        ]
        result = align_entity_results(raw)
        assert len(result) == 1
        assert result[0].instance_count == 2

    def test_synonyms_merged(self):
        raw = [
            self._raw("機器學習", degree=5),
            self._raw("machine learning", degree=3),
        ]
        result = align_entity_results(raw)
        # Both are in the same synonym group → merged into 1
        assert len(result) == 1

    def test_unrelated_entities_separate(self):
        raw = [
            self._raw("機器學習", degree=5),
            self._raw("完全不相關術語ABC", degree=3),
        ]
        result = align_entity_results(raw)
        assert len(result) == 2

    def test_sorted_by_max_degree_descending(self):
        raw = [
            self._raw("Python", degree=2),
            self._raw("深度學習", degree=10),
            self._raw("Transformer", degree=5),
        ]
        result = align_entity_results(raw)
        degrees = [r.max_degree for r in result]
        assert degrees == sorted(degrees, reverse=True)

    def test_empty_input_returns_empty(self):
        assert align_entity_results([]) == []

    def test_canonical_name_is_first_seen(self):
        raw = [self._raw("機器學習", degree=5)]
        result = align_entity_results(raw)
        assert result[0].canonical_name == "機器學習"
