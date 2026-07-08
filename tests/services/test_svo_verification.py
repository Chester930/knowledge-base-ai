"""
SVO 品質驗證機制測試 — 抽取 → 驗證 → 重試 → 本體擴充

- verify_svo_extraction       : 第二個模型（審查員）逐條三元組判定
- propose_ontology_extension  : 第三個模型（本體設計師）提議新類型
- extract_svo_verified        : 整合流程（含重試次數上限、停用開關）

全部使用 mocked LLM provider，不連線真實 Ollama/Neo4j。
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch

from models.knowledge_graph import SVOTriple
from services.svo_service import (
    verify_svo_extraction,
    propose_ontology_extension,
    extract_svo_verified,
    _extract_json_objects,
)
import services.ontology_service as ontology_service


def _mock_llm(json_response: str | None = None, raise_exc: Exception | None = None):
    llm = AsyncMock()
    if raise_exc:
        llm.generate_json.side_effect = raise_exc
        llm.generate.side_effect = raise_exc
    else:
        llm.generate_json.return_value = json_response
        llm.generate.return_value = json_response
    return llm


@pytest.fixture(autouse=True)
def _isolated_ontology_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ontology_service, "_EXT_FILE", tmp_path / "ontology_extensions.json")
    ontology_service._reset_cache_for_tests()
    yield
    ontology_service._reset_cache_for_tests()


def _sample_triples() -> list[SVOTriple]:
    return [
        SVOTriple(subject="A公司", subject_type="組織", rel_type="VIOLATES",
                   verb="違反", object="勞基法第32條", object_type="法規"),
        SVOTriple(subject="B機關", subject_type="政府機關", rel_type="APPLIES_TO",
                   verb="處分", object="A公司", object_type="組織"),
    ]


# ── _extract_json_objects（2026-07-08 人工逐階段檢驗發現的真實模型行為）───────
# 用 run_svo_pipeline_debug.py 對真實 Ollama 模型手動測試時發現：即使 prompt 明確
# 要求「只輸出 JSON 陣列」，模型有時仍直接輸出單一裸物件（不包陣列），導致原本
# 只找 `[...]` 的正則解析失敗、誤觸發保守 fallback（全數視為通過）。以下測試涵蓋
# 這個真實觀察到的行為，以及其他常見的模型輸出變形。

class TestExtractJsonObjects:
    def test_standard_array(self):
        raw = '[{"index": 0, "accepted": true}, {"index": 1, "accepted": false}]'
        items = _extract_json_objects(raw)
        assert len(items) == 2
        assert items[0]["index"] == 0
        assert items[1]["accepted"] is False

    def test_bare_single_object_not_wrapped_in_array(self):
        """重現真實觀察到的模型行為：只想評論一筆時，直接輸出裸物件而非陣列。"""
        raw = (
            '{\n'
            '    "index": 0,\n'
            '    "accepted": false,\n'
            '    "reason": "原文未提到臺灣湯淺電池股份有限公司適用於宜蘭縣政府"\n'
            '}'
        )
        items = _extract_json_objects(raw)
        assert len(items) == 1
        assert items[0]["index"] == 0
        assert items[0]["accepted"] is False

    def test_multiple_objects_without_array_wrapper(self):
        """JSON-lines 風格：多個物件並列但沒有用陣列/逗號包起來。"""
        raw = '{"index": 0, "accepted": true}\n{"index": 1, "accepted": false}'
        items = _extract_json_objects(raw)
        assert len(items) == 2
        assert items[0]["index"] == 0
        assert items[1]["index"] == 1

    def test_array_with_surrounding_preamble_text(self):
        raw = '這是我的判斷：\n[{"index": 0, "accepted": true}]\n希望有幫助！'
        items = _extract_json_objects(raw)
        assert len(items) == 1

    def test_garbage_returns_empty_list(self):
        assert _extract_json_objects("完全不是 JSON 的隨便文字") == []

    def test_empty_string_returns_empty_list(self):
        assert _extract_json_objects("") == []


# ── verify_svo_extraction ─────────────────────────────────────────────────────

class TestVerifySvoExtraction:
    async def test_empty_triples_trivially_accepted(self):
        accepted, verdicts = await verify_svo_extraction("原文", [])
        assert accepted is True
        assert verdicts == []

    async def test_all_accepted_when_judge_approves_all(self):
        triples = _sample_triples()
        response = json.dumps([
            {"index": 0, "accepted": True, "reason": "原文有提到"},
            {"index": 1, "accepted": True, "reason": "原文有提到"},
        ])
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            accepted, verdicts = await verify_svo_extraction("原文內容", triples)
        assert accepted is True
        assert len(verdicts) == 2
        assert all(v["accepted"] for v in verdicts)

    async def test_overall_rejected_when_any_triple_rejected(self):
        triples = _sample_triples()
        response = json.dumps([
            {"index": 0, "accepted": True, "reason": "ok"},
            {"index": 1, "accepted": False, "reason": "原文沒有提到處分機關"},
        ])
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            accepted, verdicts = await verify_svo_extraction("原文內容", triples)
        assert accepted is False
        assert verdicts[1]["accepted"] is False
        assert "處分機關" in verdicts[1]["reason"]

    async def test_malformed_response_conservatively_accepts(self):
        """裁判模型輸出解析失敗時，保守判定通過，避免格式異常誤殺正常抽取結果。"""
        triples = _sample_triples()
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm("不是 JSON 的亂七八糟文字")):
            accepted, verdicts = await verify_svo_extraction("原文內容", triples)
        assert accepted is True
        assert len(verdicts) == len(triples)

    async def test_bare_object_response_correctly_parsed_not_treated_as_garbage(self):
        """回歸測試：重現 2026-07-08 人工逐階段檢驗發現的真實模型行為——模型直接輸出
        裸物件而非陣列。修復前這會被判定為解析失敗、保守全數通過；修復後應正確辨識出
        index 0 被拒絕，整體判定為不通過。"""
        triples = _sample_triples()
        bare_object_response = (
            '{\n    "index": 0,\n    "accepted": false,\n'
            '    "reason": "原文未提到臺灣湯淺電池股份有限公司適用於宜蘭縣政府"\n}'
        )
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(bare_object_response)):
            accepted, verdicts = await verify_svo_extraction("原文內容", triples)
        assert accepted is False
        assert verdicts[0]["accepted"] is False
        assert "臺灣湯淺電池" in verdicts[0]["reason"]
        # index 1 未出現在回應中，保守判定通過（既有行為）
        assert verdicts[1]["accepted"] is True

    async def test_missing_index_in_response_defaults_to_accepted(self):
        """裁判模型漏掉某個 index 的判定時，該筆保守視為通過。"""
        triples = _sample_triples()
        response = json.dumps([{"index": 0, "accepted": False, "reason": "壞"}])
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            accepted, verdicts = await verify_svo_extraction("原文內容", triples)
        assert verdicts[0]["accepted"] is False
        assert verdicts[1]["accepted"] is True  # index 1 未出現在回應中，保守通過


# ── propose_ontology_extension ────────────────────────────────────────────────

class TestProposeOntologyExtension:
    async def test_parses_new_types_and_scope(self):
        response = json.dumps({
            "entity_types": ["新法條類型"],
            "rel_types": ["regulates"],
            "scope": "kg",
            "rationale": "原文涉及特殊法規類別",
        })
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            result = await propose_ontology_extension("原文", _sample_triples(), [])
        assert result["entity_types"] == ["新法條類型"]
        assert result["rel_types"] == ["REGULATES"]  # 正規化為大寫
        assert result["scope"] == "kg"

    async def test_existing_base_types_filtered_out_defensively(self):
        """即使模型違反指示、重複提議既有類型，也不應該被當成新類型回傳。"""
        response = json.dumps({
            "entity_types": ["概念", "全新類型"],  # 「概念」已在 _VALID_TYPES
            "rel_types": ["VIOLATES", "NEW_REL"],   # VIOLATES 已在 _VALID_REL_TYPES
            "scope": "kg",
            "rationale": "test",
        })
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            result = await propose_ontology_extension("原文", [], [])
        assert result["entity_types"] == ["全新類型"]
        assert result["rel_types"] == ["NEW_REL"]

    async def test_invalid_scope_defaults_to_kg(self):
        response = json.dumps({"entity_types": [], "rel_types": [], "scope": "universe", "rationale": ""})
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm(response)):
            result = await propose_ontology_extension("原文", [], [])
        assert result["scope"] == "kg"

    async def test_malformed_response_returns_no_extension(self):
        with patch("services.svo_service.get_llm_provider", return_value=_mock_llm("亂七八糟")):
            result = await propose_ontology_extension("原文", [], [])
        assert result["entity_types"] == []
        assert result["rel_types"] == []


# ── extract_svo_verified（整合流程）────────────────────────────────────────────

class TestExtractSvoVerified:
    async def test_disabled_flag_skips_verification_entirely(self):
        """svo_verify_enabled=False 時應完全略過驗證，等同直接呼叫 extract_svo_from_text。"""
        triples = _sample_triples()
        with patch("services.svo_service.extract_svo_from_text", new=AsyncMock(return_value=triples)) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction", new=AsyncMock()) as mock_verify, \
             patch("core.config.settings.svo_verify_enabled", False):
            result = await extract_svo_verified("原文", kg_id="kg-1")
        assert result == triples
        mock_verify.assert_not_called()
        assert mock_extract.call_count == 1

    async def test_accepted_on_first_try_no_retry(self):
        triples = _sample_triples()
        with patch("services.svo_service.extract_svo_from_text", new=AsyncMock(return_value=triples)) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction", new=AsyncMock(return_value=(True, []))) as mock_verify, \
             patch("services.svo_service.propose_ontology_extension", new=AsyncMock()) as mock_propose:
            result = await extract_svo_verified("原文", kg_id="kg-1")
        assert result == triples
        assert mock_extract.call_count == 1
        assert mock_verify.call_count == 1
        mock_propose.assert_not_called()

    async def test_retries_once_then_accepts(self):
        first_attempt = _sample_triples()
        second_attempt = _sample_triples()
        with patch("services.svo_service.extract_svo_from_text",
                    new=AsyncMock(side_effect=[first_attempt, second_attempt])) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction",
                    new=AsyncMock(side_effect=[(False, [{"index": 0, "accepted": False, "reason": "壞"}]), (True, [])])) as mock_verify, \
             patch("services.svo_service.propose_ontology_extension", new=AsyncMock()) as mock_propose:
            result = await extract_svo_verified("原文", kg_id="kg-1")
        assert result == second_attempt
        assert mock_extract.call_count == 2  # 初次 + 重試 1 次
        assert mock_verify.call_count == 2
        mock_propose.assert_not_called()

    async def test_escalates_to_ontology_expansion_after_retry_exhausted(self):
        attempt1 = _sample_triples()
        attempt2 = _sample_triples()
        final_attempt = _sample_triples()
        rejected_verdict = (False, [{"index": 0, "accepted": False, "reason": "缺少適當類別"}])
        with patch("services.svo_service.extract_svo_from_text",
                    new=AsyncMock(side_effect=[attempt1, attempt2, final_attempt])) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction",
                    new=AsyncMock(side_effect=[rejected_verdict, rejected_verdict])) as mock_verify, \
             patch("services.svo_service.propose_ontology_extension",
                    new=AsyncMock(return_value={
                        "entity_types": ["新類型"], "rel_types": [], "scope": "kg", "rationale": "缺類型",
                    })) as mock_propose:
            result = await extract_svo_verified("原文", kg_id="kg-1")
        # 初次抽取 + 重試 1 次 + 擴充本體後最後再抽取一次 = 3 次
        assert mock_extract.call_count == 3
        assert mock_verify.call_count == 2  # 最後一次抽取後不再驗證（有界流程）
        mock_propose.assert_called_once()
        assert result == final_attempt
        # 擴充結果應已持久化到該 KG
        assert "新類型" in ontology_service.get_extra_entity_types("kg-1")

    async def test_ontology_expansion_failure_falls_back_to_last_attempt(self):
        """本體擴充模型本身失敗時，不應讓整個流程中斷，改為沿用重試後的抽取結果。"""
        attempt1 = _sample_triples()
        attempt2 = _sample_triples()
        rejected_verdict = (False, [{"index": 0, "accepted": False, "reason": "壞"}])
        with patch("services.svo_service.extract_svo_from_text",
                    new=AsyncMock(side_effect=[attempt1, attempt2])) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction",
                    new=AsyncMock(side_effect=[rejected_verdict, rejected_verdict])), \
             patch("services.svo_service.propose_ontology_extension",
                    new=AsyncMock(side_effect=RuntimeError("模型逾時"))):
            result = await extract_svo_verified("原文", kg_id="kg-1")
        assert result == attempt2
        assert mock_extract.call_count == 2  # 擴充失敗，沒有第三次抽取

    async def test_empty_extraction_result_returns_immediately(self):
        """抽取結果本身就是空清單時，不需要驗證（無事實則無需審查）。"""
        with patch("services.svo_service.extract_svo_from_text", new=AsyncMock(return_value=[])) as mock_extract, \
             patch("services.svo_service.verify_svo_extraction", new=AsyncMock()) as mock_verify:
            result = await extract_svo_verified("原文", kg_id="kg-1")
        assert result == []
        mock_verify.assert_not_called()
        assert mock_extract.call_count == 1
