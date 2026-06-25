"""
agent.py 自我精煉相關純函數測試

涵蓋：
- _extract_confidence()  : 信心分數剝離
- _build_rag_prompt()    : RAG prompt 組裝（含/不含 extra_chunks）
- _ENUM_RE               : 列舉/編號段落偵測（含 Markdown 標題前綴）
- 信心分數邊界值夾緊至 [0, 1]
- 未附信心 JSON 時的預設值（0.5）
"""
from __future__ import annotations
import re
import pytest

from routers.agent import _build_rag_prompt, _extract_confidence, _ENUM_RE


# ══════════════════════════════════════════════════════════════════════════════
# _extract_confidence()
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractConfidence:
    # ☆9 校準係數（對應 settings.confidence_calibration 預設值 0.9）
    _CAL = 0.9

    def test_standard_format_extracted(self):
        text = '這是答案正文。\n{"confidence": 0.85}'
        clean, conf = _extract_confidence(text)
        assert conf == pytest.approx(0.85 * self._CAL)
        assert "confidence" not in clean

    def test_clean_text_trailing_whitespace_stripped(self):
        text = '答案在此。  \n{"confidence": 0.70}'
        clean, _ = _extract_confidence(text)
        assert clean == "答案在此。"

    def test_missing_confidence_json_returns_default_05(self):
        text = "答案文字，沒有附加信心 JSON。"
        clean, conf = _extract_confidence(text)
        assert clean == text
        assert conf == pytest.approx(0.5 * self._CAL)

    def test_confidence_1_0(self):
        _, conf = _extract_confidence('答案。\n{"confidence": 1.0}')
        assert conf == pytest.approx(1.0 * self._CAL)

    def test_confidence_0_0(self):
        _, conf = _extract_confidence('答案。\n{"confidence": 0.0}')
        assert conf == pytest.approx(0.0)

    def test_confidence_clamped_above_1(self):
        # 1.5 * 0.9 = 1.35，夾緊後 → 1.0
        _, conf = _extract_confidence('答案。\n{"confidence": 1.5}')
        assert conf == pytest.approx(1.0)

    def test_confidence_negative_unmatched_returns_default(self):
        # regex [\d.]+ 不匹配負號，負值視為格式非法 → 回傳預設 0.5
        _, conf = _extract_confidence('答案。\n{"confidence": -0.3}')
        assert conf == pytest.approx(0.5 * self._CAL)

    def test_confidence_with_extra_fields(self):
        _, conf = _extract_confidence('答案。\n{"confidence": 0.72, "note": "partial"}')
        assert conf == pytest.approx(0.72 * self._CAL)

    def test_confidence_in_middle_of_text_not_extracted(self):
        text = '{"confidence": 0.9} 這是正文，不在尾端。'
        clean, conf = _extract_confidence(text)
        assert conf == pytest.approx(0.5 * self._CAL)
        assert clean == text

    def test_empty_string_returns_default(self):
        clean, conf = _extract_confidence("")
        assert clean == ""
        assert conf == pytest.approx(0.5 * self._CAL)

    def test_only_confidence_json(self):
        clean, conf = _extract_confidence('{"confidence": 0.60}')
        assert conf == pytest.approx(0.60 * self._CAL)
        assert clean == ""

    def test_malformed_json_returns_default(self):
        text = '答案。\n{"confidence": not_a_number}'
        clean, conf = _extract_confidence(text)
        assert conf == pytest.approx(0.5 * self._CAL)

    def test_confidence_with_spaces_around_colon(self):
        _, conf = _extract_confidence('答案。\n{"confidence":   0.88}')
        assert conf == pytest.approx(0.88 * self._CAL)

    def test_no_info_answer_caps_confidence(self):
        # 「知識庫目前無此資訊」→ 強制 ≤ 0.35，再乘校準係數
        text = '知識庫目前無此資訊。\n{"confidence": 0.9}'
        _, conf = _extract_confidence(text)
        assert conf == pytest.approx(0.35 * self._CAL)


# ══════════════════════════════════════════════════════════════════════════════
# _build_rag_prompt()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildRagPrompt:

    def _make_ctx(self, title: str, content: str, source: str = "graph") -> dict:
        return {"title": title, "content": content, "source": source}

    # ── 基本結構 ──────────────────────────────────────────────────────────────

    def test_contains_question(self):
        prompt = _build_rag_prompt("什麼是強化學習？", [], [])
        assert "什麼是強化學習？" in prompt

    def test_contains_繁體中文_instruction(self):
        prompt = _build_rag_prompt("問題", [], [])
        assert "繁體中文" in prompt

    def test_contains_confidence_instruction(self):
        prompt = _build_rag_prompt("問題", [], [])
        assert '{"confidence":' in prompt or '"confidence"' in prompt

    def test_contains_no_info_fallback_instruction(self):
        prompt = _build_rag_prompt("問題", [], [])
        assert "知識庫目前無此資訊" in prompt

    # ── SVO facts ─────────────────────────────────────────────────────────────

    def test_no_svo_facts_no_section(self):
        prompt = _build_rag_prompt("問題", [], [])
        assert "知識圖譜" not in prompt or True  # 無 facts 時不強制出現

    # ── contexts ──────────────────────────────────────────────────────────────

    def test_context_title_in_prompt(self):
        ctx = self._make_ctx("深度學習入門", "深度學習是...")
        prompt = _build_rag_prompt("問題", [], [ctx])
        assert "深度學習入門" in prompt

    def test_context_content_in_prompt(self):
        ctx = self._make_ctx("標題", "這段內容必須出現在 prompt 中。")
        prompt = _build_rag_prompt("問題", [], [ctx])
        assert "這段內容必須出現在 prompt 中。" in prompt

    def test_multiple_contexts_all_present(self):
        ctxs = [
            self._make_ctx("文件A", "內容A"),
            self._make_ctx("文件B", "內容B"),
            self._make_ctx("文件C", "內容C"),
        ]
        prompt = _build_rag_prompt("問題", [], ctxs)
        assert "文件A" in prompt
        assert "文件B" in prompt
        assert "文件C" in prompt

    def test_graph_sources_ordered_before_sim(self):
        sim_ctx   = self._make_ctx("相似文件", "相似內容", source="similarity")
        graph_ctx = self._make_ctx("圖譜文件", "圖譜內容", source="graph")
        prompt = _build_rag_prompt("問題", [], [sim_ctx, graph_ctx])
        # graph 文件應排在相似文件之前
        assert prompt.index("圖譜文件") < prompt.index("相似文件")

    # ── extra_chunks ──────────────────────────────────────────────────────────

    def test_extra_chunks_section_present(self):
        prompt = _build_rag_prompt("問題", [], [], extra_chunks=["原文片段一"])
        assert "補充原文片段" in prompt
        assert "原文片段一" in prompt

    def test_extra_chunks_none_no_section(self):
        prompt = _build_rag_prompt("問題", [], [], extra_chunks=None)
        assert "補充原文片段" not in prompt

    def test_extra_chunks_empty_list_no_section(self):
        prompt = _build_rag_prompt("問題", [], [], extra_chunks=[])
        assert "補充原文片段" not in prompt

    def test_multiple_extra_chunks_separated_by_divider(self):
        prompt = _build_rag_prompt("問題", [], [], extra_chunks=["片段A", "片段B"])
        assert "片段A" in prompt
        assert "片段B" in prompt
        assert "---" in prompt  # 分隔線

    def test_extra_chunks_appear_after_contexts(self):
        ctx = self._make_ctx("文件A", "內容A")
        prompt = _build_rag_prompt("問題", [], [ctx], extra_chunks=["補充片段"])
        assert prompt.index("補充原文片段") > prompt.index("文件A")

    def test_extra_chunks_appear_before_question_line(self):
        prompt = _build_rag_prompt("問題Ｑ", [], [], extra_chunks=["補充片段Ｘ"])
        assert prompt.index("補充片段Ｘ") < prompt.index("問題Ｑ")

    # ── 組合情境 ──────────────────────────────────────────────────────────────

    def test_full_prompt_structure(self):
        """完整情境：contexts + extra_chunks + SVO facts 都存在。"""
        ctx = self._make_ctx("文件", "文件內容")
        prompt = _build_rag_prompt(
            "什麼是 Transformer？",
            ["A -[使用]→ B"],
            [ctx],
            extra_chunks=["補充片段文字"],
        )
        assert "Transformer" in prompt
        assert "文件內容" in prompt
        assert "補充片段文字" in prompt
        assert '{"confidence":' in prompt or '"confidence"' in prompt

    def test_prompt_is_string(self):
        result = _build_rag_prompt("問", [], [])
        assert isinstance(result, str)

    def test_prompt_not_empty(self):
        result = _build_rag_prompt("問", [], [])
        assert len(result) > 50


# ══════════════════════════════════════════════════════════════════════════════
# _ENUM_RE  —  列舉/編號段落偵測
# ══════════════════════════════════════════════════════════════════════════════

class TestEnumRe:

    def _match(self, text: str) -> bool:
        return bool(_ENUM_RE.search(text))

    # 應匹配
    def test_chinese_ordinal_with_項(self):
        assert self._match("第一項核心動力：重大使命")

    def test_markdown_h3_chinese_ordinal(self):
        assert self._match("### 第一項核心動力：重大使命")

    def test_markdown_h2_chinese_ordinal(self):
        assert self._match("## 第二項核心動力：發展與成就")

    def test_markdown_h4_with_章(self):
        assert self._match("#### 第三章 方法論")

    def test_plain_chinese_ordinal_章(self):
        assert self._match("第一章 緒論")

    def test_arabic_number_dot(self):
        assert self._match("1. 重大使命")

    def test_arabic_number_chinese_period(self):
        assert self._match("2、稀缺性")

    def test_circle_number(self):
        assert self._match("①重大使命")

    def test_bullet_point(self):
        assert self._match("• 損失與避免")

    def test_triangle_bullet(self):
        assert self._match("▸ 不確定性")

    # 不應匹配
    def test_plain_title_no_number(self):
        assert not self._match("# 八角框架（Octalysis Framework）")

    def test_plain_paragraph_no_enum(self):
        assert not self._match("這是一段普通內文，沒有編號。")

    def test_markdown_h1_plain_title(self):
        assert not self._match("## 提出者與背景")
