"""
svo_service 純函數測試

不依賴 Neo4j / LLM，只測試：
- _parse_svo_lines      : pipe 分隔格式解析（6/5/3 欄）
- _parse_svo_json       : JSON 格式解析
- _filter_hallucinated  : 幻覺過濾
- _build_ft_query       : Lucene OR 查詢字串建構
- _sentence_chunk       : 句子感知切分（委派至 chunk_store.sentence_chunk）
"""
from __future__ import annotations
import pytest

from datetime import datetime, timedelta, timezone

from services.svo_service import (
    _build_ft_query,
    _filter_hallucinated,
    _parse_svo_json,
    _parse_svo_lines,
    _sentence_chunk,
    _temporal_decay,
)
from models.knowledge_graph import SVOTriple


# ── _temporal_decay（第9節⑥時序知識圖譜衰減）───────────────────────────────

class TestTemporalDecay:
    def test_missing_created_at_no_decay(self):
        assert _temporal_decay(None) == 1.0
        assert _temporal_decay("") == 1.0

    def test_unparseable_created_at_no_decay(self):
        assert _temporal_decay("not-a-date") == 1.0

    def test_just_created_decays_to_near_one(self):
        now = datetime.now(timezone.utc).isoformat()
        assert _temporal_decay(now) == pytest.approx(1.0, abs=0.01)

    def test_older_fact_decays_below_one(self):
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        result = _temporal_decay(old, rate=0.005)
        assert 0.0 < result < 1.0
        assert result == pytest.approx(2.718281828 ** (-0.005 * 100), rel=1e-3)

    def test_more_recent_fact_scores_higher_than_older(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
        assert _temporal_decay(recent) > _temporal_decay(old)

    def test_naive_datetime_string_treated_as_utc(self):
        # Neo4j toString(datetime()) 可能不含時區資訊
        naive = (datetime.now() - timedelta(days=10)).isoformat()
        result = _temporal_decay(naive)
        assert 0.0 < result <= 1.0


# ── _build_ft_query ───────────────────────────────────────────────────────────

class TestBuildFtQuery:
    def test_single_term(self):
        result = _build_ft_query(["機器學習"])
        assert '"機器學習"' in result

    def test_multiple_terms_joined_by_or(self):
        result = _build_ft_query(["AI", "ML"])
        assert " OR " in result
        assert '"AI"' in result
        assert '"ML"' in result

    def test_special_chars_escaped(self):
        result = _build_ft_query(["C++", "hello:world"])
        # colons and plus signs should be escaped
        assert "+" not in result.replace('"', "").replace("\\+", "")
        assert ":" not in result.replace('"', "").replace("\\:", "")

    def test_empty_list_gives_empty_string(self):
        result = _build_ft_query([])
        assert result == ""


# ── _sentence_chunk（透過 svo_service 入口）──────────────────────────────────

class TestSentenceChunkViaService:
    """確認 svo_service._sentence_chunk() 正確委派至 chunk_store.sentence_chunk()。"""

    DOC_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_returns_sentence_chunk_objects(self):
        from services.chunk_store import SentenceChunk
        chunks = _sentence_chunk(self.DOC_ID, "句子一。句子二！句子三？")
        assert all(isinstance(c, SentenceChunk) for c in chunks)

    def test_empty_returns_empty(self):
        assert _sentence_chunk(self.DOC_ID, "") == []

    def test_chunk_id_format(self):
        chunks = _sentence_chunk(self.DOC_ID, "這是第一句。這是第二句！")
        assert chunks[0].chunk_id == f"{self.DOC_ID}_0001"

    def test_many_sentences_split_into_multiple_chunks(self):
        # 11 句 → ceil(11/5) = 3 個 chunk
        text = "".join(f"第{i}句話很長很長。" for i in range(1, 12))
        chunks = _sentence_chunk(self.DOC_ID, text)
        assert len(chunks) == 3


# ── _parse_svo_lines ─────────────────────────────────────────────────────────

class TestParseSvoLines:
    def test_six_column_format(self):
        raw = "強化學習|算法|USES|使用|神經網路|模型"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1
        t = triples[0]
        assert t.subject == "強化學習"
        assert t.subject_type == "算法"
        assert t.rel_type == "USES"
        assert t.verb == "使用"
        assert t.object == "神經網路"
        assert t.object_type == "模型"

    def test_five_column_format(self):
        raw = "Python|工具|用於|機器學習|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1
        t = triples[0]
        assert t.subject == "Python"
        assert t.rel_type == "RELATED_TO"

    def test_three_column_format(self):
        raw = "Docker|容器化|應用程式"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1
        t = triples[0]
        assert t.subject == "Docker"
        assert t.rel_type == "RELATED_TO"

    def test_invalid_rel_type_normalized_to_related_to(self):
        raw = "A|概念|INVALID_REL|描述|B|概念"
        triples = _parse_svo_lines(raw)
        assert triples[0].rel_type == "RELATED_TO"

    def test_invalid_entity_type_normalized_to_other(self):
        raw = "A|UNKNOWN_TYPE|IS_A|是|B|ANOTHER_INVALID"
        triples = _parse_svo_lines(raw)
        assert triples[0].subject_type == "其他"
        assert triples[0].object_type == "其他"

    def test_subject_too_long_skipped(self):
        long_subject = "A" * 51
        raw = f"{long_subject}|概念|IS_A|是|B|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 0

    def test_verb_too_long_skipped(self):
        raw = "A|概念|IS_A|" + "v" * 21 + "|B|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 0

    def test_object_too_long_skipped(self):
        raw = "A|概念|IS_A|是|" + "B" * 51 + "|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 0

    def test_duplicate_subject_rel_object_deduplicated(self):
        raw = "A|概念|IS_A|是|B|概念\nA|概念|IS_A|不同描述|B|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1

    def test_empty_lines_and_comments_skipped(self):
        raw = "\n# 這是注釋\n\nA|概念|IS_A|是|B|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1

    def test_multiple_valid_triples(self):
        raw = (
            "深度學習|算法|USES|使用|GPU|工具\n"
            "Transformer|模型|DEFINED_AS|定義為|注意力機制|算法\n"
            "BERT|模型|EXTENDS|延伸|Transformer|模型"
        )
        triples = _parse_svo_lines(raw)
        assert len(triples) == 3

    def test_fullwidth_pipe_separator(self):
        raw = "A｜概念｜IS_A｜是｜B｜算法"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 1
        assert triples[0].subject == "A"

    def test_all_valid_rel_types_accepted(self):
        valid_rels = ["IS_A", "PART_OF", "CAUSES", "USES", "REQUIRES",
                      "IMPLEMENTS", "SIMILAR_TO", "DEFINED_AS", "SOLVES"]
        for rel in valid_rels:
            raw = f"A|概念|{rel}|動詞|B|概念"
            triples = _parse_svo_lines(raw)
            assert triples[0].rel_type == rel, f"Expected {rel}, got {triples[0].rel_type}"

    def test_null_token_subject_skipped(self):
        raw = "空白|其他|CREATED_BY|由...提出|楊思枬|人物"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 0

    def test_null_token_object_skipped(self):
        raw = "A|概念|RELATED_TO|相關於|無|概念"
        triples = _parse_svo_lines(raw)
        assert len(triples) == 0


# ── _parse_svo_json ───────────────────────────────────────────────────────────

class TestParseSvoJson:
    def _item(self, **kwargs):
        base = {"s": "主詞", "st": "概念", "r": "IS_A", "v": "是", "o": "受詞", "ot": "概念"}
        base.update(kwargs)
        return base

    def test_valid_item_parsed(self):
        items = [self._item()]
        triples = _parse_svo_json(items)
        assert len(triples) == 1
        assert triples[0].subject == "主詞"

    def test_missing_subject_skipped(self):
        triples = _parse_svo_json([self._item(s="")])
        assert len(triples) == 0

    def test_missing_verb_skipped(self):
        triples = _parse_svo_json([self._item(v="")])
        assert len(triples) == 0

    def test_missing_object_skipped(self):
        triples = _parse_svo_json([self._item(o="")])
        assert len(triples) == 0

    def test_invalid_rel_type_normalized(self):
        triples = _parse_svo_json([self._item(r="BOGUS_REL")])
        assert triples[0].rel_type == "RELATED_TO"

    def test_invalid_entity_type_normalized(self):
        triples = _parse_svo_json([self._item(st="INVALID", ot="ALSO_INVALID")])
        assert triples[0].subject_type == "其他"
        assert triples[0].object_type == "其他"

    def test_long_subject_skipped(self):
        triples = _parse_svo_json([self._item(s="X" * 51)])
        assert len(triples) == 0

    def test_duplicate_s_r_o_deduplicated(self):
        item = self._item()
        triples = _parse_svo_json([item, item])
        assert len(triples) == 1

    def test_non_dict_items_skipped(self):
        triples = _parse_svo_json(["not a dict", 42, None, self._item()])
        assert len(triples) == 1

    def test_multiple_valid_items(self):
        items = [
            self._item(s="A", o="B"),
            self._item(s="C", o="D"),
            self._item(s="E", o="F"),
        ]
        triples = _parse_svo_json(items)
        assert len(triples) == 3


# ── _filter_hallucinated ──────────────────────────────────────────────────────

def _triple(s: str, o: str) -> SVOTriple:
    return SVOTriple(subject=s, subject_type="概念", rel_type="IS_A", verb="是",
                     object=o, object_type="概念")


class TestFilterHallucinated:
    def test_both_in_source_kept(self):
        t = _triple("深度學習", "神經網路")
        result = _filter_hallucinated([t], "深度學習是一種神經網路技術")
        assert len(result) == 1

    def test_subject_in_source_kept(self):
        t = _triple("深度學習", "幻覺受詞ZZZZ")
        result = _filter_hallucinated([t], "深度學習是重要技術")
        assert len(result) == 1

    def test_object_in_source_kept(self):
        t = _triple("幻覺主詞ZZZZ", "神經網路")
        result = _filter_hallucinated([t], "使用神經網路")
        assert len(result) == 1

    def test_neither_in_source_filtered(self):
        t = _triple("完全幻覺A", "完全幻覺B")
        result = _filter_hallucinated([t], "毫不相關的文字")
        assert len(result) == 0

    def test_case_insensitive_match(self):
        t = _triple("Python", "Library")
        result = _filter_hallucinated([t], "python library is great")
        assert len(result) == 1

    def test_empty_triples_returns_empty(self):
        result = _filter_hallucinated([], "any text")
        assert result == []

    def test_empty_source_filters_all(self):
        t = _triple("A", "B")
        result = _filter_hallucinated([t], "")
        assert len(result) == 0
