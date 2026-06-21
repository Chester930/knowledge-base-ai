"""
Provenance Model 測試 — Phase 3a

- SourcedFact: cite_str() 格式
- ProvenanceReport: to_dict() 結構
"""
from __future__ import annotations
import pytest

from models.provenance import ProvenanceReport, SourcedFact


# ── SourcedFact ───────────────────────────────────────────────────────────────

def _sf(**kwargs) -> SourcedFact:
    base = dict(
        fact_str="深度學習(算法) -[使用:USES]→ GPU(工具)",
        subject="深度學習", subject_type="算法",
        rel_type="USES", verb="使用",
        object="GPU", object_type="工具",
    )
    base.update(kwargs)
    return SourcedFact(**base)


class TestSourcedFact:
    def test_cite_str_with_title(self):
        sf = _sf(source_doc_title="深度學習導論", confidence=3)
        result = sf.cite_str()
        assert "深度學習導論" in result
        assert "信心 3" in result
        assert "來源" in result

    def test_cite_str_title_only_confidence_1(self):
        sf = _sf(source_doc_title="入門教材", confidence=1)
        result = sf.cite_str()
        assert "入門教材" in result
        assert "信心" not in result

    def test_cite_str_no_title_confidence_1(self):
        sf = _sf(source_doc_title="", confidence=1)
        result = sf.cite_str()
        assert result == sf.fact_str

    def test_cite_str_no_title_high_confidence(self):
        sf = _sf(source_doc_title="", confidence=5)
        result = sf.cite_str()
        assert "信心 5" in result

    def test_cite_str_contains_fact_str(self):
        sf = _sf(source_doc_title="文件A")
        result = sf.cite_str()
        assert sf.fact_str in result

    def test_default_confidence_is_1(self):
        sf = _sf()
        assert sf.confidence == 1

    def test_default_source_doc_id_empty(self):
        sf = _sf()
        assert sf.source_doc_id == ""

    def test_default_instance_id_local(self):
        sf = _sf()
        assert sf.instance_id == "local"

    def test_fields_stored_correctly(self):
        sf = _sf(source_doc_id="uuid-123", source_doc_title="教材", confidence=2,
                 created_at="2026-06-21T10:00:00", instance_id="chester")
        assert sf.source_doc_id == "uuid-123"
        assert sf.source_doc_title == "教材"
        assert sf.confidence == 2
        assert sf.created_at == "2026-06-21T10:00:00"
        assert sf.instance_id == "chester"


# ── ProvenanceReport ──────────────────────────────────────────────────────────

class TestProvenanceReport:
    def _report(self, n_facts=2) -> ProvenanceReport:
        facts = [_sf(source_doc_title=f"文件{i}", confidence=i+1) for i in range(n_facts)]
        citations = [{"doc_id": f"doc-{i}", "title": f"文件{i}", "fact_count": i+1} for i in range(n_facts)]
        return ProvenanceReport(
            query_terms=["深度學習", "GPU"],
            facts=facts,
            doc_citations=citations,
        )

    def test_to_dict_has_required_keys(self):
        d = self._report().to_dict()
        assert "query_terms" in d
        assert "fact_count" in d
        assert "doc_citations" in d
        assert "facts" in d

    def test_fact_count_matches_facts_length(self):
        report = self._report(n_facts=3)
        d = report.to_dict()
        assert d["fact_count"] == 3
        assert len(d["facts"]) == 3

    def test_each_fact_dict_has_required_fields(self):
        d = self._report().to_dict()
        for f in d["facts"]:
            assert "fact" in f
            assert "cite" in f
            assert "subject" in f
            assert "rel_type" in f
            assert "object" in f
            assert "confidence" in f
            assert "source_doc_id" in f
            assert "source_doc_title" in f
            assert "created_at" in f
            assert "instance_id" in f

    def test_query_terms_preserved(self):
        report = ProvenanceReport(query_terms=["機器學習", "神經網路"])
        d = report.to_dict()
        assert d["query_terms"] == ["機器學習", "神經網路"]

    def test_empty_facts_returns_zero_count(self):
        report = ProvenanceReport(query_terms=["A"])
        d = report.to_dict()
        assert d["fact_count"] == 0
        assert d["facts"] == []

    def test_doc_citations_included(self):
        report = self._report(n_facts=2)
        d = report.to_dict()
        assert len(d["doc_citations"]) == 2
        assert d["doc_citations"][0]["title"] == "文件0"
