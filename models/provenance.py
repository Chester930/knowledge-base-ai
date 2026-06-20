"""
Provenance Models — Phase 3a

每條 SVO 事實攜帶完整溯源資訊：來源文件、信心分數、建立時間。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourcedFact:
    """帶溯源資訊的 SVO 事實"""
    # 事實本體
    fact_str: str            # 格式化顯示字串（供 LLM Prompt 使用）
    subject: str
    subject_type: str
    rel_type: str
    verb: str
    object: str
    object_type: str
    # 溯源
    confidence: int = 1      # 同一對實體被多少文件提及（≥1）
    source_doc_id: str = ""  # Document UUID
    source_doc_title: str = ""
    created_at: str = ""     # ISO 8601（來自 Neo4j datetime）
    instance_id: str = "local"

    def cite_str(self) -> str:
        """帶引用格式的事實字串，供 LLM Prompt 使用。"""
        parts = [self.fact_str]
        if self.source_doc_title:
            parts.append(f"[來源：《{self.source_doc_title}》")
            if self.confidence > 1:
                parts.append(f"，信心 {self.confidence}")
            parts.append("]")
        elif self.confidence > 1:
            parts.append(f"[信心 {self.confidence}]")
        return "".join(parts)


@dataclass
class ProvenanceReport:
    """某個查詢的完整溯源報告"""
    query_terms: list[str]
    facts: list[SourcedFact] = field(default_factory=list)
    doc_citations: list[dict] = field(default_factory=list)  # [{doc_id, title, fact_count}]

    def to_dict(self) -> dict:
        return {
            "query_terms": self.query_terms,
            "fact_count": len(self.facts),
            "doc_citations": self.doc_citations,
            "facts": [
                {
                    "fact": f.fact_str,
                    "cite": f.cite_str(),
                    "subject": f.subject,
                    "rel_type": f.rel_type,
                    "object": f.object,
                    "confidence": f.confidence,
                    "source_doc_id": f.source_doc_id,
                    "source_doc_title": f.source_doc_title,
                    "created_at": f.created_at,
                    "instance_id": f.instance_id,
                }
                for f in self.facts
            ],
        }
