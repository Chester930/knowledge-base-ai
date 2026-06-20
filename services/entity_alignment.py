"""
Entity Alignment Service — Phase 2d

提供：
1. 查詢詞同義詞展開（zh↔en 常見 AI/CS 術語對照表）
2. 跨 instance 實體對齊候選（AlignedEntity）
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── 同義詞組（每個 frozenset 為一組等義詞）────────────────────────────────────
# 小寫儲存；查詢時 term.lower() 命中即展開整組
_SYNONYM_GROUPS: list[frozenset[str]] = [
    # AI / ML 核心
    frozenset({"強化學習", "reinforcement learning", "rl"}),
    frozenset({"機器學習", "machine learning", "ml"}),
    frozenset({"深度學習", "deep learning", "dl"}),
    frozenset({"人工智慧", "artificial intelligence", "ai"}),
    frozenset({"監督式學習", "supervised learning"}),
    frozenset({"非監督式學習", "unsupervised learning"}),
    frozenset({"遷移學習", "transfer learning"}),
    frozenset({"微調", "fine-tuning", "fine tuning", "finetuning"}),
    # 模型架構
    frozenset({"神經網路", "神經網絡", "neural network", "nn"}),
    frozenset({"卷積神經網路", "cnn", "convolutional neural network"}),
    frozenset({"遞迴神經網路", "rnn", "recurrent neural network"}),
    frozenset({"變換器", "transformer"}),
    frozenset({"注意力機制", "attention mechanism", "self-attention"}),
    frozenset({"生成對抗網路", "gan", "generative adversarial network"}),
    frozenset({"大型語言模型", "llm", "large language model"}),
    frozenset({"擴散模型", "diffusion model"}),
    # NLP / RAG
    frozenset({"自然語言處理", "nlp", "natural language processing"}),
    frozenset({"詞嵌入", "word embedding", "word2vec"}),
    frozenset({"檢索增強生成", "rag", "retrieval augmented generation",
               "retrieval-augmented generation"}),
    frozenset({"向量資料庫", "vector database", "vector db", "vector store"}),
    frozenset({"語意搜尋", "semantic search"}),
    # 知識圖譜
    frozenset({"知識圖譜", "knowledge graph", "kg"}),
    frozenset({"知識庫", "knowledge base", "kb"}),
    frozenset({"本體論", "ontology"}),
    frozenset({"實體", "entity"}),
    frozenset({"關係抽取", "relation extraction", "re"}),
    frozenset({"命名實體辨識", "ner", "named entity recognition"}),
    # 電腦視覺
    frozenset({"電腦視覺", "computer vision", "cv"}),
    frozenset({"目標偵測", "object detection"}),
    frozenset({"影像辨識", "image recognition", "image classification"}),
    # 軟體 / 系統
    frozenset({"應用程式介面", "api", "application programming interface"}),
    frozenset({"資料庫", "database", "db"}),
    frozenset({"容器化", "containerization", "docker"}),
    frozenset({"微服務", "microservice", "microservices"}),
    frozenset({"函式庫", "函數庫", "library", "lib"}),
    frozenset({"框架", "framework"}),
]

# 建立查詢索引：lower(term) → frozenset（O(1) 查找）
_TERM_INDEX: dict[str, frozenset[str]] = {}
for _grp in _SYNONYM_GROUPS:
    for _t in _grp:
        _TERM_INDEX[_t.lower()] = _grp


# ── 同義詞展開 ────────────────────────────────────────────────────────────────

def expand_terms(terms: list[str], max_expansion: int = 3) -> list[str]:
    """
    展開查詢詞的同義詞。
    - 原詞保留最前，展開詞依字典序附後
    - 每個原詞最多補 max_expansion 個同義詞
    - 去重保序
    """
    seen: set[str] = set()
    result: list[str] = []

    for term in terms:
        if term not in seen:
            seen.add(term)
            result.append(term)
        grp = _TERM_INDEX.get(term.lower())
        if not grp:
            continue
        added = 0
        for syn in sorted(grp):                     # 字典序，方便測試
            if syn.lower() == term.lower() or syn in seen:
                continue
            seen.add(syn)
            result.append(syn)
            added += 1
            if added >= max_expansion:
                break

    return result


def get_synonym_group(term: str) -> list[str]:
    """回傳 term 所在的同義詞組（含自身），未找到則回傳空列表。"""
    grp = _TERM_INDEX.get(term.lower())
    return sorted(grp) if grp else []


# ── 實體對齊 ──────────────────────────────────────────────────────────────────

@dataclass
class InstanceRef:
    """單一 instance 的實體紀錄"""
    name: str
    instance_id: str
    kg_name: str = ""
    kg_id: str = ""
    degree: int = 0
    entity_type: str = "Entity"


@dataclass
class AlignedEntity:
    """跨 instance 對齊後的實體"""
    canonical_name: str
    synonym_group: list[str] = field(default_factory=list)
    instances: list[InstanceRef] = field(default_factory=list)

    @property
    def max_degree(self) -> int:
        return max((r.degree for r in self.instances), default=0)

    @property
    def instance_count(self) -> int:
        return len({r.instance_id for r in self.instances})

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "synonym_group": self.synonym_group,
            "instance_count": self.instance_count,
            "instances": [
                {
                    "name": r.name, "instance_id": r.instance_id,
                    "kg_name": r.kg_name, "kg_id": r.kg_id,
                    "degree": r.degree, "type": r.entity_type,
                }
                for r in self.instances
            ],
            "max_degree": self.max_degree,
        }


def align_entity_results(raw_entities: list[dict]) -> list[AlignedEntity]:
    """
    將跨分片的原始實體清單做對齊：
    1. 完全同名（case-insensitive）→ 合併為同一 AlignedEntity
    2. 屬於同一同義詞組 → 同樣合併
    3. canonical_name 取最先出現的那個

    raw_entities 每筆格式：
    {name, type, kg_id, kg_name, degree, instance_id}
    """
    # canonical_key → AlignedEntity
    aligned: dict[str, AlignedEntity] = {}

    for e in raw_entities:
        name = e["name"]
        name_lower = name.lower()

        # 找同義詞組的 canonical key（字典序最小詞，確保穩定）
        grp = _TERM_INDEX.get(name_lower)
        canonical_key = min(grp) if grp else name_lower

        if canonical_key not in aligned:
            aligned[canonical_key] = AlignedEntity(
                canonical_name=name,
                synonym_group=sorted(grp) if grp else [],
            )

        aligned[canonical_key].instances.append(InstanceRef(
            name=name,
            instance_id=e.get("instance_id", "local"),
            kg_name=e.get("kg_name", ""),
            kg_id=e.get("kg_id", ""),
            degree=e.get("degree", 0),
            entity_type=e.get("type", "Entity"),
        ))

    return sorted(aligned.values(), key=lambda a: a.max_degree, reverse=True)
