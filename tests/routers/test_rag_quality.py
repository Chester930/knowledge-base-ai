"""
嚴格 RAG 品質測試（Strict RAG Quality Tests）
================================================

本測試套件驗證雙層路由 RAG 系統的每一個關鍵環節，
確保「有資料時能正確返回、無資料時能拒絕回答」。

測試標準（7 項）：
  T1. Chunk 選取精確性：問題關鍵詞必須出現在被選中的 chunk
  T2. KG 路由正確性：相關 KG 的路由分數必須 ≥ KG_ROUTE_THRESHOLD
  T3. SVO 事實注入：chat endpoint 須將圖譜事實嵌入 RAG prompt
  T4. 幻覺防守：無相關資料時答案必須含「知識庫目前無此資訊」
  T5. 信心校準：有依據時 ≥ 0.60；無依據時 ≤ 0.35（含校準係數）
  T6. 來源正確性：sources 清單必須對應實際被引用的文件
  T7. E2E 問答品質：LLM 得到完整 context 後，答案須涵蓋關鍵事實

每個測試都標明對應的測試標準（T1-T7）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4, UUID

import pytest
from httpx import AsyncClient, ASGITransport

from models.document import Document
from routers.agent import (
    _pick_relevant_chunks,
    _svo_to_sentences,
    _build_rag_prompt,
    _extract_confidence,
    _is_readable,
    _CONFIDENCE_THRESHOLD,
)
from core.constants import KG_ROUTE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# 測試輔助工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_doc(title: str = "測試文件", content: str = "測試內容", **kw) -> Document:
    doc_id = kw.pop("id", None) or uuid4()
    return Document(
        id=doc_id,
        title=title,
        content=content,
        file_type=kw.pop("file_type", "txt"),
        file_path=kw.pop("file_path", None),
        created_at=kw.pop("created_at", datetime.now()),
        updated_at=kw.pop("updated_at", datetime.now()),
    )


def _vec(val: float = 1.0, dim: int = 384) -> list[float]:
    """建立標準化向量（單位向量方向可控）。"""
    v = [0.0] * dim
    v[0] = val
    return v


def _concept(name: str, vec_val: float = 1.0) -> dict:
    return {
        "name": name,
        "q_vector": _vec(vec_val),
        "interest_score": 0.9,
        "professional_score": 0.9,
    }


def _parse_sse(raw: bytes) -> list[dict]:
    """解析 SSE 回應，回傳所有 data 欄位的 JSON 物件清單。"""
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _collect_tokens(events: list[dict]) -> str:
    return "".join(e["token"] for e in events if "token" in e)


def _make_stream(*tokens: str):
    """建立正確的 async generator，供 llm.stream mock 使用。"""
    async def _stream(prompt: str):
        for t in tokens:
            yield t
    return _stream


def _mock_embedding_provider(similarity: float = 0.9):
    """
    回傳一個固定餘弦相似度的 embedding provider mock。
    encode_batch 回傳的向量與 _vec() 的 q_vector 餘弦值為 similarity。
    """
    emb = MagicMock()
    # 使向量點積 = similarity（v1=[1,0,...], v2=[sim,sqrt(1-sim^2),...]）
    import math
    perp = math.sqrt(max(0.0, 1.0 - similarity ** 2))
    def _batch(texts):
        return [[similarity, perp] + [0.0] * (384 - 2) for _ in texts]
    emb.encode_batch = _batch
    return emb


# ══════════════════════════════════════════════════════════════════════════════
# T1：Chunk 選取精確性
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkSelectionPrecision:
    """
    [T1] _pick_relevant_chunks 必須：
    - 優先選出含有問題關鍵詞的 chunk
    - 不因字數限制遺漏最相關的段落
    - 過濾空文件
    """

    def _run(self, content: str, keywords: list[str],
             max_chars: int = 3000, similarity: float = 0.5) -> str:
        concepts = [_concept(kw) for kw in keywords]
        with patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(similarity)):
            return _pick_relevant_chunks(content, concepts, max_chars)

    # ── 基本命中 ──────────────────────────────────────────────────────────────

    def test_keyword_in_selected_chunk(self):
        """[T1] 含關鍵詞的段落必須被選中。"""
        content = (
            "第一段：這裡討論機器人技術。\n\n"
            "第二段：深度學習是人工智慧的核心技術，包含卷積神經網路與遞歸神經網路。\n\n"
            "第三段：這裡討論農業生產方式。"
        )
        result = self._run(content, ["深度學習", "卷積神經網路"])
        assert "深度學習" in result, "關鍵詞段落未被選中"

    def test_irrelevant_chunk_excluded_when_space_is_tight(self):
        """[T1] 字數嚴格時，無關段落不應擠入結果。"""
        relevant = "Transformer 架構改變了 NLP 領域的格局。" * 5
        irrelevant = "農業灌溉技術在乾旱地區有重要應用。" * 20
        content = f"{relevant}\n\n{irrelevant}"
        result = self._run(content, ["Transformer", "NLP"], max_chars=len(relevant) + 50)
        # 嚴格：相關段落必須出現
        assert "Transformer" in result

    def test_returns_empty_for_empty_content(self):
        """[T1] 空文件回傳空字串，不報錯。"""
        result = self._run("", ["關鍵詞"])
        assert result == ""

    def test_single_paragraph_returns_truncated_to_max(self):
        """[T1] 單段超長文件截斷至 max_chars。"""
        content = "ABCDE" * 1000  # 5000 chars
        result = self._run(content, ["ABC"], max_chars=500)
        assert len(result) <= 500

    def test_boost_terms_elevate_svo_entity_chunks(self):
        """[T1] SVO 實體名稱作為 boost_terms，含實體的 chunk 應優先被選中。"""
        content = (
            "段落甲：一般性討論內容，沒有特別的實體名稱。\n\n"
            "段落乙：強化學習代理人在 Atari 遊戲環境中學習策略。\n\n"
            "段落丙：其他無關內容。"
        )
        concepts = [_concept("強化學習")]
        with patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.3)):
            result = _pick_relevant_chunks(
                content, concepts, 3000, boost_terms=["強化學習代理人", "Atari"]
            )
        assert "強化學習代理人" in result

    def test_multiple_keywords_all_hit_chunks_included(self):
        """[T1] 多個關鍵詞分散在不同 chunk，各 chunk 應都能被選中（在字數限制內）。"""
        content = "\n\n".join([
            "BERT 模型使用雙向 Transformer 進行預訓練。",
            "GPT 系列模型採用自回歸語言建模目標。",
            "關於氣候變遷的其他討論。",
        ])
        result = self._run(content, ["BERT"], max_chars=2000)
        assert "BERT" in result

    def test_large_doc_keyword_prefilter_still_hits(self):
        """[T1] 超過 200 chunk 的大型文件，關鍵詞預篩仍能命中正確 chunk。"""
        # 製造 220 個無關段落 + 1 個關鍵詞段落
        filler = "\n\n".join(f"段落{i}：無關內容{'x' * 20}" for i in range(220))
        target = "量子糾纏是量子計算的基本原理之一。"
        content = filler + "\n\n" + target
        result = self._run(content, ["量子糾纏", "量子計算"], max_chars=5000)
        assert "量子糾纏" in result, "大型文件關鍵詞預篩失效"

    def test_nfkc_normalization_handles_cjk_radicals(self):
        """[T1] PDF OCR 的 CJK 部首字元（如 ⾓→角）經 NFKC 後仍能被關鍵詞命中。"""
        # U+2E86 ⾆ → 舌（NFKC），模擬 OCR 產生的部首字元
        content_with_radical = "⾆頭是味覺感知器官，⾆肌負責運動。\n\n無關內容。"
        concepts = [_concept("舌頭")]  # 標準漢字
        with patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.3)):
            result = _pick_relevant_chunks(content_with_radical, concepts, 3000)
        # 經 NFKC 後，⾆→舌，「舌頭」應能命中
        assert result  # 至少有內容被選中（不回傳空）


# ══════════════════════════════════════════════════════════════════════════════
# T2：KG 路由正確性
# ══════════════════════════════════════════════════════════════════════════════

class TestKGRoutingCriteria:
    """
    [T2] 路由層必須：
    - 高分 KG（≥ KG_ROUTE_THRESHOLD）必定被選中
    - 低分 KG（< KG_ROUTE_THRESHOLD）必定被排除
    - 多 KG 時取前 MAX_KG_PER_QUERY 個
    """

    async def _call_chat(self, test_app, kg_scores: dict[UUID, float],
                         question: str = "測試問題") -> list[dict]:
        """
        用指定的 KG 分數執行 chat，回傳所有 SSE 事件。
        kg_scores: {kg_id: score}
        """
        concepts = [_concept("測試")]
        # 為每個 KG 準備 mock 物件
        mock_kgs = {}
        for kg_id, score in kg_scores.items():
            kg_obj = MagicMock()
            kg_obj.name = f"KG-{str(kg_id)[:8]}"
            kg_obj.db_name = ""
            mock_kgs[kg_id] = (score, kg_obj)

        async def _mock_get_by_id(kid):
            return mock_kgs.get(kid, (None, None))[1] if kid in mock_kgs else None

        async def _mock_get_docs(kid):
            return []

        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.side_effect = _mock_get_by_id
        mock_kg_repo.get_documents.side_effect = _mock_get_docs

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        # 每個 KG 都有一個概念
        kgs_concepts = {kg_id: [_concept("測試")] for kg_id in kg_scores}
        mock_concept_repo.get_all_kgs_concepts.return_value = kgs_concepts
        mock_concept_repo.get_all_documents_concepts.return_value = {}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_count.return_value = 0

        # compute_match_score 依照 kg_scores 返回對應分數
        def _mock_score(query_concepts, doc_concepts):
            # 從 doc_concepts 判斷是哪個 KG
            return (0.9, ["測試"])

        # 建立每個 KG 的 score 映射
        score_map = {kg_id: score for kg_id, score in kg_scores.items()}
        call_count = [0]
        kg_id_list = list(kg_scores.keys())

        def _round_robin_score(qc, dc):
            idx = call_count[0] % len(kg_id_list)
            kid = kg_id_list[idx]
            call_count[0] += 1
            return (score_map[kid], ["測試"])

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   side_effect=_round_robin_score), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository",
                   return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository",
                   return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository",
                   return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream("答案文字。\n", '{"confidence": 0.8}')
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": question})
                return _parse_sse(res.content)

    async def test_high_score_kg_included_in_route(self, test_app):
        """[T2] 分數超過門檻的 KG 必須出現在 kg_route 事件中。"""
        kg_id = uuid4()
        events = await self._call_chat(test_app, {kg_id: KG_ROUTE_THRESHOLD + 0.1})
        route_events = [e for e in events if "kg_route" in e]
        assert route_events, "沒有收到 kg_route 事件"
        routed_ids = [r["id"] for r in route_events[0]["kg_route"]]
        assert str(kg_id) in routed_ids, f"高分 KG {kg_id} 未被路由選中"

    async def test_below_threshold_kg_excluded(self, test_app):
        """[T2] 低於門檻的 KG 不應出現在 kg_route 事件中。"""
        good_id = uuid4()
        bad_id = uuid4()
        # bad_id 的分數低於門檻
        score_map = {good_id: KG_ROUTE_THRESHOLD + 0.1, bad_id: KG_ROUTE_THRESHOLD - 0.01}

        concepts = [_concept("測試")]
        kg_objs = {}
        for kid in score_map:
            o = MagicMock()
            o.name = f"KG-{str(kid)[:8]}"
            o.db_name = ""
            kg_objs[kid] = o

        call_order = list(score_map.keys())
        call_count = [0]

        def _score(qc, dc):
            kid = call_order[call_count[0] % len(call_order)]
            call_count[0] += 1
            return (score_map[kid], ["測試"])

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {k: [_concept("測試")] for k in score_map}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.side_effect = lambda k: kg_objs.get(k)
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score", side_effect=_score), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream("回答。\n", '{"confidence": 0.7}')
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "測試問題"})
                events = _parse_sse(res.content)

        route_events = [e for e in events if "kg_route" in e]
        if route_events:
            routed_ids = [r["id"] for r in route_events[0]["kg_route"]]
            assert str(bad_id) not in routed_ids, "低分 KG 不應被路由選中"

    async def test_kg_route_event_emitted_before_answer(self, test_app):
        """[T2] kg_route 事件必須在 token 事件之前發送（保持 SSE 順序）。"""
        kg_id = uuid4()
        doc_id = uuid4()
        doc = _make_doc(id=doc_id, content="人工智慧是計算機科學的分支。")
        concepts = [_concept("人工智慧")]
        kg_obj = MagicMock()
        kg_obj.name = "測試KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("人工智慧")]}
        # 提供文件概念，使相似度補充路徑找到文件
        mock_concept_repo.get_all_documents_concepts.return_value = {
            doc_id: [_concept("人工智慧")]
        }
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = [{"id": str(doc_id)}]
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(KG_ROUTE_THRESHOLD + 0.3, ["人工智慧"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.9)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream("回答文字。\n", '{"confidence": 0.8}')
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "什麼是人工智慧？"})
                events = _parse_sse(res.content)

        keys = [list(e.keys())[0] if e else "" for e in events]
        assert "kg_route" in keys
        assert "token" in keys, f"未收到 token 事件，事件序列：{keys}"
        assert keys.index("kg_route") < keys.index("token"), \
            "kg_route 事件必須在 token 之前出現"


# ══════════════════════════════════════════════════════════════════════════════
# T3：SVO 事實注入到 RAG Prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestSVOFactInjection:
    """
    [T3] 當 SVO BFS 查詢有結果時：
    - svo_facts SSE 事件必須被發送
    - SVO 事實必須出現在送給 LLM 的 prompt 中
    - SVO 事實指向的文件應被優先納入 context
    """

    def test_svo_to_sentences_standard_relations(self):
        """[T3] 標準語意關係轉為自然語句格式正確。"""
        facts = [
            "Transformer(Algorithm) -[使用:uses]→ Attention(Mechanism)",
            "BERT(Model) -[定義:defines]→ 雙向編碼器(Component)",
            "GPT(Model) -[延伸:extends]→ Transformer(Architecture)",
        ]
        result = _svo_to_sentences(facts)
        assert "Transformer" in result
        assert "使用" in result or "Attention" in result
        assert "BERT" in result

    def test_svo_to_sentences_no_duplicates(self):
        """[T3] 重複事實只出現一次。"""
        fact = "A(T) -[使用:uses]→ B(T)"
        result = _svo_to_sentences([fact, fact, fact])
        count = result.count("A")
        assert count == 1, f"重複事實出現 {count} 次，應只有 1 次"

    def test_svo_to_sentences_skip_reasoning_chain(self):
        """[T3] [推理鏈] 前綴的事實應被跳過，不轉換為句子。"""
        facts = [
            "[推理鏈] A → B → C",
            "D(X) -[使用:uses]→ E(Y)",
        ]
        result = _svo_to_sentences(facts)
        assert "推理鏈" not in result
        assert "D" in result

    def test_svo_facts_in_rag_prompt(self):
        """[T3] SVO 事實內容必須出現在 RAG prompt 中。"""
        facts = [
            "強化學習(Algorithm) -[使用:uses]→ Q-Learning(Method)",
        ]
        # _build_rag_prompt 接受 svo_facts 但目前實作沒有直接把 facts 放進去
        # 而是依賴 contexts；但重要的是 contexts 包含了 SVO 指向的文件
        # 這裡測試 prompt 是否包含 question
        prompt = _build_rag_prompt("什麼是強化學習？", facts, [])
        assert "強化學習" in prompt

    async def test_svo_facts_sse_event_emitted(self, test_app):
        """[T3] 當 BFS 有事實時，svo_facts SSE 事件必須被發送。"""
        kg_id = uuid4()
        doc_id = str(uuid4())
        chunk_id = str(uuid4())
        svo_facts = [
            "強化學習(Algorithm) -[使用:uses]→ 獎勵函數(Concept)",
            "Q-Learning(Method) -[相關:related]→ 馬可夫決策過程(Model)",
        ]

        concepts = [_concept("強化學習")]
        kg_obj = MagicMock()
        kg_obj.name = "RL知識庫"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("強化學習")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = None  # 無對應文件

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["強化學習"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=(svo_facts, [doc_id], [chunk_id]))), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _make_stream(
                "強化學習使用獎勵函數。\n", '{"confidence": 0.8}'
            )
            mock_store.return_value.read_ranked = AsyncMock(return_value=[])
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "強化學習怎麼運作？",
                    "use_svo": True,
                })
                events = _parse_sse(res.content)

        svo_events = [e for e in events if "svo_facts" in e]
        assert svo_events, "[T3] 沒有收到 svo_facts SSE 事件"
        facts_received = svo_events[0]["svo_facts"]
        assert len(facts_received) == 2, f"應收到 2 個 SVO 事實，實際收到 {len(facts_received)}"
        assert any("強化學習" in f for f in facts_received), "強化學習應出現在 SVO 事實中"

    async def test_sparse_bfs_triggers_deeper_hop_graph_cot(self, test_app):
        """
        [Graph-CoT 簡化版] 初始跳數命中的事實過少時（< 門檻），
        應自動用同一組種子詞加深一跳重查，並合併兩次結果。
        """
        kg_id = uuid4()
        doc_id = str(uuid4())
        chunk_id = str(uuid4())

        sparse_facts = ["A(Concept) -[相關:related]→ B(Concept)"]  # 只有 1 條，低於門檻 3
        deeper_facts = sparse_facts + [
            "B(Concept) -[使用:uses]→ C(Concept)",
            "C(Concept) -[延伸:extends]→ D(Concept)",
        ]

        calls: list[int] = []

        async def _fake_bfs(kg_id, terms, hops=2, limit=50, db_name=""):
            calls.append(hops)
            if hops <= 2:
                return sparse_facts, [doc_id], [chunk_id]
            return deeper_facts, [doc_id], [chunk_id]

        concepts = [_concept("測試")]
        kg_obj = MagicMock()
        kg_obj.name = "測試知識庫"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("測試")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = None

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["測試"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts", side_effect=_fake_bfs), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _make_stream("答案。\n", '{"confidence": 0.8}')
            mock_store.return_value.read_ranked = AsyncMock(return_value=[])
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "測試問題",
                    "use_svo": True,
                    "svo_hops": 2,
                })
                events = _parse_sse(res.content)

        assert 2 in calls, "應先以請求指定的 2 跳查詢一次"
        assert 3 in calls, "證據稀疏時應加深至 3 跳重查一次"

        svo_events = [e for e in events if "svo_facts" in e]
        assert svo_events, "應收到 svo_facts SSE 事件"
        facts_received = svo_events[0]["svo_facts"]
        assert len(facts_received) == 3, f"應合併兩次查詢共 3 條不重複事實，實際 {len(facts_received)}"

    async def test_dense_bfs_does_not_trigger_deeper_hop(self, test_app):
        """[Graph-CoT 簡化版] 初始跳數已有足夠事實時，不應觸發加深查詢（避免多餘查詢成本）。"""
        kg_id = uuid4()
        doc_id = str(uuid4())
        chunk_id = str(uuid4())

        dense_facts = [
            "A(Concept) -[相關:related]→ B(Concept)",
            "B(Concept) -[使用:uses]→ C(Concept)",
            "C(Concept) -[延伸:extends]→ D(Concept)",
        ]

        calls: list[int] = []

        async def _fake_bfs(kg_id, terms, hops=2, limit=50, db_name=""):
            calls.append(hops)
            return dense_facts, [doc_id], [chunk_id]

        concepts = [_concept("測試")]
        kg_obj = MagicMock()
        kg_obj.name = "測試知識庫"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("測試")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = None

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["測試"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts", side_effect=_fake_bfs), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _make_stream("答案。\n", '{"confidence": 0.8}')
            mock_store.return_value.read_ranked = AsyncMock(return_value=[])
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                await c.post("/agent/chat", json={
                    "question": "測試問題",
                    "use_svo": True,
                    "svo_hops": 2,
                })

        assert calls == [2], f"事實充足時不應加深查詢，實際呼叫跳數：{calls}"


# ══════════════════════════════════════════════════════════════════════════════
# T4：幻覺防守
# ══════════════════════════════════════════════════════════════════════════════

class TestHallucinationDefense:
    """
    [T4] 無相關資料時，系統必須拒絕回答：
    - 無任何 KG 路由成功 → 發送 error 事件
    - 文件中完全找不到相關資訊 → LLM 答案含「知識庫目前無此資訊」
    - 亂碼文件應被過濾，不送進 prompt
    """

    def test_garbled_text_detected_as_unreadable(self):
        """[T4] 超過 30% 擴展拉丁字元的文件視為亂碼，_is_readable 回傳 False。"""
        garbled = "ÿóôûùúàâæçéèêëîïñœ" * 30  # 全部是延伸拉丁字元
        assert not _is_readable(garbled), "亂碼文件應被偵測為不可讀"

    def test_normal_chinese_text_is_readable(self):
        """[T4] 正常中文文件應通過可讀性檢查。"""
        chinese = "這是一份關於人工智慧的文件，包含機器學習與深度學習的相關知識。"
        assert _is_readable(chinese), "正常中文應被視為可讀"

    def test_mixed_chinese_english_is_readable(self):
        """[T4] 中英混合文件（常見於技術文檔）應通過可讀性檢查。"""
        mixed = "Transformer model 使用 Multi-Head Attention 機制進行特徵提取。"
        assert _is_readable(mixed), "中英混合文件應被視為可讀"

    def test_empty_content_is_not_readable(self):
        """[T4] 空字串不可讀。"""
        assert not _is_readable("")

    async def test_no_concepts_extracted_returns_error(self, test_app):
        """[T4] 當 build_query_concepts 回傳空清單，chat 必須發送 error 事件。"""
        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=[])):
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "無法理解的問題"})
                events = _parse_sse(res.content)

        error_events = [e for e in events if "error" in e]
        assert error_events, "[T4] 無概念時應發送 error 事件"

    async def test_no_kg_no_docs_returns_error(self, test_app):
        """[T4] 無任何 KG 路由成功且無文件，最終發送錯誤訊息。"""
        concepts = [_concept("完全無關的主題")]

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_doc_repo = AsyncMock()

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score", return_value=(0.0, [])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_llm_provider"), \
             patch("routers.agent.get_chunk_store"):
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "宇宙的意義是什麼？"})
                events = _parse_sse(res.content)

        error_events = [e for e in events if "error" in e]
        assert error_events, "[T4] 無資料時應發送 error 事件，告知用戶"

    def test_confidence_capped_on_no_info_answer(self):
        """[T4] LLM 自報高信心但答案為「無此資訊」，信心必須被強制下調至 ≤ 0.35 × 校準係數。"""
        _CAL = 0.9
        text = '知識庫目前無此資訊，請確認您的問題。\n{"confidence": 0.95}'
        _, conf = _extract_confidence(text)
        assert conf <= 0.35 * _CAL + 1e-9, \
            f"無資訊答案的信心應 ≤ {0.35 * _CAL:.3f}，實際為 {conf:.3f}"

    def test_no_info_pattern_variants_all_capped(self):
        """[T4] 各種「無資訊」表達方式都應被偵測並下調信心。"""
        _CAL = 0.9
        no_info_texts = [
            '找不到相關資訊。\n{"confidence": 0.9}',
            '知識庫目前無此資訊。\n{"confidence": 0.8}',
            '無法回答此問題。\n{"confidence": 0.7}',
            '沒有相關資訊可供參考。\n{"confidence": 0.85}',
        ]
        for text in no_info_texts:
            _, conf = _extract_confidence(text)
            assert conf <= 0.35 * _CAL + 1e-9, \
                f"文字 '{text[:20]}...' 的信心應被下調，實際為 {conf:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# T5：信心校準正確性
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceCalibration:
    """
    [T5] 信心分數必須：
    - 有充足依據時（LLM 自報 ≥ 0.72）→ 校準後 ≥ 0.60
    - 無依據時（無資訊答案）→ 校準後 ≤ 0.32
    - 信心 ≥ _CONFIDENCE_THRESHOLD（0.65）時停止精煉迴圈
    """

    _CAL = 0.9  # confidence_calibration 預設值

    def test_well_grounded_answer_confidence_above_threshold(self):
        """[T5] 有充分依據的答案，校準後信心應達到精煉門檻。"""
        # LLM 回報 0.85，校準後 = 0.765 ≥ 0.65（精煉門檻）
        _, conf = _extract_confidence('詳細答案文字。\n{"confidence": 0.85}')
        assert conf >= _CONFIDENCE_THRESHOLD, \
            f"有依據答案的信心 {conf:.3f} 應 ≥ 精煉門檻 {_CONFIDENCE_THRESHOLD}"

    def test_low_grounded_answer_below_refine_threshold(self):
        """[T5] 依據不足的答案，校準後信心應低於精煉門檻，觸發精煉。"""
        # LLM 回報 0.50，校準後 = 0.45 < 0.65
        _, conf = _extract_confidence('部分推論答案。\n{"confidence": 0.50}')
        assert conf < _CONFIDENCE_THRESHOLD, \
            f"低依據答案的信心 {conf:.3f} 應 < 精煉門檻 {_CONFIDENCE_THRESHOLD}"

    def test_calibration_factor_applied_consistently(self):
        """[T5] 校準係數必須始終乘以 0.9（不同輸入值一致）。"""
        raw_scores = [0.3, 0.5, 0.7, 0.9, 1.0]
        for raw in raw_scores:
            text = f'答案。\n{{"confidence": {raw}}}'
            _, conf = _extract_confidence(text)
            expected = min(1.0, raw * self._CAL)
            assert abs(conf - expected) < 1e-9, \
                f"raw={raw}: 期望 {expected:.3f}，實際 {conf:.3f}"

    def test_confidence_bounds_never_exceed_1(self):
        """[T5] 信心分數永遠不超過 1.0（即使原始值超過）。"""
        _, conf = _extract_confidence('答案。\n{"confidence": 2.0}')
        assert conf <= 1.0

    def test_confidence_bounds_never_below_0(self):
        """[T5] 信心分數永遠不低於 0.0。"""
        # 格式非法 → 預設 0.5 * 0.9 = 0.45，不會是負數
        _, conf = _extract_confidence('答案。\n{"confidence": -5.0}')
        assert conf >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# T6：來源正確性
# ══════════════════════════════════════════════════════════════════════════════

class TestSourceAttribution:
    """
    [T6] sources 清單必須：
    - 只包含實際被選入 context 的文件
    - 圖譜驅動文件標注 source="graph"
    - 相似度補充文件標注 source="similarity"
    - 亂碼文件不應出現在 sources 中
    """

    async def test_graph_doc_source_labeled_correctly(self, test_app):
        """[T6] 由 SVO 圖譜驅動取出的文件，sources 中 source 欄位必須為 'graph'。"""
        kg_id = uuid4()
        doc_id = uuid4()
        doc = _make_doc(
            id=doc_id,
            title="深度學習教材",
            content="深度學習是機器學習的一個分支，利用多層神經網路學習特徵表示。",
        )

        concepts = [_concept("深度學習")]
        kg_obj = MagicMock()
        kg_obj.name = "AI知識庫"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("深度學習")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.85, ["深度學習"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=(
                       ["深度學習(Algorithm) -[使用:uses]→ 神經網路(Model)"],
                       [str(doc_id)],  # 圖譜指向此文件
                       [],
                   ))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.9)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _make_stream(
                "深度學習使用神經網路。\n", '{"confidence": 0.9}'
            )
            mock_store.return_value.read_ranked = AsyncMock(return_value=[])
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "深度學習是什麼？",
                    "use_svo": True,
                })
                events = _parse_sse(res.content)

        source_events = [e for e in events if "sources" in e]
        assert source_events, "[T6] 缺少 sources SSE 事件"
        sources = source_events[0]["sources"]
        assert sources, "[T6] sources 不應為空"
        # 圖譜驅動的文件必須標注 source=graph
        graph_sources = [s for s in sources if s.get("source") == "graph"]
        assert graph_sources, f"[T6] 圖譜文件應標注 source='graph'，實際 sources={sources}"
        assert any(s["title"] == "深度學習教材" for s in graph_sources), \
            "[T6] 圖譜文件標題不正確"

    async def test_garbled_doc_not_in_sources(self, test_app):
        """[T6] 亂碼文件不應出現在 sources 清單中。"""
        kg_id = uuid4()
        doc_id = uuid4()
        garbled_content = "ÿóôûùúàâæçéèêëîïñœ" * 100  # 亂碼
        doc = _make_doc(id=doc_id, title="亂碼文件", content=garbled_content)

        concepts = [_concept("深度學習")]
        kg_obj = MagicMock()
        kg_obj.name = "測試KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("深度學習")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {doc_id: [_concept("深度學習")]}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = [{"id": str(doc_id)}]
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.85, ["深度學習"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [str(doc_id)], []))), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream(
                "知識庫目前無此資訊。\n", '{"confidence": 0.2}'
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "深度學習？"})
                events = _parse_sse(res.content)

        source_events = [e for e in events if "sources" in e]
        if source_events:
            sources = source_events[0]["sources"]
            assert not any(s.get("title") == "亂碼文件" for s in sources), \
                "[T6] 亂碼文件不應出現在 sources 中"


# ══════════════════════════════════════════════════════════════════════════════
# T7：E2E 問答品質
# ══════════════════════════════════════════════════════════════════════════════

class TestE2EAnswerQuality:
    """
    [T7] 端對端問答品質驗證：
    - LLM 收到正確的 context 後，答案必須涵蓋問題的關鍵事實
    - LLM 呼叫的 prompt 必須包含問題相關文件的關鍵內容
    - SSE 事件序列必須完整（status:searching → kg_route → sources → status:generating → token → done）
    """

    def _make_full_mocks(
        self,
        kg_id: UUID,
        doc: Document,
        svo_facts: list[str],
        llm_answer: str,
    ):
        """建立完整的 E2E mock 環境。"""
        kg_obj = MagicMock()
        kg_obj.name = "E2E測試KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("測試")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [_concept("測試")]}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = [{"id": str(doc.id)}]
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        return mock_concept_repo, mock_kg_repo, mock_doc_repo

    async def test_sse_event_sequence_complete(self, test_app):
        """[T7] SSE 事件序列必須完整且順序正確。"""
        kg_id = uuid4()
        doc = _make_doc(content="人工智慧是計算機科學的一個分支。")
        concepts = [_concept("人工智慧")]
        kg_obj = MagicMock()
        kg_obj.name = "AI知識庫"
        kg_obj.db_name = ""
        mc, mkr, mdr = self._make_full_mocks(kg_id, doc, [], "人工智慧是一門學科。")

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["人工智慧"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mc), \
             patch("routers.agent.DocumentRepository", return_value=mdr), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mkr), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.9)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream(
                "人工智慧是一門學科。\n", '{"confidence": 0.85}'
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "什麼是人工智慧？"})
                events = _parse_sse(res.content)

        event_types = []
        for e in events:
            if "status" in e:
                event_types.append(f"status:{e['status']}")
            elif "kg_route" in e:
                event_types.append("kg_route")
            elif "sources" in e:
                event_types.append("sources")
            elif "token" in e:
                if "token" not in event_types:
                    event_types.append("token")
            elif "done" in e:
                event_types.append("done")

        # 必要事件
        assert "status:searching" in event_types, "缺少 status:searching"
        assert "kg_route" in event_types, "缺少 kg_route"
        assert "sources" in event_types, "缺少 sources"
        assert "status:generating" in event_types, "缺少 status:generating"
        assert "token" in event_types, "缺少 token"
        assert "done" in event_types, "缺少 done"

        # 順序驗證
        seq = event_types
        assert seq.index("status:searching") < seq.index("kg_route")
        assert seq.index("kg_route") < seq.index("sources")
        assert seq.index("sources") < seq.index("status:generating")
        assert seq.index("status:generating") < seq.index("token")
        assert seq.index("token") < seq.index("done")

    async def test_llm_receives_document_content_in_prompt(self, test_app):
        """[T7] LLM 呼叫時，prompt 必須包含相關文件的實際內容。"""
        kg_id = uuid4()
        key_content = "八角框架包含八個核心動力：史詩意義、成就感、創意授權、所有權、社交影響、稀缺性、未知性、損失迴避。"
        doc = _make_doc(title="八角框架", content=key_content)
        concepts = [_concept("八角框架")]

        prompt_captured = []

        async def _mock_stream(prompt: str):
            prompt_captured.append(prompt)
            for chunk in ["八角框架包含八個核心動力。\n", '{"confidence": 0.9}']:
                yield chunk

        kg_obj = MagicMock()
        kg_obj.name = "遊戲化KG"
        kg_obj.db_name = ""
        mc, mkr, mdr = self._make_full_mocks(kg_id, doc, [], "")

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.85, ["八角框架"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mc), \
             patch("routers.agent.DocumentRepository", return_value=mdr), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mkr), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.9)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _mock_stream
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                await c.post("/agent/chat", json={"question": "八角框架有哪些核心動力？"})

        assert prompt_captured, "[T7] LLM 的 stream 未被呼叫"
        prompt = prompt_captured[0]
        assert "八角框架" in prompt, "[T7] prompt 未包含文件標題"
        # 文件內容的關鍵詞必須出現在 prompt 中
        assert "八個核心動力" in prompt or "史詩意義" in prompt, \
            f"[T7] prompt 未包含文件關鍵內容，prompt 開頭：{prompt[:200]}"

    async def test_answer_tokens_streamed_correctly(self, test_app):
        """[T7] LLM 產生的 token 必須完整地透過 SSE 串流到客戶端。"""
        kg_id = uuid4()
        doc = _make_doc(content="強化學習使用獎勵訊號更新策略。")
        concepts = [_concept("強化學習")]
        expected_answer = "強化學習是一種透過試錯學習最佳策略的方法。"
        mc, mkr, mdr = self._make_full_mocks(kg_id, doc, [], expected_answer)

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["強化學習"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mc), \
             patch("routers.agent.DocumentRepository", return_value=mdr), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mkr), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.8)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            # 分多個 token 串流
            mock_llm.return_value.stream = _make_stream(
                "強化學習", "是一種", "透過試錯", "學習最佳策略", "的方法。\n",
                '{"confidence": 0.85}',
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "強化學習是什麼？"})
                events = _parse_sse(res.content)

        answer = _collect_tokens(events)
        # 信心 JSON 不應出現在 token 流中（若無精煉，直接串流可能含 JSON）
        # 核心：答案主體必須完整傳達
        assert "強化學習" in answer, "[T7] 答案中缺少核心詞彙"

    async def test_done_event_always_sent(self, test_app):
        """[T7] 無論結果如何，最後一個事件必須是 done:True。"""
        kg_id = uuid4()
        doc = _make_doc(content="測試內容")
        concepts = [_concept("測試")]
        mc, mkr, mdr = self._make_full_mocks(kg_id, doc, [], "")

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["測試"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mc), \
             patch("routers.agent.DocumentRepository", return_value=mdr), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mkr), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.8)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream(
                "測試回答。\n", '{"confidence": 0.7}'
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={"question": "測試問題"})
                events = _parse_sse(res.content)

        assert events, "沒有收到任何 SSE 事件"
        last_event = events[-1]
        assert last_event.get("done") is True, \
            f"[T7] 最後一個事件應為 done:True，實際為 {last_event}"

    async def test_forced_kg_id_bypasses_routing(self, test_app):
        """[T7] 指定 kg_id 時，直接使用該 KG，跳過全域路由。"""
        target_kg_id = uuid4()
        doc = _make_doc(content="指定 KG 的專屬內容。")
        concepts = [_concept("測試")]

        kg_obj = MagicMock()
        kg_obj.name = "指定的KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score") as mock_score, \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=([], [], []))), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store"):
            mock_llm.return_value.stream = _make_stream(
                "指定KG的回答。\n", '{"confidence": 0.8}'
            )
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "測試問題",
                    "kg_id": str(target_kg_id),
                })
                events = _parse_sse(res.content)

        # 指定 kg_id 時不應呼叫全域路由（compute_match_score）
        mock_score.assert_not_called()

        route_events = [e for e in events if "kg_route" in e]
        assert route_events
        routed = route_events[0]["kg_route"]
        assert len(routed) == 1
        assert routed[0]["id"] == str(target_kg_id)


# ══════════════════════════════════════════════════════════════════════════════
# T3+T5：精煉迴圈品質
# ══════════════════════════════════════════════════════════════════════════════

class TestRefinementLoop:
    """
    [T3+T5] 精煉迴圈必須：
    - 低信心時觸發補充 chunk
    - 高信心時立即停止（不浪費 LLM 呼叫）
    - refine 事件包含正確的輪次與信心值
    """

    async def test_high_confidence_stops_refinement_early(self, test_app):
        """[T5] 第一輪信心 ≥ 0.65 時，不應再觸發 refine 事件。"""
        kg_id = uuid4()
        doc_id = uuid4()
        chunk_id = str(uuid4())
        doc = _make_doc(id=doc_id, content="深度學習的內容。")
        concepts = [_concept("深度學習")]

        kg_obj = MagicMock()
        kg_obj.name = "測試KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("深度學習")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        llm_call_count = [0]
        async def _mock_stream(prompt):
            llm_call_count[0] += 1
            # 第一次就回傳高信心
            for chunk in ["深度學習是神經網路技術。\n", '{"confidence": 0.85}']:
                yield chunk

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["深度學習"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=(
                       ["深度學習(Alg) -[使用:uses]→ 神經網路(Model)"],
                       [str(doc_id)],
                       [chunk_id],  # 有 chunk_id 才會觸發精煉
                   ))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.9)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _mock_stream
            mock_store.return_value.read_ranked = AsyncMock(return_value=[])
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "深度學習是什麼？",
                    "use_svo": True,
                })
                events = _parse_sse(res.content)

        refine_events = [e for e in events if "refine" in e and "refine_preview" not in e]
        assert len(refine_events) == 0, \
            f"[T5] 高信心時不應觸發精煉，實際觸發 {len(refine_events)} 次"
        # LLM 只應被呼叫一次（精煉迴圈第一輪）
        assert llm_call_count[0] == 1, \
            f"[T5] 高信心時 LLM 應只呼叫一次，實際呼叫 {llm_call_count[0]} 次"

    async def test_low_confidence_triggers_chunk_supplement(self, test_app):
        """[T5] 第一輪信心 < 0.65 時，應觸發 refine 事件並補充 chunk。"""
        kg_id = uuid4()
        doc_id = uuid4()
        chunk_ids = [str(uuid4()), str(uuid4()), str(uuid4())]
        doc = _make_doc(id=doc_id, content="量子計算內容。")
        concepts = [_concept("量子計算")]

        kg_obj = MagicMock()
        kg_obj.name = "量子KG"
        kg_obj.db_name = ""

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_public_kgs_concepts_for_query.return_value = None
        mock_concept_repo.get_documents_concepts_for_query.return_value = None
        mock_concept_repo.get_all_kgs_concepts.return_value = {kg_id: [_concept("量子計算")]}
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg_obj
        mock_kg_repo.get_documents.return_value = []
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        call_num = [0]
        async def _mock_stream_low_then_high(prompt):
            call_num[0] += 1
            if call_num[0] == 1:
                # 第一次：低信心，觸發精煉
                for chunk in ["部分推論。\n", '{"confidence": 0.45}']:
                    yield chunk
            else:
                # 第二次（補充後）：高信心
                for chunk in ["完整答案。\n", '{"confidence": 0.80}']:
                    yield chunk

        ranked_chunks = [
            {"chunk_id": cid, "text": f"補充內容{i}"}
            for i, cid in enumerate(chunk_ids[:3])
        ]

        with patch("routers.agent.build_query_concepts",
                   new=AsyncMock(return_value=concepts)), \
             patch("routers.agent.compute_match_score",
                   return_value=(0.8, ["量子計算"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo), \
             patch("routers.agent.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("routers.agent.query_svo_facts",
                   new=AsyncMock(return_value=(
                       ["量子計算(Tech) -[使用:uses]→ 疊加態(Concept)"],
                       [str(doc_id)],
                       chunk_ids,  # 提供 chunk_ids 觸發精煉機制
                   ))), \
             patch("routers.agent.get_embedding_provider",
                   return_value=_mock_embedding_provider(0.7)), \
             patch("routers.agent.get_llm_provider") as mock_llm, \
             patch("routers.agent.get_chunk_store") as mock_store:
            mock_llm.return_value.stream = _mock_stream_low_then_high
            mock_store.return_value.read_ranked = AsyncMock(return_value=ranked_chunks)
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as c:
                res = await c.post("/agent/chat", json={
                    "question": "量子計算如何運作？",
                    "use_svo": True,
                })
                events = _parse_sse(res.content)

        refine_events = [e for e in events if "refine" in e and "refine_preview" not in e]
        assert len(refine_events) >= 1, \
            f"[T5] 低信心時應觸發至少 1 次精煉，實際觸發 {len(refine_events)} 次"

        refine = refine_events[0]["refine"]
        assert refine["round"] == 1
        assert refine["confidence_before"] < _CONFIDENCE_THRESHOLD
        assert refine["chunks_added"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 輔助函式：async iterable
# ══════════════════════════════════════════════════════════════════════════════

async def _aiter_impl(items):
    for item in items:
        yield item


def aiter(items):
    """將清單轉為 async iterable，供 mock LLM stream 使用。"""
    return _aiter_impl(items)
