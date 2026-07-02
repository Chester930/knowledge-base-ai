from __future__ import annotations
import json
import logging
import re
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.constants import KG_ROUTE_THRESHOLD, MAX_KG_PER_QUERY
from core.database import get_driver
from core.config import settings
from core.providers.factory import get_llm_provider, get_embedding_provider
from models.document import AgentQueryRequest, AgentQueryResponse, AgentContext, ChatMessage, ChatRequest
from repositories.concept_repo import ConceptRepository
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from services.chunk_store import get_chunk_store
from services.concept_engine import build_query_concepts, compute_match_score, route_documents, route_kgs
from services.svo_service import query_svo_facts

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)

_CHUNK_SIZE = 400   # 每個段落切片的目標字元數
_CHUNK_ENCODE_CAP = 512  # 送進 embedding 的最大字元數（超過截斷）
_GARBLED_THRESHOLD = 0.3  # 非可讀字元比例超過此值視為亂碼
# 列舉型 chunk 偵測：數字/圓圈編號/中文數字/圓點開頭的列舉行
_ENUM_RE = re.compile(r'(?m)^\s*(?:#{1,6}\s+)?(?:[①-⑩]|[1-9][0-9]?(?:[.)）]\s|、)|[•·▪▸]\s|第[一二三四五六七八九十百]+[章節項])')

# 自我精煉迴圈參數
_CONFIDENCE_THRESHOLD = 0.65  # 達到此信心值即停止精煉
_MAX_REFINE_ROUNDS = 3        # 最多補充幾輪
_CHUNKS_PER_ROUND = 3         # 每輪補充的 chunk 數
_CONFIDENCE_RE = re.compile(r'\{"confidence":\s*([\d.]+)[^}]*\}\s*$', re.DOTALL)
_NO_INFO_RE = re.compile(r"知識庫目前無此資訊|找不到相關|無法回答|沒有相關資訊", re.IGNORECASE)


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (n1 * n2)))


def _is_readable(content: str) -> bool:
    """
    中文知識庫文件應以 ASCII 或 CJK 為主。
    延伸拉丁字元（0x80-0xFF，如 ÿ、ý、ï）在 UTF-8 正確解碼的中文文件中不應大量出現，
    若比例超過門檻則視為亂碼（如 Big5/GBK 被誤讀為 UTF-8）。
    """
    if not content:
        return False
    sample = content[:500]
    bad = sum(
        1 for c in sample
        if not (
            ord(c) < 0x80           # 基礎 ASCII
            or '一' <= c <= '鿿'   # CJK 統一表意文字
            or '　' <= c <= '〿'   # CJK 標點
            or '＀' <= c <= '￯'   # 全角字元
            or '⺀' <= c <= '⻿'   # CJK 部首補充
            or c in ' \t\n\r'
        )
    )
    return (bad / len(sample)) < _GARBLED_THRESHOLD


_MAX_CHUNKS_EMBED = 200   # 超過此數量時，先用關鍵詞預篩再 embed


def _pick_relevant_chunks(
    content: str,
    query_concepts: list[dict],
    max_chars: int = 3000,
    boost_terms: list[str] | None = None,
) -> str:
    """
    將文件切成段落塊，用 query embedding + 關鍵詞 boost 挑出最相關的段落，
    按原文順序拼回，總長不超過 max_chars。
    大文件（>_MAX_CHUNKS_EMBED 塊）先用關鍵詞預篩，保留命中塊 + 均勻抽樣塊再 embed，
    最後用 encode_batch 批量計算，避免逐塊呼叫 embedding 的效能問題。
    """
    if not content:
        return ""
    if not query_concepts:
        return content[:max_chars]

    # PDF OCR 常產生 CJK 部首字元（U+2E80-U+2FFF，如 ⾓→角），
    # NFKC 正規化統一成標準漢字，讓 keyword match 與 embedding 正常比對
    import unicodedata as _ud
    content = _ud.normalize("NFKC", content)

    # 偵測段落分隔符：優先用 \n\n；若段落數太少或平均段落過長，降級用 \n
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    avg_len = sum(len(p) for p in paragraphs) / max(1, len(paragraphs))
    if len(paragraphs) < 5 or avg_len > _CHUNK_SIZE:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        return content[:max_chars]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        if buf_len + len(p) > _CHUNK_SIZE and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [p], len(p)
        else:
            buf.append(p)
            buf_len += len(p)
    if buf:
        chunks.append("\n\n".join(buf))

    if len(chunks) <= 1:
        return content[:max_chars]

    # q_names：查詢關鍵詞（高權重 0.4/hit）
    # boost_terms：SVO 實體（低權重 0.02/hit，僅輔助）
    q_vecs = [c["q_vector"] for c in query_concepts]
    q_names = [n for c in query_concepts if (n := c.get("name", "").strip()) and len(n) > 1]
    svo_terms = [t for t in (boost_terms or []) if t]
    all_filter = list({*q_names, *svo_terms})

    # 大文件預篩：先用查詢關鍵字找命中 chunk（含前後各 1 個相鄰 chunk），再均勻抽樣補齊
    if len(chunks) > _MAX_CHUNKS_EMBED:
        raw_hits = [i for i, c in enumerate(chunks)
                    if any(t in c for t in all_filter)]
        # 擴展到相鄰 chunk，捕捉「定義-列舉」跨段落結構
        expanded: set[int] = set(raw_hits)
        for i in raw_hits:
            if i > 0: expanded.add(i - 1)
            if i < len(chunks) - 1: expanded.add(i + 1)
        keyword_hit = sorted(expanded)
        remaining = _MAX_CHUNKS_EMBED - len(keyword_hit)
        step = max(1, len(chunks) // max(1, remaining))
        sampled = list(range(0, len(chunks), step))
        candidate_indices = sorted(set(keyword_hit + sampled))[:_MAX_CHUNKS_EMBED]
    else:
        candidate_indices = list(range(len(chunks)))

    # 批量 embed（一次呼叫，快 10-50x）
    embedding = get_embedding_provider()
    candidate_texts = [chunks[i][:_CHUNK_ENCODE_CAP] for i in candidate_indices]
    try:
        cvecs = embedding.encode_batch(candidate_texts)
    except Exception:
        cvecs = []

    scored: list[tuple[int, float]] = []
    for local_idx, global_idx in enumerate(candidate_indices):
        chunk = chunks[global_idx]
        emb_score = max(_cosine(cvecs[local_idx], qv) for qv in q_vecs) if local_idx < len(cvecs) else 0.0
        # 查詢詞命中：高權重，讓直接含有問題核心詞的 chunk 排前
        q_hits = sum(1 for t in q_names if t in chunk)
        # SVO 實體命中：提升至 0.10，讓知識圖譜實體更有效引導 chunk 選取
        svo_hits = sum(1 for t in svo_terms if t in chunk)
        # 列舉型加分：chunk 含有條列格式（1. 2. 3.）且有關鍵詞命中時優先選取
        enum_bonus = 0.25 if (q_hits > 0 or svo_hits > 0) and _ENUM_RE.search(chunk) else 0.0
        score = emb_score + q_hits * 0.4 + svo_hits * 0.10 + enum_bonus
        scored.append((global_idx, score))

    # 重新排序：q_name 命中數（降冪）→ 得分（降冪）
    # 直接用 list comprehension 避免 closure 作用域問題
    scored_ext = [
        (gi, sc, sum(1 for t in q_names if t in chunks[gi]))
        for gi, sc in scored
    ]
    scored_ext.sort(key=lambda x: (x[2], x[1]), reverse=True)
    logger.info(
        f"[chunk_pick] q_reranked top5: "
        f"{[(gi, qh, round(sc,3)) for gi,sc,qh in scored_ext[:5]]}"
    )

    # 按相關度取 chunk，直到達到 max_chars
    selected_ordered: list[int] = []
    total = 0
    for gi, sc, qh in scored_ext:
        chunk_len = len(chunks[gi])
        if total + chunk_len > max_chars:
            break
        selected_ordered.append(gi)
        total += chunk_len

    logger.info(f"[chunk_pick] selected_ordered={selected_ordered[:8]}")
    result = "\n\n---\n\n".join(chunks[i] for i in selected_ordered)
    return result or content[:max_chars]


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_confidence(text: str) -> tuple[str, float]:
    """
    從 LLM 輸出末尾剝離 {"confidence": x} JSON，
    回傳 (乾淨答案文字, 校準後信心分數)。
    校準（☆9）：無資訊答案強制下調；全域係數再縮放。
    """
    m = _CONFIDENCE_RE.search(text)
    if m:
        try:
            conf = float(m.group(1))
            clean = text[:m.start()].rstrip()
        except ValueError:
            clean, conf = text, 0.5
    else:
        clean, conf = text, 0.5

    # 若答案明確表示無資訊，強制下調信心（避免 LLM 自報高分但實際無答）
    if _NO_INFO_RE.search(clean):
        conf = min(conf, 0.35)

    # 全域校準係數（預設 0.9，讓模型稍微保守）
    from core.config import settings
    conf *= settings.confidence_calibration

    return clean, max(0.0, min(1.0, conf))


_TRIPLE_RE = __import__("re").compile(
    r"([^(]+)\([^)]*\)\s*-\[([^:]+):[^\]]*\]→\s*([^(]+)\([^)]*\)"
)

# 需要翻轉主賓語才自然的關係（verb 在 object 前）
_FLIP_RELS = {"歸屬"}


def _svo_to_sentences(facts: list[str]) -> str:
    """
    將 SVO triple 轉為自然中文句。
    歸屬關係翻轉語序：A -[歸屬]→ B  =>  「B 由 A 提出。」
    其餘：A -[rel]→ B  =>  「A rel B。」
    """
    templates = {
        "歸屬": lambda s, o: f"{s} 由 {o} 提出。",
        "階層": lambda s, o: f"{s} 屬於 {o}。",
        "組成": lambda s, o: f"{s} 的組成包含 {o}。",
        "包含": lambda s, o: f"{s} 包含 {o}。",
        "定義": lambda s, o: f"{s} 定義為：{o}。",
        "相關": lambda s, o: f"{s} 與 {o} 相關。",
        "因果": lambda s, o: f"{s} 導致 {o}。",
        "需求": lambda s, o: f"{s} 需要 {o}。",
        "優越": lambda s, o: f"{s} 優於 {o}。",
        "前置": lambda s, o: f"{o} 的前置條件是 {s}。",
        "延伸": lambda s, o: f"{s} 建構於 {o}。",
        "相似": lambda s, o: f"{s} 類似於 {o}。",
        "改善": lambda s, o: f"{s} 可提升 {o}。",
        "使用": lambda s, o: f"{s} 使用 {o}。",
    }
    seen: set[str] = set()
    sentences: list[str] = []
    for f in facts:
        if f.startswith("[推理鏈]"):
            continue
        m = _TRIPLE_RE.match(f.strip())
        if not m:
            continue
        subj = m.group(1).strip()
        rel_key = m.group(2).strip()
        obj = m.group(3).strip()
        tmpl = templates.get(rel_key)
        sent = tmpl(subj, obj) if tmpl else f"{subj} {rel_key} {obj}。"
        if sent not in seen:
            seen.add(sent)
            sentences.append(sent)
    return "\n".join(sentences)


_HISTORY_TURNS = 4  # 納入 prompt 的最近對話輪數


def _build_rag_prompt(
    question: str,
    svo_facts: list[str],
    contexts: list[dict],
    extra_chunks: list[str] | None = None,
    history: list[ChatMessage] | None = None,
) -> str:
    import unicodedata as _ud
    svo_facts = [_ud.normalize("NFKC", f) for f in svo_facts]

    parts: list[str] = ["你是知識庫問答助手，請用繁體中文回答。\n\n"]

    # 多輪對話歷史（☆10）：最近 N 輪，避免 context 過長
    if history:
        recent = history[-_HISTORY_TURNS * 2:]  # 每輪2條（user+assistant）
        parts.append("=== 對話歷史（供理解追問用）===\n")
        for msg in recent:
            role = "使用者" if msg.role == "user" else "助手"
            parts.append(f"{role}：{msg.content}\n\n")
        parts.append("=== 以上為歷史對話 ===\n\n")

    if contexts:
        graph_docs = [c for c in contexts if c.get("source") == "graph"]
        sim_docs   = [c for c in contexts if c.get("source") != "graph"]
        ordered = graph_docs + sim_docs
        docs = "".join(
            f"=== 文件 {i}：{c['title']} ===\n{c['content']}\n\n"
            for i, c in enumerate(ordered, 1)
        )
        parts.append(f"以下是知識庫中的相關文件內容：\n\n{docs}")

    if extra_chunks:
        chunks_text = "\n---\n".join(extra_chunks)
        parts.append(f"\n\n[補充原文片段]\n{chunks_text}\n\n")

    parts.append(
        f"---\n問題：{question}\n\n"
        "請仔細閱讀上方文件，找出與問題直接相關的段落，用繁體中文回答。\n"
        "若文件中列舉了具體項目（例如框架的組成要素、動力列表等），請完整列出每一項原文。\n"
        "若所有文件都找不到相關資訊，回答「知識庫目前無此資訊」。\n\n"
        "最後一行請附上你的回答信心評估（不要包含在回答正文中）：\n"
        '{"confidence": 0.85}\n'
        "（1.0=完全確定有依據，0.7=有間接線索，0.5=部分推論，0.3=幾乎無資訊）"
    )
    return "".join(parts)


# ── /agent/chat（雙層路由）────────────────────────────────────────────────────

@router.post("/chat", summary="雙層路由 RAG 問答（SSE）")
async def chat(req: ChatRequest):
    """
    雙層路由 RAG 問答：
    1. ConceptNode 路由層 → 選出相關 KG
    2. SVO 知識層 → BFS 圖遍歷取得知識事實
    3. 文件層 → 取得相關文件原文片段
    4. 統合成 RAG prompt → LLM 串流回答

    SSE 事件序列：
      status:searching → kg_route → svo_facts → sources → status:generating → token... → done
    """

    async def generate():
        try:
            yield _sse({"status": "searching"})

            # ── Step 1：提取問題概念 ──────────────────────────────────────────
            query_concepts = await build_query_concepts(req.question)
            if not query_concepts:
                yield _sse({"error": "無法理解問題，請換個說法"})
                return
            logger.info(f"[DEBUG query_concepts] {[(c.get('name'), round(max(c.get('q_vector',[0])[:3]),3)) for c in query_concepts]}")

            concept_repo = ConceptRepository(get_driver())
            doc_repo = DocumentRepository(get_driver())
            kg_repo = KnowledgeGraphRepository(get_driver())

            # ── Step 2：KG 路由層 ─────────────────────────────────────────────
            # 若請求指定了 kg_id，直接強制使用該 KG，跳過全域路由
            if req.kg_id:
                selected_kgs = [(req.kg_id, 1.0, [])]
            else:
                all_kg_concepts = await route_kgs(concept_repo, query_concepts)

                kg_scores: list[tuple[UUID, float, list[str]]] = []
                for kg_id, kg_concepts in all_kg_concepts.items():
                    score, matched = compute_match_score(query_concepts, kg_concepts)
                    if score >= KG_ROUTE_THRESHOLD:
                        kg_scores.append((kg_id, score, matched))

                kg_scores.sort(key=lambda x: x[1], reverse=True)
                selected_kgs = kg_scores[:MAX_KG_PER_QUERY]

            # 推送 KG 路由結果，同時快取 KG 物件（含 db_name）
            kg_route_info = []
            selected_kg_objects: dict[UUID, object] = {}
            for kg_id, score, matched in selected_kgs:
                kg = await kg_repo.get_by_id(kg_id)
                if kg:
                    selected_kg_objects[kg_id] = kg
                    kg_route_info.append({
                        "id": str(kg_id),
                        "name": kg.name,
                        "score": round(score, 3),
                        "matched_concepts": matched[:5],
                    })
            yield _sse({"kg_route": kg_route_info})

            # ── Step 3：SVO 知識層（BFS 圖遍歷，同時收集來源文件與 chunk_ids）──
            svo_facts: list[str] = []
            graph_doc_ids: list[str] = []   # 圖譜指向的文件 ID
            svo_chunk_ids: list[str] = []   # BFS 實體對應的 chunk_ids（精煉用）

            if req.use_svo and selected_kgs:
                q_terms = [c["name"] for c in query_concepts]
                matched_terms = []
                for _, _, matched in selected_kgs:
                    matched_terms.extend(matched)
                terms = list(dict.fromkeys(q_terms + matched_terms))
                
                # 多語言實體同義詞對齊展開（查詢期動態展開）
                from services.entity_alignment import expand_terms
                terms = await expand_terms(terms)
                
                seen_facts: set[str] = set()
                seen_doc_ids: set[str] = set()
                seen_cids: set[str] = set()

                # 多 KG 並行 BFS（☆4 優化）
                async def _bfs_kg(kg_id):
                    kg_obj = selected_kg_objects.get(kg_id)
                    db_name = getattr(kg_obj, "db_name", "") if kg_obj else ""
                    return await query_svo_facts(
                        kg_id, terms, hops=req.svo_hops, limit=50, db_name=db_name
                    )

                import asyncio as _asyncio
                bfs_results = await _asyncio.gather(
                    *[_bfs_kg(kg_id) for kg_id, _, _ in selected_kgs],
                    return_exceptions=True,
                )
                for res in bfs_results:
                    if isinstance(res, Exception):
                        logger.warning(f"BFS 並行錯誤：{res}")
                        continue
                    facts, src_ids, cids = res
                    for f in facts:
                        if f not in seen_facts:
                            seen_facts.add(f)
                            svo_facts.append(f)
                    for doc_id in src_ids:
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            graph_doc_ids.append(doc_id)
                    for cid in cids:
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            svo_chunk_ids.append(cid)

            if svo_facts:
                yield _sse({"svo_facts": svo_facts})

            # ── Step 4：混合搜尋（圖譜驅動 + 相似度補充）────────────────────────
            from uuid import UUID as _UUID
            contexts: list[dict] = []
            sources: list[dict] = []
            seen_doc_ids_ctx: set[str] = set()

            # SVO 實體名稱作為 chunk 關鍵詞 boost（從 svo_facts 解析出節點名稱）
            import re as _re
            svo_entity_names: list[str] = list({
                m for f in svo_facts
                for m in _re.findall(r'([^\-\[\]→]+?)(?:\([^)]*\))', f)
                if m.strip() and len(m.strip()) > 1
            })

            # 4a. 圖譜驅動：SVO 指向的文件（限制數量 + 過濾亂碼）
            # 最多取 top_k * 2 篇（後面由 prompt_guard 公平裁減字數），確保相關文件有機會進入
            _graph_quota = min(req.top_k * 2, 10)
            for doc_id_str in graph_doc_ids:
                if len(contexts) >= _graph_quota:
                    break
                try:
                    doc = await doc_repo.get_by_id(_UUID(doc_id_str))
                except Exception:
                    continue
                if not doc:
                    continue
                content = doc.content or ""
                if not _is_readable(content):
                    continue
                snippet = _pick_relevant_chunks(
                    content, query_concepts, req.max_chars_per_doc,
                    boost_terms=svo_entity_names,
                )
                logger.info(f"[DEBUG graph chunk] {doc.title}: len={len(snippet)} snippet_start={snippet[:120].replace(chr(10),' ')!r}")
                contexts.append({"title": doc.title, "content": snippet, "source": "graph"})
                sources.append({"title": doc.title, "score": None, "source": "graph"})
                seen_doc_ids_ctx.add(doc_id_str)

            # 4b. 相似度補充：有圖譜文件時縮減補充量，並過濾亂碼
            sim_quota = max(1, req.top_k - len(contexts))  # 圖譜已覆蓋時少補
            all_doc_concepts = await route_documents(concept_repo, query_concepts)
            allowed: set[str] = set()
            if selected_kgs:
                for kg_id, _, _ in selected_kgs:
                    for d in await kg_repo.get_documents(kg_id):
                        allowed.add(str(d["id"]))

            # Ollama 離線時 query 為整句（非短概念），分數系統性偏低；0.28 為合理下限
            _SIM_MIN_SCORE = 0.28  # 相似度補充文件的最低門檻，低於此分不納入 RAG context
            logger.info(f"[DEBUG] all_doc_concepts size={len(all_doc_concepts)}, allowed={len(allowed)}, query_concepts={[c['name'] for c in query_concepts]}")
            scored_docs = []
            _debug_scores = []
            for doc_id, dc in all_doc_concepts.items():
                if allowed and str(doc_id) not in allowed:
                    continue
                score, matched = compute_match_score(query_concepts, dc)
                _debug_scores.append((str(doc_id), score))
                if score >= _SIM_MIN_SCORE:
                    scored_docs.append((doc_id, score, matched))
            _debug_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[DEBUG] top5 scores: {_debug_scores[:5]}, total_matched={len(scored_docs)}")

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            sim_added = 0
            for doc_id, score, matched in scored_docs:
                if sim_added >= sim_quota:
                    break
                if str(doc_id) in seen_doc_ids_ctx:
                    continue  # 已由圖譜納入，跳過
                doc = await doc_repo.get_by_id(doc_id)
                if not doc:
                    continue
                content = doc.content or ""
                if not _is_readable(content):
                    continue
                snippet = _pick_relevant_chunks(
                    content, query_concepts, req.max_chars_per_doc,
                    boost_terms=svo_entity_names,
                )
                contexts.append({"title": doc.title, "content": snippet, "source": "similarity"})
                sources.append({"title": doc.title, "score": round(score, 3),
                                "matched": matched, "source": "similarity"})
                seen_doc_ids_ctx.add(str(doc_id))
                sim_added += 1

            yield _sse({"sources": sources})

            # ── Step 5：自我精煉 RAG 迴圈 ────────────────────────────────────
            if not svo_facts and not contexts:
                yield _sse({"error": "知識庫中沒有找到相關資訊，請先建立知識圖譜或匯入文件"})
                return

            # prompt 大小保護：每篇公平分配字數，避免前幾篇佔滿 LLM context
            # phi4 num_ctx=8192 ≈ 12,000 chars；扣除 system+instructions~1500，留 ~7500 給 context
            _MAX_TOTAL_CTX_CHARS = 7500
            if contexts:
                _per_doc = max(500, _MAX_TOTAL_CTX_CHARS // len(contexts))
                _total_before = sum(len(c["content"]) for c in contexts)
                contexts = [{**c, "content": c["content"][:_per_doc]} for c in contexts]
                _total_after = sum(len(c["content"]) for c in contexts)
                if _total_before > _total_after:
                    logger.info(f"[prompt_guard] 公平裁減 {len(contexts)} 篇：{_total_before} → {_total_after} chars")

            llm = get_llm_provider()
            chunk_store = get_chunk_store()
            extra_chunks: list[str] = []
            used_cids: set[str] = set()
            final_answer: str | None = None
            last_clean_answer: str | None = None

            # 精煉迴圈（只在有 chunk_ids 且啟用 SVO 時進行）
            if req.use_svo and svo_chunk_ids:
                for round_num in range(_MAX_REFINE_ROUNDS):
                    prompt = _build_rag_prompt(
                        req.question, svo_facts, contexts,
                        extra_chunks if extra_chunks else None,
                        history=req.history,
                    )
                    try:
                        # ☆3：改用 stream() 收集，讓前端即時看到精煉中間結果
                        raw_parts: list[str] = []
                        async for tok in llm.stream(prompt):
                            raw_parts.append(tok)
                        raw = "".join(raw_parts)
                        clean, confidence = _extract_confidence(raw)
                        last_clean_answer = clean
                        # 送出本輪的預覽答案（信心 JSON 已剝離）
                        yield _sse({"refine_preview": {"round": round_num + 1, "answer": clean}})
                    except Exception as e:
                        logger.warning(f"精煉 round {round_num} 失敗：{e}")
                        break

                    if confidence >= _CONFIDENCE_THRESHOLD:
                        final_answer = clean
                        logger.info(f"精煉 round {round_num}：信心={confidence:.2f} ≥ 門檻，停止")
                        break

                    # 信心不足 → 依語意相似度選下一批 chunk（☆6 優化）
                    remaining_ids = [c for c in svo_chunk_ids if c not in used_cids]
                    if not remaining_ids:
                        final_answer = clean
                        logger.info(f"精煉 round {round_num}：無更多 chunk 可補充，停止")
                        break

                    q_vec = query_concepts[0]["q_vector"] if query_concepts else []
                    ranked_chunks = await chunk_store.read_ranked(remaining_ids, q_vec)
                    top_chunks = ranked_chunks[:_CHUNKS_PER_ROUND]
                    next_ids = [c["chunk_id"] for c in top_chunks]
                    extra_chunks.extend(c["text"] for c in top_chunks)
                    used_cids.update(next_ids)
                    chunks_data = top_chunks
                    logger.info(
                        f"精煉 round {round_num+1}：信心={confidence:.2f}，"
                        f"補充 {len(chunks_data)} 個 chunk"
                    )
                    yield _sse({
                        "refine": {
                            "round": round_num + 1,
                            "confidence_before": round(confidence, 2),
                            "chunks_added": len(chunks_data),
                        }
                    })

            yield _sse({"status": "generating"})

            # 如果精煉完畢但未達門檻，直接採用最後一輪的答案，避免再次重複呼叫 LLM
            if final_answer is None and last_clean_answer is not None:
                final_answer = last_clean_answer
                logger.info("精煉輪次耗盡，直接採用最後一輪的備份答案")

            if final_answer is not None:
                # 有精煉產生的最終答案 → 分段 emit 緩衝答案
                for i in range(0, len(final_answer), 80):
                    yield _sse({"token": final_answer[i:i + 80]})
            else:
                # 未觸發任何精煉迴圈 → 呼叫 LLM 生產
                prompt = _build_rag_prompt(
                    req.question, svo_facts, contexts,
                    extra_chunks if extra_chunks else None,
                    history=req.history,
                )
                raw_parts: list[str] = []
                async for token in llm.stream(prompt):
                    raw_parts.append(token)
                raw = "".join(raw_parts)
                clean, _conf = _extract_confidence(raw)
                for i in range(0, len(clean), 80):
                    yield _sse({"token": clean[i:i + 80]})

            yield _sse({"done": True})

        except Exception as e:
            logger.exception("chat 發生錯誤")
            yield _sse({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /agent/query（非串流，向下相容）──────────────────────────────────────────

@router.post("/query", response_model=AgentQueryResponse, summary="RAG 查詢（非串流）")
async def agent_query(req: AgentQueryRequest):
    """非串流版本，回傳最相關文件片段，供程式整合使用。"""
    query_concepts = await build_query_concepts(req.question)
    if not query_concepts:
        return AgentQueryResponse(question=req.question, context=[], sources=[])

    concept_repo = ConceptRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())
    all_doc_concepts = await route_documents(concept_repo, query_concepts)

    scored = []
    for doc_id, doc_concepts in all_doc_concepts.items():
        score, matched = compute_match_score(query_concepts, doc_concepts)
        if score >= settings.score_threshold * 0.5:
            scored.append((doc_id, score, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: req.top_k]

    context_list, sources = [], []
    for doc_id, score, matched in top:
        doc = await doc_repo.get_by_id(doc_id)
        if not doc:
            continue
        snippet = doc.content[: req.max_content_chars] if req.include_content else ""
        context_list.append(
            AgentContext(
                title=doc.title, content_snippet=snippet,
                score=score, file_path=doc.file_path,
            )
        )
        sources.append(doc.title)

    return AgentQueryResponse(
        question=req.question, context=context_list, sources=sources
    )


# ── /agent/health ─────────────────────────────────────────────────────────────

@router.get("/health", summary="Agent 健康狀態")
async def agent_health():
    doc_count = await DocumentRepository(get_driver()).get_count()
    kg_count_result = await get_driver().execute_query(
        "MATCH (kg:KnowledgeGraph) RETURN count(kg) AS cnt"
    )
    entity_count_result = await get_driver().execute_query(
        "MATCH (e:Entity) RETURN count(e) AS cnt"
    )
    return {
        "status": "ok",
        "document_count": doc_count,
        "kg_count": kg_count_result.records[0]["cnt"],
        "entity_count": entity_count_result.records[0]["cnt"],
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "dual_layer_routing": True,
    }
