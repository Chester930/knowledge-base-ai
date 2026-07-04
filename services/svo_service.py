from __future__ import annotations
import asyncio
import logging
import re
import time
import unicodedata as _unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncIterator
from uuid import UUID, uuid4

from core.database import get_driver
from core.providers.factory import get_llm_provider
from models.knowledge_graph import SVOTriple
from repositories.document_repo import DocumentRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from services.chunk_store import SentenceChunk, get_chunk_store, sentence_chunk

logger = logging.getLogger(__name__)

_SENTENCES_PER_CHUNK = 5  # 每個 Chunk 的句子數（可調整）
_SVO_CONCURRENCY = 2      # 每份文件最大平行 LLM 呼叫數（對應 OLLAMA_NUM_PARALLEL=2）
_BFS_CACHE_TTL = 300      # BFS 查詢快取 TTL（秒）
_BFS_CACHE_MAX = 1000     # LRU 容量上限，避免長時間運行無上限增長造成記憶體洩漏
_bfs_cache: OrderedDict = OrderedDict()  # (kg_id, terms, hops, min_conf) → (facts, docs, chunk_ids, ts)


def _bfs_cache_get(key: tuple):
    cached = _bfs_cache.get(key)
    if cached is None:
        return None
    _bfs_cache.move_to_end(key)
    return cached


def _bfs_cache_set(key: tuple, value: tuple) -> None:
    if key in _bfs_cache:
        _bfs_cache.move_to_end(key)
    else:
        if len(_bfs_cache) >= _BFS_CACHE_MAX:
            _bfs_cache.popitem(last=False)
    _bfs_cache[key] = value


# ── 進度事件（供 SSE 串流使用）────────────────────────────────────────────────

@dataclass
class BuildProgress:
    event: str          # chunk_start | chunk_done | merge_done | error | done
    chunk_idx: int = 0
    total_chunks: int = 0
    triples_extracted: int = 0
    triples_merged: int = 0
    message: str = ""


# ── 公開 API ──────────────────────────────────────────────────────────────────

async def _get_kg_db(kg_id: UUID) -> str:
    """取得 KG 的專用資料庫名稱（空白 = 主資料庫）。"""
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    return await KnowledgeGraphRepository(get_driver()).get_db_name(kg_id)


async def _get_fallback_model(current_model: str) -> str | None:
    """動態查詢本地 Ollama 已有的較輕量模型作為降級選擇。"""
    if "phi4" not in current_model.lower():
        # 如果當前模型本來就很輕量 (e.g. qwen2.5:7b 或 llama3.2:3b)，不需特別降級
        return None
    try:
        import httpx
        from core.config import settings
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{settings.ollama_base_url}/api/tags")
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                # 依大小與效果尋找備用模型
                for fallback in ["qwen2.5:7b", "qwen2.5-coder:7b", "llama3.2:3b", "llama3.2:latest"]:
                    if fallback in models:
                        return fallback
    except Exception as e:
        logger.debug(f"動態獲取 Ollama fallback 模型失敗: {e}")
    return None


async def build_graph_for_kg(
    kg_id: UUID,
    doc_ids: list[UUID] | None = None,
    force_rebuild: bool = False,
    rebuild_relations_only: bool = False,
) -> AsyncIterator[BuildProgress]:
    """
    對 KG 下的所有（或指定）文件執行 SVO 提取，直接 MERGE 進 Neo4j。
    - force_rebuild=True: 清除後重建（rebuild_relations_only=True 時只清關係保留節點）
    - force_rebuild=False: 增量模式，跳過已有 svo_processed_at 的文件
    - 平行處理：每份文件的 chunks 以 settings.svo_concurrency 限流並行送 LLM
    """
    from core.config import settings
    kg_repo = KnowledgeGraphRepository(get_driver())
    doc_repo = DocumentRepository(get_driver())

    kg = await kg_repo.get_by_id(kg_id)
    if kg is None:
        yield BuildProgress(event="error", message=f"KG 不存在：{kg_id}")
        return

    db_name = kg.db_name

    if doc_ids:
        docs = [d for d in [await doc_repo.get_by_id(did) for did in doc_ids] if d]
    else:
        raw = await kg_repo.get_documents(kg_id)
        docs = [await doc_repo.get_by_id(UUID(r["id"])) for r in raw]
        docs = [d for d in docs if d]

    if not docs:
        yield BuildProgress(event="done", message="此 KG 下沒有文件")
        return

    if force_rebuild:
        if rebuild_relations_only:
            await _clear_kg_relations(kg_id, db_name)
        else:
            await _clear_kg_entities(kg_id, db_name)
        await _reset_docs_svo_processed(kg_id)
        # 清除快取
        to_del = [k for k in _bfs_cache.keys() if k[0] == str(kg_id)]
        for k in to_del:
            _bfs_cache.pop(k, None)
    else:
        docs = await _filter_unprocessed_docs(docs)
        if not docs:
            # 即使沒有文件要處理，也主動清一次快取確保一致性
            to_del = [k for k in _bfs_cache.keys() if k[0] == str(kg_id)]
            for k in to_del:
                _bfs_cache.pop(k, None)
            yield BuildProgress(event="done", message="所有文件已是最新，無需重建")
            return

    sem = asyncio.Semaphore(settings.svo_concurrency)
    total_merged = 0

    chunk_store = get_chunk_store()
    progress_lock = asyncio.Lock()

    for doc in docs:
        try:
            text = doc.content or ""
            _doc_id = doc.id
            _doc_title = doc.title

            sent_chunks = sentence_chunk(str(_doc_id), text)
            total_chunks = len(sent_chunks)

            # 1. 建立/讀取本地進度檔
            import json as _json
            progress_dir = chunk_store._base / str(kg_id) / str(_doc_id)
            progress_file = progress_dir / "svo_progress.json"
        
            if force_rebuild:
                if progress_file.exists():
                    try:
                        progress_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                progress_data = {}
            else:
                try:
                    if progress_file.exists():
                        with open(progress_file, "r", encoding="utf-8") as f:
                            progress_data = _json.load(f)
                    else:
                        progress_data = {}
                except Exception as pe:
                    logger.warning(f"讀取進度檔失敗，重設：{pe}")
                    progress_data = {}

            # 持久化 Chunk 檔案，同時計算並儲存 embedding 向量（☆6 優化）
            try:
                from core.providers.factory import get_embedding_provider
                _emb = get_embedding_provider()
                _texts = [c.text for c in sent_chunks]
                # encode()/encode_batch() 為同步呼叫，丟進 thread pool 避免阻塞事件迴圈
                if hasattr(_emb, "encode_batch"):
                    _vectors = await asyncio.to_thread(_emb.encode_batch, _texts)
                else:
                    _vectors = await asyncio.to_thread(
                        lambda: [_emb.encode(t) for t in _texts]
                    )
            except Exception as _e:
                logger.warning(f"Chunk 向量計算失敗，略過持久化向量：{_e}")
                _vectors = None
            await chunk_store.write(kg_id, _doc_id, sent_chunks, vectors=_vectors)

            # 2. 分類：哪些 chunk 已完成，哪些待處理
            to_process = []
            skipped_results = []
            for sc in sent_chunks:
                chunk_status = progress_data.get(str(sc.idx))
                if chunk_status and chunk_status.get("processed"):
                    skipped_results.append((sc.idx, chunk_status.get("triples_count", 0)))
                else:
                    to_process.append(sc)

            yield BuildProgress(
                event="chunk_start",
                chunk_idx=0, total_chunks=total_chunks,
                message=f"[{_doc_title}] 句子感知切分 {total_chunks} 個 Chunk，已跳過 {len(skipped_results)} 個已處理 Chunk，剩餘 {len(to_process)} 個開始並行提取…",
            )

            # 3. 先把已跳過的進度 yield 給 UI 顯示
            for idx, triples_cnt in sorted(skipped_results, key=lambda x: x[0]):
                total_merged += triples_cnt
                yield BuildProgress(
                    event="chunk_done",
                    chunk_idx=idx, total_chunks=total_chunks,
                    triples_extracted=triples_cnt, triples_merged=triples_cnt,
                    message=f"[{_doc_title}] 第 {idx} 段已跳過（已於先前提取）",
                )

            _MAX_CHUNK_RETRIES = 2

            # 4. 用 Lock 保護進度寫入的 async 函式
            async def _save_chunk_progress(chunk_idx: int, triples_count: int):
                async with progress_lock:
                    try:
                        current_progress = {}
                        if progress_file.exists():
                            with open(progress_file, "r", encoding="utf-8") as f:
                                current_progress = _json.load(f)
                        current_progress[str(chunk_idx)] = {
                            "processed": True,
                            "triples_count": triples_count,
                            "ts": time.time()
                        }
                        progress_dir.mkdir(parents=True, exist_ok=True)
                        with open(progress_file, "w", encoding="utf-8") as f:
                            _json.dump(current_progress, f, ensure_ascii=False, indent=2)
                    except Exception as pe:
                        logger.warning(f"寫入 chunk 進度檔失敗: {pe}")

            async def _save_chunk_failure(chunk_idx: int, error_msg: str):
                async with progress_lock:
                    try:
                        current_progress = {}
                        if progress_file.exists():
                            with open(progress_file, "r", encoding="utf-8") as f:
                                current_progress = _json.load(f)
                        current_progress[str(chunk_idx)] = {
                            "processed": False,
                            "error": error_msg[:500],
                            "ts": time.time(),
                        }
                        progress_dir.mkdir(parents=True, exist_ok=True)
                        with open(progress_file, "w", encoding="utf-8") as f:
                            _json.dump(current_progress, f, ensure_ascii=False, indent=2)
                    except Exception as pe:
                        logger.warning(f"寫入 chunk 失敗記錄失敗: {pe}")

            async def _process_chunk(
                sc: SentenceChunk,
                __doc_id=_doc_id, __doc_title=_doc_title,
            ):
                async with sem:
                    last_err = None
                    for attempt in range(1 + _MAX_CHUNK_RETRIES):
                        try:
                            # 決定是否使用 fallback model
                            model_override = None
                            if attempt == _MAX_CHUNK_RETRIES:
                                # 最後一次重試，嘗試使用 fallback 模型
                                try:
                                    current_model = settings.ollama_llm_model
                                    model_override = await _get_fallback_model(current_model)
                                    if model_override:
                                        logger.warning(
                                            f"SVO 提取重試 [{__doc_title} chunk {sc.idx}] "
                                            f"最後一次重試：降級至備用模型 {model_override}"
                                        )
                                except Exception:
                                    pass

                            triples = await extract_svo_from_text(sc.text, model_override=model_override)
                            merged = await merge_triples_to_neo4j(
                                triples, kg_id, __doc_id, db_name, chunk_id=sc.chunk_id
                            )
                            # 成功提取與寫入，保存進度
                            await _save_chunk_progress(sc.idx, len(triples))
                            return sc.idx, triples, merged, None
                        except Exception as e:
                            last_err = e
                            if attempt < _MAX_CHUNK_RETRIES:
                                wait = 2 ** attempt
                                logger.warning(
                                    f"SVO 提取重試 [{__doc_title} chunk {sc.idx}] "
                                    f"第 {attempt + 1} 次（等 {wait}s）：{e}"
                                )
                                await asyncio.sleep(wait)
                    logger.warning(
                        f"SVO 提取失敗 [{__doc_title} chunk {sc.idx}]"
                        f"（已重試 {_MAX_CHUNK_RETRIES} 次）：{last_err}"
                    )
                    return sc.idx, [], 0, last_err

            if to_process:
                max_doc_attempts = 3
                doc_attempt = 0
                last_errors: dict[int, str] = {}
                while to_process and doc_attempt < max_doc_attempts:
                    doc_attempt += 1
                    if doc_attempt > 1:
                        yield BuildProgress(
                            event="chunk_start",
                            chunk_idx=0, total_chunks=total_chunks,
                            message=f"[{_doc_title}] 重新對上一步失敗的 {len(to_process)} 個段落發起原地重試抽取（第 {doc_attempt} 次）...",
                        )

                    chunk_results = await asyncio.gather(
                        *[_process_chunk(sc) for sc in to_process]
                    )

                    success_indices = set()
                    doc_has_error = False
                    for idx, triples, merged, err in sorted(chunk_results, key=lambda x: x[0]):
                        if err:
                            doc_has_error = True
                            last_errors[idx] = str(err)
                            yield BuildProgress(event="error", chunk_idx=idx, message=str(err))
                        else:
                            success_indices.add(idx)
                            total_merged += merged
                            yield BuildProgress(
                                event="chunk_done",
                                chunk_idx=idx, total_chunks=total_chunks,
                                triples_extracted=len(triples), triples_merged=merged,
                                message=f"[{_doc_title}] 第 {idx} 段完成：{len(triples)} 組三元組",
                            )

                    # 僅保留失敗的段落作為下一輪的 to_process
                    to_process = [sc for sc in to_process if sc.idx not in success_indices]

                    if not to_process:
                        break

                if not to_process:
                    # 100% 成功，標記為已處理
                    await _set_doc_svo_processed(_doc_id)
                else:
                    logger.warning(
                        f"[{_doc_title}] 達到原地重試上限 {max_doc_attempts} 次，仍有 {len(to_process)} 個 chunk 失敗，暫時跳過此文檔..."
                    )
                    for sc in to_process:
                        await _save_chunk_failure(sc.idx, last_errors.get(sc.idx, "未知錯誤"))
            else:
                await _set_doc_svo_processed(_doc_id)
        except Exception as _doc_err:
            logger.exception(f"文件處理發生未預期錯誤，已跳過此文件：{doc.title}")
            yield BuildProgress(
                event="error",
                message=f"[{doc.title}] 未預期錯誤，已跳過此文件：{_doc_err}",
            )
            continue

    # 清除本次 build 可能遺留的孤兒 Entity 節點（無任何關係邊）
    orphans_removed = await cleanup_orphan_entities(kg_id, db_name)

    await kg_repo.refresh_counts(kg_id)
    # 再次清除快取以反映最新合併的資料
    to_del = [k for k in _bfs_cache.keys() if k[0] == str(kg_id)]
    for k in to_del:
        _bfs_cache.pop(k, None)
    done_message = f"圖譜建立完成，共合併 {total_merged} 組三元組"
    if orphans_removed:
        done_message += f"，清除 {orphans_removed} 個孤兒節點"
    yield BuildProgress(
        event="done",
        triples_merged=total_merged,
        message=done_message,
    )


async def extract_svo_from_text(text: str, model_override: str | None = None) -> list[SVOTriple]:
    """呼叫 LLM 從單段文字提取本體論知識三元組，依設定選用 JSON 或 Pipe 模式。"""
    if not text.strip():
        return []

    prompt = (
        "請從以下文字中提取知識關係，以六欄格式輸出，每行一組：\n"
        "主詞|主詞類型|關係類別|動詞|受詞|受詞類型\n\n"
        "【實體類型】選最接近：概念、算法、技術、方法、工具、框架、模型、系統、人物、組織、資料集、指標、其他\n\n"
        "【關係類別】必須從以下 30 種選一，不可自造：\n"
        "層級/組成：\n"
        "  IS_A        → 是一種、屬於、屬類\n"
        "  PART_OF     → 是...的部分、隸屬於\n"
        "  CONTAINS    → 包含、涵蓋、由...組成\n"
        "  INSTANCE_OF → 是...的例子、如、例如\n"
        "因果/效應：\n"
        "  CAUSES      → 導致、引起、造成、觸發\n"
        "  PREVENTS    → 防止、阻止、避免\n"
        "  ENABLES     → 使能、支援、允許、實現\n"
        "  IMPROVES    → 優化、提升、改進、增強\n"
        "  INHIBITS    → 降低、妨礙、限制、抑制\n"
        "功能/操作：\n"
        "  USES        → 使用、調用、依賴、基於\n"
        "  REQUIRES    → 需要、依賴於、前提是\n"
        "  PRODUCES    → 輸出、生成、產生、建立\n"
        "  IMPLEMENTS  → 實現、實作、落地、執行\n"
        "  REPLACES    → 取代、替換、棄用\n"
        "  EXTENDS     → 擴展、延伸、建構於、衍生自\n"
        "比較：\n"
        "  CONTRASTS   → 不同於、相比、相對於\n"
        "  SIMILAR_TO  → 類似、相似、等同於\n"
        "  OUTPERFORMS → 優於、超越、勝過\n"
        "描述/定義：\n"
        "  DEFINED_AS  → 定義為、稱為、指的是\n"
        "  HAS_PROPERTY → 具有、是...的特點、特性為\n"
        "  MEASURED_BY → 用...衡量、以...評估、指標為\n"
        "  APPLIES_TO  → 應用於、適用於、用於\n"
        "時序：\n"
        "  PRECEDES    → 先於、之前執行、前置步驟\n"
        "  FOLLOWS     → 接著、後於、之後發生\n"
        "  CO_OCCURS   → 伴隨、同時、共同出現\n"
        "資料流：\n"
        "  INPUTS      → 接收、輸入、讀取\n"
        "  TRANSFORMS  → 轉換、處理、映射、編碼\n"
        "歸屬/解決：\n"
        "  CREATED_BY  → 由...提出、由...開發、創建者為\n"
        "  SOLVES      → 解決、處理、應對、克服\n"
        "  RELATED_TO  → 【絕對最後手段】以上 29 種皆完全不適用時才能用，目標使用率 < 5%\n\n"
        "規則：\n"
        "- 主詞與受詞為名詞或名詞短語（2-15字），去除冗餘後綴（如「XX技術」→「XX」、「XX方法」→「XX」）\n"
        "- 動詞欄位盡量引用原文中的實際措辭（2-8字），忠實反映原文用語，不要自行概括替換\n"
        "- 嚴禁濫用 RELATED_TO：有明確語意關係必須選精確類別，每段文字最多使用 1 次\n"
        "- 必須只輸出六欄格式（用 | 分隔），絕對不要使用 ``` 或 ```python 等 Markdown 程式碼方塊包裹輸出，不要輸出任何前言、引言、註釋或解釋\n"
        "- 行數上限 30 行，優先抽取最重要的知識關係，若無符合內容則直接輸出空白\n\n"
        "範例：\n"
        "Q-Learning|算法|IS_A|屬於|強化學習|概念\n"
        "工具呼叫|方法|PART_OF|包含於|代理迴圈|概念\n"
        "注意力機制|技術|PART_OF|組成|Transformer|模型\n"
        "提示快取|技術|ENABLES|使能|跨請求重用|技術\n"
        "Context 超限|概念|CAUSES|導致|回應延遲增加|指標\n"
        "Transformer|模型|USES|使用|多頭注意力|技術\n"
        "GPT-4|模型|OUTPERFORMS|超越|GPT-3.5|模型\n"
        "梯度消失|概念|PREVENTS|阻止|深層網路收斂|概念\n"
        "BatchNorm|技術|IMPROVES|穩定|訓練過程|概念\n"
        "ResNet|模型|EXTENDS|建構於|CNN|框架\n"
        "Adam|算法|REPLACES|取代|SGD|算法\n"
        "損失函數|概念|MEASURED_BY|衡量|模型表現|指標\n"
        "反向傳播|算法|PRECEDES|先於|權重更新|方法\n"
        "Tokenizer|工具|TRANSFORMS|將文字轉換為|Token|資料集\n"
        "BERT|模型|CREATED_BY|由 Google 提出|Google|組織\n"
        "Dropout|技術|PREVENTS|防止|過擬合|概念\n\n"
        f"文字：\n{text}"
    )
    import json as _json
    from core.config import settings

    if model_override:
        from core.providers.llm.ollama import OllamaLLMProvider
        llm = OllamaLLMProvider(base_url=settings.ollama_base_url, model=model_override)
    else:
        llm = get_llm_provider()

    # 優先嘗試 JSON 模式（如果設定為 json）
    if settings.svo_format == "json":
        json_prompt = prompt + (
            "\n\n【輸出格式】請以 JSON 陣列輸出，每項為：\n"
            '{"s":"主詞","st":"主詞類型","r":"關係類別","v":"動詞","o":"受詞","ot":"受詞類型"}\n'
            "不要輸出任何其他文字，只輸出 JSON 陣列。"
        )
        try:
            raw = await llm.generate_json(json_prompt)
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                items = _json.loads(match.group())
                triples = _parse_svo_json(items)
                if triples:
                    return _filter_hallucinated(triples, text)
        except Exception as e:
            logger.debug(f"JSON 模式失敗，回退 pipe 格式：{e}")

    # Fallback：傳統 pipe-delimited 格式
    raw = await llm.generate(prompt)
    triples = _parse_svo_lines(raw)
    return _filter_hallucinated(triples, text)


def _normalize_entity(name: str) -> str:
    """正規化實體名稱（☆8）：NFKC + 去頭尾空白 + 合併連續空格 + 修正康熙部首偏僻字。"""
    name = _unicodedata.normalize("NFKC", name).strip()
    name = re.sub(r"\s+", " ", name)
    # 修正康熙部首偏僻字對應
    mapping = {
        "\u2fd3": "龍",
        "\u2fe3": "龍",
        "\u2f9d": "門",
        "\u2f08": "人",
        "\u2f8b": "魚",
        "\u2f8c": "鳥",
        "\u2f94": "麥",
        "\u2f9c": "黃",
    }
    for k, v in mapping.items():
        name = name.replace(k, v)
    return name


async def merge_triples_to_neo4j(
    triples: list[SVOTriple],
    kg_id: UUID,
    doc_id: UUID,
    db_name: str = "",
    chunk_id: str = "",
) -> int:
    """
    將帶型別的三元組以 rel_type 作為真正的 Neo4j relationship type 寫入。
    每個 rel_type 一批 UNWIND（最多 8 批），確保邊標籤有語意意義。
    db_name 不為空 → 寫入 KG 專用資料庫
    db_name 為空  → 寫入主資料庫並以 kg_id 隔離
    chunk_id 不為空 → Entity / Relationship 節點追加 source_chunk_ids
    """
    if not triples:
        return 0

    from collections import defaultdict
    driver = get_driver()
    doc_id_str = str(doc_id)
    kg_id_str = str(kg_id)

    # 按 rel_type 分組，rel_type 已通過 _VALID_REL_TYPES 驗證，可安全嵌入 Cypher
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        groups[t.rel_type].append({
            "subject": _normalize_entity(t.subject),
            "s_type":  t.subject_type,
            "object":  _normalize_entity(t.object),
            "o_type":  t.object_type,
            "verb":    t.verb,
            "s_id":    str(uuid4()),
            "o_id":    str(uuid4()),
        })

    total_merged = 0
    for rel_type, rows in groups.items():
        try:
            if db_name:
                result = await driver.execute_query(
                    f"""
                    UNWIND $rows AS r
                    MERGE (s:Entity {{name: r.subject}})
                    ON CREATE SET s.id = r.s_id, s.type = r.s_type, s.created_at = datetime(),
                                  s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN r.s_type ELSE s.type END,
                                  s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(s.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(s.source_chunk_ids, []) END
                    MERGE (o:Entity {{name: r.object}})
                    ON CREATE SET o.id = r.o_id, o.type = r.o_type, o.created_at = datetime(),
                                  o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN r.o_type ELSE o.type END,
                                  o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(o.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(o.source_chunk_ids, []) END
                    MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                    ON CREATE SET rel.verb = r.verb, rel.confidence = 1, rel.created_at = datetime(),
                                  rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  rel.confidence = rel.confidence + 1,
                                  rel.verb = r.verb, rel.updated_at = datetime(),
                                  rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(rel.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(rel.source_chunk_ids, []) END
                    RETURN count(rel) AS merged
                    """,
                    rows=rows, doc_id=doc_id_str, chunk_id=chunk_id,
                    database_=db_name,
                )
            else:
                result = await driver.execute_query(
                    f"""
                    UNWIND $rows AS r
                    MERGE (s:Entity {{name: r.subject, kg_id: $kg_id}})
                    ON CREATE SET s.id = r.s_id, s.type = r.s_type, s.created_at = datetime(),
                                  s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN r.s_type ELSE s.type END,
                                  s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(s.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(s.source_chunk_ids, []) END
                    MERGE (o:Entity {{name: r.object, kg_id: $kg_id}})
                    ON CREATE SET o.id = r.o_id, o.type = r.o_type, o.created_at = datetime(),
                                  o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN r.o_type ELSE o.type END,
                                  o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(o.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(o.source_chunk_ids, []) END
                    MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                    ON CREATE SET rel.verb = r.verb, rel.confidence = 1, rel.created_at = datetime(),
                                  rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                    ON MATCH SET  rel.confidence = rel.confidence + 1,
                                  rel.verb = r.verb, rel.updated_at = datetime(),
                                  rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                      [x IN coalesce(rel.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                      ELSE coalesce(rel.source_chunk_ids, []) END
                    RETURN count(rel) AS merged
                    """,
                    rows=rows, doc_id=doc_id_str, kg_id=kg_id_str, chunk_id=chunk_id,
                )
            total_merged += result.records[0]["merged"] if result.records else len(rows)
        except Exception as e:
            logger.warning(f"批次 MERGE 失敗 [{rel_type}]，回退逐條模式：{e}")
            for row in rows:
                try:
                    if db_name:
                        await driver.execute_query(
                            f"""
                            MERGE (s:Entity {{name: $subject}})
                            ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime(),
                                          s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END,
                                          s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(s.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(s.source_chunk_ids, []) END
                            MERGE (o:Entity {{name: $object}})
                            ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime(),
                                          o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END,
                                          o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(o.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(o.source_chunk_ids, []) END
                            MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                            ON CREATE SET rel.verb = $verb, rel.confidence = 1, rel.created_at = datetime(),
                                          rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  rel.confidence = rel.confidence + 1,
                                          rel.verb = $verb, rel.updated_at = datetime(),
                                          rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(rel.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(rel.source_chunk_ids, []) END
                            """,
                            subject=row["subject"], object=row["object"],
                            verb=row["verb"], s_type=row["s_type"], o_type=row["o_type"],
                            doc_id=doc_id_str, s_id=row["s_id"], o_id=row["o_id"],
                            chunk_id=chunk_id, database_=db_name,
                        )
                    else:
                        await driver.execute_query(
                            f"""
                            MERGE (s:Entity {{name: $subject, kg_id: $kg_id}})
                            ON CREATE SET s.id = $s_id, s.type = $s_type, s.created_at = datetime(),
                                          s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN $s_type ELSE s.type END,
                                          s.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(s.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(s.source_chunk_ids, []) END
                            MERGE (o:Entity {{name: $object, kg_id: $kg_id}})
                            ON CREATE SET o.id = $o_id, o.type = $o_type, o.created_at = datetime(),
                                          o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN $o_type ELSE o.type END,
                                          o.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(o.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(o.source_chunk_ids, []) END
                            MERGE (s)-[rel:{rel_type} {{source_doc_id: $doc_id}}]->(o)
                            ON CREATE SET rel.verb = $verb, rel.confidence = 1, rel.created_at = datetime(),
                                          rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN [$chunk_id] ELSE [] END
                            ON MATCH SET  rel.confidence = rel.confidence + 1,
                                          rel.verb = $verb, rel.updated_at = datetime(),
                                          rel.source_chunk_ids = CASE WHEN $chunk_id <> '' THEN
                                              [x IN coalesce(rel.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
                                              ELSE coalesce(rel.source_chunk_ids, []) END
                            """,
                            subject=row["subject"], object=row["object"],
                            verb=row["verb"], s_type=row["s_type"], o_type=row["o_type"],
                            kg_id=kg_id_str, doc_id=doc_id_str,
                            s_id=row["s_id"], o_id=row["o_id"], chunk_id=chunk_id,
                        )
                    total_merged += 1
                except Exception as inner_e:
                    logger.warning(f"MERGE 失敗 [{row['subject']}|{rel_type}|{row['object']}]: {inner_e}")

    return total_merged


async def get_kg_graph(
    kg_id: UUID,
    limit: int = 200,
    min_confidence: int = 1,
) -> dict:
    """取得 KG 的 Entity 節點與 RELATION 邊清單，供前端視覺化或 API 輸出。"""
    import asyncio

    driver = get_driver()
    db_name = await _get_kg_db(kg_id)
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    params = {"kg_id": str(kg_id), "limit": limit, "min_conf": min_confidence}

    entities_q = driver.execute_query(
        f"""
        MATCH (e:Entity {kg_filter})
        OPTIONAL MATCH (e)-[r:{_ALL_REL_PATTERN}]->()
        WITH e, count(r) AS out_degree
        RETURN e.id AS id, e.name AS name, e.type AS type, out_degree
        ORDER BY out_degree DESC LIMIT $limit
        """,
        **params, **db_kw,
    )
    relations_q = driver.execute_query(
        f"""
        MATCH (s:Entity {kg_filter})-[r:{_ALL_REL_PATTERN}]->(o:Entity {kg_filter})
        WHERE r.confidence >= $min_conf
        RETURN s.name AS subject, s.type AS subject_type,
               type(r) AS rel_type, r.verb AS verb,
               o.name AS object, o.type AS object_type,
               r.confidence AS confidence, r.source_doc_id AS source_doc_id
        ORDER BY r.confidence DESC LIMIT $limit
        """,
        **params, **db_kw,
    )

    entities_result, relations_result = await asyncio.gather(entities_q, relations_q)

    entities = [dict(r) for r in entities_result.records]
    relations = [dict(r) for r in relations_result.records]

    return {
        "kg_id": str(kg_id),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "entities": entities,
        "relations": relations,
    }


def _build_ft_query(terms: list[str]) -> str:
    """將 terms 轉成 Lucene OR 查詢字串，特殊字元逸出。"""
    escape = str.maketrans({c: f"\\{c}" for c in r'+-&|!(){}[]^"~*?:\/'})
    return " OR ".join(f'"{t.translate(escape)}"' for t in terms)


async def query_svo_facts(
    kg_id: UUID,
    terms: list[str],
    hops: int = 2,
    limit: int = 50,
    db_name: str | None = None,
    min_confidence: int = 1,
) -> tuple[list[str], list[str]]:
    """
    BFS 遍歷 SVO 圖，回傳：
    - facts:       知識事實字串清單（供 LLM prompt 使用）
    - source_docs: 這些事實來源的 doc_id 字串清單
    - min_confidence: 只回傳 confidence >= 此值的邊（過濾低品質單次抽取）
    結果快取 _BFS_CACHE_TTL 秒，key = (kg_id, sorted_terms, hops, min_confidence)。
    """
    if not terms:
        return [], []

    cache_key = (str(kg_id), tuple(sorted(terms)), hops, min_confidence)
    cached = _bfs_cache_get(cache_key)
    if cached and time.time() - cached[3] < _BFS_CACHE_TTL:
        return cached[0], cached[1], cached[2]

    if db_name is None:
        db_name = await _get_kg_db(kg_id)

    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_id_str = str(kg_id)
    ft_query_str = _build_ft_query(terms)

    if db_name:
        seed_cypher_ft = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e RETURN e"
        )
        seed_params_ft = {"ft_q": ft_query_str}
    else:
        seed_cypher_ft = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e WHERE e.kg_id = $kg_id RETURN e"
        )
        seed_params_ft = {"ft_q": ft_query_str, "kg_id": kg_id_str}

    try:
        seed_result = await driver.execute_query(seed_cypher_ft, **seed_params_ft, **db_kw)
        use_ft = True
    except Exception:
        use_ft = False

    kg_filter = "" if db_name else "{kg_id: $kg_id}"

    if use_ft and seed_result.records:
        seed_ids = [r["e"].element_id for r in seed_result.records]

        # 查詢展開：將 seed 節點的 1-hop 鄰居也納入種子，擴大覆蓋範圍
        if seed_ids:
            try:
                expand_result = await driver.execute_query(
                    f"""
                    MATCH (seed)-[:{_ALL_REL_PATTERN}]-(neighbor:Entity {kg_filter})
                    WHERE elementId(seed) IN $seed_ids
                    RETURN DISTINCT elementId(neighbor) AS nid
                    LIMIT 20
                    """,
                    seed_ids=seed_ids,
                    **({"kg_id": kg_id_str} if not db_name else {}),
                    **db_kw,
                )
                extra = [r["nid"] for r in expand_result.records]
                seed_ids = list(dict.fromkeys(seed_ids + extra))  # 去重保序
            except Exception:
                pass

        bfs_params: dict = {"seed_ids": seed_ids, "limit": limit, "min_conf": min_confidence}
        if not db_name:
            bfs_params["kg_id"] = kg_id_str
        result = await driver.execute_query(
            f"""
            MATCH (seed)
            WHERE elementId(seed) IN $seed_ids
            WITH collect(seed) AS seeds
            UNWIND seeds AS seed
            MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity {kg_filter})
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            WHERE r.confidence >= $min_conf
            RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                   type(r) AS rel_type, r.verb AS verb,
                   o.name AS object, o.type AS object_type,
                   r.confidence AS confidence, r.source_doc_id AS source_doc_id
            ORDER BY confidence DESC LIMIT $limit
            """,
            **bfs_params, **db_kw,
        )
    else:
        where_clauses = " OR ".join(
            f"toLower(e.name) CONTAINS toLower($term{i})" for i in range(len(terms))
        )
        fallback_params: dict = {f"term{i}": t for i, t in enumerate(terms)}
        fallback_params["limit"] = limit
        fallback_params["min_conf"] = min_confidence
        if db_name:
            result = await driver.execute_query(
                f"""
                MATCH (e:Entity)
                WHERE {where_clauses}
                WITH collect(e) AS seeds
                UNWIND seeds AS seed
                MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity)
                UNWIND relationships(path) AS r
                WITH startNode(r) AS s, r, endNode(r) AS o
                WHERE r.confidence >= $min_conf
                RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                       type(r) AS rel_type, r.verb AS verb,
                       o.name AS object, o.type AS object_type,
                       r.confidence AS confidence, r.source_doc_id AS source_doc_id
                ORDER BY confidence DESC LIMIT $limit
                """,
                database_=db_name, **fallback_params,
            )
        else:
            fallback_params["kg_id"] = kg_id_str
            result = await driver.execute_query(
                f"""
                MATCH (e:Entity {{kg_id: $kg_id}})
                WHERE {where_clauses}
                WITH collect(e) AS seeds
                UNWIND seeds AS seed
                MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(neighbor:Entity {{kg_id: $kg_id}})
                UNWIND relationships(path) AS r
                WITH startNode(r) AS s, r, endNode(r) AS o
                WHERE r.confidence >= $min_conf
                RETURN DISTINCT s.name AS subject, s.type AS subject_type,
                       type(r) AS rel_type, r.verb AS verb,
                       o.name AS object, o.type AS object_type,
                       r.confidence AS confidence, r.source_doc_id AS source_doc_id
                ORDER BY confidence DESC LIMIT $limit
                """,
                **fallback_params,
            )

    # 以 (subject, object) 為 key 收集同一對節點的所有關係，組成推理鏈
    edge_map: dict[tuple[str, str], list[str]] = {}
    source_docs: list[str] = []
    seen_docs: set[str] = set()
    entity_freq: dict[str, int] = {}    # 實體出現頻率，供 chunk_ids 排序

    for r in result.records:
        rel_type = r.get("rel_type") or "RELATED_TO"
        verb = r.get("verb") or rel_type
        label = REL_TYPE_LABELS.get(rel_type, rel_type)
        s = r["subject"]
        o = r["object"]
        st = r.get("subject_type") or "概念"
        ot = r.get("object_type") or "概念"
        edge_str = f"{s}({st}) -[{label}:{verb}]→ {o}({ot})"
        edge_map.setdefault((s, o), []).append(edge_str)
        entity_freq[s] = entity_freq.get(s, 0) + 1
        entity_freq[o] = entity_freq.get(o, 0) + 1
        doc_id = r.get("source_doc_id")
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            source_docs.append(doc_id)

    # 輸出：單邊事實 + 多跳推理鏈（找出可串接的路徑 A→B→C）
    facts: list[str] = []
    for edges in edge_map.values():
        facts.extend(edges)

    # 推理鏈：嘗試把 A→B 和 B→C 串成 A→B→C
    end_nodes: dict[str, str] = {}   # subject → edge_str（供串接）
    start_nodes: dict[str, str] = {} # object  → edge_str
    for (s, o), edges in edge_map.items():
        end_nodes[o] = edges[0]
        start_nodes[s] = edges[0]
    chains: list[str] = []
    for (s, o), edges in edge_map.items():
        if o in start_nodes and (o, None) != (s, None):
            next_edge = start_nodes[o]
            chain = f"{edges[0]} → {next_edge.split('→', 1)[-1].strip()}"
            chains.append(f"[推理鏈] {chain}")
    facts.extend(chains[:10])  # 最多附加 10 條推理鏈

    # 收集 BFS 實體節點的 source_chunk_ids（依出現頻率排序，高頻 → 高優先）
    chunk_ids: list[str] = []
    entity_names = list(entity_freq.keys())
    if entity_names:
        try:
            if db_name:
                ci_result = await driver.execute_query(
                    "UNWIND $names AS n MATCH (e:Entity {name: n}) "
                    "RETURN e.name AS name, coalesce(e.source_chunk_ids, []) AS cids",
                    names=entity_names, database_=db_name,
                )
            else:
                ci_result = await driver.execute_query(
                    "UNWIND $names AS n MATCH (e:Entity {name: n, kg_id: $kg_id}) "
                    "RETURN e.name AS name, coalesce(e.source_chunk_ids, []) AS cids",
                    names=entity_names, kg_id=kg_id_str,
                )
            rows = sorted(
                [(r["name"], r["cids"]) for r in ci_result.records],
                key=lambda x: entity_freq.get(x[0], 0), reverse=True,
            )
            seen_cids: set[str] = set()
            for _, cids in rows:
                for cid in (cids or []):
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        chunk_ids.append(cid)
        except Exception as e:
            logger.debug(f"chunk_ids 收集失敗（非必要）：{e}")

    _bfs_cache_set(cache_key, (facts, source_docs, chunk_ids, time.time()))
    return facts, source_docs, chunk_ids


async def _batch_get_doc_titles(doc_ids: list[str]) -> dict[str, str]:
    """批次查詢 Document 標題（主資料庫），回傳 {doc_id: title}。"""
    if not doc_ids:
        return {}
    driver = get_driver()
    try:
        result = await driver.execute_query(
            "MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.title AS title",
            ids=doc_ids,
        )
        return {r["id"]: r["title"] for r in result.records if r["title"]}
    except Exception as e:
        logger.warning(f"批次查詢 Document 標題失敗：{e}")
        return {}


async def query_svo_facts_with_provenance(
    kg_id: UUID,
    terms: list[str],
    hops: int = 2,
    limit: int = 50,
    db_name: str | None = None,
    min_confidence: int = 1,
    instance_id: str = "local",
) -> list:
    """
    Phase 3a：帶完整溯源資訊的 BFS 查詢。
    回傳 list[SourcedFact]，每條事實包含：
      - 來源文件標題（JOIN Document 節點）
      - 信心分數（confidence）
      - 建立時間（created_at，ISO 8601）
    """
    from models.provenance import SourcedFact

    if not terms:
        return []

    if db_name is None:
        db_name = await _get_kg_db(kg_id)

    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_id_str = str(kg_id)
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    ft_query_str = _build_ft_query(terms)

    # ── 種子節點（Full-text → fallback CONTAINS）─────────────────────────────
    use_ft = False
    seed_ids: list[str] = []
    if db_name:
        seed_cypher = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e RETURN e"
        )
        seed_params: dict = {"ft_q": ft_query_str}
    else:
        seed_cypher = (
            "CALL db.index.fulltext.queryNodes('entity_name_ft', $ft_q) "
            "YIELD node AS e WHERE e.kg_id = $kg_id RETURN e"
        )
        seed_params = {"ft_q": ft_query_str, "kg_id": kg_id_str}

    try:
        seed_res = await driver.execute_query(seed_cypher, **seed_params, **db_kw)
        if seed_res.records:
            seed_ids = [r["e"].element_id for r in seed_res.records]
            use_ft = True
    except Exception:
        pass

    # ── BFS 查詢（帶 created_at）──────────────────────────────────────────────
    raw_records = []
    if use_ft and seed_ids:
        bfs_params: dict = {"seed_ids": seed_ids, "limit": limit, "min_conf": min_confidence}
        if not db_name:
            bfs_params["kg_id"] = kg_id_str
        res = await driver.execute_query(
            f"""
            MATCH (seed) WHERE elementId(seed) IN $seed_ids
            WITH collect(seed) AS seeds
            UNWIND seeds AS seed
            MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(nb:Entity {kg_filter})
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS o
            WHERE r.confidence >= $min_conf
            RETURN DISTINCT
                s.name AS subject, s.type AS subject_type,
                type(r) AS rel_type, r.verb AS verb,
                o.name AS object, o.type AS object_type,
                r.confidence AS confidence,
                r.source_doc_id AS source_doc_id,
                toString(r.created_at) AS created_at
            ORDER BY confidence DESC LIMIT $limit
            """,
            **bfs_params, **db_kw,
        )
        raw_records = res.records
    else:
        where = " OR ".join(
            f"toLower(e.name) CONTAINS toLower($term{i})" for i in range(len(terms))
        )
        fb_params: dict = {f"term{i}": t for i, t in enumerate(terms)}
        fb_params.update({"limit": limit, "min_conf": min_confidence})
        if not db_name:
            fb_params["kg_id"] = kg_id_str
        cypher = f"""
        MATCH (e:Entity {kg_filter}) WHERE {where}
        WITH collect(e) AS seeds
        UNWIND seeds AS seed
        MATCH path = (seed)-[:{_ALL_REL_PATTERN}*1..{hops}]-(nb:Entity {kg_filter})
        UNWIND relationships(path) AS r
        WITH startNode(r) AS s, r, endNode(r) AS o
        WHERE r.confidence >= $min_conf
        RETURN DISTINCT
            s.name AS subject, s.type AS subject_type,
            type(r) AS rel_type, r.verb AS verb,
            o.name AS object, o.type AS object_type,
            r.confidence AS confidence,
            r.source_doc_id AS source_doc_id,
            toString(r.created_at) AS created_at
        ORDER BY confidence DESC LIMIT $limit
        """
        res = await driver.execute_query(cypher, **fb_params, **db_kw)
        raw_records = res.records

    # ── 批次查詢文件標題 ──────────────────────────────────────────────────────
    doc_ids = list({r.get("source_doc_id") for r in raw_records if r.get("source_doc_id")})
    title_map = await _batch_get_doc_titles(doc_ids)

    # ── 組裝 SourcedFact ──────────────────────────────────────────────────────
    sourced: list[SourcedFact] = []
    seen: set[tuple] = set()
    for r in raw_records:
        rel_type = r.get("rel_type") or "RELATED_TO"
        verb = r.get("verb") or rel_type
        label = REL_TYPE_LABELS.get(rel_type, rel_type)
        s = r["subject"]
        o = r["object"]
        st = r.get("subject_type") or "概念"
        ot = r.get("object_type") or "概念"
        key = (s, rel_type, o)
        if key in seen:
            continue
        seen.add(key)
        doc_id = r.get("source_doc_id") or ""
        sourced.append(SourcedFact(
            fact_str=f"{s}({st}) -[{label}:{verb}]→ {o}({ot})",
            subject=s, subject_type=st,
            rel_type=rel_type, verb=verb,
            object=o, object_type=ot,
            confidence=r.get("confidence") or 1,
            source_doc_id=doc_id,
            source_doc_title=title_map.get(doc_id, ""),
            created_at=r.get("created_at") or "",
            instance_id=instance_id,
        ))

    return sourced


async def create_entity_index() -> None:
    """在 lifespan 啟動時建立 Entity 的 Neo4j 索引，加速查詢。"""
    driver = get_driver()
    await driver.execute_query(
        "CREATE INDEX entity_kg_name IF NOT EXISTS FOR (e:Entity) ON (e.kg_id, e.name)"
    )
    # fulltext index：加速 query_svo_facts 的關鍵字模糊搜尋
    try:
        await driver.execute_query(
            "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]"
        )
    except Exception as e:
        logger.debug(f"fulltext index 建立跳過（可能已存在）：{e}")


# type 屬性 → Neo4j 附加標籤的映射（保留 Entity 主標籤，加上語義標籤）
_TYPE_LABEL_MAP: dict[str, str] = {
    "概念":  "Concept",
    "算法":  "Algorithm",
    "技術":  "Technology",
    "方法":  "Method",
    "工具":  "Tool",
    "框架":  "Framework",
    "模型":  "Model",
    "系統":  "System",
    "人物":  "Person",
    "組織":  "Organization",
    "資料集": "Dataset",
    "指標":  "Metric",
    "其他":  "Other",
}


async def apply_type_labels(kg_id: UUID, db_name: str = "") -> dict[str, int]:
    """
    依 Entity.type 屬性為節點附加語義標籤（e.g. :Algorithm, :Model）。
    不移除 :Entity 主標籤，只疊加。
    回傳各標籤打上的節點數 {label: count}。
    """
    driver = get_driver()
    db_kw = {"database_": db_name} if db_name else {}
    kg_filter = "" if db_name else "{kg_id: $kg_id}"
    params_base: dict = {} if db_name else {"kg_id": str(kg_id)}

    stats: dict[str, int] = {}
    for type_name, label in _TYPE_LABEL_MAP.items():
        result = await driver.execute_query(
            f"MATCH (e:Entity {kg_filter}) WHERE e.type = $type "
            f"SET e:`{label}` RETURN count(e) AS n",
            type=type_name, **params_base, **db_kw,
        )
        n = result.records[0]["n"] if result.records else 0
        if n:
            stats[label] = n

    # 為 rel_type 也加關係類型索引標籤（RELATION 本身已有，這裡確保完整性）
    logger.info(f"KG {kg_id} 標籤完成：{stats}")
    return stats


# ── 內部工具 ──────────────────────────────────────────────────────────────────

def _sentence_chunk(doc_id: str, text: str) -> list[SentenceChunk]:
    """句子感知切分，委派至 chunk_store.sentence_chunk()。保留此入口供模組內統一呼叫。"""
    return sentence_chunk(doc_id, text)


_VALID_TYPES = {
    "概念", "算法", "技術", "方法", "工具", "框架",
    "模型", "系統", "人物", "組織", "資料集", "指標", "其他",
}

_VALID_REL_TYPES = {
    # 層級/組成
    "IS_A", "PART_OF", "CONTAINS", "INSTANCE_OF",
    # 因果/效應
    "CAUSES", "PREVENTS", "ENABLES", "IMPROVES", "INHIBITS",
    # 功能/操作
    "USES", "REQUIRES", "PRODUCES", "IMPLEMENTS", "REPLACES", "EXTENDS",
    # 比較
    "CONTRASTS", "SIMILAR_TO", "OUTPERFORMS",
    # 描述/定義
    "DEFINED_AS", "HAS_PROPERTY", "MEASURED_BY", "APPLIES_TO",
    # 時序
    "PRECEDES", "FOLLOWS", "CO_OCCURS",
    # 資料流
    "INPUTS", "TRANSFORMS",
    # 歸屬/解決
    "CREATED_BY", "SOLVES", "RELATED_TO",
}

# Cypher relationship type pattern（供 MATCH 使用）
_ALL_REL_PATTERN = (
    "IS_A|PART_OF|CONTAINS|INSTANCE_OF|"
    "CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
    "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|"
    "CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
    "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|"
    "PRECEDES|FOLLOWS|CO_OCCURS|"
    "INPUTS|TRANSFORMS|"
    "CREATED_BY|SOLVES|RELATED_TO"
)

# 關係類別的中文顯示名稱（供 UI 顯示）
REL_TYPE_LABELS = {
    "IS_A":         "階層",
    "PART_OF":      "組成",
    "CONTAINS":     "包含",
    "INSTANCE_OF":  "實例",
    "CAUSES":       "因果",
    "PREVENTS":     "阻止",
    "ENABLES":      "賦能",
    "IMPROVES":     "改善",
    "INHIBITS":     "抑制",
    "USES":         "使用",
    "REQUIRES":     "需求",
    "PRODUCES":     "產出",
    "IMPLEMENTS":   "實作",
    "REPLACES":     "取代",
    "EXTENDS":      "延伸",
    "CONTRASTS":    "對比",
    "SIMILAR_TO":   "相似",
    "OUTPERFORMS":  "優越",
    "DEFINED_AS":   "定義",
    "HAS_PROPERTY": "屬性",
    "MEASURED_BY":  "量測",
    "APPLIES_TO":   "應用",
    "PRECEDES":     "前置",
    "FOLLOWS":      "後置",
    "CO_OCCURS":    "共現",
    "INPUTS":       "輸入",
    "TRANSFORMS":   "轉換",
    "CREATED_BY":   "歸屬",
    "SOLVES":       "解決",
    "RELATED_TO":   "相關",
}


def _parse_svo_json(items: list[dict]) -> list[SVOTriple]:
    """解析 LLM 回傳的 JSON 格式三元組陣列。"""
    triples: list[SVOTriple] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        s = str(item.get("s", "")).strip()
        st = str(item.get("st", "其他")).strip()
        r = str(item.get("r", "RELATED_TO")).strip().upper()
        v = str(item.get("v", "")).strip()
        o = str(item.get("o", "")).strip()
        ot = str(item.get("ot", "其他")).strip()
        if not s or not v or not o:
            continue
        if len(s) > 50 or len(v) > 20 or len(o) > 50:
            continue
        if st not in _VALID_TYPES:
            st = "其他"
        if ot not in _VALID_TYPES:
            ot = "其他"
        if r not in _VALID_REL_TYPES:
            r = "RELATED_TO"
        key = (s, r, o)
        if key in seen:
            continue
        seen.add(key)
        triples.append(SVOTriple(subject=s, subject_type=st, rel_type=r, verb=v, object=o, object_type=ot))
    return triples


def _filter_hallucinated(triples: list[SVOTriple], source_text: str) -> list[SVOTriple]:
    """過濾主詞與受詞皆不出現於原文的幻覺三元組（寬鬆：至少一個詞出現即保留）。"""
    text_lower = source_text.lower()
    kept = []
    for t in triples:
        s_hit = t.subject.lower() in text_lower
        o_hit = t.object.lower() in text_lower
        if s_hit or o_hit:
            kept.append(t)
        else:
            logger.debug(f"幻覺過濾：{t.subject}|{t.rel_type}|{t.object}")
    return kept


def _parse_svo_lines(raw: str) -> list[SVOTriple]:
    """解析 LLM 回傳的知識三元組行。

    6欄（新）：主詞|主詞類型|關係類別|關係描述|受詞|受詞類型
    5欄（舊）：主詞|主詞類型|關係描述|受詞|受詞類型
    3欄（舊）：主詞|關係描述|受詞
    """
    triples: list[SVOTriple] = []
    seen: set[tuple[str, str, str]] = set()

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[|｜]", line)]

        if len(parts) == 6:
            s, s_type, rel_type, v, o, o_type = parts
            if s_type not in _VALID_TYPES:
                s_type = "其他"
            if o_type not in _VALID_TYPES:
                o_type = "其他"
            if rel_type not in _VALID_REL_TYPES:
                rel_type = "RELATED_TO"
        elif len(parts) == 5:
            s, s_type, v, o, o_type = parts
            rel_type = "RELATED_TO"
            if s_type not in _VALID_TYPES:
                s_type = "其他"
            if o_type not in _VALID_TYPES:
                o_type = "其他"
        elif len(parts) == 3:
            s, v, o = parts
            s_type = o_type = "概念"
            rel_type = "RELATED_TO"
        else:
            continue

        if not s or not v or not o:
            continue
        if len(s) > 50 or len(v) > 20 or len(o) > 50:
            continue
        key = (s, rel_type, o)   # 用 rel_type 去重（同類別的同主受詞只保留一條）
        if key in seen:
            continue
        seen.add(key)
        triples.append(SVOTriple(
            subject=s, subject_type=s_type,
            rel_type=rel_type,
            verb=v,
            object=o, object_type=o_type,
        ))

    return triples


async def _clear_kg_entities(kg_id: UUID, db_name: str = "") -> None:
    """force_rebuild 時清除此 KG 的所有 Entity 與 RELATION（路由層 ConceptNode 不動）。"""
    driver = get_driver()
    if db_name:
        await driver.execute_query(
            "MATCH (e:Entity) DETACH DELETE e",
            database_=db_name,
        )
    else:
        await driver.execute_query(
            "MATCH (e:Entity {kg_id: $kg_id}) DETACH DELETE e",
            kg_id=str(kg_id),
        )
    logger.info(f"KG {kg_id} Entity 清除完成（force_rebuild）")


async def _clear_kg_relations(kg_id: UUID, db_name: str = "") -> None:
    """rebuild_relations_only 時只清除關係邊，保留 Entity 節點。"""
    driver = get_driver()
    if db_name:
        await driver.execute_query(
            f"MATCH ()-[r:{_ALL_REL_PATTERN}]->() DELETE r",
            database_=db_name,
        )
    else:
        await driver.execute_query(
            f"MATCH (s:Entity {{kg_id: $kg_id}})-[r:{_ALL_REL_PATTERN}]->() DELETE r",
            kg_id=str(kg_id),
        )
    logger.info(f"KG {kg_id} 關係邊清除完成（rebuild_relations_only）")


async def cleanup_orphan_entities(kg_id: UUID, db_name: str = "") -> int:
    """
    清除沒有任何關係邊的孤兒 Entity 節點（degree = 0）。
    典型成因：rebuild_relations_only 重建後舊實體不再被任何新三元組引用、
    merge_triples_to_neo4j 批次 MERGE 部分失敗遺留的殘留節點。
    每次 build_graph_for_kg 完成後自動呼叫一次，回傳實際刪除的節點數。
    """
    driver = get_driver()
    if db_name:
        result = await driver.execute_query(
            """
            MATCH (e:Entity) WHERE NOT (e)--()
            WITH e LIMIT 5000
            DETACH DELETE e
            RETURN count(e) AS removed
            """,
            database_=db_name,
        )
    else:
        result = await driver.execute_query(
            """
            MATCH (e:Entity {kg_id: $kg_id}) WHERE NOT (e)--()
            WITH e LIMIT 5000
            DETACH DELETE e
            RETURN count(e) AS removed
            """,
            kg_id=str(kg_id),
        )
    removed = result.records[0]["removed"] if result.records else 0
    if removed:
        logger.info(f"[SVO] KG {kg_id} 清除 {removed} 個孤兒 Entity 節點（degree=0）")
    return removed


async def _filter_unprocessed_docs(docs: list) -> list:
    """增量模式：回傳尚未標記 svo_processed_at 的文件清單。"""
    if not docs:
        return []
    driver = get_driver()
    result = await driver.execute_query(
        """
        UNWIND $ids AS doc_id
        MATCH (d:Document {id: doc_id})
        WHERE d.svo_processed_at IS NULL
        RETURN d.id AS id
        """,
        ids=[str(d.id) for d in docs],
    )
    unprocessed_ids = {r["id"] for r in result.records}
    return [d for d in docs if str(d.id) in unprocessed_ids]


async def _set_doc_svo_processed(doc_id: UUID) -> None:
    """文件 SVO 處理完畢後，記錄時間戳以供增量跳過。"""
    try:
        await get_driver().execute_query(
            "MATCH (d:Document {id: $id}) SET d.svo_processed_at = datetime()",
            id=str(doc_id),
        )
    except Exception as e:
        logger.warning(f"設定 svo_processed_at 失敗：{e}")


async def _reset_docs_svo_processed(kg_id: UUID) -> None:
    """force_rebuild 時清除 KG 下所有文件的 svo_processed_at，讓增量追蹤重置。"""
    try:
        await get_driver().execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[:CONTAINS]->(d:Document)
            SET d.svo_processed_at = null
            """,
            kg_id=str(kg_id),
        )
    except Exception as e:
        logger.warning(f"重置 svo_processed_at 失敗：{e}")
