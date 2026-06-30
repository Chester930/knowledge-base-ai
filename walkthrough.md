# 智慧知識庫 — 驗收報告（Walkthrough）

> 建立日期：2026-06-30  
> 測試狀態：**495 tests passed**

---

## 一、專案完成總覽

| Phase | 項目 | 狀態 |
|-------|------|------|
| Phase 1 | 本機多場景知識圖譜 | ✅ |
| Phase 2 | 世界知識聯邦 | ✅ |
| Phase 3 | 知識深化（溯源/圖視覺化/版控/訂閱） | ✅ |
| Chunk 溯源 Phase 1 | 句子感知切分 `_sentence_chunk()` | ✅ |
| Chunk 溯源 Phase 2 | ChunkStore 持久化 | ✅ |
| Chunk 溯源 Phase 3 | Neo4j `source_chunk_ids` 溯源標記 | ✅ |
| Chunk 溯源 Phase 4 | 自我精煉 RAG 迴圈 | ✅ |
| Chunk 溯源 Phase 5 | 前端精煉進度 UI | ✅ |
| Session 4 | 生產環境 Bug 修復（LLM 掛機 + NaN%） | ✅ |

---

## 二、Chunk 溯源任務（TASK_CHUNK_PROVENANCE.md）驗收

### Phase 1：句子感知切分

**位置**：`services/svo_service.py:1077`、`services/chunk_store.py`（`SentenceChunk` 資料類別）

```
文章全文 → [。！？!?\n] 斷句 → 每 5 句一組 → SentenceChunk(chunk_id, idx, text, char_start, char_end)
```

**驗收通過**：
- `sentence_chunk()` 函式可匯入自 `chunk_store.py`
- 中英混排、空文、餘句不足 5 句均正確處理
- `char_start / char_end` 準確追蹤原文偏移

---

### Phase 2：ChunkStore 持久化

**位置**：`services/chunk_store.py`

```
chunk_store/{kg_id}/{doc_id}/chunk_XXXX.json
每個 JSON 包含：chunk_id、idx、doc_id、kg_id、sentences、text、char_start、char_end
```

**驗收通過**：
- `write / read / read_many / delete_doc` 均有測試（`tests/services/test_chunk_store.py`）
- 強制重建（`--force`）時舊 Chunk 自動清除後重寫
- `read(chunk_id)` 不需呼叫者提供 kg_id（chunk JSON 內記錄 kg_id）

---

### Phase 3：Neo4j 溯源標記

**位置**：`services/svo_service.py:418–491`

```cypher
ON CREATE SET e.source_chunk_ids = [$chunk_id]
ON MATCH  SET e.source_chunk_ids = [x IN coalesce(e.source_chunk_ids, []) WHERE x <> $chunk_id] + [$chunk_id]
```

**驗收通過**：
- Entity 節點、Relationship 邊均帶 `source_chunk_ids`
- 增量重建不重複累加相同 chunk_id

---

### Phase 4：自我精煉 RAG 迴圈

**位置**：`routers/agent.py`

**參數**：

| 常數 | 值 |
|------|----|
| `_CONFIDENCE_THRESHOLD` | 0.65 |
| `_MAX_REFINE_ROUNDS` | 3 |
| `_CHUNKS_PER_ROUND` | 3 |

**SSE 事件流程**：

```
status:searching → kg_route → svo_facts → sources
  → refine_preview(round=1) → refine(round=1)   # 信心 < 0.65 時觸發
  → refine_preview(round=2) → refine(round=2)
  → status:generating → token... → done
```

**驗收通過**：
- 低信心問題觸發精煉，補充 `source_chunk_ids` 指向的原文 Chunk
- `use_svo=False` 跳過精煉迴圈
- 最終答案不含 `{"confidence": ...}` 原始 JSON
- 測試：`tests/routers/test_agent_refine.py`

---

### Phase 5：前端 UI

**位置**：`ui/templates/index.html`

**實作**：
- `addRefineNote(uid, refine)` — 每輪在訊息泡泡下顯示精煉進度
- `finalizeRefineNotes(uid)` — 完成後顯示「✅ 共精煉 N 輪」
- `.refine-note` CSS class（含深色模式）
- `refine_preview` 事件：顯示半透明中間答案，讓使用者不需黑盒等待

---

## 三、Session 4 生產環境 Bug 修復

### 問題 1：LLM 完全沒有回答（掛機 300 秒後 timeout）

**根本原因**：
- BFS 遍歷跨 5 個 KG，取得 ~14 個 graph 文件
- `max_chars_per_doc=8000`（Session 3 設定）× 14 docs = **112,000 chars**
- phi4 `num_ctx=8192` ≈ 12,000 chars 容量，遠超上限
- Ollama 收到超大 prompt 無限等待，httpx 300 秒後 timeout

**修復**（`models/document.py` + `routers/agent.py`）：

| 修復 | 前 | 後 |
|------|----|----|
| `max_chars_per_doc` 預設值 | 8000 | **2000** |
| graph 文件數量上限 | 無限制 | `min(top_k × 2, 10)` |
| 總 context 字數保護 | 無 | **公平分配 7500 chars**（每篇均分，所有文件都有份額） |

**公平分配邏輯**：

```python
_per_doc = max(500, 7500 // len(contexts))
contexts = [{**c, "content": c["content"][:_per_doc]} for c in contexts]
```

舊邏輯（先到先得截斷）會讓後面的文件完全沒有份額（問八角框架時第 5 篇恰好是相關文件卻被踢出）。

---

### 問題 2：來源文件顯示 NaN%

**根本原因**：
- 圖譜驅動文件（graph source）沒有相似度分數
- 前端 `Math.round(s.score * 100)` → `Math.round(null * 100)` = NaN

**修復**：

Backend（`agent.py`）— 圖譜來源加 `"score": None`：
```python
sources.append({"title": doc.title, "score": None, "source": "graph"})
```

Frontend（`index.html`）— 空分數顯示「圖譜」紫色標籤：
```javascript
const pctStr = s.score != null ? `<span class="source-score">${Math.round(s.score * 100)}%</span>` : '';
const srcLabel = s.source === 'graph' ? '<span class="source-score" style="color:#a78bfa">圖譜</span>' : pctStr;
```

---

## 四、測試狀態

```
tests/services/test_chunk_store.py     ✅
tests/services/test_svo_service.py     ✅
tests/routers/test_agent_refine.py     ✅
tests/routers/test_rag_quality.py      ✅ (38 RAG 品質測試)
...（共 24 個測試檔案）

總計：495 passed, 0 failed, 0 errors
```

---

## 五、待辦（未排期）

| 項目 | 優先 | 備注 |
|------|------|------|
| Neo4j AuraDB 雲端同步設定 | 低 | 需用戶申請免費帳號後手動設定 |
| 使用者帳號系統 | 低 | 貢獻追蹤 + 知識授權 |
| 多語言實體對齊 | 低 | Phase 2d 延伸 |
| 知識品質評分 | 低 | 社群回饋機制 |
| KG 合併工具 | 低 | 衝突仲裁 |
| Server 重啟（本機） | **現在** | 套用 Session 4 的 agent.py / index.html 修改 |

---

## 六、如何重啟 Server 套用修改

```powershell
# 方法 A：Docker 環境
docker cp "C:\Users\666\Desktop\智慧知識庫\routers\agent.py" kg-api:/app/routers/agent.py
docker cp "C:\Users\666\Desktop\智慧知識庫\models\document.py" kg-api:/app/models/document.py
docker cp "C:\Users\666\Desktop\智慧知識庫\ui\templates\index.html" kg-api:/app/ui/templates/index.html
docker restart kg-api

# 方法 B：本地 uvicorn（已有 --reload 時自動套用）
# 前端 index.html 需手動重新整理瀏覽器（Ctrl+F5）
```
