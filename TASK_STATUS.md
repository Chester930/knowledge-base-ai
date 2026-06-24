# 任務書：智慧知識庫 RAG 問答品質優化

> 最後更新：2026-06-23（Session 3 進行中）

---

## 專案背景

多場景知識圖譜 RAG 系統，後端為 FastAPI + Neo4j + 多 LLM Provider，  
知識圖譜以 30 種語意關係 SVO 三元組儲存，並透過雙層路由（ConceptNode + BFS 圖遍歷）提供問答。

---

## ✅ Phase 1–3 已完成（上個 Session 驗收）

### Phase 1 — 本機多場景知識圖譜 ✅

- 多 KG 管理、SVO 知識提取、雙層 RAG 問答
- 公開/私有 KG、World Agent、實體探索 UI
- 多 Provider 支援（Ollama / OpenAI / Anthropic / Gemini / Grok）
- 語音/影片轉譯（faster-whisper）

### Phase 2 — 世界知識聯邦 ✅

- 免費 Neo4j AuraDB 分片 + GitHub Registry
- FederationCache 快取、並行查詢引擎、35 組 zh↔en 同義詞對齊

### Phase 3 — 知識深化 ✅

- 3a 知識溯源（SourcedFact + 信心分數）
- 3b 互動式 D3.js 力導向圖
- 3c KG 版本控制（changelog / diff / snapshot）
- 3d KG 訂閱自動同步（APScheduler 每 6 小時）

---

## ✅ Session 2 已完成事項（2026-06-23）

### SVO 知識圖譜建構驗收

| KG 名稱 | 文件 | Entity | Relation | 狀態 |
|---------|-----:|-------:|---------:|------|
| AI商業與科技趨勢 | 9 | 24,769 | 21,463 | ✅ |
| AI工作流與工具應用 | 19 | 927 | 638 | ✅ |
| AI影音內容創作 | 17 | 2,505 | 1,729 | ✅ |
| 教育與學習資源 | 25 | 5,027 | 4,403 | ✅ |
| 本體論與知識表示 | 10 | 12,843 | 12,221 | ✅ |
| 機器人與強化學習 | 8 | 4,353 | 3,391 | ✅ |
| 軟體架構與專案開發 | 11 | 3,704 | 2,935 | ✅ |
| 邊緣AI與工業視覺 | 13 | 2,219 | 1,529 | ✅ |
| **合計** | **112** | **56,347** | **48,309** | ✅ |

### 孤兒節點清除 ✅

- 原 224 個 Document 節點中 112 個為殭屍節點（重複匯入、無 CONTAINS 關係）
- 已清除，現剩 112 個節點全數有效

### RAG 問答品質修復（本 Session 核心工作）

#### 問題根源分析

測試問題：「遊戲化全書中的八角框架是什麼？」（kg_id: `3de0f63b-b7b3-46ed-8c52-603766752fd0`）

| 問題 | 原因 | 修復 |
|------|------|------|
| Chunk 切割過粗 | `\n\n` 分段後 avg_len >_CHUNK_SIZE，12 大段只能放 2 段，chunk[8]（含8動力列表）被排除 | `avg_len > _CHUNK_SIZE` 時降級用 `\n` 切割，取得細粒度 chunks |
| 命中 log 缺失 | 無法確認 selected chunk 範圍 | 加入 `[chunk_pick] selected_ordered` log |

#### 程式碼變更（`routers/agent.py`）

```python
# 修改前
if len(paragraphs) < 5:
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

# 修改後（加入平均段落長度偵測）
avg_len = sum(len(p) for p in paragraphs) / max(1, len(paragraphs))
if len(paragraphs) < 5 or avg_len > _CHUNK_SIZE:
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
```

```python
# 新增 chunk 選取 log
logger.info(f"[chunk_pick] selected_ordered={selected_ordered[:8]}")
```

#### 驗收結果

- ✅ chunk 切割策略改善，selected_ordered 已能選到含「八角框架八項核心動力」的 chunk[79]
- ⚠️ 但 chunk[8]（含完整 8 大動力列表）仍未進入 selected_ordered（embedding 相似度不足）
- ⚠️ 根本原因：SVO 從未提取「使命感/成就感/創意與反饋/所有權與佔有欲/社會影響與關聯性/稀缺性與迫切/未知性與好奇/損失與避免」等動力名稱，無法 boost

#### SVO 建構嘗試

- 執行 `run_build_kg.py --kg 3de0f63b...` → **0 triples**（書已被標記 `svo_processed_at`，視為已處理）
- 執行驗證查詢 → 遭遇 Claude session limit，本 Session 中斷

---

## ✅ Session 3 完成事項（2026-06-23 ~ 2026-06-24）

### 任務 1：遊戲化全書 SVO 強制重建 ✅

- 執行 `run_build_kg.py --kg 3de0f63b... --force`，耗時 221 分鐘
- 結果：1,556 triples，2,240 Entity 節點，17 份文件全部重提取
- 確認「使命感/成就感/損失與避免/八項核心動力」等 Entity 成功存入 DB

### 任務 2：max_chars_per_doc 調整 ✅

- `models/document.py` → 預設值 6000 → **8000**，上限 8000 → **12000**

### 任務 3：RELATED_TO 精化 ✅

- 執行 `run_reclassify_related_to.py`，完成時間 2026-06-23 19:59
- 全部 8 個 KG，合計重分類 **3,506 條**，維持 RELATED_TO 2,366 條（精化率 60%）

| KG | 重分類 | 維持 |
|----|-------:|-----:|
| AI商業與科技趨勢 | 1,076 | 1,289 |
| AI工作流與工具應用 | 34 | 17 |
| AI影音內容創作 | 80 | 73 |
| 教育與學習資源 | 454 | 183 |
| 本體論與知識表示 | 1,162 | 451 |
| 機器人與強化學習 | 297 | 183 |
| 軟體架構與專案開發 | 320 | 138 |
| 邊緣AI與工業視覺 | 83 | 32 |
| **合計** | **3,506** | **2,366** |

### RAG 問答品質根本修復 ✅（Session 3 最終驗收通過）

**根本原因**（本 Session 發現）：Ollama 呼叫未設 `num_ctx`，預設 2048 token context window 截斷了整個文件內容，LLM 根本看不到任何文件！

**修復清單**（`routers/agent.py` + `core/providers/llm/ollama.py`）：

| 修復項目 | 變更 | 效果 |
|----------|------|------|
| **Ollama num_ctx** | 加入 `"options": {"num_ctx": 8192}` | LLM 能讀到完整文件內容（根本修復）|
| **列舉型 chunk 加分** | `_ENUM_RE` 偵測 + `enum_bonus=0.25` | 含「1. 2. 3.」的定義段被優先選取 |
| **相鄰 chunk 擴展** | keyword_hit 加入 ±1 相鄰 chunk | 定義段+列舉段不再分離 |
| **SVO boost 提升** | svo_hits × 0.02 → × 0.10 | KG 實體名稱更有效引導 chunk 選取 |
| **相似度文件過濾** | `_SIM_MIN_SCORE = 0.38` | 移除 veo3/nano banana 等不相關文件，釋放 context 空間 |
| **max_chars_per_doc** | 預設 6000 → 8000，上限 12000 | 每份文件可納入更多 chunk |

**驗收測試結果**（2026-06-24 02:00）：

問：「遊戲化全書中的八角框架是什麼？包含哪些核心動力？」

答：✅ 正確列出全部八項核心動力：
1. 重大使命與呼召
2. 發展與成就
3. 賦予創造力與回饋
4. 所有權與占有欲
5. 社會影響力與同理心
6. 稀缺性與迫切
7. 不確定性與好奇心
8. 知覺（損失與避免）

---

### 中期（架構優化）

#### 4. RAG chunk 精選策略 ✅ 已完成

- 列舉型加分（Session 3 完成）
- 相鄰 chunk 擴展（Session 3 完成）
- SVO boost 提升（Session 3 完成）
- max_chars_per_doc 擴大到 8000（Session 3 完成）

#### 5. Neo4j AuraDB 雲端同步設定

**目的**：Phase 2 設計的世界聯邦需要個人 AuraDB 分片  
**步驟**：

1. 在 Neo4j AuraDB 建立免費帳號（200K 節點 / 400K 關係）
2. 填入 `.env`：`AURA_URI / AURA_USER / AURA_PASSWORD`
3. 執行 `python run_sync_public_kgs.py` 推送公開 KG
4. 開 PR 將 AuraDB URI 加入 `registry.json`

---

### 長期（未排期）

- [ ] 使用者帳號系統（貢獻追蹤 + 知識授權）
- [ ] 多語言實體對齊（Phase 2d 延伸，目前只有 zh↔en）
- [ ] 知識品質評分（基於社群回饋調整 confidence 分數）
- [ ] KG 合併工具（兩個本機 KG 智慧合併 + 衝突仲裁）

---

## 技術備注

### KG 專用資料庫對照

| KG 名稱 | DB Name | KG ID |
|---------|---------|-------|
| AI商業與科技趨勢 | kgai9ef9fb0c | 7d86a961-4bc2-4f0c-8a7d-fe2158b0c3a3 |
| AI工作流與工具應用 | kgai8782d79e | b5f883d2-4275-47ea-847d-9ef1c8dea896 |
| AI影音內容創作 | kgai901d32e7 | 3de0f63b-b7b3-46ed-8c52-603766752fd0 |
| 教育與學習資源 | kgdd3f0293 | 298aa949-9761-4c66-b3ad-f7666b462a5f |
| 本體論與知識表示 | kga19ee491 | c848724c-1f91-4519-9f9c-7401ed9c2613 |
| 機器人與強化學習 | kg2c78b8d5 | dc9a6af0-ca1d-45c1-afa7-c4d0dffbc7a8 |
| 軟體架構與專案開發 | kgb1d814ce | f7c7662b-1c24-4e6c-b74a-1aa65fbc456d |
| 邊緣AI與工業視覺 | kgai6bf953b2 | dd491b8d-d130-41bd-aa3d-e4d88c8114d2 |

### 常用維護指令

```bash
# 檢查 API 健康
curl http://localhost:8000/agent/health

# Docker 重啟 API
docker restart kg-api

# 查看 API 即時 log（最近 30 行）
docker logs kg-api --since 60s 2>&1 | Select-Object -Last 30

# 手動測試問答
$body = '{"question":"...","kg_id":"..."}';
Invoke-RestMethod -Uri "http://localhost:8000/agent/chat" -Method POST -Body $body -ContentType "application/json; charset=utf-8"

# copy 修改的檔案進 container
docker cp "C:\Users\666\Desktop\智慧知識庫\routers\agent.py" kg-api:/app/routers/agent.py && docker restart kg-api
```

### 重要常數（`routers/agent.py`）

| 常數 | 目前值 | 說明 |
|------|-------:|------|
| `_CHUNK_SIZE` | 400 | 每段落目標字元數 |
| `_CHUNK_ENCODE_CAP` | 512 | embedding 最大字元數 |
| `_GARBLED_THRESHOLD` | 0.3 | 亂碼過濾閾值 |
| `_MAX_CHUNKS_EMBED` | 200 | 大文件預篩閾值 |

### chunk 排序邏輯

```
q_name 命中數（↓）→ embedding 相似度（↓）→ SVO 實體命中 × 0.02
```

> **注意**：當所有 chunk 都含問題關鍵詞時（q_hit 全部 = 1），退化為純 embedding 分數競爭，可能導致「有標題無內容」的 chunk 勝出。  
> 這是本 Session 遇到的問題，後續需改進排序策略。
