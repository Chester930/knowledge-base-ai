# 路線圖 · Roadmap

> 最後更新：2026-06-21（Phase 3b 互動式圖視覺化完成）

本文件記錄「智慧知識庫」的長期願景與開發計畫。  
若你對某個階段的方向有興趣，歡迎填寫下方意願表單或直接開 Issue 討論。

---

## 願景

**任何人都能建立自己的知識圖譜，並選擇將知識公開給世界。**

個人維護的知識庫 → 標記公開 → 自動匯入世界知識圖譜 → 所有人皆可探索與提問。

---

## 現況（已完成）

### ✅ Phase 1 — 本機多場景知識圖譜

- **多 KG 管理**：建立多個場景知識圖譜，各自獨立
- **SVO 知識提取**：文件 → 30 種語意關係三元組 → Neo4j 圖譜
- **雙層 RAG 問答**：ConceptNode 路由層 + BFS 圖遍歷，精準問答
- **公開 / 私有 KG**：標記 `is_public`，World Agent 只讀公開 KG
- **World Agent**：跨所有公開 KG 問答，支援 SSE 串流
- **實體探索 UI**：搜尋實體 → 點擊展開鄰居關係 → 遞迴圖探索
- **多 Provider 支援**：Ollama / OpenAI / Anthropic / Gemini / Grok
- **語音 / 影片轉譯**：faster-whisper 支援 MP3 / MP4 / WAV / M4A 等格式
- **Docker 一鍵部署**、完整文件（SETUP / PROVIDERS / ARCHITECTURE / API）

---

## 已完成

### ✅ Phase 2 — 世界知識聯邦（World Knowledge Federation）

**解決問題**：Phase 1 的 World Agent 只能看到「本機上」的公開 KG。  
Phase 2 讓不同人、不同機器的公開知識透過聯邦架構互相流通。

#### 設計方向比較

| 模式 | 優點 | 缺點 |
|------|------|------|
| 純聯邦 API（各 instance 互查）| 去中心化，個人完全控制 | 對方 instance 離線就查不到 |
| 單一大型雲端 DB | 永遠在線，查詢快 | 需中心化管理、超過免費額度需付費 |
| **免費帳號分片聯邦（本計畫）** ✓ | 零費用、去中心化、離線容錯 | 需 registry 協調、AuraDB 閒置會暫停 |

---

#### 採用方案：免費帳號分片 + GitHub Registry

每位貢獻者使用自己的 **免費 Neo4j AuraDB**（200K 節點 / 400K 關係）存放公開 KG。  
World Agent 透過一份存放在 GitHub 的 **registry.json** 發現所有分片，並行查詢後合併結果。

這就像 **Git + GitHub** 的關係，但沒有中央付費伺服器：
- 本地 instance = 你的 source of truth
- 個人 AuraDB = 你的公開分片（你的免費帳號，你控制）
- registry.json = 所有人的 AuraDB 清單（GitHub 管理，開 PR 加入）
- World Agent = 讀 registry → 並行查詢所有分片

#### 架構圖

```
Chester 的 local  ──push→  AuraDB #1（Chester 的免費帳號）──┐
用戶 B 的 local   ──push→  AuraDB #2（B 的免費帳號）        ├──→ World Agent
用戶 C 的 local   ──push→  AuraDB #3（C 的免費帳號）        │    asyncio.gather()
...                                                          │    並行查詢，合併結果
                    GitHub registry.json ───────────────────┘
                    [{ "name": "Chester", "uri": "neo4j+s://xxx..." },
                     { "name": "UserB",   "uri": "neo4j+s://yyy..." }]
```

#### 加入社群分片的流程

1. 在 [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) 建立免費帳號
2. 在本機 `.env` 填入 AuraDB 連線資訊
3. 執行 `python run_sync_public_kgs.py`，將公開 KG 推送到你的 AuraDB
4. 開 Pull Request，將你的 AuraDB 連線資訊加入 `registry.json`
5. 完成 — 所有人的 World Agent 都能查詢到你的知識

#### 主要設計決策

**1. 同步格式（NDJSON）**

每次同步匯出公開 KG 的所有 SVO triple：

```json
{"subject": "Claude Code", "subject_type": "工具", "rel_type": "USES", "verb": "使用", "object": "Agent", "object_type": "概念", "kg_id": "xxx", "instance_id": "chester"}
{"subject": "強化學習",    "subject_type": "算法", "rel_type": "REQUIRES", "verb": "需要", "object": "大量訓練資料", "object_type": "資源", "kg_id": "yyy", "instance_id": "chester"}
```

**2. 實體命名空間隔離**

不同 instance 的同名實體以 `instance_id` 區隔，避免知識混雜：

```
Chester 的「Claude」(instance: chester) ≠ UserB 的「Claude」(instance: userb)
```

World Agent 預設各自獨立呈現，未來可選擇性合併同義實體。

**3. 離線容錯**

某個 AuraDB 暫停或網路不通 → World Agent 靜默跳過該分片，在回應 metadata 中標記哪些分片未回應，不阻塞整體查詢。

```json
{ "answer": "...", "shards_queried": 3, "shards_offline": ["UserC"] }
```

**4. 同步頻率**

知識不需要實時更新。建議：
- 手動標記 KG 為公開時觸發一次推送
- 或每日定時推送（cron job）

#### 待解決問題

| 問題 | 狀態 |
|------|------|
| AuraDB 免費版閒置暫停 | 已處理：5 秒超時 + 離線標記，不阻塞整體查詢 |
| registry.json 的讀取憑證 | 設計中：`read_token` 欄位已保留，建議只存 read-only token |
| 實體同義詞合併 | ✅ Phase 2d 完成：35 組 zh↔en 同義詞字典 + `expand_terms()` |
| 資料下架機制 | 重新推送空資料覆蓋即可（AuraDB 端操作） |
| 查詢延遲隨分片數增加 | 已處理：asyncio.gather() 並行，10 個分片約 1-3 秒 |

#### 實作計畫

```
Phase 2a：同步協議 ✅
  → run_sync_public_kgs.py：將公開 KG 匯出 NDJSON 並推送到個人 AuraDB
  → 支援全量推送（初期）和 delta 推送（後期優化）

Phase 2b：GitHub Registry ✅
  → registry.json：記錄所有貢獻者的 AuraDB 連線資訊（含 fingerprint_vector）
  → services/federation_service.py：FederationCache 單例，30 分鐘快取
  → World Agent 啟動時背景預取 GitHub registry，不阻塞啟動
  → GET /world/federation/status：分片狀態（online / offline / pending）
  → GET /world/federation/registry：合併本機 + 遠端 registry
  → POST /world/federation/refresh：強制重新下載遠端 registry
  → .env GITHUB_REGISTRY_URL：指向 GitHub Raw URL，留空則只用本機

Phase 2c：並行查詢引擎 ✅
  → services/shard_query.py：ShardResult + query_shards_parallel()
  → asyncio.gather() 同時查詢所有分片（本機 + 遠端 AuraDB）
  → 每個分片超時 5 秒（asyncio.wait_for），超時後靜默跳過並標記離線
  → 遠端 BFS 使用 CONTAINS fallback（不依賴 FTS index）
  → 事實去重：同名實體保留多個 instance 來源前綴
  → /world/chat SSE 新增 shard_meta、shards_queried、shards_offline 欄位
  → 遠端路由：top_concepts 關鍵字比對（本機 KG 仍用 ConceptRepository 向量路由）

Phase 2d：實體對齊 ✅
  → services/entity_alignment.py：同義詞字典（35 組 zh↔en AI/CS 術語）
  → expand_terms()：查詢前自動展開同義詞（e.g. 強化學習 → RL → Reinforcement Learning）
  → AlignedEntity + align_entity_results()：跨 instance 同名/同義詞實體分組
  → shard_query.py：query_shards_parallel() 加 expand_synonyms=True 參數
  → explore/entities 回傳補 instance_id、新增 expand_synonyms 查詢參數
  → GET /world/align/synonyms：查詢術語的同義詞組
  → GET /world/align/entities：跨 instance 實體對齊搜尋（含同義詞展開）
```

---

## 計畫中

### 🔄 Phase 3 — 知識深化（Knowledge Deepening）

Phase 3 聚焦於「讓知識更可信、更可視、更可追蹤、更易共享」，分為四個子階段：

---

#### Phase 3a：知識溯源（Provenance）✅

**目標**：每條 SVO 事實都能追蹤到來源文件、信心分數、建立時間。

```
事實：強化學習(算法) -[需要:REQUIRES]→ 大量訓練資料(資源)
溯源：來源《深度學習導論》，信心分數 3（三份文件均提及），建立於 2026-06-20
```

```
→ models/provenance.py：SourcedFact（事實 + 來源 + 信心 + 時間）、ProvenanceReport
→ services/svo_service.py：query_svo_facts_with_provenance()
   - BFS + 批次 JOIN Document 節點取得標題（_batch_get_doc_titles）
→ services/shard_query.py：ShardResult.sourced_facts、query_shards_parallel() 回傳第 4 值
→ GET /world/provenance/facts：查詢詞 → 完整溯源報告（含 doc_citations）
→ /world/chat SSE 新增 provenance 事件（Top 10 文件引用清單）
→ LLM Prompt：改用 cite_str() 格式「事實 [來源：《文件名》，信心 N]」
   - 指示模型在回答中提及出處
```

---

#### Phase 3b：互動式圖視覺化（Interactive Graph）✅

**目標**：在前端渲染力導向圖，取代純列表的實體探索。

```
→ 前端引入 D3.js v7 force simulation（CDN，無需打包）
→ 節點：Entity（依 type 著色，8 種類型色系）
→ 邊：關係標籤（rel_type / verb）+ 方向箭頭
→ 互動：點擊節點展開鄰居（呼叫 /world/explore/neighbors）
→ 縮放 / 平移（d3.zoom，⊙ 重置按鈕）
→ Hover tooltip（顯示實體類型 + KG 名稱）
→ 節點拖曳（d3.drag，釋放後繼續模擬）
→ 圖形 / 列表 雙視圖切換（≡ 列表 ↔ ◎ 圖形）
→ 圖例（左下角自動顯示當前節點類型）
→ ✕ 清除按鈕（重置圖形狀態）
```

---

#### Phase 3c：KG 版本控制（Version Control）

**目標**：像 Git 一樣追蹤 KG 的每次變更。

Neo4j 邊上已有 `created_at`，Phase 3c 補充：

```
→ 邊新增 updated_at 欄位（MERGE 時更新）
→ GET /kg/{id}/changelog：列出近期新增/修改的 SVO 事實（含時間戳）
→ GET /kg/{id}/diff?since=ISO8601：指定時間點之後的所有變更
→ GET /kg/{id}/snapshot?at=ISO8601：某時間點的知識快照（過濾 created_at）
```

---

#### Phase 3d：KG 訂閱 / 自動同步（Subscription）

**目標**：訂閱別人的公開 KG，定時拉取並合併到本機。

```
→ subscriptions.json：本機訂閱清單（instance_id + kb_id + 上次同步時間）
→ POST /world/subscribe：新增訂閱
→ DELETE /world/subscribe/{kb_id}：取消訂閱
→ GET /world/subscriptions：列出所有訂閱 + 同步狀態
→ POST /world/sync-subscriptions：手動觸發全部訂閱同步
→ 背景排程（APScheduler）：每 N 小時自動同步
```

---

### 💡 未來探索方向（尚未排期）

以下是社群提出或作者思考中的方向，尚未進入正式計畫：

- **使用者帳號系統**：追蹤個人貢獻與知識授權
- **多語言實體對齊**：跨語言知識連結（Phase 2d 的延伸）
- **知識品質評分**：基於社群回饋調整 confidence 分數
- **KG 合併工具**：將兩個本機 KG 智慧合併，衝突時人工仲裁

---

## 參與意願調查

> **Google 表單連結（即將開放）**
>
> 如果你對 Phase 2「世界知識聯邦」有興趣，或想加入早期測試，  
> 請填寫意願調查表單。你的回應將直接影響開發優先順序。
>
> 📋 **[意願調查表單 — 連結待補]**

或直接在 GitHub 開 Issue 留言：[github.com/Chester930/knowledge-base-ai/issues](https://github.com/Chester930/knowledge-base-ai/issues)

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| v0.1 | 2026-06-21 | Phase 1 完成，公開 Phase 2 計畫草案 |
| v0.2 | 2026-06-21 | Phase 2 架構更新：採用免費帳號分片 + GitHub Registry 方案 |
| v0.3 | 2026-06-21 | Phase 2 全部完成（2a 同步協議 / 2b GitHub Registry / 2c 並行查詢 / 2d 實體對齊）|
| v0.4 | 2026-06-21 | Phase 3 規劃：知識溯源 / 圖視覺化 / 版本控制 / 訂閱同步 |
| v0.5 | 2026-06-21 | Phase 3a 知識溯源 + Phase 3b 互動式 D3.js 力導向圖完成 |
