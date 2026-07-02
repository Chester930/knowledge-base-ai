# 智慧知識庫 / World Knowledge Hub 專案架構健檢報告
# (System Architecture & Health Audit Report)

本報告針對**「智慧知識庫 / World Knowledge Hub」**的專案代碼進行深度架構審查（Code & Architecture Audit），評估其「已做到的先進特性」、「未做到的設計缺陷」以及「潛在效能瓶頸」，並給出具體的重構與優化建議。

---

## 1. 健檢等級評定 (Overall Health Grading)

### 評級：`A-` ( SOTA 架構設計 / 演示級性能優化 )

* **評語**：本專案在**「神經符號 AI」**的理論落地、**「圖譜混合 RAG」** 的功能實作、以及**「建圖容錯機制（Fallback）」** 上，達到了極高的學術水準與演示體驗。然而，其資料庫讀寫與內存概念比對的邏輯，暴露出其仍屬於「開發/展示（Demo）」級別，在大規模（Scale-up）與高併發（High-concurrency）的生產環境下，存在著嚴重的記憶體與 CPU 瓶頸。

---

## 2. 專案已做到的先進特性 (What's Done Well)

本系統在 RAG 架構上做到了許多優於一般市面 RAG 專案的亮點：

### ① 圖譜專家門控路由 (Graph-level Gating MoE)
* **實現**：利用問題概念向量（Query Concept）對各個公開 KG（Key）進行對齊比對（Match Score），大於閾值才激活該圖譜。
* **學術價值**：有效將「單點海量知識檢索」降維為「局部圖譜專家檢索」，成功將 **Mixture of Experts (MoE)** 理論落地於圖譜 RAG。

### ② 符號-物理源頭回溯 (Symbolic-to-Physical Source Backtracking)
* **實現**：當 SVO 事實被檢索但資訊太精簡時，系統不拋棄 SVO，而是沿著節點綁定的 `chunk_id` 座標，回溯拉取 `ChunkStore` 中對應的原始文件段落。
* **學術價值**：打通了「離散符號邊」與「連續上下文細節」，解決了傳統 KG RAG 丟失脈絡的缺陷。

### ③ 句子感知與向量批量運算 (Sentence-Aware & Batch Embedding)
* **實現**：切分 Chunk 時利用句子邊界，保證語意不被硬性物理長度切斷。在重排階段使用 `encode_batch` 一次計算（比逐個 encode 快 10-50 倍）。

### ④ 大規模建圖的 Dynamic Fallback 降級容錯
* **實現**：在 `svo_service.py` 中，若高能力模型（如 `phi4`）發生 API 限流或超時，系統會原地重試，並在最後一次重試時**動態降級使用本地已有的輕量模型（如 `qwen/llama`）**，確保大規模建圖不因單點 API 失敗而中斷。

---

## 3. 架構設計缺陷與潛在效能瓶頸 (Gaps & Flaws)

在深入閱讀 `concept_repo.py`、`svo_service.py` 與 `agent.py` 後，審查出以下四個核心缺陷：

### 🚨 缺陷一：內存全量加載與 O(N*M) 比對瓶頸 (Memory & CPU Bottleneck)
* **問題檔案**：`repositories/concept_repo.py` 中的 [get_all_documents_concepts](file:///c:/Users/666/Desktop/智慧知識庫/repositories/concept_repo.py#L83-L102)、`services/concept_engine.py` 中的 [compute_match_score](file:///c:/Users/666/Desktop/智慧知識庫/services/concept_engine.py#L102-L137)。
* **分析**：
  在 `world/chat` 路由中，系統調用 `get_all_documents_concepts()`，這會執行 `MATCH (d:Document)-[e:EFFECTIVE]->(c:ConceptNode)` 將**全資料庫所有文件的概念和特徵向量一次性拉進內存**。隨後，系統在 Python 中使用**雙重迴圈**對這些向量逐一進行 Cosine 相似度與對齊度計算。
* **風險**：
  * **內存爆炸**：當文件數大於 1,000 份、Concept 節點大於 10,000 個時，Python `dict` 將吃光伺服器內存。
  * **CPU 阻塞**：Python 對大矩陣做 $O(N \times M)$ 雙重迴圈餘弦運算效能極低，線上問答可能卡死數秒至數十秒。
* **改進建議**：
  應利用 Neo4j 內建的 **Vector Index** 進行向量粗篩，只返回最相關的 Top-K 個概念節點，將計算交給資料庫（C++ 底層運算），在 Python 中僅做精篩。

### 🚨 缺陷二：無容量上限的內存 dict 快取 (Memory Leak Risk)
* **問題檔案**：`services/svo_service.py` 中的 `_bfs_cache` 字典 ([svo_service.py:L23](file:///c:/Users/666/Desktop/智慧知識庫/services/svo_service.py#L23))。
* **分析**：
  `_bfs_cache` 是一個標準的 Python `dict`。雖然系統在寫入時會判斷 `TTL = 300s` 是否過期，但**沒有設定快取的容量上限**，且沒有背景定時清理線程（GC）。
* **風險**：
  若系統在線上運行，隨着併發請求增加，這個 `dict` 將持續增長。即便快取過期，舊的 Key-Value 依然殘留在 `dict` 中，導致**嚴重的記憶體洩漏 (Memory Leak)**。
* **改進建議**：
  使用 `collections.OrderedDict` 實作 **LRU Cache（近期最少使用淘汰）**，並在寫入時限制最大 Capacity（例如最多 1000 條），多餘的自動彈出。

### 🚨 缺陷三：聯邦 Registry 同步阻塞 GitHub 故障
* **問題檔案**：`routers/world.py` 中的 [world_chat](file:///c:/Users/666/Desktop/智慧知識庫/routers/world.py#L255-L486)。
* **分析**：
  問答接口 `/world/chat` 是直接在 Streaming 執行線程中，調用 `await get_federation_cache().merged_registry()` 去下載 GitHub 遠端的 registry 進行聯邦查詢。
* **風險**：
  一旦 GitHub 網絡連接緩慢或 DNS 解析失敗，該接口會直接卡在 HTTP 請求上，造成使用者問答請求超時掛起（Timeout）。
* **改進建議**：
  應將遠端 Registry 的同步交給 **FastAPI Lifespan 背景定時任務（Background Scheduler）** 每 10 分鐘下載一次，將結果快取在內存。路由層 `/world/chat` 只讀取快取，絕對不應同步發起外部網路請求。

### 🚨 缺陷四：社群版 Neo4j 大資料量下的全表掃描
* **問題檔案**：`repositories/concept_repo.py` 以及建圖查詢。
* **分析**：
  在 Community 社群版 Neo4j 下，系統使用 `kg_id` 標籤屬性來區隔不同的獨立知識圖譜。但在目前的 Cypher 語句中，有大量的 `MATCH (e:Entity {kg_id: $kg_id})` 查詢。
* **風險**：
  如果沒有對 `(:Entity {kg_id})` 與 `(:ConceptNode {kg_id})` 建立複合索引（Composite Index），當數據量增大時，每次查詢都會引發 **Full Table Scan（全表掃描）**，資料庫 CPU 將直接飆到 100%。
* **改進建議**：
  必須在資料庫初始化 lifespan 中，執行 `CREATE INDEX` 為 `kg_id` 建立屬性索引。

---

## 4. 架構重構優化路線圖 (Refactoring Roadmap)

為了將此專案從「Demo 級別」提升至「Production-Ready（生產級別）」，建議進行以下三步重構：

```
【第一步：資料庫索引與計算下移】
  利用 Neo4j Vector Index 做 KNN 粗篩
  利用 Cypher 的 gds 或 vector 函數計算 cosine score 
                   │
                   ▼
【第二步：安全快取與背景任務】
  將 _bfs_cache 改為 OrderedDict (LRU, max_size=1000)
  GitHub Registry 同步交由背景 Scheduler 異步執行，避免阻塞 chat API
                   │
                   ▼
【第三步：社群版 / 企業版 適應最佳化】
  Lifespan 中強制執行 CREATE INDEX FOR (e:Entity) ON (e.kg_id)
  引入 Redis 作為 ConceptNode 的分散式緩存
```
