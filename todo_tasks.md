# 智慧知識庫 ── 待辦任務清單 (Architecture & Refactoring TODO List)

本文件根據專案架構健檢（`docs/SYSTEM_HEALTH_AUDIT.md`）與路線圖（`ROADMAP.md`）整理，臚列目前系統的四大核心缺陷、技術債以及未來的學術前沿優化方向。

---

## 🚨 第一階段：高優先級缺陷修復 (High-Priority Gaps - Production-Ready)

這部分任務旨在解決系統從「展示級（Demo）」過渡到「生產級（Production）」的致命缺陷（內存與 CPU 瓶頸）。

### 1. [ ] 解決內存全載與 $O(N \times M)$ 路由比對瓶頸
* **問題模組**：
  * [repositories/concept_repo.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/repositories/concept_repo.py#L83) (`get_all_documents_concepts`)
  * [services/concept_engine.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/services/concept_engine.py) (`compute_match_score`)
* **工作目標**：
  * 重構檢索為 **二階段檢索架構（Two-Stage Retrieval）**。
  * **第一階段（粗篩）**：利用 Neo4j 內建的向量索引 `concept_q_vector` 進行近似最近鄰搜尋（KNN），僅返回相似度最高的 Top-100 概念節點（由 Neo4j 底層 C++ 執行）。
  * **第二階段（精篩）**：僅在 Python 內存中對粗篩出的 100 個候選節點進行複雜的對齊遮罩（Align）與強度振幅（Mag）比對計算。
  * **預期效果**：避免全量加載全庫 ConceptNode，時間複雜度由 $O(N)$ 降為 $O(100)$ 常數級，消除內存爆炸與 CPU 阻塞風險。

### 2. [ ] 重構 BFS 快取，避免記憶體洩漏 (Memory Leak)
* **問題模組**：
  * [services/svo_service.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/services/svo_service.py#L23) (`_bfs_cache` 字典)
* **工作目標**：
  * 將無限制的 Python `dict` 改用 `collections.OrderedDict` 封裝，實現帶有容量限制的 **LRU Cache（近期最少使用淘汰）**。
  * 設定最大快取筆數限制（如 `maxsize=1000`），當快取超出時自動彈出最舊的快取。
  * 確保在記憶體快取過期被覆蓋時，舊數據能被垃圾回收（GC）正確回收。

### 3. [ ] 消除聯邦 Registry 下載對問答串流的同步阻塞
* **問題模組**：
  * [routers/world.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/routers/world.py#L312) 與 [services/federation_service.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/services/federation_service.py#L38)
* **工作目標**：
  * 將遠端 GitHub `registry.json` 的下載同步邏輯，從 `/world/chat` 的請求響應鏈中抽離。
  * 引入 FastAPI Lifespan 的**異步背景定時任務**（使用 APScheduler 或 asyncio task），每 10-30 分鐘自動下載一次遠端 Registry，並更新至內存的 `_github_registry` 快取中。
  * 讓 `/world/chat` 與 `/world/federation/*` 等路由只讀取內存快取，在任何情況下都不在請求線程內發起對外部網路（GitHub）的同步請求。

### 4. [ ] 建立社群版 Neo4j 的屬性複合索引，防止全表掃描
* **問題模組**：
  * [services/svo_service.py](file:///C:/Users/mycena/Desktop/knowledge-base-ai/services/svo_service.py#L1088) (`create_entity_index`)
* **工作目標**：
  * 針對社群版（Community Edition）將所有 KG 混合在同一資料庫以 `kg_id` 區隔的情況，在 `create_entity_index` 初始化時，補上單獨針對 `kg_id` 的屬性索引建置：
    * `CREATE INDEX FOR (e:Entity) ON (e.kg_id)`
    * `CREATE INDEX FOR (c:ConceptNode) ON (c.kg_id)`（若 ConceptNode 帶有 kg_id）
  * 執行 Cypher 查詢性能分析（EXPLAIN / PROFILE），確保 `MATCH (e:Entity {kg_id: $kg_id})` 確實走 Index Scan 而非 NodeByLabelScan。

---

## 📈 第二階段：系統評估與學術指標實施 (Academic Validation & Evaluation)

這部分任務旨在讓系統具備量化實證能力，補齊學術報告所需的數據鏈。

### 5. [ ] 導入 RAGAS / TruLens 自動化評估 Pipeline
* **工作目標**：
  * 新增一個評估指令碼（如 `run_evaluation.py`），整合 **RAGAS** 程式庫。
  * 使用測試問題集，自動評估並記錄系統的：
    * **Faithfulness（忠實度）**：檢測自我精煉迴圈前後答案的幻覺率。
    * **Answer Relevance（答案相關性）**。
    * **Context Recall（檢索召回率）**：特別是 BFS 多跳檢索的召回成效。

### 6. [ ] 撰寫消融實驗測試腳本與統計報告
* **工作目標**：
  * 實作一個基準對比指令碼，能在相同測試集下，切換「純向量檢索」、「純圖譜檢索（SVO）」與「雙層混合檢索（本系統）」。
  * 自動輸出比較表格，以 RAGAS 分數證明本專案「雙層路由與源頭回溯」機制的實質優勢。

---

## 🚀 第三階段：未來前沿功能開發 (Future Extensions)

對應學術報告中的優化展望，適合做為專案進一步深化的方向。

### 7. [ ] 圖拓撲感知共嵌入空間 (GraphSAGE / Node2Vec)
* **工作目標**：
  * 結合圖神經網絡（GNN）演算法，使 ConceptNode 不僅包含文字 embedding，更能融合 Neo4j 圖譜中 SVO 邊的拓撲結構強度。

### 8. [ ] 圖譜鏈式思考推理 (Graph Chain-of-Thought / G-CoT)
* **工作目標**：
  * 將 BFS 擴充為自主 Agent 尋路推理，使 LLM 能在多步推理中沿著關係邊主動發起 Cypher 跳轉查詢。

### 9. [ ] 主動自適應檢索 (Active & Adaptive Retrieval)
* **工作目標**：
  * 在 LLM 串流回答過程中，動態偵測信心度，若缺失邏輯鏈則自發性觸發新圖譜查詢，實現一邊生成一邊檢索的 Self-RAG。
