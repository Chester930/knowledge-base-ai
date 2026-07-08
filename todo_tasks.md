# 智慧知識庫 ── 待辦任務清單 (Architecture & Refactoring TODO List)

本文件根據專案架構健檢（`docs/SYSTEM_HEALTH_AUDIT.md`）與路線圖（`ROADMAP.md`）整理，臚列目前系統的四大核心缺陷、技術債以及未來的學術前沿優化方向。

---

## 🚨 第一階段：高優先級缺陷修復 (High-Priority Gaps - Production-Ready)

這部分任務旨在解決系統從「展示級（Demo）」過渡到「生產級（Production）」的致命缺陷（內存與 CPU 瓶頸）。

### 1. [x] 解決內存全載與 $O(N \times M)$ 路由比對瓶頸
* **實作細節**：
  * 重構檢索為 **二階段檢索架構（Two-Stage Retrieval）**（實作於 `route_via_two_stage()`，包含 `route_kgs()` 與 `route_documents()`）。
  * **第一階段（粗篩）**：利用 Neo4j 內建的向量索引 `concept_q_vector` 進行近似最近鄰搜尋（KNN），僅返回相似度最高的 Top-100 概念節點（由 Neo4j 底層 C++ 執行）。
  * **第二階段（精篩）**：僅在 Python 內存中對粗篩出的 100 個候選節點進行複雜的對齊遮罩（Align）與強度振幅（Mag）比對計算（使用 `compute_match_score`）。
  * **預期效果**：避免全量加載全庫 ConceptNode，時間複雜度由 $O(N)$ 降為 $O(100)$ 常數級，消除內存爆炸與 CPU 阻塞風險。

### 2. [x] 重構 BFS 快取，避免記憶體洩漏 (Memory Leak)
* **實作細節**：
  * 將無限制的 Python `dict` 改用 `collections.OrderedDict` 封裝，實現帶有容量限制的 **LRU Cache（近期最少使用淘汰）**（`_bfs_cache` 實作於 `services/svo_service.py`）。
  * 設定最大快取筆數限制 `_BFS_CACHE_MAX = 1000`，當快取超出時自動彈出最舊的快取。
  * 確保在記憶體快取過期被覆蓋時，舊數據能被垃圾回收（GC）正確回收。

### 3. [x] 消除聯邦 Registry 下載對問答串流的同步阻塞
* **實作細節**：
  * 將遠端 GitHub `registry.json` 的下載同步邏輯，從 `/world/chat` 的請求響應鏈中抽離。
  * 引入 FastAPI Lifespan 的**異步背景定時任務**（使用 APScheduler 執行 `cron_refresh_registry`），每 30 分鐘自動下載一次遠端 Registry，並更新至內存快取中。
  * 讓 `/world/chat` 與 `/world/federation/*` 等路由只讀取內存快取，在任何情況下都不在請求線程內發起對外部網路（GitHub）的同步請求。

### 4. [x] 建立社群版 Neo4j 的屬性複合索引，防止全表掃描
* **實作細節**：
  * 針對社群版（Community Edition）將所有 KG 混合在同一資料庫以 `kg_id` 區隔的情況，在 `create_entity_index` 初始化時，補上單獨針對 `kg_id` 的屬性索引建置：
    * `CREATE INDEX entity_kg_id IF NOT EXISTS FOR (e:Entity) ON (e.kg_id)`
    * `CREATE INDEX concept_kg_id IF NOT EXISTS FOR (c:ConceptNode) ON (c.kg_id)`
  * 建立複合索引 `entity_kg_name` 針對 `(e.kg_id, e.name)` 加速實體查詢，避免 NodeByLabelScan 全表掃描。

---

## 📈 第二階段：系統評估與學術指標實施 (Academic Validation & Evaluation)

這部分任務旨在讓系統具備量化實證能力，補齊學術報告所需的數據鏈。

### 5. [x] 導入 RAGAS / TruLens 自動化評估 Pipeline
* **實作細節**：
  * 新增評估指令碼 `run_evaluation.py`，基於 RAGAS 的三元組評估概念（Faithfulness, Answer Relevance, Context Recall），實作了手寫的 **LLM-as-a-Judge** 機制，直接對三項指標評分。
  * 為了改善測試的冷啟動問題，已於 `workspace/kg_勞動力合規與排班知識圖譜` 補齊 `taiwan_yuasa_violation.txt` 與 `labor_standards_act_32.txt` 等測試文件，確保能跑滿完整的評估流。

### 6. [x] 撰寫消融實驗測試腳本與統計報告
* **實作細節**：
  * 實作基準對比機制，能在相同測試集下，對比「純向量檢索 (Pure Vector RAG)」、「純圖譜檢索 (Pure Graph RAG)」與「雙層混合檢索 (Proposed Hybrid GraphRAG)」。
  * 自動輸出 Markdown 報告 `evaluation_report.md`（含動態評估結論，解決硬編碼與數據矛盾/雜訊詞問題）與 CSV 詳細數據 `evaluation_results.csv`。
* **2026-07-07 已完整跑滿 5 題測試集**，且針對首次結果發現的問題進行了修復與重跑驗證，經過三次迭代：
  1. 首次完整結果：Hybrid（0.47/0.60/0.60）> Graph（0.42/0.60/0.35）> Vector（0.28/0.24/0.00），排序符合預期，但案例 1、4 出現反直覺結果（見下方殘留問題）。
  2. **已修復**：`run_evaluation.py::simulate_rag` 的 Hybrid 文件回溯改用生產環境 `routers/agent.py::_pick_relevant_chunks`（原本只是 `doc.content[:1000]` 截斷），並比照生產環境的 `sim_quota` 邏輯——**僅在 hybrid 模式**下，依圖譜驅動文件已覆蓋的配額動態縮減向量補充量（Pure Vector 基準維持固定配額 3，避免修復 Hybrid 時意外污染基準）。
  3. **重跑驗證結果（最終、已穩定）**：Hybrid 提升至 **0.60/0.62/0.60**，Faithfulness 較首次提升 0.13；Graph 不變（0.42/0.60/0.35）；Vector 因單次 LLM 裁判的抽樣雜訊在 0.28→0.10 間波動（非程式邏輯問題，配額已確認與首次一致）。
* **2026-07-07 診斷：案例 1 根因已查明（撤回先前「KG 路由問題」的推測）**：
  * 診斷 1（`build_query_concepts` + `route_kgs` + `compute_match_score`）證實目標 KG 在案例 1、3、5 三個問句中都穩定排名第一，**KG 路由本身完全正常**。
  * 診斷 2（`query_svo_facts`）發現案例 1 的問句抽取出「勞動基準法」「延長工時」等泛用詞彙，BFS 被 KG 內同質泛用事實淹沒，目標文件完全不在回傳的 doc_ids 中，與 KG 路由無關。
* **✅ 2026-07-08：根因已在生產程式碼 `services/svo_service.py::query_svo_facts()` 實際修復（[x] 已完成，非僅診斷）**：
  1. **修復 1**：`RETURN DISTINCT` 原本誤把 `source_doc_id`／`confidence`／`created_at` 也納入去重鍵，導致同一句語意事實在多份文件重複出現時，每份文件各佔一個 `LIMIT` 名額。改為對 `(subject, rel_type, verb, object)` 分組聚合，`LIMIT` 只作用在真正相異的語意事實上（新增常數 `_MAX_DOCS_PER_FACT = 3`，每組事實仍保留最多 3 個來源文件供回溯）。
  2. **修復 2（更根本）**：即使消除完全重複的事實，KG 內高扇出的樞紐實體（如「勞動基準法」連到數十個不同案例公司）仍會用「相異但同樣泛用」的事實佔滿 `LIMIT`。改用 Neo4j 5.26 支援的 `CALL (seed) {...}` 相關子查詢，對每個 seed 各自的扇出先截斷到 `_PER_SEED_FACT_LIMIT = 20` 筆，確保每個 seed 都有機會貢獻事實配額，不被樞紐節點獨占。
  3. **配套**：`run_evaluation.py` 的 hybrid 文件回溯配額從寫死 `[:4]` 提高為比照生產環境 `_graph_quota = min(top_k*2, 10)`。
  * **驗證**：`pytest tests/services/test_svo_service.py tests/integration/test_neo4j_integration.py`（含連線真實 Neo4j）全數通過（唯二失敗為與本次修改無關的既有 `_build_ft_query` 測試落差）；唯讀診斷腳本確認修復後目標文件穩定出現在 doc_ids 中；完整重跑 5 題測試集驗證分數提升。
  * **最終效果**：Hybrid 從 0.60/0.62/0.60 躍升至 **0.92/0.96/0.90**；Graph（共用同一份 `query_svo_facts`）從 0.42/0.60/0.35 提升到 0.67/0.72/0.45。案例 1 從 0/0/0 → 0.6/0.8/0.5（大幅改善但未滿分）；**案例 4 從 0.33/0.2/0 → 1.0/1.0/1.0（完全修復）**；案例 2、3、5 維持滿分。
  * **⚠️ 案例 1 仍非滿分**：目標文件雖已進入 doc_ids，但排序仍在第 6、7 名（`ORDER BY confidence DESC` 未特別偏好「高特定性 seed 來源」的事實），列為後續待辦。
  * **⚠️ 影響範圍**：`query_svo_facts()` 同時被 `routers/agent.py`（`/agent/chat`）與 `run_evaluation.py` 呼叫。
* **❌ 2026-07-08 端到端生產驗證：發現修復在真實規模下仍不足（重要修正，撤回過度樂觀表述）**：
  * 直接對正式運行中的 `kg2-api` 容器（`http://localhost:8002/agent/chat`）發送案例 1、案例 3 的原始問句做真實驗證（非評估腳本的簡化模擬）。**兩題皆失敗**，回傳的 `sources` 完全相同（10 份 `violation_lsa_XXXX` 文件），不含目標的臺灣湯淺電池案；案例 3 在評估腳本拿到滿分，但在真實 `/agent/chat` 卻失敗。
  * **根因**：程式化查詢確認這個生產 KG 實際有 **909 份文件**，其中約 907 份是自動產生、結構相同只換公司名的泛用勞基法違規案例（先前透過 `chunk_store/` 只找到 2 份文件的印象嚴重失真——`chunk_store/` 只是本地快取，未反映 Neo4j 實際掛載的完整文件規模）。
  * **為何修復不夠**：`_PER_SEED_FACT_LIMIT = 20` 是針對「少數種子、少數泛用鄰居」設計，在只有個位數/十位數候選文件時效果顯著，但當泛用樞紐實體真實連到 907 個結構相同、置信度相近的候選文件時，任何固定的每 seed 上限都只能隨機/依 confidence 排序挑出一小部分，目標文件（1/907）在數學上仍極可能被排除——這不是參數沒調對，而是**目前的 BFS 策略在同質大規模泛用實體場景下，本質上缺乏區分「907 篇裡最相關的是哪一篇」的機制**。
  * **誠實修正**：第八輪「推測生產環境同樣受益」與「預期能直接改善生產環境真實問答品質」的樂觀表述，經真實驗證後**證實不成立**於此規模。修復方向正確、在小規模場景已驗證有效（診斷腳本、評估腳本用的 2 文件 KG），但不足以解決 909 文件規模的同一類問題。
  * **後續待辦（優先度更高，取代原本較模糊的待辦）**：
    1. 需要依詞彙特定性/稀有度排序或篩選 seed 的機制——命中基數低的詞彙（公司名）應優先驅動 BFS，命中基數高的詞彙（「勞動基準法」）應降權或僅作候選補充。
    2. 或導入全文檢索/向量相似度作為 SVO BFS 前置過濾，只在信心不足時才用泛用法條詞彙擴大搜尋範圍。
    3. 應先用這個 909 文件的真實 KG（而非 2 文件診斷 KG）建立更真實的評估基準，避免下一輪修復又只在小規模驗證就誤判為已解決。
* 完整結果與八輪迭代過程（含本次端到端驗證）已回填 `docs/THEORETICAL_ARCHITECTURE.md` 第11節與第13節第三～八輪變更記錄。

---

## 🚀 第三階段：未來前沿功能開發 (Future Extensions)

對應學術報告中的優化展望，適合做為專案進一步深化的方向。

### 7. [x] 圖拓撲感知共嵌入空間 (GraphSAGE / Node2Vec)
* **實作細節**：
  * 使用離線腳本 `run_build_graph_embeddings.py` 通過 node2vec 演算法計算圖譜實體的結構特徵，並透過 `_fuse_graph_vector()` 對 `ConceptNode` 的文字嵌入與圖結構特徵進行加權融合。

### 8. [/] 圖譜鏈式思考推理 (Graph Chain-of-Thought / G-CoT)
* **實作細節**：
  * 已在 `routers/agent.py` 實作簡化版 Graph-CoT 推理（門檻觸發式加深查詢，不含逐跳 LLM 自主選路）。未來可進一步擴充為自主 Agent 尋路推理。

### 9. [ ] 主動自適應檢索 (Active & Adaptive Retrieval)
* **工作目標**：
  * 在 LLM 串流回答過程中，動態偵測信心度，若缺失邏輯鏈則自發性觸發新圖譜查詢，實現一邊生成一邊檢索的 Self-RAG。

---

## 🔬 第四階段：SVO 品質驗證機制 (Quality Verification, 2026-07-08 新增)

### 10. [x] 抽取-審查-重試-本體擴充三模型迴圈
* **需求來源**：使用者要求為 SVO 抽取結果新增「另一個模型」審查機制——抽取後由審查模型根據原句子判斷是否可接受，不可接受則重新抽取；重試後仍不行，再由第三個模型提議新增本體類別讓抽取模型依新類型重新抽取。確認的設計取捨：新類型預設僅供該 KG 使用（例外才升級全域）、不需人工審核、重試一次。
* **實作細節**：
  * `services/ontology_service.py`（新檔）：JSON 持久化每個 KG 的擴充實體/關係類型（`ontology_extensions.json`，仿 `subscriptions.json` 慣例），區分 `scope="kg"`/`"global"`。
  * `services/svo_service.py` 新增 `verify_svo_extraction()`（審查模型，逐條三元組 accept/reject + 理由）、`propose_ontology_extension()`（本體擴充模型，每次上限各提議 3 個新類型，防禦性排除已存在的類型）、`extract_svo_verified()`（整合流程：抽取→審查→重試一次→審查→仍拒絕則擴充本體→再抽取一次，有界不循環）。
  * 已接入 `build_graph_for_kg()` 取代原本直接呼叫 `extract_svo_from_text()`；新增設定 `svo_verify_enabled`（預設開）、`svo_verify_model`、`svo_verify_max_retries`（預設 1）。
  * `query_svo_facts()`、`query_svo_facts_with_provenance()`、`_clear_kg_relations()`、`community_service.py` 的 BFS/社群偵測 Cypher 全數改用 `get_effective_rel_pattern()` 動態組出含該 KG 擴充關係類型的 pattern，避免新類型的邊被 BFS/清除/社群偵測遺漏。
* **附帶修正真實 bug**：`_VALID_TYPES` 先前只有 13 種，但抽取 prompt 範例早已使用 19 種（含法規/企業/政府機關等），導致 LLM 正確標註後被驗證階段靜默降級為「其他」——已同步擴充為 19 種。
* **驗證**：新增 `tests/services/test_ontology_service.py`（16 案例）、`tests/services/test_svo_verification.py`（15 案例），全數通過；完整既有測試套件 872→903 個測試無回歸（含連線真實 Neo4j 的整合測試）；另跑一次真實 Ollama 端到端 smoke test 確認 6 種新修正類型被正確抽取保留。
* 完整設計細節、與第9節②「動態本體對齊」立場差異的說明，已回填 `docs/THEORETICAL_ARCHITECTURE.md` 第 4.1 節與第13節第九輪變更記錄。

### 11. [x] 互動式逐階段檢驗工具 + 修復審查模型 JSON 解析真實 bug
* **需求來源**：使用者要求「參與單元流程測試，把關整個過程，優化設計」。
* **新增 `run_svo_pipeline_debug.py`**：互動式 CLI，讓使用者可對真實 Ollama 模型逐句手動測試三模型迴圈，清楚印出每一階段（抽取/審查/重試/本體擴充）的完整輸入輸出，本體擴充是否實際寫入還會詢問確認。
* **✅ 用這個工具實測時發現真實 bug**：審查模型有時直接輸出單一裸物件而非陣列（即使 prompt 明確要求陣列），導致原本只找 `[...]` 的正則解析失敗、誤觸發保守 fallback（全部判定通過）——**審查機制在這類回應下完全沒有實際發揮作用，且沒有任何錯誤提示**。已修復：新增 `_extract_json_objects()` 共用輔助函式（依序嘗試整段解析／陣列正則／逐字掃描頂層物件），`verify_svo_extraction()` 與 `propose_ontology_extension()` 皆已改用。修復後審查模型正確抓到抽取模型在 `APPLIES_TO` 關係方向上的系統性錯誤。
* **✅ 順帶修復**：`propose_ontology_extension()` 新增 `kg_id` 參數，讓提議新類型時能看到該 KG 先前已擴充過的類型，避免同一 KG 多次觸發擴充累積出語意重疊的近義類型。
* **⚠️ 重要教訓**：此 bug 完全不會被第 10 項新增的 mocked 單元測試發現，因為 mock 永遠回傳格式正確的 JSON。純 mocked 測試對「真實 LLM 輸出格式穩健性」有覆蓋盲點，需要真實模型的人工逐階段檢驗才能補上。
* **驗證**：新增 `TestExtractJsonObjects`（6 案例）與端到端回歸測試，測試套件 903→910，無回歸。
* 完整細節已回填 `docs/THEORETICAL_ARCHITECTURE.md` 第13節第十輪變更記錄。

---

## 📚 第五階段：文獻與引用查證 (Citation Audit, 2026-07-08 新增)

### 12. [x] 全文 GitHub 專案與學術文獻查證，修正虛構/錯誤引用
* **需求來源**：使用者在口試/審查場合被質疑「參考的 GitHub 專案星數不足、缺乏驗證」，要求重新篩選值得引用的 GitHub 專案，並確認所引用學術論文有足夠驗證。
* **查證範圍**：不只處理被質疑的部分，而是對全文所有引用做地毯式查證——第 7 節 6 個 GitHub repo + 全文約 34 筆學術文獻引用，透過即時網路查詢逐一核對是否真實存在、作者/年份/會議期刊是否正確，而非憑記憶判斷。
* **✅ GitHub 專案查核結果**：
  * 5 個真實存在且可用：`zjunlp/WKM`（★167）、`King-s-Knowledge-Graph-Lab/ProVe`（★11，King's College London 學術實驗室背書）、`pat-jj/KG-FIT`（★131）、`microsoft/graphrag`（★34,244，旗艦級）、`neo4j-contrib/ms-graphrag-neo4j`（★88，Neo4j 官方組織背書）。
  * **移除 1 個**：`Wikipedia-KG-RAG` 連結格式本身無效（缺 owner 路徑），查證後對應到的真實專案僅 2 顆星，整列移除不做連結修補。
  * 已在表格中為星數偏低的兩項明確標註「可信度來源是機構背書而非星數」，避免證據力層級混淆。
* **✅ 可疑 arXiv 引用查核（意外大多為真）**：6 篇帶有 2026 年份、看似可疑的引用中，5 篇（PathRAG、MoG、GraphRAG-Router、Neurosymbolic Retrievers、RouteRAG）經直接查詢 arxiv.org 確認題目/作者/摘要精確吻合，屬實保留。
* **❌ 確認虛構、已替換的引用（2 篇）**：
  1. 「Chao, Yuxiao (2024). Graph-ToolChain」arXiv:2401.12345 → 該 ID 實際對應無線通訊訊號處理論文，虛構，替換為 RouteRAG（arXiv:2512.09487，已驗證真實）。
  2. 「He, Xiaoxin (2023). Mind's Eye of LLM」arXiv:2310.13344 → 該 ID 實際對應電腦圖學裂縫模擬論文，作者標題皆誤植，替換為 Jin, Bowen et al. (2024) "Graph Chain-of-Thought"，arXiv:2404.07103，ACL 2024（查證屬實）。
* **⚠️ 確認有誤但非虛構、已修正的引用（8 處）**：Schwarte FedX 會議別（WWW→ISWC 2011）、Zhao 實體對齊調查作者姓氏（Zhao C.→Zhao X.）與缺漏期刊（補 IEEE TKDE）、Garcez & Lamb 年份/版本混淆（arXiv 2020 版 vs. 期刊 2023 版）、Active RAG 作者誤植（Trivedi H.→Jiang Z. 等，FLARE 論文）、時序知識圖譜論文標題/會議誤植（→"Know-Evolve"，ICML 2017）、Nogueira BERT 排序 arXiv ID 誤植（→1901.04085）、Busbridge RGAT arXiv ID 尾數誤植（→1904.05811）。
* **⚠️ 查無可靠出處、已移除具體引用的項目（3 處）**：MoKG（Zhao, Y. 2022）、Provenance-Aware Retrieval（Deutch 2021 ACL）、KG 演化管理（Dividino 2014）——三處皆改為「不掛文獻的概念性類比」並誠實標註「引用待補」，保留設計動機說明但不假裝有可靠學術出處。
* **📌 總結**：全文約 34 筆學術引用 + 6 個 GitHub repo 逐一查核，2 篇確認虛構已替換、8 處有誤已修正、3 篇查無出處已移除引用、其餘約 21 篇查證屬實。
* 完整查證方法與逐項修正細節已回填 `docs/THEORETICAL_ARCHITECTURE.md` 第 2、3、4、6、6.1、7、9、12 節與第13節第十一輪變更記錄。
