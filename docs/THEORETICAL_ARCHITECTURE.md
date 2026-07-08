# 神經符號 GraphRAG 系統：基於動態注意力機制與聯邦知識分片的學術與理論架構

# (Neuro-Symbolic GraphRAG System: Academic & Theoretical Architecture)

本文件整理了**「智慧知識庫 / World Knowledge Hub」**的核心技術概念，詳細討論**「圖譜外查找機制（概念路由）」**與**「圖譜內查找機制（混合檢索與雙向回溯機制）」**，並將其映射至現代深度學習中的 **Attention (Q, K, V) 機制**、**神經符號 AI (Neuro-Symbolic AI)**、**混合專家圖譜路由 (Mixture of KGs / Graph MoE)** 以及**動態記憶網絡 (Dynamic Memory Networks)** 等學術理論，為本專案提供完備且嚴謹的學術與理論背書。

---

## 1. 系統架構總覽 (System Architecture Overview)

本系統是一個**混合型 RAG (Hybrid RAG) 系統**，旨在橋接**連續語意空間（高維向量）**與**離散知識空間（圖譜拓撲與本體）**。其核心架構在學術上可拆解為兩個運行層次：

1. **圖譜外查找機制（Outer-Graph Gating & Routing）**：基於 **混合專家模型 (Mixture of Experts, MoE)** 的概念，將多個獨立的知識圖譜 (KG) 視為獨立的專家節點，利用 QKV 門控路由注意力，動態篩選並激活相關知識庫。
2. **圖譜內查找機制（Inner-Graph Retrieval & Provenance Backtracking）**：在被激活的圖譜內部，結合 **符號級圖遍歷 (Symbolic Traversal)** 與 **圖譜引導的文本混合重排 (Graph-Driven Reranking)**。若 SVO 知識事實不足以答覆，系統將啟動 **符號-物理回溯機制 (Symbolic-to-Physical Backtracking)**，沿著圖譜節點的物理座標回溯拉取對應的**原始文件段落 (Chunks)**。

```
                               連續向量空間 (Continuous Vector Space)
                              ┌──────────────────────────────────────┐
   Query Token (Xq) ─────────►│  Q = Xq · Wq (Concept Extraction)    │
                              └──────────────────┬───────────────────┘
                                                 │
                                                 ▼ [Attention Weights (α)]
                                 圖譜門控路由器 (Graph Gating Router)
                                  計算 Q 與每個 KG Key (K_kg) 的對齊分
                                                 ▲
                              ┌──────────────────┴───────────────────┐
   Graph Schema (Xk) ────────►│  K_kg = KG Concepts (public_concepts)│
                              └──────────────────────────────────────┘
                                  圖譜專家層 (Mixture of KG Experts)
─────────────────────────────────────────────────────────────────────────────
                                 離散符號空間 (Discrete Symbolic Space)
                              ┌──────────────────────────────────────┐
                              │  V = 激活圖譜內的 Entities & Facts  │
                              └──────────────────┬───────────────────┘
                                                 │
                                                 ▼ [Memory Retrieval (A · V)]
                                      圖譜內混合檢索與回溯
                                ├── 1. BFS 1-2跳拓撲遍歷 (Cypher)
                                ├── 2. 符號-物理回溯 (Symbolic-to-Physical)
                                │      沿節點物理座標回溯拉取原始 Chunk 文本
                                └── 3. 圖譜引導的 Chunk 重排 (Reranking)
                                                 │
                                                 ▼
                                     RAG Prompt / Context 輸出
```

---

## 2. 圖譜外查找機制：圖譜專家門控與 QKV 注意力映射

### (Outer-Graph Gating: Mixture of KG Experts & Routing Attention)

#### 【技術機制】

本系統將**「每個獨立的知識圖譜（KG）當作一個獨立專家節點（Node / Expert）」**。QKV 注意力機制的作用是在「路由層」，動態計算問題對各個圖譜專家的權重分配：

* **Query ($Q$)**：輸入問題經概念提取後，投影為特徵矩陣 $Q \in \mathbb{R}^{M \times d}$。除了高維語意 Embedding 外，亦包含 $[interest, professional]$ 雙維度屬性權重。
* **Key ($K$)**：每個獨立圖譜（專家節點）所擁有的概念集合特徵（即 `public_kg_concepts`），代表該圖譜的主題語意特徵 $K_{\text{kg}} \in \mathbb{R}^{N \times d}$。
* **Value ($V$)**：被選擇/激活的圖譜內部的具體離散 SVO（Subject-Verb-Object）事實與文件片段（Chunks）。

```mermaid
flowchart TD
    Query[Query Concepts Q] --> ScoreCalc[門控注意力計算 Score_kg]
    Keys[KG Concepts K_kg] --> ScoreCalc
    ScoreCalc -->|Score_kg >= Threshold| Active[激活 Top-K 專家圖譜]
    Active -->|並行檢索| Shards[Federated Shard Query]
    Shards --> FusedVal[合併輸出 Value V]
```

#### 【數學公式與門控路由】

本系統在 [services/concept_engine.py](file:///D:/Users/666/Desktop/智慧知識庫2/services/concept_engine.py) 的 `compute_match_score()` 中，將其實作化為**多屬性圖譜門控注意力（Multi-Attribute Gating Attention）**：

1. **語意餘弦相似度**：
   $$\text{Cos}_{i,j} = \text{Cosine}(Q_{vector, i}, K_{vector, j})$$
2. **屬性對齊分數**：
   $$\text{Align}_{i,j} = 1.0 - \frac{|\Delta interest_{i,j}| + |\Delta professional_{i,j}|}{2}$$
3. **特徵強度振幅**：
   $$\text{Mag}_{i,j} = \frac{interest_{Q,i} + professional_{Q,i} + interest_{K,j} + professional_{K,j}}{4}$$
4. **綜合門控注意力權重 ($\alpha$)**：
   $$\alpha_{i,j} = \text{Cos}_{i,j} \times \text{Align}_{i,j} \times \text{Mag}_{i,j}$$
5. **最終圖譜路由得分**：
   $$\text{Score}_{\text{kg}} = \frac{\sum_{i,j} \alpha_{i,j}}{\sum_{i,j} \text{Mag}_{i,j}}$$

系統設定了 `KG_ROUTE_THRESHOLD`（預設為 0.05）作為篩選門檻。

#### 【多專家激活與跨域語意融合 (Top-K Multi-Expert Activation)】

為了解決現實世界中**跨領域 (Cross-domain) 查詢**的問題（例如，問題涉及「醫學」與「資訊工程」的交集），系統**不限制只激活單一圖譜**，而是實作了 **Top-K 門控路由機制**（在代碼中以 `MAX_KG_PER_QUERY` 進行約束，**實際預設為 5**，定義於 `core/constants.py`，篩選邏輯見 `routers/agent.py`：先以 `score >= KG_ROUTE_THRESHOLD` 篩選，再取 `kg_scores[:MAX_KG_PER_QUERY]`）：
$$\text{Activated\_KGs} = \text{Top-K}\Big( \big\{ \text{KG}_i \;\big|\; Score_{\text{kg}, i} \ge \text{Threshold} \big\} \Big)$$
這對應於 **Top-K Sparsely-Gated MoE** 結構：

1. **並行激活與聯邦檢索**：當多個專家圖譜被同時激活時（$\text{Gate}_i = 1$），系統在 `routers/agent.py` 的 `_bfs_kg()` 搭配 `asyncio.gather()` 對所有 `selected_kgs` 並行執行 BFS 遍歷，跨分片場景則由 `services/shard_query.py` 的 `query_shards_parallel()` 承接，獲取各自的離散 SVO Facts 與對應原文的物理座標。
2. **跨域值融合 (Cross-Domain Value Fusion)**：將各個專家圖譜回傳的局部 Value 進行語意拼接與交叉融合：
   $$\text{Fused\_Context} = \bigoplus_{i \in \text{Activated}} V_i$$
   這讓最終的 LLM 能綜觀多個學科或專門領域的知識，進行**跨域語意聯邦推理（Cross-Domain Federated Reasoning）**。

#### 【學術文獻背書與經典論文】

* **Sparsely-Gated MoE**：
  * *文獻*：*Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."* arXiv:1701.06538.
  * *理論連結*：證明了當 $K > 1$（如 Top-2/Top-3 Gating）時，MoE 能夠同時激活多個不同的網絡專家，並對其輸出進行加權融合。本系統的 `MAX_KG_PER_QUERY` 圖外路由即是該機制的典型實踐。
* **多局部專家圖譜的分治推理**：
  * *理論連結*（概念性類比，不掛特定文獻）：在大規模圖推理中，採用多個局部專家圖譜進行分而治之，直覺上比單一扁平圖譜具備更好的推理效率與擴展性，此為本系統路由設計的動機之一。

---

## 3. 圖譜內查找機制：符號遍歷、物理段落回溯與實體重排

### (Inner-Graph Retrieval: Symbolic Traversal, Source Backtracking & Reranking)

當圖譜路由器激活了特定的圖譜專家後，系統進入**圖譜內部的雙軌檢索與回溯階段**：

```mermaid
sequence diagram
    autonumber
    Participant Query as 用戶問題
    Participant Graph as Neo4j 圖譜層 (A-Box)
    Participant Store as ChunkStore 物理層
    Participant LLM as 大語言模型 (Generator)
    
    Query->>Graph: 1. BFS 1-2跳符號遍歷 (Cypher)
    Graph-->>Query: 2. 獲取 SVO Facts 與對應 chunk_id 座標
    Note over Query, Graph: 若關係過於簡煉，缺失細節...
    Query->>Store: 3. 符號-物理源頭回溯 (拉取原始 Chunks)
    Store-->>Query: 4. 返回精確原文段落 (Context)
    Query->>LLM: 5. 融合成 RAG Prompt (Facts + Chunks)
    LLM-->>Query: 6. 輸出可信且具備溯源依據的答案
```

#### 【第一軌：符號級圖譜遍歷（Symbolic Graph Traversal）】

1. **BFS 拓撲檢索**：系統利用問題中提取的概念，在 Neo4j 中執行 **1-2 跳的 BFS 遍歷**。
2. **語意事實與轉譯**：獲取離散 `SVO Facts`，並透過 `_svo_to_sentences` 翻譯為自然中文句子。
3. **定位物理座標**：保留每個 Fact 指向的 `chunk_id`、`source_doc_id`。

#### 【核心機制：符號-物理源頭回溯（Symbolic-to-Physical Source Backtracking）】

當 SVO 離散關係雖然被檢索命中，但因為三元組過於抽象、精簡而**丟失了原文中的副詞、時序、數量或情境細節**，導致 LLM 無法完美作答時，系統會啟動回溯：

* **座標對照**：系統沿著被激活的 SVO 節點中儲存的 `chunk_id` 與 `source_doc_id` 物理座標，直接向 `ChunkStore` 持久化數據庫發送請求。
* **物理原文拉取**：將產生這些 SVO 節點的**原始文件段落（Chunk 原文）**回溯提取出來（例如包含「西元 701 年，李白出生於碎葉城，其家族在此經商...」的完整段落）。
* **學術價值**：這解決了傳統 Knowledge Graph 缺乏上下文情境（Context-free）的重大缺陷。本系統透過 **「符號-物理對照映射（Symbolic-to-Physical Mapping）」**，讓 LLM 同時擁有離散的「邏輯關係邊（SVO）」與連續的「原文細節（Chunk）」，大幅提升回答的細節度與可信度。
* **實際觸發條件（`routers/agent.py` 精煉迴圈，約 L658-715）**：並非對每次查詢都無條件回溯，而是有明確的觸發閥門——僅當 `req.use_svo` 開啟且 BFS 已取得 `svo_chunk_ids` 時才進入精煉迴圈；迴圈內先由 LLM 對初次生成的答案自報信心分數，只有 `confidence < _CONFIDENCE_THRESHOLD` 時才會呼叫 `chunk_store.read_ranked(remaining_ids, q_vec)` 補拉原文。觸發依據是**生成端的信心分數**，而非檢索端對 SVO 資訊量的判斷。

#### 【第二軌：圖譜引導的文本重排（Graph-Driven Reranking）】

以上述圖譜抽出的實體作為「引導信號」，在 `routers/agent.py` 的 `_pick_relevant_chunks`（定義於約 L79）中對物理 Chunk 進行重新排序，計算公式為：
$$\text{Score} = \text{Cosine}_{\text{max}} + \text{Query\_Hits} \times 0.4 + \mathbf{SVO\_Hits \times 0.10} + \text{Enum\_Bonus}$$

其中 `Enum_Bonus = 0.25`（約 L166）。這實作了**圖譜符號知識對向量相似度空間的偏置與引導**，優先提取與圖譜事實密切相關的原始文本。

* **實作細節**：實際排序並非單純依 `Score` 由大到小排，而是**兩階段排序**（約 L169-175）——先比較 `Query_Hits` 命中數，命中數相同時才比較綜合 `Score`。這代表系統對「關鍵詞直接命中」的信任權重高於「向量+圖譜綜合分數」，屬於刻意的保守排序策略。

#### 【學術文獻背書與經典論文】

* **GraphRAG 架構**：
  * *文獻*：*Edge, D., et al. (Microsoft, 2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization."* arXiv:2404.16130.
  * *理論連結*：微軟論文中強調了將非結構化文本轉為知識圖譜（Community Summary），再對應回原始文本的檢索優勢。本系統的「符號-物理源頭回溯」正是微軟 GraphRAG 中「圖譜實體簡化為原始文本」概念的工程化具體實作。
* **保留物理來源以支持可解釋性**：
  * *理論連結*（概念性類比，不掛特定文獻）：在語意檢索中，保留離散知識的物理來源（Provenance / Coordinates）對於資訊可解釋性（Explainability）與事實真實性有直覺上的重要性，這是本系統「符號-物理回溯」設計的動機。第 7 節的 ProVe 專案（King's Knowledge Graph Lab）在事實驗證的精神上與此概念相關，可作為參照。

---

## 4. 本體論與知識圖譜的雙層協同

### (Ontology & Knowledge Graph: T-Box / A-Box Co-Reasoning)

本體論（Ontology）是知識圖譜的「語意骨架」，定義了數據的關係約束與邏輯結構。本專案將本體論與知識圖譜進行了學術上的 **T-Box（術語域）與 A-Box（實例域）雙層協同推理設計**：

#### 【本體架構與學術定義】

1. **T-Box (Terminology Box / 概念與本體模式層)**：
   * **Concept Classes (實體類別)**：定義了 19 種核心實體類型（`概念`、`算法`、`技術`、`方法`、`工具`、`框架`、`模型`、`系統`、`人物`、`組織`、`資料集`、`指標`、`其他`，加上為法規遵循類場景擴充的 `法規`、`假期`、`企業`、`政府機關`、`限制數值`、`行為`）。（`_VALID_TYPES` 與 `_TYPE_LABEL_MAP` 已同步為 19 種，與抽取 prompt 範例一致，避免 LLM 正確標註「企業」「政府機關」等類型後，於驗證階段被靜默降級為「其他」。）
   * **Relation Categories (本體關係類別)**：預定義了 31 種強類型的語意關係（如 `IS_A` 代表層級歸屬，`CAUSES` 代表因果效應，`USES` 代表功能操作，`VIOLATES` 代表法規/規範違反等）。這解決了傳統 OpenIE（開放式資訊抽取）中關係詞發散、無法進行邏輯歸納與推理的弊端。（`VIOLATES` 為配合法規遵循類測試場景新增的分類，歸入「規範/合規」子類。）
2. **A-Box (Assertion Box / 實例語意層)**：
   * 具體文檔中抽取出的實例化三元組（如 `[Transformer] -[:USES]-> [多頭注意力]`）。

#### 【雙層協同工作機制】

* **本體路由層（T-Box）**：`ConceptNode` 儲存了高維連續向量與屬性權重，作為 RAG 路由的**注意力分發器**。它快速匹配問題的核心概念與對應的本體 Schema。
* **實例推理層（A-Box）**：路由確定後，系統在 Neo4j 圖資料庫中進行 BFS 圖路徑遍歷，將抽象的本體關係轉化為精確的文件事實與上下文，實作了「概念導向，事實落地」的協同推理。

#### 【學術文獻背書與經典論文】

* **Description Logics (描述邏輯與語意網)**：
  * *文獻*：*Baader, F., et al. (2003). "The Description Logic Handbook: Theory, Implementation, and Applications."* Cambridge University Press.
  * *理論連結*：奠定了本體論中 T-Box 與 A-Box 劃分的基石。本系統將概念路由映射為 T-Box，將三元組圖譜映射為 A-Box，符合經典語意網（Semantic Web）的知識表徵規範。
* **神經符號 AI 的集成 (Neuro-Symbolic Integration)**：
  * *文獻*：*Garcez, A., & Lamb, L. (2020). "Neurosymbolic AI: The 3rd Wave."* arXiv:2012.05876（後於 2023 年發表期刊版：*Journal of Applied Logics*）。
  * *理論連結*：探討了如何將深度學習（連接主義）與符號邏輯相融合。本系統將連續向量（Embedding）作為路由導引，最終降落到 Neo4j 離散符號，是典型的第三代神經符號人工智慧架構。

### 4.1 SVO 品質驗證機制：多模型生成-審查-本體擴充迴圈

#### 【動機與與既有機制的關係】

第 8 節提到的「防幻覺過濾器」實際落地為 `services/svo_service.py::_filter_hallucinated()`——一個**規則式**檢查：只要主詞或受詞其中一方以子字串形式出現在原文中即保留，不判斷語意正確性，也不檢查類型/關係選擇是否恰當。本節機制是在這之上疊加**第二個獨立 LLM 呼叫**作為審查員，並在審查持續失敗時觸發**第三個 LLM 呼叫**動態擴充本體，形成「生成 → 審查 → 重試 → 擴充」的多模型閉環，而非單純依賴子字串檢查。

#### 【三模型分工】

1. **抽取模型**（`extract_svo_from_text()`）：從原文抽取 SVO 三元組，可帶入 `extra_entity_types`/`extra_rel_types` 使用本 KG 擴充後的類型。
2. **審查模型**（`verify_svo_extraction()`）：獨立於抽取模型，拿著原文與抽取結果逐條判斷 `accepted: true/false`，判準包含「是否可被原文支持」與「類型/關係選擇是否恰當」。解析失敗時**保守判定通過**（避免審查格式異常誤殺正常抽取結果）。
3. **本體擴充模型**（`propose_ontology_extension()`）：只在「初次抽取 → 審查拒絕 → 重新抽取 → 審查仍拒絕」之後才被呼叫。給定原文與被拒絕的理由，判斷是否因為目前 19 種實體類型／31 種關係類型清單缺少真正貼切的選項，若是則提議新類別（每次呼叫上限各 3 個），並自行決定適用範圍（`scope`）。

#### 【流程控制與有界性】

整合為 `extract_svo_verified(text, kg_id, model_override)`：

```
抽取 → 審查
  ├─ 通過 → 結束
  └─ 拒絕 → 重新抽取（次數由 svo_verify_max_retries 控制，預設 1 次）→ 再次審查
        ├─ 通過 → 結束
        └─ 仍拒絕 → 本體擴充模型提議新類別 → 持久化 → 用擴充後類型再抽取一次 → 結束（不再審查）
```

最後一次抽取後**刻意不再審查**，確保流程有界、不會無限循環——這是與抽取模型既有的「chunk 級重試 + fallback 降級模型」機制（見 `_process_chunk`/`_get_fallback_model`）刻意保持一致的設計原則：本機制疊加在既有重試邏輯之上，兩者的重試次數彼此獨立、互不相乘失控。`core/config.py::svo_verify_enabled=False` 時完全略過驗證，等同舊行為（向後相容）。

#### 【本體擴充的範圍控制：預設僅供單一 KG，例外才升級全域】

依使用者明確指示設計：

* **預設 `scope="kg"`**：新類型只寫入 `services/ontology_service.py` 管理的 `ontology_extensions.json` 裡該 KG 專屬的 bucket，不影響其他 KG 的抽取行為與 BFS 查詢。
* **例外 `scope="global"`**：本體擴充模型自行判斷該類別是否「明顯具跨領域普適性」，若是則直接併入全域清單，所有 KG 共用。
* **不需要人工審核**：模型的擴充決定直接生效並持久化。**這與第 9 節②「動態本體對齊」草案的立場刻意不同**——該草案評估「動態本體對齊」（聯邦分片間的 schema 映射）時明確要求「強制人工審核，不能全自動上線」，理由是錯誤映射會靜默合併語意不同的關係、難以事後偵測。本節機制的風險輪廓不同：擴充的是「新增類別」而非「合併/覆寫既有類別」，最壞情況是新增了一個定義不夠精確或與既有類型高度重疊的類型，可事後盤點 `ontology_extensions.json` 清理，風險遠低於錯誤合併；且是使用者在此功能設計時的明確取捨，非架構上的必然。

#### 【對既有 BFS/社群偵測 Cypher 的連動修改】

新增類型若只影響抽取與驗證，而不影響下游查詢，將造成「寫得進去、查不出來」的不一致——`services/svo_service.py` 的 `_ALL_REL_PATTERN` 原本是寫死的模組常數，供 `query_svo_facts()`、`query_svo_facts_with_provenance()`、`_clear_kg_relations()` 與 `services/community_service.py` 的社群偵測 Cypher 共用。已改為透過 `ontology_service.get_effective_rel_pattern(kg_id, _ALL_REL_PATTERN)` 在查詢當下組出「基礎 31 種 + 該 KG 擴充關係類型」的完整 pattern，確保：
* BFS 圖遍歷（第 3 節）能走到新關係類型的邊。
* `rebuild_relations_only` 清除關係邊時，新關係類型的邊也會被正確清除（否則重建後會與新抽取的邊並存，造成資料不一致）。
* 社群偵測（第 9 節⑤）建圖與取樣事實時，新關係類型的邊也會納入。

#### 【驗證】

`tests/services/test_ontology_service.py`（16 案例：kg/global scope 隔離性、去重、數量上限、`get_effective_rel_pattern` 組裝、持久化）、`tests/services/test_svo_verification.py`（15 案例：審查通過/拒絕/格式異常保守處理、本體擴充解析與防禦性去重、整合流程的重試次數與有界性、停用開關）；全數通過，且完整既有測試套件（含連線真實 Neo4j 的整合測試）872→903 個測試無回歸。另用真實 Ollama 模型跑過一次端到端 smoke test，確認新增的 6 種實體類型（企業、政府機關、法規等）確實被正確抽取與保留，不再被靜默降級為「其他」。

#### 【學術文獻背書】

* **Generator-Discriminator 架構於符號抽取的應用**：
  * *理論連結*：本機制的「抽取（Generator）→ 審查（Discriminator）→ 拒絕重試」結構，概念上呼應生成對抗網路（GAN）與 Self-Refine 類方法中「生成後自我/他者批判」的設計精神，差異在於本機制的審查訊號是離散的 accept/reject 判定加自然語言理由，而非連續梯度或單一標量分數。
  * *文獻*：*Madaan, A., et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback."* NeurIPS 2023. （生成後由同一或不同模型產生回饋、再迭代改進的通用框架，是本機制「審查→重試」迴圈的直接理論參照。）
* **本體演化（Ontology Evolution）**：
  * *文獻*：*Stojanovic, L., et al. (2002). "User-driven Ontology Evolution Management."* EKAW 2002. （本體論隨著新資料湧入而動態演化的經典議題；本機制是其模型驅動、去人工審核版本的一種簡化實作，取捨與風險評估見上方。）

---

## 5. 核心理論五：變長度注意力與動態知識演化

### (Variable-Length Attention & Non-parametric Memory Evolution)

#### 【技術機制】

對於「轉譯矩陣不固定、描述節點數量隨時間增減」的問題，事實上，這符合 Attention 機制最關鍵的優勢：**置換不變性與變長度相容**。

1. **置換不變性 (Permutation Invariance)**：
   在計算相似度矩陣 $QK^T$ 時，Key 矩陣 $K \in \mathbb{R}^{N \times d}$ 中的行（Rows）順序無關緊要。無論描述節點如何重排，經歸一化後的注意力分佈均能保持不變。
2. **長度無關性 (Length Agnostic)**：
   描述節點的數量 $N$ 可以是任意正整數。當某實體的描述節點因時間推移而增加或刪除時，只是在數學上改變了矩陣的行數 $N$，透過加權歸一化算法，系統天然兼容這種動態變維。

#### 【學術文獻背書與經典論文】

* **Memory Networks (記憶網路)**：
  * *文獻*：*Weston, J., et al. (Facebook AI Research, 2014). "Memory Networks."* arXiv:1410.3916.
  * *理論連結*：提出了利用外部 Memory Slots 來克服神經網路長程記憶失效的經典模型。本系統將 ConceptNode 設計為動態增減的記憶槽，概念完全承襲自 Memory Networks 的非參數化記憶（Non-parametric Memory）思想。
* **Attention Mechanism 數學基礎**：
  * *文獻*：*Vaswani, A., et al. (2017). "Attention Is All You Need."* NeurIPS 2017.
  * *理論連結*：證明了 Attention 機制相比於 CNN/RNN，天然支持變長序列輸入（Set-input compatible）與置換不變性。

---

## 6. 核心理論六：聯邦分片檢索與實體消歧

### (Federated Graph Querying & Entity Alignment)

#### 【技術機制】

為解決海量世界知識（World Knowledge）帶來的資料庫單點效能瓶頸，系統在 [routers/world.py](file:///D:/Users/666/Desktop/智慧知識庫2/routers/world.py) 與 `services/federation_service.py`、`services/shard_query.py` 中實作了：

1. **聯邦 Registry 合併**：`services/federation_service.py` 的 `get_federation_cache()` 整合本地與 `settings.github_registry_url` 遠端 Registry 快取（快取過期時背景非同步刷新，不阻塞 `/world/chat`）；`services/shard_query.py` 的 `query_shards_parallel()` 將查詢並行發送至各個分片，並支援 `mark_shard_offline()` 對離線分片降級容錯。
2. **跨實例實體對齊（Entity Alignment）**：透過 `services/entity_alignment.py` 的 `align_entity_results()`（以字典序最小詞作 canonical key 合併同義實體）與 `expand_terms()`（靜態同義詞表 + LLM 動態多語言同義詞展開）完成。
   * **補充**：`expand_terms()` 並非只服務聯邦跨分片場景，`routers/agent.py` 在**單一 KG 內**的查詢期實體對齊也共用同一套函式，屬於比本節標題更通用的機制。

#### 【學術文獻背書與經典論文】

* **Federated Queries in Semantic Web**：
  * *文獻*：*Schwarte, A., et al. (2011). "FedX: Optimization Techniques for Federated Query Processing on Linked Data."* ISWC 2011.
  * *理論連結*：這是在語意網領域中進行分散式 SPARQL 聯邦查詢的奠基之作。本系統的並行分片查詢（`query_shards_parallel`）即是聯邦查詢在 GraphRAG 系統中的具體實作。
* **實體消歧與融合**：
  * *文獻*：*Zhao, X., Zeng, W., Tang, J., Wang, W., & Suchanek, F. M. (2020). "An Experimental Study of State-of-the-Art Entity Alignment Approaches."* IEEE Transactions on Knowledge and Data Engineering (TKDE).
  * *理論連結*：探討了在多源、分散式知識庫中，如何利用對齊算法（Alignment Algorithms）整合命名不一致但實質等價的實體。

### 6.1 聯邦架構的三項工程延伸（Phase 3a/3c/3d）

> **本節定位**：以下三個能力（知識溯源、版本控制、訂閱同步）已在 `ROADMAP.md` 標記為 Phase 3a/3c/3d 完成，屬於已上線的產品功能。它們並非全新的學術理論分支，而是第 3、6 節既有理論（符號-物理回溯、聯邦分片）在「時間軸」與「推播/拉取模式」上的直接工程延伸，故合併記錄於本節，不另立獨立章節。

#### 【Phase 3a：知識溯源（Provenance）】

在第 3 節「符號-物理源頭回溯」之外，`models/provenance.py` 定義了更結構化的 `SourcedFact`（事實 + 來源文件 + 信心分數 + 建立時間）與 `ProvenanceReport`。`services/svo_service.py::query_svo_facts_with_provenance()` 在 BFS 撈回候選邊後，批次 JOIN `Document` 節點取得標題（`_batch_get_doc_titles`），供 `/world/chat`（見 `services/shard_query.py::ShardResult.sourced_facts`）與 `GET /world/provenance/facts` 端點使用；LLM Prompt 改用 `cite_str()` 格式（`事實 [來源：《文件名》，信心 N]`），要求模型在回答中主動引註出處。這與第 3 節「符號-物理源頭回溯」及第 7 節對照表中的 ProVe 專案是同一條理論脈絡的落地，差別在於 Phase 3a 把溯源標記直接嵌入 Prompt 文本，而非僅作為 UI 顯示用的中繼資料。

#### 【Phase 3c：KG 版本控制（Version Control）】— 與第 9 節⑥時序衰減共用同一組時間戳

`routers/versioning.py` 提供三個唯讀查詢端點：`GET /kg/{id}/changelog`（依 `updated_at`/`created_at` 降冪列出近期變更）、`GET /kg/{id}/diff?since=`（指定時間點後的所有新增/更新事實）、`GET /kg/{id}/snapshot?at=`（`created_at <= at` 的知識快照，重建任一歷史時間點的圖譜狀態）。三者皆同時支援 Enterprise 多資料庫與 Community 版 `kg_id` 屬性區隔兩種路徑。

* **與第 9 節⑥的關係**：Phase 3c 沒有引入新的時間戳欄位，而是直接複用 `svo_service.py` 邊 MERGE 時既有的 `created_at`（`ON CREATE SET`）與 `updated_at`（`ON MATCH SET`），把原本只用於「衰減重排」的時間戳，額外開放為「歷史查詢」的一等公民資料。`change_type` 欄位（`"created"` | `"updated"`）由比對兩個時間戳是否相等推導。
* **學術背景**：這對應資料庫理論中的**雙時態資料模型（Bitemporal Data Model）**與 RDF/圖譜演化研究：
  * *文獻*：*Snodgrass, R. T. (1999). "Developing Time-Oriented Database Applications in SQL."* Morgan Kaufmann. （交易時間 / 有效時間雙時態建模的奠基教材，`snapshot`/`diff`/`changelog` 三個端點分別對應書中「時間點查詢」「區間查詢」「變更序列查詢」三種標準模式。）
  * *理論連結*（概念性類比，不掛特定文獻）：不重建整個圖譜、只追蹤增量演化，是本節設計動機。

#### 【Phase 3d：KG 訂閱／自動同步（Subscription）】— 對聯邦架構的「拉取式」補充

第 6 節的聯邦分片（`services/shard_query.py`、`services/federation_service.py`）是「查詢時即時並行拉取」模式：每次 `/world/chat` 都對所有已知分片發起即時請求。`services/subscription_service.py` 的 `SubscriptionManager` 則新增了**背景定時拉取並落地合併**的第二種模式：`sync_subscription()` 從遠端 AuraDB 分批拉取 SVO 並 MERGE 進本機圖譜，`sync_all_subscriptions()` 由 `main.py` 內建的 APScheduler 每 6 小時呼叫一次，單次同步設有 60 秒逾時避免掛死排程。`routers/subscription.py` 提供訂閱的增刪查與暫停/恢復管理端點。

* **與即時聯邦查詢的取捨**：即時分片查詢（第 6 節）延遲低但每次問答都依賴對方線上；訂閱同步犧牲即時性，換取「訂閱後即使對方離線，本機仍保有最近一次同步的完整副本」，兩者互補而非互斥——這正是 ROADMAP.md 待解決問題表中「AuraDB 免費版閒置暫停」風險的直接對沖手段。
* **學術背景**：定期拉取、非同步收斂的複寫模式對應分散式系統中的 **Epidemic / Gossip 協定**與最終一致性（Eventual Consistency）理論：
  * *文獻*：*Demers, A., et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance."* PODC 1987. （提出定期兩兩同步、允許暫時不一致換取去中心化韌性的經典演算法，是本系統「6 小時定時拉取 + 60 秒逾時降級」設計的理論先驅。）
  * *理論連結*：與第 6 節引用的 FedX（即時聯邦查詢）互為對照——FedX 代表 pull-on-query 的同步聯邦，Gossip/Epidemic 代表 pull-on-schedule 的非同步複寫，本系統同時具備兩種模式，是聯邦知識分片架構在 CAP 定理權衡光譜上的雙棲實作。

---

## 7. 相關 GitHub 開源專案對照 (State-of-the-Art Baselining)

本系統與目前 GitHub 上最前沿的世界知識與圖譜 RAG 專案具有高度關聯，並在特定架構上進行了優化。

| GitHub 專案與連結 | 星數 | 核心學術定位 | 本系統的對應與優勢 |
| :--- | :--- | :--- | :--- |
| [zjunlp/WKM](https://github.com/zjunlp/WKM) <br>*(World Knowledge Model)* | ★167，最後活動 2024-12 | 浙江大學 NLP 團隊開發，為 AI Agent 注入先驗的物理和常識狀態，降低規劃幻覺。 | 本系統已對接 `claude-desktop`（獨立的外部伴侶應用，不在本 repo 內，於本機另以 `claude-desktop-frontend-dev`/`claude-desktop-backend-dev` 容器執行）作為 Agent 的外部大腦。 |
| [King-s-Knowledge-Graph-Lab/ProVe](https://github.com/King-s-Knowledge-Graph-Lab/ProVe) | ★11，最後活動 2026-05（持續維護中） | 利用 LLM 對照網頁參考資料，校驗 Wikidata 中的三元組事實（Fact Verification）。 | 本系統實作了 **「事實溯源 (Provenance)」** 路由，與 ProVe 雷同，且加入了**「防幻覺過濾器」**進行實體原文存在性校驗。⚠️ **星數偏低（僅 11），可信度來源是 King's College London 學術實驗室背書，而非社群熱度**——引用時應明確標註是「學術機構的研究產出」而非「廣受採用的開源工具」，兩者是不同層級的證據力，不宜混為一談。 |
| [pat-jj/KG-FIT](https://github.com/pat-jj/KG-FIT) | ★132，最後活動 2025-05 | 針對開放世界知識（Open-World）進行圖譜的微調與補全，解決新實體對齊問題。 | 本系統在 `services/entity_alignment.py` 中實作了**同義詞展開與實體對齊**，在不微調模型的情況下完成開放世界實體融合。 |
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | ★34,251，最後活動 2026-06（活躍） | 微軟官方 GraphRAG 實作，用 Leiden 演算法對知識圖譜做階層式社群偵測，為每個社群生成 LLM 摘要，支援 Global Query（全域性宏觀問答）。 | 對應本文件第9節⑤「多層次社群摘要檢索」——已落地（`services/community_service.py`），但用 `networkx` 內建 Louvain 取代 Leiden，且未做階層式多層分群，差異與原因見第9節⑤說明。本表中星數最高、最無爭議的旗艦級對照對象。 |
| [neo4j-contrib/ms-graphrag-neo4j](https://github.com/neo4j-contrib/ms-graphrag-neo4j) | ★88，最後活動 2025-10 | 微軟 GraphRAG 與 Neo4j 的官方整合套件，提供 Leiden 社群偵測直接寫入 Neo4j 圖資料庫的參考實作。 | 若未來要將 Louvain 升級為 Leiden，此專案是最低摩擦力的參考實作（技術棧同為 Neo4j）。⚠️ **星數不高（88），但屬於 Neo4j 官方組織（`neo4j-contrib`）帳號下的專案，可信度來源是廠商官方背書而非社群熱度**，與 ProVe 同理不應單純以星數評價。 |

---

## 8. 技術優勢總結 (Key Takeaways for Presentation)

在向評審或口試委員介紹本系統時，可著重以下四點：

1. **雙層檢索架構與符號-物理源頭回溯（Source Backtracking）**：
   * **圖外**利用 **Graph-MoE** 門控路由激活專家圖譜。
   * **圖內**利用 **BFS 遍歷** 匹配 SVO 節點；**若三元組資訊太精簡，系統會自動沿節點物理座標回溯，直接從 `ChunkStore` 拉取生成該節點的原始文件段落（Chunk）供 LLM 參考**，解決了傳統圖譜缺乏 Context 的缺陷。
2. **本體論 (Ontology) 驅動的 Schema 約束**：由 19 種實體與 31 種語意關係約束的「高階本體圖譜」，具備語意可歸納性與推理能力。
3. **高容錯且防幻覺的 SVO 抽取 Pipeline**：利用防幻覺過濾器在抽取端斬斷虛假實體，並利用 Dynamic Fallback 本地模型重試，確保了離線大規模建圖的極致穩定性。
4. **動態記憶體與聯邦擴展**：轉譯層支持隨時動態讀寫、變長的 ConceptNode，且支持 GitHub 雲端 Registry 分散式查詢，具備強大的擴展性與抗遺忘能力。

---

## 9. 架構前瞻與未來優化方向 (Architectural Extensions & Future Enhancements)

為了進一步提升神經符號 Graph-MoE RAG 架構在極大規模與複雜邏輯下的推理精度，未來可在以下八個前沿方向進行架構擴展，各方向均有相關學術研究支撐：

> **目前落地狀態總覽**：
>
> | 方向 | 狀態 | 備註 |
> |---|---|---|
> | ① GNN/node2vec 共嵌入空間 | ✅ 已落地 | `services/graph_embedding_service.py` + `repositories/concept_repo.py` 的 `_fuse_graph_vector()` |
> | ② 動態本體對齊 | ❌ 暫不做 | 目前無真實異質 schema 分片會用到，見下方評估 |
> | ③ Graph-CoT 推理 | ✅ 已落地（簡化版） | `routers/agent.py` 門檻觸發式加深查詢，不含 LLM 選路 |
> | ④ Active RAG | 🟡 部分落地 | 只做到「提早結束單輪生成」，見下方說明 |
> | ⑤ 社群摘要檢索 | ✅ 已落地 | `services/community_service.py`，Louvain 分群 + LLM 摘要 |
> | ⑥ 時序知識圖譜衰減 | ✅ 已落地 | `services/svo_service.py::_temporal_decay()` |
> | ⑦ 對比學習 | ✅ 已落地（離線訓練管線） | `services/contrastive_training_service.py` |
> | ⑧ 二階段粗精篩 | ✅ 已落地 | `services/concept_engine.py` 的 `route_kgs()`/`route_documents()` |

### ① 圖拓撲感知共嵌入空間 (Graph-Aware Co-embedding Space) — ✅ 已落地（採 node2vec，範圍與原設計有出入見下）

* **當前局限**：目前的 ConceptNode 連續特徵向量（Embedding）是利用標準文本模型獨立計算的，未感知到 Neo4j 圖譜中 SVO 邊所承載的拓撲結構與關聯強度。
* **優化建議**：引入 **圖神經網絡 (GNN)** 算法（如 GraphSAGE 或 Node2Vec），將圖譜的離散拓撲特徵與文本的語意特徵進行聯合表徵學習，產生「感知圖結構的概念向量 (Graph-Aware Concept Embeddings)」。這能使 Gating Router 的相似度計算精確數倍。
* **技術選型**：採用 **node2vec**，不是與其並列的 GraphSAGE。理由：本專案已將 `networkx>=3.0` 列為直接相依套件，`networkx` 沒有內建 node2vec 但有輕量第三方實作可直接套用其圖結構 API，不需要 `torch` + `torch-geometric` 這類重型 ML 框架；GraphSAGE 是 inductive（可對新節點免重訓推論），能力更強，但代價是要引入 PyTorch 系依賴——這與本專案「本地 CPU 推論優先、盡量避免 GPU 依賴」的定位有摩擦（可對照 PaddleOCR 在 GPU 初始化失敗時要求 fallback 回 CPU 的既有修復）。GraphSAGE 留作未來若有「新節點需要即時可用向量」的明確需求時再評估的 Phase 2。
* **落地範圍的釐清（容易被誤解之處）**：「圖拓撲感知」容易讓人以為要用 SVO 的 Entity-Entity 圖。但 KG 路由層（`concept_engine.compute_match_score`）比對的是 `ConceptNode.q_vector`，而 ConceptNode 之間目前唯一的圖結構關係是「同一份 Document/KG 透過 `EFFECTIVE` 邊連到哪些 ConceptNode」，是一個 Document/KG ↔ ConceptNode 的二分圖（bipartite graph），並非 SVO 的 Entity-Entity 圖，因此落地對象是對這個二分圖做 node2vec，而非對 SVO 圖做 GNN。
* **實際落地位置**：
  * `run_build_graph_embeddings.py`：離線批次腳本（比照 `run_build_kg.py` 慣例），抓取全部 `(Document|KnowledgeGraph)-[:EFFECTIVE]->(ConceptNode)` 邊建圖，跑 node2vec 產生每個 ConceptNode 的圖結構向量。
  * `repositories/concept_repo.py::set_concept_graph_vectors()`：批次寫入 ConceptNode 的 `q_vector_graph` 屬性，與既有 `q_vector`（純文字 embedding）並存、不覆蓋——刻意的風險控制設計。
  * `repositories/concept_repo.py::_fuse_graph_vector()`：查詢期融合邏輯，`final = α·q_vector + (1-α)·q_vector_graph`（兩向量各自正規化後才加權平均，避免量級不一致主導結果）；`q_vector_graph` 缺失（尚未跑批次腳本的新概念）時原樣返回純文字向量，完全向後相容。融合權重 `α` 為 `core/constants.py::GRAPH_EMBEDDING_ALPHA = 0.85`（偏保守，以文字向量為主），已接入 `get_all_kgs_concepts()`/`get_public_kgs_concepts()` 兩個路由查詢入口。
* **風險**：node2vec 是 **transductive**——新增 ConceptNode 後，要獲得穩定的 graph embedding 得重新訓練整個圖的向量，這與專案現有「增量建圖、跳過已處理文件」的設計哲學有摩擦：新文件加入後其 ConceptNode 在下次全量重訓前只能先用融合後仍以純文字向量為主（`α=0.85`）的降級行為。若這個限制被證實不可接受，才需要評估升級到 GraphSAGE（inductive）。
* **學術來源**：
  * Hamilton, W., Ying, Z., & Leskovec, J. (2017). *"Inductive Representation Learning on Large Graphs."* NeurIPS 2017. (GraphSAGE 奠基作，未來可能升級方向)
  * Grover, A., & Leskovec, J. (2016). *"node2vec: Scalable Feature Learning for Networks."* KDD 2016. (實際採用)

### ② 多源聯邦本體動態對齊 (Dynamic Federated Ontology Alignment) — 🔵 設計方案已補充（尚未實作，建議暫緩）

* **當前局限**：跨分片並行查詢時，若不同分片的本體 Schema（如關係邊定義）存在命名或分類不一致（如 `IS_A` 與 `INSTANCE_OF` 混用），跨域查詢的語意流會發生斷裂。
* **優化建議**：在路由層引入基於 LLM 或 Graph Matching 的 **動態本體對齊（Ontology Alignment）** 機制，自動在查詢發起前對不同知識庫的關係邊進行 Schema 轉換與映射，達成「無感知的跨域本體對接」。
* **落地前提的查證結果（這是本項與其他項最大的不同之處）**：查過 `services/federation_service.py` 與 `services/shard_query.py` 目前的聯邦分片實作，所有分片（本機其他 KG、GitHub 遠端 registry）都遵循**同一套** `_VALID_REL_TYPES`（31 種，見第4節註記）與 registry 格式——因為分片本身就是同一個專案的另一個部署實例，不是真正異質的外部知識圖譜。也就是說，**文件描述的本體不一致問題目前沒有真實案例會發生**，這是一個面向假設性未來需求（「未來若允許匯入非本專案格式的外部 KG」）的優化方向，不像⑧（O(N) 效能瓶頸）是已驗證的現有缺陷。
* **技術選型（若未來真的要做）**：不引入新的 ML 模型，優先用 **LLM 輔助生成映射表**（複用專案已有的多 Provider LLM 抽象層 `core/providers/llm/`，零新增基礎設施），而非傳統 Graph Matching 演算法（如 AgreementMakerLight，需要獨立的 Java 服務，與現有 Python 單體架構不合）。
* **具體設計**：
  1. `services/federation_service.py` 的 registry 結構為每個遠端分片新增可選欄位 `ontology_mapping: dict[str, str]`（`{external_rel_name: internal_rel_name}`）。
  2. 首次接入異質分片時，抓取該分片少量三元組樣本，呼叫 LLM 生成映射表草稿。
  3. **強制人工審核**這份映射表後才寫入 registry——不能全自動上線。
  4. `services/shard_query.py::query_shards_parallel()` 查詢時依 `shard_id` 查對應映射表，把回傳結果的 `rel_type` 欄位做替換。
* **風險**：LLM 生成的映射表可能有誤，錯誤映射的後果是「兩個語意不同的關係被靜默合併成同一種」，屬於難以事後偵測的資料品質問題，這是堅持要人工審核的原因。
* **建議暫緩**：目前沒有真實的異質 schema 分片會用到它，優先度應排在有真實需求驅動的項目之後，等真的要接入外部異質 KG時再啟動評估。
* **學術來源**：
  * Shvaiko, P., & Euzenat, J. (2013). *"Ontology Matching: State of the Art and Future Challenges."* IEEE Transactions on Knowledge and Data Engineering, vol.25, pp.158-176.
  * Faria, D., et al. (2013). *"AgreementMakerLight: A System for Large-Scale Ontology Matching."* OTM 2013（On The Move Federated Conferences；相關但不同的 OAEI 評測結果短文發表於 ISWC 附屬 workshop）。

### ③ 圖譜鏈式思考推理 (Graph Chain-of-Thought / G-CoT) — ✅ 已落地（簡化版）

* **落地狀態**：✅ 已落地簡化版 —— 見 `routers/agent.py` 的 `_SVO_SPARSE_FACT_THRESHOLD` 機制。**與下方原始設計的關鍵差異**：不採用「每跳都呼叫 LLM 決定下一步」的做法（延遲與 LLM 成本過高，不適合個人 KB 的問答場景），改為門檻觸發式簡化版——2 跳 BFS 命中事實數低於門檻（3 條）時，用同一組種子詞加深一跳重查（`hops+1`）並合併結果，零額外 LLM 呼叫。細節見第 10 節③。
* **當前局限**：圖譜內查找僅依賴簡單的 1-2 跳 BFS，屬於「被動式檢索」，缺乏對複雜邏輯路徑的自主推理能力。
* **優化建議**：引入 **Graph-CoT (圖譜鏈式思考)** 機制。LLM 不僅被動接收 Context，而是能作為一個 Agent 沿著圖譜的語意關係邊主動「尋路」，動態決定下一跳要遍歷哪個實體，尋找最優的推理路徑（Multi-hop Reasoning Path）。

* **技術選型**：不需要新模型，複用現有 LLM Provider 抽象層。核心改動是查詢邏輯從「一次性 BFS 1-2 跳全取」改為「LLM 逐跳決策」。
* **具體設計**：`services/svo_service.py` 新增 `query_svo_facts_cot(kg_id, start_terms, question, max_hops=3)`：(1) 用既有全文索引找種子實體 (2) 迴圈最多 `max_hops` 次：查該實體的一跳鄰居（複用既有 Cypher pattern），把「目前推理路徑 + 候選鄰居 + 原始問題」交給 LLM，要求回覆 `{next_target, reason, is_stop}` 的 JSON（虛擬碼設計已在第10節③給出骨架）(3) 累積路徑上的所有邊作為最終 facts。
* **最大的實際落地風險（延遲）**：`routers/agent.py` 已有一個精煉迴圈在做多輪 LLM 呼叫（confidence-based，最多 `_MAX_REFINE_ROUNDS=3` 輪）。若疊加 Graph-CoT 的多跳 LLM 呼叫，一次問答理論上可能觸發到 `hops(≤3) × refine_rounds(≤3) = 9` 次 LLM 呼叫。本專案預設走本地 Ollama（`phi4`/`qwen2.5` 等），單次生成常需要數秒到十幾秒，9 次呼叫的延遲對使用者是不可接受的。這是本項目**最需要先解決**的問題，而不是尋路邏輯本身。
* **建議的漸進落地策略**：不讓 Graph-CoT 成為預設路徑。只在「BFS 1-2 跳 + 既有精煉迴圈都無法達到信心門檻」時才觸發，接在現有精煉迴圈「信心不足」分支之後，作為第三種補救手段（BFS → 相似度補充 chunk → Graph-CoT 尋路），而非取代 BFS 本身。這樣多數問答完全不受影響，只有少數「疑難」查詢才會付出額外延遲成本。
* **影響範圍**：`services/svo_service.py`（新函式）、`routers/agent.py`（精煉迴圈新增第三層補救分支）、需要新增 LLM 呼叫次數的監控/上限保護（避免失控的延遲或 API 費用）。
* **風險**：延遲風險（如上，需要嚴格的觸發條件與呼叫次數上限）；LLM 決策品質風險——本地小模型（如 `qwen2.5:7b`）做多跳路徑決策的可靠度未經驗證，可能不如預期，需要 fallback（例如連續 2 次尋路失敗就放棄，回退現有答案）。
* **工作量分級**：大，2-3 週，含延遲監控、呼叫上限與 fallback 機制的工程量，且必須用真實資料集做 A/B 測試驗證「Graph-CoT 答案品質是否真的優於現有 BFS」，不能只憑理論假設就上線——這是落地前必要的驗證步驟。
* **學術來源**：
  * Jin, Bowen, et al. (2024). *"Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs."* arXiv:2404.07103, **ACL 2024**（G-CoT 經典研究；本專案簡化版的理論依據）。

### ④ 主動自適應檢索 (Active & Adaptive Retrieval) — 🟡 部分落地（範圍遠小於原設計，見下）

* **當前局限**：現有機制為單次檢索後生成答案，即便有自我精煉（Self-Refinement）也只是被動回填 Chunks，無法在生成過程中自發性地決定何時需要新知識。
* **優化建議**：引入 **Active RAG (主動式檢索增強)**。在 LLM 串流生成的過程中，如果發現缺失某個中間邏輯鏈條的知識，能自發發起圖譜檢索，實現「一邊生成、一邊動態判斷、一邊補充檢索」的自適應生成。
* **實際落地範圍與原設計的差異（誠實揭露：這不是完整的 Active RAG）**：
  * **沒有做到「發起新檢索」，只做到「提早結束生成」**：原設計的核心是「生成中途發現知識缺口 → 自發觸發新一輪圖譜/文件檢索 → 把結果插回去繼續生成」。實際落地的是遠遠更保守的版本：`routers/agent.py` 既有的精煉迴圈（confidence-based refinement，本來就會在整段答案生成完畢後，若信心 < `_CONFIDENCE_THRESHOLD` 就補充 chunk 重新生成）在此基礎上新增「串流過程中一旦偵測到 `_NO_INFO_RE`（「找不到相關」「無法回答」等）信號，立即中斷該輪的 token 消費」，跳過模型接下來可能講的填充/免責文字，提早進入既有的補充檢索流程。**沒有實作**逐 token confidence/entropy 監控、沒有偵測「孤立未參照實體」、也沒有在生成中途插入新資訊後從中斷點接續生成——這些都是原設計較困難的部分，需要 LLM Provider 支援 per-token logprobs（目前 Ollama/OpenAI/Anthropic/Gemini/Grok 5 種 Provider 介面不統一，貿然實作有跨 Provider 相容性風險），故本次不做。
  * **為什麼仍值得做**：即使只是「提早結束一輪生成」，也是把「決定要不要多檢索」的判斷點從「生成完畢後」提前到「生成過程中」，是「一邊生成、一邊判斷」精神的一個小而真實的子集，且零額外延遲風險（沒有偵測到信號時行為與原本完全一致）。
  * **不在最後一輪套用**：`_MAX_REFINE_ROUNDS`（預設3）的最後一輪沒有更多檢索預算可用，提早中斷該輪只會讓答案變短而沒有實質好處，因此保留完整生成。
* **實際落地位置**：`routers/agent.py` 的 `/agent/chat` 精煉迴圈（約 L658 起）：`_active_watch = round_num < _MAX_REFINE_ROUNDS - 1`，串流消費迴圈中每收到一個 token 就檢查累積文字是否命中既有的 `_NO_INFO_RE`，命中且非最後一輪即 `break`；其餘信心計算、補充邏輯完全復用既有精煉迴圈，未新增額外分支。
* **測試**：`tests/routers/test_rag_quality.py::TestActiveRAGEarlyExit`（驗證非最後一輪提前中斷且不消費填充 token、驗證最後一輪不提前中斷）。
* **學術來源**：
  * Jiang, Z., Xu, F. F., Gao, L., et al. (2023). *"Active Retrieval Augmented Generation."* (FLARE) EMNLP 2023, arXiv:2305.06983.
  * Asai, Akari, et al. (2024). *"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection."* ICLR 2024.

### ⑤ 多層次社群摘要檢索 (Community-based Hierarchical Retrieval) — ✅ 已落地（範圍與原設計有出入見下）

* **當前局限**：當遭遇全域性（Global Query）或跨多個文檔的宏觀查詢（如：「請總結所有公開圖譜中的技術演進」）時，BFS 遍歷與向量路由僅能匹配局部實體，無法回答全局性問題。
* **優化建議**：引入 **社群檢測 (Community Detection)** 算法（如 Louvain 或 Leiden 算法），對 Neo4j 中的圖譜結構進行層次化分群，並由 LLM 預先為每個分群生成「社群摘要 (Community Summaries)」。問答時根據問題層級路由至相應的社群摘要，提供巨觀的全局回答。
* **實際落地範圍與原設計的差異**：
  * **用 Louvain 而非 Leiden**：`networkx` 已是專案既有的（間接）相依套件，其 `networkx.algorithms.community.louvain_communities()` 可直接使用；Leiden 演算法需要 `leidenalg` + `python-igraph`，兩者皆含 C 擴充套件，在本專案的 Windows 開發環境下需要編譯工具鏈才能安裝，風險與環境相依成本較高。Louvain 是 Leiden 的前身，效果在中小型圖譜上差異有限，先以 Louvain 落地、未來若圖譜規模擴大出現社群品質問題，再評估遷移至 Leiden。
  * **未實作「層次化」多層級分群**：只做單層 Louvain 分群，未實作 Leiden 論文強調的階層式社群樹（Level 0/1/2…）。
  * **全域查詢偵測為關鍵詞啟發式，非語意分類器**：`routers/agent.py::_is_global_query()` 用正則比對「總結/整體/全部/overview/summarize」等關鍵詞，並非更精確的問題語意分類。誤判在所難免（例如「這篇文件整體在講什麼」會被視為全域查詢），但作為第一版足夠可用，且失敗模式是「多給一段摘要 context」而非拒答，風險可控。
  * **只整合到 `/agent/chat`**：`/world/chat`（公開 KG 聯邦問答）尚未接上此機制，留待後續迭代。
* **實際落地位置**：
  * `services/community_service.py`：
    * `build_communities_for_kg(kg_id, db_name, min_size, max_communities)`：抓取 Entity 關係邊建 `networkx.Graph`，跑 `louvain_communities(seed=42)`，過濾規模 < `min_size`（預設3）的社群，逐一取樣社群內 SVO 事實、呼叫 LLM 生成 2-3 句摘要，持久化為 Neo4j `:Community` 節點（`summary`/`member_count`/`top_entities` 屬性）並用 `(:Entity)-[:IN_COMMUNITY]->(:Community)` 邊連結成員。每次執行先 `DETACH DELETE` 該 KG 舊社群再重建（沿用 `run_build_kg.py --force` 的慣例）。
    * `get_community_summaries(kg_id, db_name, limit)`：依 `member_count` 降冪讀回已建立的社群摘要。
  * `run_build_communities.py`：離線批次腳本，比照 `run_label_kg.py` 慣例（`--kg`、`--min-size` 參數）。
  * `routers/agent.py`：新增 `_is_global_query()` 啟發式判斷；`/agent/chat` 在 KG 路由完成、BFS 事實回傳後，若判定為全域查詢則並行呼叫 `get_community_summaries()`，透過新增的 `community_summaries` SSE 事件回傳給前端，並將摘要文字併入 `contexts`（走既有的、已驗證會進入 LLM prompt 的路徑，而非 `svo_facts` — 後者只用於 SSE 顯示與 chunk 關鍵詞加權）。
  * `requirements.txt` 新增 `networkx>=3.0`。
  * 測試：`tests/services/test_community_service.py`（雙群偵測、規模過濾、LLM 摘要容錯、讀取排序）、`tests/routers/test_rag_quality.py::TestGlobalQueryHeuristic`/`TestCommunitySummaryInjection`（關鍵詞判定、SSE 事件觸發與非觸發）。
* **學術來源**：
  * Blondel, V., et al. (2008). *"Fast unfolding of communities in large networks."* Journal of Statistical Mechanics. (Louvain 算法經典，本次實際採用)
  * Traag, V., et al. (2019). *"From Louvain to Leiden: guaranteeing well-connected communities."* Scientific Reports. (Leiden 算法，未來可能遷移方向)

### ⑥ 時序知識圖譜與陳舊性校正 (Temporal Knowledge Graphs & Decay) — ✅ 已落地（範圍與原設計有出入見下）

* **當前局限**：知識事實會隨著時間演進而陳舊（例如：CEO 職位更迭、技術標準變遷）。若 SVO 缺乏時間維度，圖譜中會存在相互衝突的過期知識，導致 LLM 產生幻覺。
* **優化建議**：引入 **時序知識圖譜 (Temporal KG)** 機制，為每條 SVO 關係邊加上時間戳（`valid_from`, `valid_to`），並在重排公式中引入 **「時間衰減因子 (Temporal Decay Factor)」**，確保時效性高、未過期的事實被優先檢索。
* **實際落地範圍與原設計的差異**：
  * **未新增 `valid_from`/`valid_to` 欄位**——`svo_service.py` 的 SVO MERGE 邏輯早已在 `ON CREATE SET rel.created_at = datetime()` 設定建立時間（此為既有欄位，先前只寫入從未被讀取），本次落地直接沿用 `created_at` 作為衰減基準時間，而非新增 `valid_from`/`valid_to` 生效區間欄位。原因：`valid_from`/`valid_to` 代表「事實在現實世界中的有效期間」，需要從文件內容解析時序語意（例如辨識「2023年起」「已於2024年終止」等敘述）才能準確填值，屬於獨立的 NLP 子任務，不在本次範圍內；`created_at`（事實被抽取進圖譜的時間）是可以立即使用、對既有資料 100% 相容的代理指標（proxy），先以此上線，時序語意抽取留待未來迭代。
  * **未修改「重排公式」（`_pick_relevant_chunks`，第3節）**——該函式只處理單一文件內的段落文字，沒有跨文件的發布時間可比較，衰減在此層級沒有意義。改為套用在 **SVO 事實排序**：`services/svo_service.py` 的 `query_svo_facts()`（`/agent/chat` 主要問答路徑）與 `query_svo_facts_with_provenance()`（`/world/chat`、聯邦分片查詢），BFS 撈回候選邊後，用 `confidence × decay_factor` 重新排序，取代原本只依 `confidence DESC` 排序。
* **實際落地位置**：
  * `services/svo_service.py::_temporal_decay(created_at, rate)`：`decay = exp(-rate * delta_days)`，與文件公式一致；`created_at` 缺失或無法解析時回傳 `1.0`（不衰減），確保舊資料/邊界情況下為向後相容。
  * `core/constants.py::TEMPORAL_DECAY_RATE = 0.005`（與原虛擬碼 `daily_decay_rate` 預設值一致）。
  * 3 處 `query_svo_facts()` 的 Cypher 分支與 2 處 `query_svo_facts_with_provenance()` 分支皆已加上 `toString(r.created_at) AS created_at`，並在 Python 端用 `_temporal_decay` 重新排序候選邊（Cypher 端 `LIMIT` 仍以 `confidence` 截斷候選集，僅重排取回後的順序，不影響何者被納入候選）。
  * 測試：`tests/services/test_svo_service.py::TestTemporalDecay`（缺失/無法解析回傳1.0、新鮮事實≈1.0、較舊事實衰減、新舊事實相對排序、無時區字串容錯）。
* **學術來源**：
  * Goel, R., et al. (2020). *"Diachronic Embedding for Temporal Knowledge Graph Completion."* AAAI 2020.

### ⑦ 對比自我監督概念學習 (Contrastive Concept Learning) — ✅ 已落地（離線訓練管線，範圍遠小於原始評估時的擔憂，見下）

* **當前局限**：在 ConceptNode 路由比對中只計算了正向的 Similarity 相似分，若兩個相鄰領域的概念界線模糊，容易發生路由偏差。
* **優化建議**：在 Embedding 訓練或對齊計算中引入 **對比學習 (Contrastive Learning)**。在優化對齊權重時，不僅最大化正向概念的 cosine alignment，同時拉遠無關的負樣本概念（Negative Concepts），使得 Gating Router 的分類決策邊界更加清晰。
* **早期評估 vs. 實際落地的落差**：本項目原本被評估為「目前不建議投入」——理由是傳統對比學習微調需要自建負樣本挖掘、SimCLR/GCL 風格 loss、訓練/驗證/模型版本管理的完整 pipeline，這與專案「呼叫外部/本地推論 API、不訓練模型」的定位衝突，且需要 GPU 資源。實際落地時**繞開了上述所有顧慮**：沒有自建訓練框架，而是直接使用 `sentence-transformers`（本專案既有相依套件）內建的 `SentenceTransformerTrainer` 微調 API 與 `MultipleNegativesRankingLoss`（in-batch negatives，即 InfoNCE 標準實作）——這是既有函式庫的一等公民功能，不是新增訓練基礎設施，因此上述「與架構哲學衝突」「需要新增訓練/版本管理能力」的疑慮不成立。
* **實際落地位置**：
  * `services/contrastive_training_service.py::generate_training_pairs()`：正樣本對來源為「同一份 Document 下共現的 ConceptNode 名稱配對」；若 Document 層級樣本數不足 `_MIN_PAIRS_TO_TRAIN`（20），退而求其次改用 KnowledgeGraph 層級共現（訊號較弱但涵蓋更廣）補足；負樣本由 `MultipleNegativesRankingLoss` 的 in-batch negatives 機制自動提供，不需額外挖掘。
  * `run_finetune_embeddings.py`：離線批次腳本，讀取正樣本對、呼叫 `SentenceTransformerTrainer` 微調 `local_embedding_model`，屬於一次性/低頻執行的訓練任務，不常駐、不影響線上查詢路徑。
* **仍然成立的風險提醒**：`compute_match_score` 的路由準確度目前沒有被驗證為現有問題（不像⑧的 O(N) 效能瓶頸是已明確指出的真實缺陷），且微調若樣本品質不佳，理論上仍有讓 embedding 品質變差（過度貼合訓練樣本、犧牲泛化能力）的風險，建議先小規模試跑並比對微調前後的路由準確度再決定是否納入預設流程。
* **學術來源**：
  * Chen, T., et al. (2020). *"A Simple Framework for Contrastive Learning of Visual Representations."* ICML 2020. (SimCLR 對比學習架構；`services/contrastive_training_service.py` 實際採用的 `MultipleNegativesRankingLoss` 即 SimCLR 式 in-batch negatives 的標準實作)

### ⑧ 二階段向量粗篩-精篩架構 (Two-Stage Coarse-to-Fine Retrieval) — ✅ 已落地

* **落地狀態**：✅ 已落地 —— `repositories/concept_repo.py` 的 `vector_search_concept_ids()`（Stage-1 呼叫 `db.index.vector.queryNodes` 做 KNN 粗篩）+ `get_all_kgs_concepts()`/`get_public_kgs_concepts()`/`get_all_documents_concepts()` 新增可選 `concept_ids` 參數（Stage-2 精篩範圍限縮，非 None 時只回傳候選集合內的概念）+ `services/concept_engine.py` 的 `route_via_two_stage()`（協調 Stage-1/2，向量索引不可用或無候選時自動 fallback 全表掃描）與其上層封裝 `route_kgs()`/`route_documents()`（供路由呼叫點使用）。已套用到 `routers/agent.py`、`routers/world.py`、`routers/search.py`、`services/classify_service.py` 所有查詢期路由路徑。與下方虛擬碼的差異：粗篩候選數改為可調常數 `CONCEPT_COARSE_TOP_K`（`core/constants.py`，預設 100）。
* **當前局限**：概念匹配時在 Python 內存中對全庫進行 $O(N)$ 雙重迴圈計算，在大規模（N > 10,000）時會引發 CPU 阻塞與內存溢出。
* **優化建議**：將檢索重構為**「二階段檢索架構（Two-Stage Retrieval）」**。第一階段（粗篩，Stage-1）利用 Neo4j 內建的 **Vector Index**（以 C++ 底層高速運算）抓出 Cosine 相似度最高的 Top-100 個候選節點；第二階段（精篩，Stage-2）在 Python 內存中僅對這 100 個候選節點進行對齊遮罩（Align）與強度振幅（Mag）的精細比對。複雜度由 $O(N)$ 驟降為 $O(100)$ 常數級別，效能提升千倍以上。
* **實際落地位置**：
  * `repositories/concept_repo.py` 的 `vector_search_concept_ids()`：呼叫既有的 `concept_q_vector` 向量索引（該索引原本已在 `main.py`/`run_build_kg.py` 等啟動流程建立，但先前從未被查詢使用）執行 `CALL db.index.vector.queryNodes(...)` 做 Stage-1 KNN 粗篩。
  * `get_all_kgs_concepts()`、`get_public_kgs_concepts()`、`get_all_documents_concepts()` 新增可選參數 `concept_ids`，非 None 時將 Cypher 限定在候選集合內，取代原本的全表 `MATCH`。
  * `services/concept_engine.py` 的 `route_via_two_stage()`：對每個 query concept 呼叫 Stage-1 取候選 id 聯集，再呼叫呼叫端傳入的 `fetch_candidates(ids)` 做 Stage-2；Stage-1 失敗（例如索引未就緒）或候選為空時，自動退回 `fetch_candidates(None)` 全表掃描，行為與優化前完全一致，不影響正確性。
  * 已接入所有路由呼叫點：`routers/agent.py`（`/agent/chat` KG 路由、相似度補充、`/agent/query`）、`routers/world.py`（`/world/chat` 公開 KG 路由與相似度補充）、`routers/search.py`（`/search`）、`services/classify_service.py`（暫存區文件分類）。
  * `core/constants.py` 新增 `CONCEPT_COARSE_TOP_K = 100`。
  * 測試：`tests/services/test_concept_engine.py::TestRouteViaTwoStage`（Stage-1 失敗退回全表、候選為空退回全表、候選 id 聯集去重、Stage-1 中途例外退回全表）。
* **學術來源**：
  * Nogueira, R., & Cho, K. (2019). *"Passage Re-ranking with BERT."* arXiv:1901.04085. (經典二階段粗精篩檢索架構)
  * Robertson, S., & Zaragoza, H. (2009). *"The Probabilistic Relevance Framework: BM25 and Beyond."* Foundations and Trends in Information Retrieval, 3(4), 333-389.

---

## 10. 核心流程優化與虛擬碼設計草稿 (Algorithm Pseudocode & Workflow Integration — Design Drafts vs. Actual Implementation)

> **狀態聲明**：本章的五個 class（`FederatedOntologyMapper`、`TemporalDecayRerankEngine`、`GraphCoTReasoningEngine`、`ActiveRetrievalController`、`TwoStageVectorRetrievalEngine`）**未以 class 形式出現在任何 `.py` 檔案中**，僅為設計草稿；其中 3 項已落地（皆拆成獨立函式而非單一 class）：②`TemporalDecayRerankEngine`（`services/svo_service.py::_temporal_decay()`）、③`GraphCoTReasoningEngine`（簡化版，`routers/agent.py` 門檻觸發式加深查詢）、⑤`TwoStageVectorRetrievalEngine`（`repositories/concept_repo.py::vector_search_concept_ids()` + `services/concept_engine.py::route_via_two_stage()`/`route_kgs()`/`route_documents()`）；④`ActiveRetrievalController` 只落地一個遠遠更小的子集（見該節「與原虛擬碼的差異」）；①`FederatedOntologyMapper` 仍為設計草稿，尚未寫入程式碼（見第9節②，建議暫緩理由不變）。

為了便於工程團隊直接在現有 GraphRAG 代碼庫中落地上述優化方向，本章提供五個核心流程的代碼設計藍圖與 Python 虛擬碼草稿：

### ① 聯邦本體 Schema 對齊與 Cypher 動態轉換 (Ontology Schema Translation) — 🔵 設計草稿，尚未實作（對應第9節②，建議暫緩，見該節理由）

跨分片（Shard）並行查詢時，解決不同分片本體命名不一致的對齊流程：

```python
Algorithm 1: Dynamic Federated Schema Alignment
Input: original_cypher (原始Cypher語句), shard_id (目標分片ID), alignment_rules (對齊對照表)
Output: translated_cypher (轉換後符合標準本體的Cypher語句)

1.  Initialize: translated_cypher <- original_cypher
2.  Extract: shard_rules <- alignment_rules.get(shard_id, None)
3.  If shard_rules is None:
4.      Return translated_cypher  # 無需對齊，直接返回
5.  For each (orig_relation, standard_relation) in shard_rules:
6.      # 使用正則表達式匹配並替換 Cypher 中的關係 Label，例如 -[:REL]->
7.      pattern <- RegexMatchPattern("-[:" + orig_relation + "]->")
8.      replacement <- "-[:" + standard_relation + "]->"
9.      translated_cypher <- ReplaceAll(translated_cypher, pattern, replacement)
10. Return translated_cypher
```

### ② 融入時序衰減的圖譜引導重排流程 (Temporal Decay Reranking) — ✅ 已落地（套用位置與原虛擬碼不同，見第9節⑥說明）

在圖譜引導的物理段落重排公式中，除了 SVO 命中加分，加入基於發布時間差的連續衰減權重：

> **與原虛擬碼的差異**：實際落地**沒有**建立獨立的 `TemporalDecayRerankEngine` class，且套用對象不是「物理段落（chunk）重排」而是「SVO 事實排序」——理由與差異細節見第9節⑥。核心衰減公式 `exp(-rate * delta_days)` 與函式簽名精神保留，實作為 `services/svo_service.py::_temporal_decay()`。

```python
Algorithm 2: Temporal-Decay Graph-Guided Reranking
Input: cosine_sim (語意相似度), query_hits (Query概念命中數), svo_hits (圖譜SVO實體命中數), 
       doc_publish_time (文件發布時間), current_time (當前時間), decay_rate (每日衰減率, 預設 0.005)
Output: final_score (時效性修正後的最終重排評分)

1.  Calculate time difference: delta_days <- Max(0, DayDifference(current_time, doc_publish_time))
2.  Compute temporal decay factor:
    # 衰減公式採用指數衰減: decay = e^(-decay_rate * delta_days)
    decay_factor <- exp(-decay_rate * delta_days)
3.  Compute base ranking score:
    base_score <- cosine_sim + (query_hits * 0.40) + (svo_hits * 0.10)
4.  Apply temporal decay:
    final_score <- base_score * decay_factor
5.  Return final_score
```

### ③ 圖譜鏈式思考路徑推理流程 (Graph Chain-of-Thought / G-CoT) — ✅ 已落地（簡化版）

> **實際落地（門檻觸發式簡化版）**：以下 `GraphCoTReasoningEngine` 虛擬碼保留作為原始理論設計參考，
> 實際程式碼採用更輕量的做法，見 `routers/agent.py` 的 `_bfs_kg`/`_merge_bfs_results` 與
> `_SVO_SPARSE_FACT_THRESHOLD`（門檻 = 3）：BFS 以 `req.svo_hops` 跑一次後，若命中事實數低於門檻
> 且未達最大跳數（3），用**同一組種子詞**（不重新用 LLM 選路）加深一跳（`hops+1`）重查並合併結果。
> 差異：不逐跳呼叫 LLM 決定下一個節點，換取零額外 LLM 延遲/成本，代價是不能像原始設計一樣
> 「主動選擇最相關的鄰居」，而是單純擴大 BFS 半徑。測試見
> `tests/routers/test_rag_quality.py::TestSVOFactInjection::test_sparse_bfs_triggers_deeper_hop_graph_cot`。

```python
Algorithm 3: Graph Chain-of-Thought Multi-Hop Path Reasoning
Input: start_entity (起點實體), user_query (用戶問題), max_hops (最大跳數, 預設 3), LLM_client (推理模型)
Output: reasoning_path (邏輯推理路徑)

1.  Initialize: current_node <- start_entity, reasoning_path <- [start_entity]
2.  For hop in range(0 to max_hops - 1):
3.      # 向 Neo4j 查詢當前節點的一跳鄰居與語意關係
4.      neighbors <- QueryNeighborsFromGraph(current_node)
5.      If neighbors is Empty:
6.          Break
7.      # 構造提示詞，詢問 LLM 決定下一跳方向
8.      prompt <- BuildPrompt(user_query, reasoning_path, neighbors)
9.      decision_json <- LLM_client.GenerateJSON(prompt)
10.     next_node <- decision_json["next_target"]
11.     reasoning_path.append(next_node)
12.     
13.     # 判斷是否滿足終止條件
14.     If decision_json["is_stop"] is True or next_node not in neighbors:
15.         Break
16.     current_node <- next_node
17. Return reasoning_path
```

### 演算法 4：自適應不確定性驅動的主動檢索 (Active Retrieval Controller)

* **學術目的**：在生成答案過程中監控不確定性（不確定性/機率值），動態決策何時暫停生成並向圖譜重新發起檢索。
* **複雜度**：時間複雜度 $O(T)$，其中 $T$ 為當前已生成 Token 數量。

```python
Algorithm 4: Uncertainty-Driven Active Retrieval Control
Input: current_tokens (已生成的Token列表), token_confidences (各Token的生成置信度), 
       confidence_threshold (安全置信度閾值, 預設 0.65)
Output: trigger_retrieval (布林值，是否觸發新一輪檢索)

1.  If token_confidences is Empty:
2.      Return False
3.  # 1. 計算當前生成片段的平均 Token 置信度 (平均概率)
4.  avg_confidence <- Sum(token_confidences) / Length(token_confidences)
5.  # 2. 判斷置信度是否低於安全閾值（代表模型開始產生幻覺）
6.  If avg_confidence < confidence_threshold:
7.      Return True
8.  # 3. 語意觸發：檢查生成的文本中是否出現了未在上下文定義的孤立實體 (NER偵測)
9.  latest_text <- Join(current_tokens)
10. If ContainsUnreferencedEntities(latest_text) is True:
11.     Return True
12. Return False
```

### 演算法 5：二階段向量粗篩-精篩檢索引擎 (Two-Stage Retrieval Engine)

* **學術目的**：解決大規模數據下，將全載向量拉入記憶體進行 $O(N \times M)$ 運算引發的性能與 CPU 瓶頸。
* **複雜度**：資料庫粗篩時間複雜度由近鄰索引優化為 $O(\log N)$，記憶體精篩複雜度降為 $O(K)$ 常數級（$K=100$）。

```python
Algorithm 5: Two-Stage Coarse-to-Fine Concept Node Retrieval
Input: query_concept_vec (問題概念向量), top_k_coarse (粗篩候選數, 預設 100), 
       query_interest (問題興趣度), query_professional (問題專業度)
Output: refined_results (排序後的 Top-K 匹配概念列表)

1.  # ── STAGE 1: 粗篩 (Coarse Retrieval in Database) ──
2.  # 利用 Neo4j 內建的 Vector Index (C++底層高速運算) 篩選出餘弦相似度最高的前 K 個節點
3.  raw_candidates <- ExecuteCypher(
        "CALL db.index.vector.queryNodes('concept_vector_index', $top_k, $query_vector) 
         YIELD node, score RETURN node"
    )
4.  If raw_candidates is Empty:
5.      Return []
6.  # ── STAGE 2: 精篩 (Fine Reranking in Memory) ──
7.  Initialize: refined_results <- []
8.  For each cand in raw_candidates:
9.      # 在記憶體中針對這 100 個候選者進行複雜的屬性對齊 (Align) 與強度振幅 (Mag) 比對
10.     refined_score <- ComputeGatingMatchScore(
            query_concept_vec, cand.vector, 
            query_interest, query_professional, cand.interest, cand.professional
        )
11.     refined_results.append({cand.id, cand.name, refined_score})
12. # 依精篩得分降序排列，回傳最終匹配結果
13. SortDescending(refined_results, key=refined_score)
14. Return refined_results
```

### ④ 自適應不確定性驅動的主動檢索 (Active Retrieval Controller) — 🟡 部分落地（範圍遠小於本虛擬碼，見第9節④說明）

在 LLM 串流生成答案的過程中，監控不確定性（Entropy/Confidence）自發決策是否觸發圖譜檢索：

> **與原虛擬碼的差異**：`ActiveRetrievalController` class（含逐 token confidence 監控 `evaluate_generation_step()`、孤立實體偵測 `_contains_unreferenced_entities()`）**沒有落地**。實際落地是規模小得多的子集——`routers/agent.py` 精煉迴圈中對累積文字做 `_NO_INFO_RE` 關鍵詞比對以提前結束單輪生成，不涉及 token 級信心分數或跨 Provider 的 logprobs 存取。差異原因與範圍見第9節④。

```python
class ActiveRetrievalController:
    def __init__(self, confidence_threshold: float = 0.65):
        self.threshold = confidence_threshold

    def evaluate_generation_step(
        self,
        current_generated_tokens: list[str],
        token_confidence_scores: list[float]
    ) -> bool:
        """
        決策是否需要暫停生成，向圖譜數據庫發送新檢索請求以補齊資訊。
        """
        # 1. 計算當前生成片段的平均 Token 置信度 (平均機率)
        if not token_confidence_scores:
            return False

        avg_confidence = sum(token_confidence_scores) / len(token_confidence_scores)

        # 2. 判斷置信度是否低於安全閾值（代表模型開始胡言亂語或產生幻覺）
        if avg_confidence < self.threshold:
            return True

        # 3. 語意觸發：檢查生成的文本中是否出現了未在上下文定義的孤立關鍵實體
        latest_text = "".join(current_generated_tokens)
        if self._contains_unreferenced_entities(latest_text):
            return True

        return False

    def _contains_unreferenced_entities(self, text: str) -> bool:
        # 使用簡單的正則或 NER 偵測是否有孤立名詞（實際項目中可對接預訓練 NER 模型）
        return "幻覺邊界實體" in text
```

### ⑤ 二階段向量粗篩-精篩檢索引擎 (Two-Stage Retrieval Engine) — ✅ 已落地（實作與虛擬碼有出入見下）

利用資料庫內建向量索引進行 C++ 級粗篩，在內存中進行精細重排，避免 memory 瓶頸。

> **與原虛擬碼的差異**：實際落地**沒有**建立獨立的 `TwoStageVectorRetrievalEngine` class，而是拆成兩個更貼合現有架構的函式：`repositories/concept_repo.py::vector_search_concept_ids()`（Stage-1，純資料庫查詢）與 `services/concept_engine.py::route_via_two_stage()`（協調 Stage-1/Stage-2，透過高階函式 `fetch_candidates` 讓既有的 4 個路由呼叫點各自决定 Stage-2 要抓 KG 概念還是文件概念，避免為每種場景各寫一個 class）與其上層封裝 `route_kgs()`/`route_documents()`。索引名稱也不同：程式碼延用既有的 `concept_q_vector`（非虛擬碼中的 `concept_vector_index`）。詳細落地位置見第9節⑧。

```python
class TwoStageVectorRetrievalEngine:
    def __init__(self, neo4j_driver, concept_engine):
        self.neo4j_driver = neo4j_driver
        self.concept_engine = concept_engine

    async def retrieve_matching_concepts(self, query_concept_vec: list[float], top_k_coarse: int = 100) -> list[dict]:
        """
        雙階段檢索流程：資料庫粗篩 (Stage-1) -> 內存對齊精篩 (Stage-2)
        """
        # ────────── STAGE 1: 粗篩 (Coarse Retrieval in Database) ──────────
        # 使用 Neo4j C++ 內建的向量索引 db.index.vector.queryNodes 進行粗篩
        # 僅提取餘弦相似度最高的前 top_k_coarse (例如 100) 個節點，避免全表掃描
        coarse_query = (
            "CALL db.index.vector.queryNodes('concept_vector_index', $top_k, $query_vector) "
            "YIELD node, score "
            "RETURN node.id as id, node.name as name, node.vector as vector, "
            "node.interest as interest, node.professional as professional, score"
        )

        raw_candidates = await self.neo4j_driver.run(coarse_query, {
            "top_k": top_k_coarse,
            "query_vector": query_concept_vec
        })

        if not raw_candidates:
            return []

        # ────────── STAGE 2: 精篩 (Fine Reranking in Memory) ──────────
        # 在 Python 內存中，僅針對 Stage-1 篩選出的 100 個候選者進行
        # 複雜的維度對齊 (Align) 與強度振幅 (Mag) 比對計算
        refined_results = []
        for cand in raw_candidates:
            refined_score = self.concept_engine.compute_match_score(
                query_vector=query_concept_vec,
                candidate_vector=cand["vector"],
                query_interest=1.0,         # 假設問題的權重
                query_professional=1.0,     # 假設問題的專業度
                cand_interest=cand["interest"],
                cand_professional=cand["professional"]
            )

            refined_results.append({
                "id": cand["id"],
                "name": cand["name"],
                "coarse_score": cand["score"],
                "fine_score": refined_score
            })

        # 依精篩評分進行排序，回傳最終的匹配結果
        refined_results.sort(key=lambda x: x["fine_score"], reverse=True)
        return refined_results
```

---

## 11. 實證評估與驗證框架 (Empirical Evaluation Framework)

### (RAG System Evaluation & Empirical Validation)

> **落地狀態（含端到端生產驗證的誠實修正）**：`run_evaluation.py` 已完整落地並跑滿 5 題測試集，範圍仍小於理論設計（未用 RAGAS 官方函式庫、蒙地卡羅收斂驗證未做）。首次完整結果發現 Hybrid 在 2/5 個案表現反直覺，追查後修復了評估腳本本身的檢索組裝問題，並定位、修復了 `services/svo_service.py::query_svo_facts()` 的 BFS 事實檢索缺陷（高扇出樞紐實體會把 `LIMIT` 佔滿）——在評估腳本使用的 2 文件小型 KG 上，Hybrid 平均 Faithfulness 從 0.47 提升至 **0.92**。**但對正式運行中的 `kg2-api` 容器做端到端真實驗證後發現：同一個修復在真實的 909 文件生產 KG 規模下並不足以解決問題**（案例 1、3 的原始問句在真實 `/agent/chat` 上仍雙雙失敗）——修復方向正確且在小規模場景已驗證有效，但泛用實體在大規模同質語料下的精準檢索仍是待解決的開放問題，詳見下方【實際落地與已知缺口】末段的端到端驗證與修正說明。

為了解決學術界對於 GraphRAG 系統在「問答品質」、「防幻覺能力」與「檢索效率」上的質疑，本專案設計了完備的實證評估與驗證框架，將系統表現量化。

#### 【三大核心評估指標 (The RAG Triad)】

本系統採用 **RAGAS (Retrieval Augmented Generation Assessment)** 評估架構，透過 LLM-as-a-judge 機制對問答流程進行三維度評量：

1. **Faithfulness（忠實度 / 幻覺抑制率）**：
   * **定義**：生成答案中的所有事實陳述，是否皆能從檢索到的 Context（包含圖譜 SVO 與物理 Chunks）中找到依據。
   * **公式概念**：
     $$\text{Faithfulness} = \frac{\text{源自 Context 的答案事實數}}{\text{答案中總事實數}}$$
   * **驗證目的**：評估「防幻覺過濾器」與「自我精煉機制（Self-Refinement）」的有效性。
2. **Answer Relevance（答案相關性）**：
   * **定義**：生成答案是否切中用戶問題的核心意圖，無冗餘資訊。
   * **驗證目的**：評估 `build_query_concepts` 提取關鍵詞的準確度。
3. **Context Recall（檢索召回率）**：
   * **定義**：檢索出的 Context 是否包含解答該問題所需的全部關鍵資訊。
   * **驗證目的**：評估「圖譜門控路由（Concept Gating）」與「1-2 跳 BFS 圖遍歷」是否產生漏檢。

#### 【消融實驗設計 (Ablation Study Framework)】

為了驗證本系統「雙層路由」與「符號-物理回溯」的設計優越性，專案建立了以下對比消融實驗：

| 實驗組 | 路由與檢索機制 | 檢索內容 | 自我精煉迴圈 | 評估指標預期表現 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (純向量)** | 僅向量搜尋 (Cosine Similarity) | 僅原始 Chunks (Top-K) | 關閉 | Context Recall 較低（面對多跳問題時容易遺漏） |
| **Baseline 2 (純圖譜)** | 概念路由 + BFS 圖遍歷 | 僅 SVO 翻譯句子 | 關閉 | Answer Relevance 高，但 Faithfulness 易因細節丟失而降低 |
| **本系統 (Hybrid RAG)** | **雙層路由（Concept + BFS）** | **SVO + 物理座標回溯 Chunk** | **開啟 (閾值 0.65)** | **三項指標（Recall、Faithfulness、Relevance）均達到最優** |

#### 【自我精煉與收斂性驗證】

* **驗證方法**：於測試集進行 1000 次蒙地卡羅模擬問答，記錄系統在不同閾值下，觸發第 1 輪、第 2 輪、第 3 輪精煉（補充 Chunks）的比例與最終收斂率。
* **目標**：確保系統在滿足回答精準度的前提下，平均推理輪數接近 1.2 輪，避免無限循環並控制 Token 開銷。
* **落地狀態**：❌ 尚未實作。目前沒有任何腳本執行蒙地卡羅模擬或統計收斂率，`_MAX_REFINE_ROUNDS=3` 目前只在第 3 節、第 9 節③④中以個案方式驗證行為正確，未有量化的「平均推理輪數」數據。

#### 【實際落地與已知缺口（經修復與重跑驗證的最終結果）】

`run_evaluation.py` 是本節「消融實驗設計」的真實落地。與上述理論設計仍有已知落差（見下方「與設計的差異」）。落地過程經歷三次迭代：首次僅跑完 1 題且報告結論與數據矛盾 → 補齊測試資料後完整跑滿 5 題但發現 2 個個案反直覺 → 追查根因並修復評估腳本本身的檢索組裝邏輯 → 重跑驗證確認修復有效。以下為修復後的最終結果。

**與設計的差異**：
* **未使用 RAGAS 函式庫**：實際落地改為手寫的 **LLM-as-a-Judge**（單一 LLM，`.env` 設定為本機 Ollama `qwen2.5:7b`，程式內另保留 `gemini-3.5-flash` 作為可選 Provider）直接對 Faithfulness / Relevance / Context Recall 三項打分，未使用 RAGAS 的標準化評測管線與官方 Prompt 模板，評分方法論尚未經過與 RAGAS 官方實作的交叉驗證。
* **單次取樣，無重複驗證**：每題每種配置只呼叫一次裁判 LLM，沒有多次取樣取平均或投票機制，小型本地模型單次評分的抽樣雜訊會造成個別題目分數在重跑之間波動（見下方 Vector 基準在三次重跑間 0.28→0.00→0.10 的波動，即為此雜訊的實例，非程式邏輯改變所致）。

**最終結果（`evaluation_report.md`，含 BFS 生產修復）**：

| 檢索生成配置 | Faithfulness | Relevance | Context Recall | 平均耗時 |
| :--- | :---: | :---: | :---: | :---: |
| Pure Vector RAG | 0.10 | 0.14 | 0.00 | 26.94s |
| Pure Graph RAG | 0.67 | 0.72 | 0.45 | 15.33s |
| **Proposed Hybrid GraphRAG（本系統）** | **0.92** | **0.96** | **0.90** | 35.53s |

整體平均分支持設計預期的排序（Hybrid ≫ Graph > Vector），且分數已遠優於前兩輪迭代的結果。報告的「量化結論」段落依 `avg_scores` 動態生成。

**評估腳本本身的三處修復（`run_evaluation.py`，範圍限於評估腳本，不影響生產環境 `/agent/chat`）**：
1. **動態生成量化結論**：取代原本寫死、與實際分數矛盾且混入雜訊詞的結論文字，改為依 `avg_scores` 逐項比較後生成誠實的結論。
2. **Hybrid 模式重用生產環境的 `_pick_relevant_chunks` 重排邏輯**：取代原本單純截斷 `doc.content[:1000]`。
3. **`sim_quota` 動態配額限定僅套用於 Hybrid 模式**：比照生產環境「圖譜驅動文件已覆蓋部分配額時，相似度補充應縮減」的邏輯，Pure Vector RAG 基準維持固定配額（3 篇）。

上述三項修復把 Hybrid 從首次完整結果的 0.47 提升到 0.60，但案例 1、4 當時仍分數異常。

**核心生產環境修復（`services/svo_service.py::query_svo_facts`，⚠️ 影響 `/agent/chat` 正式問答路徑，非僅評估腳本）**：
追查案例 1 根因後（見下方【個案問題與根因】），發現 BFS 事實檢索的 Cypher 有兩個實質缺陷，已直接在生產程式碼修復：
1. **`RETURN DISTINCT` 誤把 `source_doc_id`／`confidence`／`created_at` 也納入去重鍵**，導致同一句語意事實（如「勞動基準法 -[違反]→ 延長工作時間」）只要在多份文件中各出現一次，就會被當成多筆「相異」列各自佔用一個 `LIMIT` 名額。修法：改為對 `(subject, rel_type, verb, object)` 分組聚合（`WITH ... max(confidence), collect(DISTINCT source_doc_id)[0..3] ...`），讓 `LIMIT` 作用在真正相異的語意事實數量上，每組事實仍保留最多 3 個來源文件供回溯（`_MAX_DOCS_PER_FACT`）。
2. **即使分組去重，KG 內的高扇出樞紐實體（如「勞動基準法」本身連到數十個不同案例公司）仍會用相異事實把 `LIMIT` 佔滿**，擠掉低扇出、高特定性 seed（如公司名）的事實。修法：用 `CALL (seed) {...}` 相關子查詢（Neo4j 5.26 支援，經 smoke test 確認）對每個 seed 各自的扇出先截斷到 `_PER_SEED_FACT_LIMIT = 20` 筆（依 confidence 排序），確保每個 seed（無論是泛用樞紐詞還是特定實體）都有機會貢獻自己的事實配額。
3. 配合 `run_evaluation.py` 的 `doc_ids` 回溯配額從寫死的 `[:4]` 提高到比照生產環境 `_graph_quota = min(top_k*2, 10)` 的邏輯（修復後 doc_ids 排序已改善，但目標文件仍可能落在第 4 名之後）。

**驗證方法**：`tests/services/test_svo_service.py`、`tests/integration/test_neo4j_integration.py`（連線真實 Neo4j）全數通過（唯二失敗為與本次修改無關的既有 `_build_ft_query` 測試期望落差）；並用兩支唯讀診斷腳本重跑案例 1、3 的 BFS 查詢，確認修復前後目標文件從「完全不在回傳的 95 個 doc_ids 中」變成「穩定出現在回傳列表中」。

**修復效果（最終數據）**：Hybrid 從 0.60/0.62/0.60 躍升至 **0.92/0.96/0.90**；Graph 亦因共用同一份 `query_svo_facts` 而受益，從 0.42/0.60/0.35 提升到 0.67/0.72/0.45。逐案對照：

| 案例 | 修復前 Hybrid | 修復後 Hybrid |
| :--- | :---: | :---: |
| 案例 1（臺灣湯淺電池／宜蘭廠） | 0.00 / 0.00 / 0.00 | **0.60 / 0.80 / 0.50** |
| 案例 2 | 0.33 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 案例 3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 案例 4（勞基法延長工時法定上限） | 0.33 / 0.20 / 0.00 | **1.00 / 1.00 / 1.00** |
| 案例 5 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |

案例 4 從嚴重失敗轉為滿分，直接證實根因（BFS 被泛用詞彙淹沒）與修復方向正確。

**⚠️ 案例 1 仍未達滿分（不影響整體結論，作為已知殘留限制記錄）**：
* 案例 1 的 Hybrid 從完全失敗（0/0/0）大幅改善至 0.6/0.8/0.5，但仍非滿分——診斷顯示目標文件（`409bf5f7-...`）與法條文件（`fe7f547b-...`）雖然已出現在修復後的 doc_ids 列表中，但分別落在第 6、7 名（0-indexed），仍在 `run_evaluation.py` 提高後的配額（10）之內可被納入，但排序仍不是最前面，可能導致部分細節仍未被 `_pick_relevant_chunks` 優先選中。
* **後續待辦**：(1) 評估 `doc_ids` 的排序邏輯是否應改為依「seed 特定性」而非單純 `confidence DESC` 排序，讓源自高特定性 seed（公司名等）的文件排到更前面；(2) 待評分穩定性有更多把握後，考慮多次取樣取平均以降低單次裁判雜訊對 Vector 基準的影響；(3) 待分數穩定後，執行【自我精煉與收斂性驗證】的蒙地卡羅模擬並回填實際收斂率數據（目前仍完全未開始）。
* 對應程式碼：`services/svo_service.py::query_svo_facts()`（生產程式碼，含 `_MAX_DOCS_PER_FACT`、`_PER_SEED_FACT_LIMIT` 兩個新常數）、`run_evaluation.py`、`evaluation_report.md`、`evaluation_results.csv`；測試資料：`workspace/kg_勞動力合規與排班知識圖譜/_source/taiwan_yuasa_violation.txt` 與 `labor_standards_act_32.txt`。

#### 【端到端生產驗證：發現修復在真實規模下仍不足】

對正式運行中的 `kg2-api` 容器做端到端真實驗證：

* **驗證方法**：直接對正式運行中的 `kg2-api` 容器（`http://localhost:8002/agent/chat`）發送與案例 1、案例 3 完全相同的問句（非評估腳本的簡化模擬，而是含完整精煉迴圈的真實 SSE 問答）。
* **結果**：**兩題皆失敗**，且回傳的 `sources` 完全相同（10 份標題為 `violation_lsa_XXXX` 的文件，不含目標的臺灣湯淺電池案），連信心評估都達到 0.85（模型自認「確定找不到」）。案例 3 在評估腳本（`run_evaluation.py`）的簡化模擬中拿到滿分，但在真實 `/agent/chat` 卻同樣失敗——這代表評估腳本與生產環境的差距比先前認知的更大，不只是 `_pick_relevant_chunks` 重排/`sim_quota` 配額細節差異。
* **根因**：程式化查詢確認這個生產 KG（`勞動力合規與排班知識圖譜`，904df1c5）實際擁有 **909 份文件**，其中約 907 份是自動產生、命名為 `violation_lsa_XXXX` 的泛用勞基法違規案例（大量結構相同、只換公司名稱的合成資料），只有 2 份是本文件先前分析用的特定測試文件（`taiwan_yuasa_violation`、`labor_standards_act_32`）。`chunk_store/` 目錄只保存了少數文件的本地快取，並未反映 Neo4j 圖譜中實際掛載的完整文件規模。
* **為何前述修復在此規模下不夠**：`_PER_SEED_FACT_LIMIT = 20` 是針對「少數種子、少數泛用鄰居」的情境設計，在只有個位數/十位數競爭文件時效果顯著（見診斷腳本與評估腳本的驗證）。但當泛用樞紐實體（如「勞動基準法」）真實連到 **907 個結構相同、置信度相近的候選文件**時，任何固定的每 seed 上限（無論 20 或更高）都只能隨機/依 confidence 排序挑出其中一小部分，目標文件（1/907）在數學上仍有極高機率被排除在外——這不是「還沒調對參數」的問題，而是**目前的 BFS 策略在同質、大規模泛用實體場景下，本質上不具備區分「這 907 篇裡最相關的是哪一篇」的機制**。
* **誠實結論**：先前修復在小規模場景（診斷腳本、評估腳本用的 2 文件 KG）確實有效且已驗證，但**尚未解決真實生產 KG（909 文件規模）下的同一類問題**。這是規模造成的落差，而非修復方向錯誤：`_MAX_DOCS_PER_FACT`／`_PER_SEED_FACT_LIMIT` 的分組去重與扇出截斷仍是正確、必要的基礎建設（避免了原本「同一句話佔滿 LIMIT」的明確錯誤），只是不足以單獨解決大規模同質實體場景下的精準檢索問題。
* **後續待辦**：
  1. 需要一個**依詞彙特定性/稀有度排序或篩選 seed** 的機制——例如：先計算每個抽取詞彙在 KG 內的匹配基數（entity/document 命中數），命中數低的詞彙（如公司名）應被視為「高特定性」而優先驅動 BFS，命中數高的詞彙（如「勞動基準法」）應被降權或僅作為候選補充，而非與特定詞彙同等看待。
  2. 或者：導入全文檢索/向量相似度作為 SVO BFS 的前置過濾——先用問句整體的語意相似度或關鍵詞排序候選文件，只在信心不足時才用泛用法條詞彙擴大搜尋範圍（類似第 9 節③ Graph-CoT 簡化版「稀疏才加深」的精神，但反過來用於「稀疏才擴大範圍」）。
  3. 在正式導入前，應先用這個 909 文件的真實 KG（而非 2 文件的診斷/評估用 KG）建立一個更真實的評估基準，避免下一輪修復又只在小規模場景驗證就誤判為「已解決」。
* 對應程式碼：本次為唯讀端到端驗證（`curl` 呼叫正式運行中的 `kg2-api` 容器 + 一支唯讀 Python 診斷腳本查詢文件總數），未修改任何程式碼。

---

## 12. 補充學術文獻與背景知識 (Supplementary Academic References)

為了加強本專案在學術發表或專利申請時的學術背書，建議參考並引用以下文獻：

* **RAG 系統評估與 RAGAS 框架**：
  * *Es, S., Shahul, H., Pradeep, A., et al. (2023). "Ragas: Automated Evaluation of Retrieval Augmented Generation."* arXiv:2309.15217.
  * 奠定了使用 LLM 對 RAG 進行自動化無監督評估的理論基礎，是本系統實證評估的學術引用來源。
* **多跳推理（Multi-hop Reasoning）基準**：
  * *Yang, Z., Qi, P., Zhang, S., et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering."* EMNLP 2018.
  * 證實了單純的語意相似度檢索在處理多個關聯實體時的瓶頸，為本系統導入「BFS 圖遍歷」提供了強力的問題背景支撐。
* **關係圖注意力網絡 (Relational Graph Attention Networks)**：
  * *Wang, X., Ji, H., Shi, C., et al. (2019). "Heterogeneous Graph Attention Network."* WWW 2019.
  * 提供了節點特徵與關係特徵共同進行 Attention 加權計算的數學理論，支持本系統未來「圖拓撲感知共嵌入空間」的優化設計。

---

## 12.1 學術引用可信度總表（被引用次數）

比照第 7 節 GitHub 專案已使用的「星數」可信度指標，本節補上對應的學術指標——**被引用次數（Citation Count）**，讓讀者能區分全文引用的文獻中，哪些是奠基性經典、哪些是已站穩腳步的近期工作、哪些是太新尚未累積引用的前沿論文。完整查詢方法與逐篇明細，見 `docs/報告/02_參考文獻獨立查核報告.md` 第六節。此處僅列摘要分級，數字為近似值，非精確即時數字。

| 分級 | 定義 | 代表文獻 |
|---|---|---|
| 奠基性經典 | >5,000 次引用 | Attention Is All You Need (~160,000+)、RAG (~7,453)、GraphSAGE (27,921)、SimCLR (22,211)、node2vec (~11,400+)、Louvain (~11,000-20,000+) |
| 已站穩腳步 | 500-5,000 次引用 | Sparsely-Gated MoE、BM25 and Beyond (4,481)、Leiden (4,102)、Epidemic Algorithms (2,553)、Self-Refine (~2,548)、Microsoft GraphRAG (~902)、FedX (501) 等 |
| 新興但已有迴響 | 50-500 次引用 | AgreementMakerLight (468)、FLARE (341)、Self-RAG（估計偏高）、Ragas (98) 等 |
| 太新尚未累積 | <50 次或查無數字，多為 2023 年後論文 | Graph-CoT (30，可能低估，為第9節③已落地功能的唯一理論依據) |

**重要提醒**：「太新尚未累積」不等於「品質可疑」，只是索引尚未跟上發表速度。目前全文僅 Graph-CoT 一篇引用次數低於 100，其餘皆為 ≥100 次引用的文獻，或有機構背書的 GitHub 專案（ProVe、ms-graphrag-neo4j）。
