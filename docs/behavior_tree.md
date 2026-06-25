# 智慧知識庫 — 完整 Behavior Tree

> 本文件描述系統所有主要行為與子行為，並標注優化整合點（★ = 已實作優化，☆ = 待整合優化）。

## 符號說明

| 符號 | 節點類型 | 語意 |
|------|----------|------|
| `[→]` | Sequence | 依序執行所有子節點，任一失敗則中止 |
| `[?]` | Selector | 依序嘗試子節點，任一成功則停止 |
| `[∀]` | ForEach | 對集合中每個元素執行子樹 |
| `[！]` | Parallel | 同時執行所有子節點 |
| `[D]` | Decorator | 修改單一子節點的行為 |
| `◇` | Condition | 純條件判斷（不改變狀態） |
| `□` | Action | 葉節點動作 |
| `★` | 已整合優化 | 此節點已有對應優化實作 |
| `☆` | 待整合優化 | 建議新增的優化點 |

---

## 根節點

```
[ROOT] 智慧知識庫系統
├── [?] 觸發來源判斷
│   ├── [→] A. 知識圖譜管理行為
│   ├── [→] B. 文件攝取行為
│   ├── [→] C. 暫存區分群行為
│   ├── [→] D. SVO 知識圖譜建構行為
│   ├── [→] E. 智慧問答行為（核心）
│   └── [→] F. 系統維護行為
```

---

## A. 知識圖譜管理行為

```
A. 知識圖譜管理
├── A1. [→] 建立 KG（POST /knowledge-graphs）
│   ├── ◇ KG 名稱不重複
│   ├── □ 建立 workspace 資料夾（kg_{slug}/_source, /_text）
│   ├── [?] Neo4j 資料庫模式選擇
│   │   ├── [→] Enterprise 模式
│   │   │   ├── □ create_kg_database(db_name)         # 為此 KG 建立獨立 DB
│   │   │   ├── □ CREATE INDEX entity_name
│   │   │   └── □ CREATE FULLTEXT INDEX entity_name_ft
│   │   └── □ fallback：使用主資料庫（db_name=""）    ★ 自動降級，Community 版友好
│   ├── □ kg_repo.create() → KnowledgeGraph 節點
│   └── □ add_watch_dir(source_dir)                   # 啟動目錄監聽
│
├── A2. [→] 更新 KG
│   ├── □ kg_repo.update(name, description, is_public)
│   └── [?] 公開狀態同步
│       ├── [→] is_public=True → generate_skill + upsert_skill   ★ KB Skill 自動同步
│       └── □ is_public=False → remove_skill(kg_id)
│
├── A3. [→] 刪除 KG
│   ├── □ drop_kg_database(db_name)
│   ├── □ kg_repo.delete(kg_id)
│   └── [?] delete_files=True → shutil.rmtree(folder_path)
│
├── A4. [→] 刷新 KG 路由層概念（POST /{kg_id}/refresh）
│   ├── □ refresh_kg_concepts(kg_id)                  # 見 A4-Detail
│   └── [?] is_public → generate_skill + upsert_skill ★ refresh 後自動更新 registry
│
└── A4-Detail. refresh_kg_concepts()
    ├── [?] 來源 1：Document EFFECTIVE ConceptNode
    │   ├── □ concept_repo.get_all_documents_concepts()
    │   └── □ 過濾屬於此 KG 的 doc_id 集合
    ├── [?] 來源 2 fallback（若來源1為空）：Entity 高頻名詞 top-50
    │   ├── ◇ concept_names 為空
    │   └── □ Cypher: MATCH Entity by type IN [...] ORDER BY degree DESC LIMIT 50
    ├── [?] 來源 3（可選）：text LLM 概念提取
    │   ├── ◇ text 參數非空
    │   └── □ extract_concepts(text)
    ├── [∀] 每個概念名稱
    │   ├── □ embedding.encode(name)
    │   ├── □ concept_repo.get_or_create(name, domain, vec)
    │   └── □ concept_repo.init_kg_concept(kg_id, name, INTEREST_INIT, PROFESSIONAL_INIT)
    └── □ concept_repo.sync_kg_effective(kg_id)
        □ kg_repo.refresh_counts(kg_id)
```

---

## B. 文件攝取行為

```
B. 文件攝取
├── B1. [?] 攝取路徑選擇
│   ├── [→] B1a. API 直接建立（POST /documents，含 content）
│   │   ├── □ repo.create(title, content, file_path, file_type)
│   │   ├── □ background: extract_and_init_document_concepts(doc.id, content)
│   │   └── [?] kg_id 非空 → kg_repo.add_document(kg_id, doc.id)   ★ 自動關聯 KG
│   │
│   ├── [→] B1b. 檔案上傳（POST /documents/upload）
│   │   ├── □ 驗證副檔名在 SUPPORTED_EXTENSIONS
│   │   ├── □ 寫入 temp 檔
│   │   └── □ ingest_file(tmp_path)                 # → B2
│   │
│   ├── [→] B1c. 批次目錄（POST /documents/ingest-dir 或 run_ingest.py）
│   │   └── [∀] 目錄下每個支援格式檔案 → ingest_file(f)   # → B2
│   │
│   ├── [→] B1d. 搬移並匯入（POST /documents/move-and-ingest）
│   │   ├── [∀] source_dir 每個檔案
│   │   │   ├── □ shutil.move(src, dst)
│   │   │   └── [?] 檔名衝突 → 加流水號
│   │   └── [∀] 已移動檔案 → ingest_file(dest_path)
│   │       └── [?] 匯入失敗 → shutil.move(dst, src)   # 自動還原
│   │
│   ├── [→] B1e. 暫存區分配（classify_service.assign_document_to_kg）
│   │   ├── □ shutil.move(_staging/, kg_folder/_text/)
│   │   ├── □ doc_repo.create(title, content)
│   │   ├── □ extract_and_init_document_concepts(doc.id)
│   │   ├── □ kg_repo.add_document(kg_id, doc.id)
│   │   ├── □ refresh_kg_concepts(kg_id)
│   │   └── □ asyncio.create_task(_auto_svo())       # 背景觸發 SVO
│   │
│   └── [→] B1f. 目錄監聽（file_watcher_service）
│       ├── ◇ workspace/_source/ 有新檔案
│       └── □ ingest_file(new_file)                  # → B2
│
└── B2. ingest_file(file_path)
    ├── ◇ 檔案存在
    ├── [?] 格式解析
    │   ├── [→] .pdf → _read_pdf()                   # → B3 三層備援
    │   ├── [→] .md / .txt → _read_text()            # 多編碼嘗試
    │   ├── [→] .docx → _read_docx()
    │   ├── [→] .pptx → _read_pptx()
    │   ├── [→] .doc → _read_doc() (COM/Word)
    │   └── [→] .ppt → _read_ppt() (COM/PPT)
    ├── □ _sanitize_text()                           ★ 過濾控制字元、合併多餘空行
    ├── ◇ content.strip() 非空
    ├── □ repo.create(title, content, file_path, file_type)
    └── □ extract_and_init_document_concepts(doc.id, content)

B3. _read_pdf() 三層備援
├── [?] 層 1：pypdf.PdfReader → extract_text
├── [?] 層 2（若層1 < 50字）：pdfminer.high_level.extract_text
└── [?] 層 3（若層2仍 < 50字）：OCR                 ★ easyocr 繁中+英文，2× 縮放
    ├── □ 每頁 fitz.Matrix(2.0,2.0) 渲染
    └── □ reader.readtext(img, detail=0, paragraph=True)
```

---

## C. 暫存區分群行為

```
C. 暫存區分群
├── C1. [→] 分析單一文件（POST /staging/{filename}/classify）
│   ├── □ build_query_concepts(text[:4000])
│   ├── [∀] 所有 KG 的 ConceptNode
│   │   └── □ compute_match_score(doc_concepts, kg_concepts)
│   ├── □ 過濾 score < CLASSIFY_MIN_THRESHOLD
│   ├── □ 排序候選 KG（降冪）
│   └── [?] auto_assign=True 且 top_score ≥ threshold
│       └── □ assign_document_to_kg(filename, top.kg_id)   # → B1e
│
├── C2. [→] 批次自動分配（POST /staging/classify-all）
│   └── [∀] _staging/ 中每個 .txt → C1
│
├── C3. [→] LLM 分群建議（POST /staging/suggest-kgs）
│   ├── [!] 並行掃描
│   │   ├── □ 讀取 _staging/*.txt 清單
│   │   └── □ doc_repo.get_orphan_documents()        # 未關聯 KG 的孤立文件
│   ├── □ 建立 d1,d2... 短索引（避免 LLM 難以回傳 UUID）
│   ├── □ LLM stream：分群 + 命名 + 一句話描述
│   ├── □ re.search JSON array 解析
│   └── □ 未分配者歸入「其他」
│
├── C4. [→] 確認分群方案（POST /staging/approve-suggestion 或 /knowledge-graphs/auto-cluster/confirm）
│   ├── [∀] 每個 cluster
│   │   ├── □ create_kg(name, description)           # → A1
│   │   ├── [∀] staging files → assign_document_to_kg  # → B1e
│   │   └── [∀] doc_ids → kg_repo.add_document + refresh_kg_concepts
│   └── ☆ 建議：confirm 後自動觸發 build-graph（目前須手動）
│
└── C5. [→] 手動指定（POST /staging/{filename}/assign）
    └── □ assign_document_to_kg(filename, kg_id)     # → B1e
```

---

## D. SVO 知識圖譜建構行為

```
D. SVO 知識圖譜建構
├── D1. [→] 觸發建圖（POST /{kg_id}/build-graph 或 run_build_kg.py）
│   ├── ◇ KG 存在
│   └── □ build_graph_for_kg(kg_id, doc_ids, force_rebuild)   # → D2

D2. build_graph_for_kg()  [AsyncIterator<BuildProgress>]
├── [?] force_rebuild=True → _clear_kg_entities(kg_id)
├── □ 取得待處理文件清單
│   └── ◇ 增量模式：只取 svo_processed_at=null 的文件
├── [∀] 每份文件 doc (semaphore=_SVO_CONCURRENCY=2)   ★ 並發控制
│   ├── □ 讀取 doc.content
│   ├── D3. sentence_chunk(doc.id, content)
│   │   ├── □ re.split([。！？!?\n], text)
│   │   ├── □ 每 _SENTENCES_PER_CHUNK=5 句組成一個 chunk
│   │   ├── □ chunk_id = f"{doc_id}_{idx:04d}"      ★ 1-based idx，唯一 provenance key
│   │   └── □ 記錄 char_start / char_end
│   ├── [∀] 每個 chunk                               ★ 批次 SVO 提取
│   │   ├── D4. extract_svo_from_text(chunk.text)
│   │   │   ├── [?] JSON 模式
│   │   │   │   ├── □ LLM 生成 JSON（含 6 欄 SVO）
│   │   │   │   └── □ _parse_svo_json()
│   │   │   ├── [?] fallback：pipe 格式
│   │   │   │   └── □ _parse_svo_lines()
│   │   │   └── □ _filter_hallucinated()             ★ 過濾幻覺三元組
│   │   │       ├── ◇ rel_type 在 _VALID_REL_TYPES (30種)
│   │   │       └── ◇ entity_type 在 _VALID_TYPES (13種)
│   │   ├── [D] retry up to _MAX_CHUNK_RETRIES=2 (exponential backoff)   ★ 失敗重試
│   │   │   └── [?] 失敗 → log warning，doc 保留 svo_processed_at=null 供下次補跑
│   │   └── D5. merge_triples_to_neo4j(triples, kg_id, doc_id, db_name, chunk_id)
│   │       ├── □ UNWIND batch per rel_type           ★ 批次 UNWIND，效能優化
│   │       ├── □ MERGE (s:Entity {name, kg_id})
│   │       ├── □ MERGE (o:Entity {name, kg_id})
│   │       ├── □ MERGE (s)-[r:REL_TYPE]->(o)
│   │       └── □ ON CREATE SET source_chunk_ids=[chunk_id]
│   │           ON MATCH SET source_chunk_ids = CASE...  ★ chunk_id provenance 陣列
│   ├── D6. chunk_store.write(kg_id, doc_id, chunks)  ★ 檔案持久化
│   │   ├── □ chunk_store/{kg_id}/{doc_id}/chunk_{idx:04d}.json
│   │   └── □ _docs/{doc_id} ref 檔（記錄所屬 kg_id）
│   └── □ doc_repo.mark_svo_processed(doc.id)
├── □ SSE: yield BuildProgress (chunk_start/chunk_done/done/error)
├── □ apply_type_labels(kg_id)                       # → D7
└── [?] KG is_public → generate_skill + upsert_skill  ★ build 完成自動更新 registry

D7. apply_type_labels(kg_id)
├── □ Cypher: MATCH Entity → 用 LLM 判斷語意型別
└── □ SET e.type = inferred_type                     # Concept/Algorithm/Tool 等13種
```

---

## E. 智慧問答行為（核心）

```
E. 智慧問答 [POST /agent/chat → SSE]
│
├── E1. [→] Step 1：問題概念提取
│   ├── □ extract_concepts(question)                 # LLM：最多 8 個核心概念
│   ├── [∀] 每個概念名稱
│   │   └── □ embedding.encode(name)                # local sentence-transformers
│   ├── □ 組裝 query_concepts [{name, q_vector, interest=0.8, professional=0.8}]
│   └── ◇ query_concepts 非空（否則 error）
│
├── E2. [→] Step 2：KG 雙層路由
│   ├── □ concept_repo.get_all_kgs_concepts()        # 取所有 KG 的 ConceptNode
│   ├── [∀] 每個 KG
│   │   └── □ compute_match_score(query_concepts, kg_concepts)
│   │       ├── [∀] 每對 (query_concept, kg_concept)
│   │       │   ├── □ _cosine(q_vector, kg_vector)
│   │       │   ├── □ _alignment(interest/professional scores)
│   │       │   └── □ _magnitude(scores)
│   │       └── □ weighted_score = Σ(cos × align × mag) / Σ(mag)
│   ├── □ 過濾 score < KG_ROUTE_THRESHOLD
│   ├── □ 排序選出 top MAX_KG_PER_QUERY 個 KG
│   └── □ SSE: yield {kg_route: [...]}
│
├── E3. [→] Step 3：SVO 知識層（BFS 圖遍歷）
│   ├── ◇ req.use_svo=True 且 selected_kgs 非空
│   ├── [∀] 每個 selected_kg
│   │   └── □ query_svo_facts(kg_id, terms, hops, limit=50, db_name)
│   │       ├── E3a. [?] 快取命中                    ★ BFS in-memory cache TTL=300s
│   │       │   ├── ◇ key=(kg_id, sorted_terms, hops, min_confidence) 在 _bfs_cache
│   │       │   └── □ 直接回傳快取結果
│   │       └── E3b. Neo4j BFS 查詢
│   │           ├── □ FULLTEXT INDEX entity_name_ft 種子擴展（FT 查詢每個 term）
│   │           ├── □ 1-hop 展開鄰居（擴充種子集合）
│   │           ├── □ BFS MATCH 1..hops 遍歷
│   │           ├── □ 去重 + 按 confidence DESC 排序
│   │           ├── □ 額外查詢：從 entity 節點收集 source_chunk_ids（按實體頻率排序）
│   │           └── □ 寫入 _bfs_cache + 設定 TTL
│   ├── □ 合併去重 svo_facts / source_doc_ids / chunk_ids
│   └── □ SSE: yield {svo_facts: [...]}
│
├── E4. [→] Step 4：混合文件檢索
│   ├── □ 從 svo_facts 解析 SVO 實體名稱（regex）→ svo_entity_names（boost_terms）
│   ├── E4a. [→] 圖譜驅動（Graph-Driven）
│   │   └── [∀] graph_doc_ids（SVO BFS 指向的文件）
│   │       ├── □ doc_repo.get_by_id(doc_id)
│   │       ├── ◇ _is_readable(content)              ★ 亂碼過濾（CJK比例判斷）
│   │       └── □ _pick_relevant_chunks(content, query_concepts, boost_terms=svo_entity_names)
│   │           # → E4c
│   └── E4b. [→] 相似度補充（Similarity Fallback）
│       ├── □ concept_repo.get_all_documents_concepts()
│       ├── □ 過濾：只取 allowed KG 範圍內的文件
│       ├── [∀] 候選文件
│       │   └── □ compute_match_score(query_concepts, doc_concepts)
│       ├── □ 過濾 score < _SIM_MIN_SCORE=0.38
│       ├── □ 排序；取 top sim_quota（= max(1, top_k - graph_count)）
│       ├── ◇ _is_readable(content)                  ★ 亂碼過濾
│       └── □ _pick_relevant_chunks(content, query_concepts, boost_terms=svo_entity_names)
│           # → E4c
│
├── E4c. _pick_relevant_chunks() 細節
│   ├── □ NFKC 正規化（OCR CJK部首字元統一）          ★ PDF OCR 相容性修復
│   ├── [?] 段落分割策略
│   │   ├── □ 嘗試 \n\n 分割
│   │   └── [?] 段落數<5 或平均段落>400字 → 降級為 \n 分割
│   ├── □ 合併成 400 字以內的 chunk（buf 策略）
│   ├── [?] 大文件預篩（len(chunks) > _MAX_CHUNKS_EMBED=200）  ★ 效能優化
│   │   ├── □ keyword 命中 chunk（含前後相鄰各1個）
│   │   └── □ 均勻抽樣補齊至 200 塊
│   ├── □ embedding.encode_batch(candidate_texts)    ★ 批量 embed，10-50x 加速
│   ├── [∀] 每個候選 chunk 計算分數
│   │   ├── □ emb_score = max cosine(chunk_vec, q_vec)
│   │   ├── □ q_hits = Σ(q_name in chunk) × 0.4
│   │   ├── □ svo_hits = Σ(svo_term in chunk) × 0.10  ★ SVO 實體 boost
│   │   └── □ enum_bonus = 0.25 if (hit>0 and _ENUM_RE.match(chunk))  ★ 列舉加分
│   │       # _ENUM_RE: 中文數字/阿拉伯數字/圓點/Markdown標題前綴
│   ├── □ 二次排序：q_hits 命中數（降冪）→ 分數（降冪）
│   └── □ 貪婪選取直到 max_chars
│
├── E5. [→] Step 5：自我精煉迴圈（Self-Refine Loop）
│   ├── ◇ req.use_svo=True 且 svo_chunk_ids 非空（否則跳至 E6 直接串流）
│   ├── [D] for round in range(_MAX_REFINE_ROUNDS=3)
│   │   ├── □ _build_rag_prompt(question, svo_facts, contexts, extra_chunks)
│   │   │   ├── □ graph_docs 排在 similarity_docs 之前
│   │   │   ├── [?] extra_chunks 非空 → 加入 [補充原文片段] 區塊
│   │   │   └── □ 末尾加信心評估指令：{"confidence": 0.85}
│   │   ├── □ llm.generate(prompt)                   # 同步呼叫（非串流）
│   │   ├── □ _extract_confidence(raw)               ★ 尾端 regex 剝離信心 JSON
│   │   │   └── _CONFIDENCE_RE = r'\{"confidence":\s*([\d.]+)[^}]*\}\s*$'
│   │   ├── [?] confidence ≥ _CONFIDENCE_THRESHOLD=0.65
│   │   │   └── □ final_answer = clean；break
│   │   ├── □ next_ids = svo_chunk_ids - used_cids（取前 _CHUNKS_PER_ROUND=3 個）
│   │   ├── [?] next_ids 為空 → break
│   │   ├── □ chunk_store.read_many(next_ids)        ★ ChunkStore 讀取原文
│   │   ├── □ extra_chunks.extend(chunk.text)
│   │   └── □ SSE: yield {refine: {round, confidence_before, chunks_added}}
│   └── □ SSE: yield {status: "generating"}
│
└── E6. [→] Step 6：串流輸出
    ├── [?] final_answer 非空（精煉後有足夠信心）
    │   └── □ 分段 emit 緩衝答案（每批 80 字元）
    ├── [?] 否則（精煉輪次耗盡 or 未觸發精煉）
    │   └── □ _build_rag_prompt() → llm.stream() → yield token
    └── □ SSE: yield {done: true}
```

---

## F. 系統維護行為

```
F. 系統維護
├── F1. [→] Entity 語意型別標籤套用（run_label_kg.py）
│   └── □ apply_type_labels(kg_id)                   # → D7
│
├── F2. [→] RELATED_TO 邊重新分類（run_reclassify_related_to.py）
│   ├── [?] --dry-run → 僅預覽不寫入
│   ├── □ MATCH Entity-[RELATED_TO]->Entity
│   ├── □ LLM 判斷更精確的語意類型
│   └── [?] 非 dry-run → MERGE 新關係邊 + DELETE 舊 RELATED_TO
│
├── F3. [→] 清除 KG 知識層（DELETE /{kg_id}/graph）
│   ├── □ _clear_kg_entities(kg_id, db_name)
│   └── □ kg_repo.refresh_counts(kg_id)
│
└── F4. [→] Agent 健康檢查（GET /agent/health）
    ├── □ doc_count / kg_count / entity_count
    └── □ llm_provider / embedding_provider 回報
```

---

## 優化整合清單

### ★ 已實作優化

| # | 優化項目 | 位置 | 說明 |
|---|----------|------|------|
| 1 | **Sentence-Aware Chunking** | `chunk_store.py::sentence_chunk()` | 按句子邊界分割（5句/chunk），chunk_id=`{doc_id}_{idx:04d}` |
| 2 | **Chunk Provenance** | `svo_service.py::merge_triples_to_neo4j()` | Entity/Relation 節點記錄 `source_chunk_ids[]` 陣列 |
| 3 | **ChunkStore 持久化** | `services/chunk_store.py` | 檔案型持久化 `chunk_store/{kg_id}/{doc_id}/chunk_{idx:04d}.json` |
| 4 | **自我精煉迴圈** | `routers/agent.py` lines 478-515 | 最多3輪，信心 ≥0.65 停止，每輪補3個原文 chunk |
| 5 | **信心分數剝離** | `_extract_confidence()` | `_CONFIDENCE_RE` 尾端錨定，夾緊至 [0,1] |
| 6 | **BFS in-memory 快取** | `svo_service.py::_bfs_cache` | TTL=300s，key=(kg_id, sorted_terms, hops, min_conf) |
| 7 | **批量 embed** | `_pick_relevant_chunks()::encode_batch` | 替代逐塊呼叫，10-50x 加速 |
| 8 | **列舉加分 (enum_bonus)** | `routers/agent.py::_ENUM_RE` | 含 Markdown 標題前綴，命中+0.25 |
| 9 | **SVO 實體 boost** | `_pick_relevant_chunks()` | svo_hits×0.10，SVO 實體引導 chunk 選取 |
| 10 | **大文件關鍵詞預篩** | `_pick_relevant_chunks()` | >200塊先 keyword filter+均勻抽樣，再 embed |
| 11 | **亂碼過濾** | `_is_readable()` | CJK/ASCII 比例判斷，過濾 Big5/GBK 誤讀 |
| 12 | **PDF 三層備援** | `ingestion_service.py::_read_pdf()` | pypdf → pdfminer → easyocr |
| 13 | **OCR NFKC 正規化** | `_pick_relevant_chunks()` | 統一 CJK 部首字元（⾓→角）讓 keyword match 正常 |
| 14 | **SVO 提取並發控制** | `svo_service.py::_SVO_CONCURRENCY=2` | asyncio.Semaphore 限制並發 |
| 15 | **指數退避重試** | `svo_service.py::_MAX_CHUNK_RETRIES=2` | SVO 提取失敗最多重試2次 |
| 16 | **增量建圖** | `build_graph_for_kg()` | 只處理 `svo_processed_at=null` 的文件 |
| 17 | **Enterprise/Community 自動降級** | `knowledge_graph_service.py::create_kg()` | Enterprise 版用獨立 DB，否則 fallback 主 DB |
| 18 | **圖譜驅動優先** | `routers/agent.py` Step 4a | graph_doc 排在 similarity_doc 之前進入 prompt |
| 19 | **相似度補充縮減** | `routers/agent.py` | 已有圖譜文件時 sim_quota = top_k - graph_count |
| 20 | **KG 路由多來源概念** | `refresh_kg_concepts()` | Doc EFFECTIVE → Entity fallback → LLM text 提取 |
| 21 | **DocumentCreate.kg_id** | `models/document.py` | POST /documents 同時傳 kg_id 可自動關聯 |
| 22 | **build-graph 後自動 registry** | `routers/knowledge_graph.py` | build 完成若 is_public 自動更新 KB Skill |
| 23 | **refresh 後自動 registry** | `routers/knowledge_graph.py::refresh()` | refresh 後自動 upsert_skill |
| 24 | **自動分群 SVO 觸發** | `classify_service.py::assign_document_to_kg()` | 分配後背景 asyncio.create_task(_auto_svo()) |
| 25 | **LLM 短索引分群** | `knowledge_graph_service.py::auto_cluster_kgs()` | 用 d1,d2 代替 UUID，避免 LLM 回傳錯誤 |

### ☆ 建議新增優化

| # | 優化項目 | 建議位置 | 說明 |
|---|----------|----------|------|
| 1 | **approve-suggestion 後自動 build-graph** | `staging.py::approve_suggestion()` | 目前 approve 後須手動觸發建圖；可在 assign 完成後背景觸發 build_graph_for_kg |
| 2 | **BFS 快取 Redis 化** | `svo_service.py::_bfs_cache` | 目前僅 in-memory，多 worker 下失效；可替換為 Redis 或 Memcached |
| 3 | **精煉輪次 SSE 進度 token 預覽** | `routers/agent.py` Step 5 | 精煉時 LLM generate 為黑盒；可改為 stream，讓前端看到中間思考過程 |
| 4 | **query_svo_facts 多 KG 並行** | `routers/agent.py` Step 3 | 目前多 KG 逐一 BFS；可 asyncio.gather() 並行 |
| 5 | **概念提取結果快取** | `concept_engine.py::extract_concepts()` | 相同文件重複提取；可 LRU cache by doc_id+domain |
| 6 | **chunk embed 向量持久化** | `chunk_store.py` | 目前每次查詢重新 embed；可將 chunk vector 存入 chunk JSON，避免重複計算 |
| 7 | **Staging 自動分類 Cron** | 新增 `services/staging_cron.py` | 定期掃描 _staging/，對新檔案自動執行 classify（auto_assign=True） |
| 8 | **Entity 去重合併** | `svo_service.py::merge_triples_to_neo4j()` | 同義詞/別名實體（如「強化學習」vs「RL」）目前各自獨立 MERGE；可加語意相似度合併 |
| 9 | **Confidence calibration** | `_extract_confidence()` + `_CONFIDENCE_THRESHOLD` | 目前 LLM 自評信心，但不同模型校準不一；可加 calibration layer 或自適應門檻 |
| 10 | **多輪對話 Context Window** | `routers/agent.py::ChatRequest` | 目前每次問答獨立；可加 conversation history 至 prompt，支援追問 |

---

## 資料流總覽

```
使用者文件
    │
    ▼
[B. 文件攝取]
  Parser (PDF/DOCX/PPTX/TXT)
  ↓ _sanitize_text
  ↓ OCR fallback
  ↓ DocumentRepository.create()
  ↓ extract_and_init_document_concepts()
         │ LLM: extract 8 concepts
         │ embed each concept
         │ ConceptNode.get_or_create()
         │ EFFECTIVE edge: Doc→Concept
         ▼
[D. SVO 建圖] ─────────────────────────────────────────────
  sentence_chunk()  → SentenceChunk [chunk_id, text]
       ↓
  extract_svo_from_text()  → SVOTriple [subj, rel, obj]
       ↓
  merge_triples_to_neo4j()  → Entity nodes + 30種語意邊
       ↓ source_chunk_ids[]
  chunk_store.write()  → chunk_store/{kg_id}/{doc_id}/*.json

使用者問題
    │
    ▼
[E. 智慧問答] ─────────────────────────────────────────────
  build_query_concepts()
  ↓ extract + embed
  ↓
  KG 路由層 compute_match_score()
  ↓ top KG 選擇
  ↓
  query_svo_facts() [BFS + Cache]
  ↓ facts, source_docs, chunk_ids
  ↓
  Hybrid Retrieval
  ├── Graph-Driven: SVO指向文件 → _pick_relevant_chunks()
  │     batch_embed + keyword/svo_boost + enum_bonus
  └── Similarity: concept match → _pick_relevant_chunks()
  ↓
  Self-Refine Loop (max 3 rounds)
  ├── generate() → _extract_confidence()
  ├── conf ≥ 0.65 → 停止
  └── conf < 0.65 → chunk_store.read_many() → extra_chunks
  ↓
  llm.stream() → SSE token stream → 使用者
```

---

## 關鍵常數一覽

| 常數 | 值 | 位置 | 說明 |
|------|-----|------|------|
| `_SENTENCES_PER_CHUNK` | 5 | svo_service.py | 每個 SVO chunk 的句子數 |
| `_SVO_CONCURRENCY` | 2 | svo_service.py | 同時處理的文件數（Semaphore） |
| `_MAX_CHUNK_RETRIES` | 2 | svo_service.py | SVO 提取失敗重試次數 |
| `_BFS_CACHE_TTL` | 300s | svo_service.py | BFS 結果快取存活時間 |
| `_VALID_REL_TYPES` | 30種 | svo_service.py | 允許的語意關係類型 |
| `_VALID_TYPES` | 13種 | svo_service.py | 允許的實體語意類型 |
| `_CONFIDENCE_THRESHOLD` | 0.65 | routers/agent.py | 精煉停止門檻 |
| `_MAX_REFINE_ROUNDS` | 3 | routers/agent.py | 最大精煉輪次 |
| `_CHUNKS_PER_ROUND` | 3 | routers/agent.py | 每輪補充的 chunk 數 |
| `_CHUNK_SIZE` | 400 chars | routers/agent.py | chunk 目標字元數 |
| `_MAX_CHUNKS_EMBED` | 200 | routers/agent.py | 超過此數啟用關鍵詞預篩 |
| `_GARBLED_THRESHOLD` | 0.3 | routers/agent.py | 非可讀字元比例門檻 |
| `_SIM_MIN_SCORE` | 0.38 | routers/agent.py | 相似度補充最低門檻 |
| `KG_ROUTE_THRESHOLD` | constants.py | core/constants.py | KG 路由最低分數 |
| `MAX_KG_PER_QUERY` | constants.py | core/constants.py | 每次問答最多查詢的 KG 數 |
| `CLASSIFY_AUTO_THRESHOLD` | constants.py | core/constants.py | 暫存區自動分配門檻 |
| `concept_extraction_max` | 8 | core/config.py | 每次 LLM 提取概念上限 |
