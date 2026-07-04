# 系統架構說明

## 整體架構

```
┌─────────────────────────────────────────────────────────────┐
│                     瀏覽器（Vanilla JS）                       │
│  問答 │ KG 管理 │ 暫存區 │ 文件上傳 │ 深色模式                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI 後端                               │
│  /agent  /knowledge-graphs  /documents  /staging  /search   │
└───┬──────────────┬─────────────────┬───────────────┬────────┘
    │              │                 │               │
    ▼              ▼                 ▼               ▼
 LLM Provider  Embedding       Neo4j 主資料庫    KG 專用 DB (x N)
 (ollama/      Provider        KnowledgeGraph    Entity 節點
  openai/      (local/         Document 節點     RELATION 邊
  anthropic/   openai/         ConceptNode       （Enterprise）
  gemini/grok) ollama)         路由層
```

---

## 雙層知識結構

系統並存兩套 Neo4j 結構：

### 路由層（主資料庫）

```cypher
(:KnowledgeGraph {id, name, description, db_name, entity_count})
    -[:EFFECTIVE {interest_score, professional_score}]->
(:ConceptNode {name, domain, q_vector})

(:Document {id, title, content, svo_processed_at})
    -[:EFFECTIVE]->(:ConceptNode)

(:KnowledgeGraph)-[:CONTAINS]->(:Document)
```

**用途**：問答時快速路由到相關 KG，使用 cosine similarity 比對 `q_vector`

### 知識層（KG 專用資料庫）

```cypher
(:Entity {id, name, type, kg_id})
    -[:IS_A | :CAUSES | :USES | ...]->    ← 30 種語意類型作為 edge label
(:Entity)
    {verb, source_doc_id, confidence, created_at}
```

**用途**：BFS 圖遍歷，取出結構化知識事實供 RAG 使用

---

## 雙層問答流程

```
POST /agent/chat { question }
         │
         ▼
① build_query_concepts(question)
   → LLM 提取問題中的關鍵概念詞（最多 8 個）
         │
         ▼
② compute_match_score(query_concepts, kg_concepts)
   → 對每個 KG 計算 cosine similarity
   → 選出 score > KG_ROUTE_THRESHOLD 的 KG（最多 5 個）
   → SSE: {"kg_route": [...]}
         │
         ▼
③ query_svo_facts(query_concepts, selected_kgs)
   → 全文索引搜尋 Entity 節點（BFS seed）
   → 廣度優先遍歷 1-2 跳
   → 收集知識事實：「主詞 [語意類別:動詞] 受詞」
   → SSE: {"svo_facts": [...]}
         │
         ▼
④ 圖譜驅動文件選取
   → 由 SVO 事實中的 source_doc_id 直接定位相關文件
   → 補充向量相似度搜尋（top-k 文件）
   → SSE: {"sources": [...]}
         │
         ▼
⑤ RAG Prompt → LLM 串流
   → 知識事實 + 文件原文片段 → 構成 Prompt
   → LLM.stream() → SSE: {"token": "..."}
```

---

## SVO 知識提取流程

```
文件文字
   │ _chunk_text()（每段約 1500 字）
   ▼
[並行處理] asyncio.gather() with Semaphore(_SVO_CONCURRENCY=2)
   │ extract_svo_from_text(chunk)
   │   → JSON mode：LLM 回傳 [{subject, subject_type, rel_type, verb, object, object_type}]
   │   → 失敗備援：pipe mode（| 分隔格式）
   │   → 自動重試 2 次（exponential backoff）
   ▼
merge_triples_to_neo4j(triples, kg_id, doc_id, db_name)
   → 依 rel_type 分組
   → UNWIND 批次 MERGE
   → ON MATCH: confidence += 1
   ▼
_set_doc_svo_processed(doc_id)
   → 設定 svo_processed_at = datetime()
   → 下次增量跑時跳過此文件
```

**失敗處理**：chunk 最終失敗時不設 `svo_processed_at`，下次增量跑自動補提取

---

## 30 種語意關係類型

所有關係直接以語意類型作為 Neo4j edge label（非屬性），支援高效 pattern matching：

```
層級/組成：IS_A  PART_OF  CONTAINS  INSTANCE_OF
因果/效應：CAUSES  PREVENTS  ENABLES  IMPROVES  INHIBITS
功能/操作：USES  REQUIRES  PRODUCES  IMPLEMENTS  REPLACES  EXTENDS
比較：    CONTRASTS  SIMILAR_TO  OUTPERFORMS
描述/定義：DEFINED_AS  HAS_PROPERTY  MEASURED_BY  APPLIES_TO
時序：    PRECEDES  FOLLOWS  CO_OCCURS
資料流：  INPUTS  TRANSFORMS
歸屬/解決：CREATED_BY  SOLVES  RELATED_TO（最後手段）
```

BFS 查詢使用 OR pattern：
```cypher
MATCH (seed)-[:IS_A|PART_OF|CONTAINS|...|RELATED_TO*1..2]-(neighbor)
```

---

## Neo4j 資料庫分層

### Enterprise 模式（預設）

每個 KG 有獨立的 Neo4j 資料庫（`db_name` 欄位存儲）：

```
主資料庫（neo4j）
├── KnowledgeGraph 節點：CLIArchitecture (db: kgcliarchi0049706b)
├── KnowledgeGraph 節點：SpatialIntelligence (db: kgspatiali0b46abc6)
└── Document / ConceptNode 節點（共用）

kgcliarchi0049706b 資料庫
└── Entity 節點 + 30 種 RELATION 邊

kgspatiali0b46abc6 資料庫
└── Entity 節點 + 30 種 RELATION 邊
```

**優點**：KG 完全隔離，查詢不互相干擾，可獨立備份

### Community 版 Fallback

若 Neo4j 不支援多資料庫，自動 fallback：
- 所有 Entity 存主資料庫
- 以 `{kg_id: $kg_id}` 屬性區隔
- 功能不受影響，但效能較低

---

## Provider 工廠

```python
# core/providers/factory.py
init_providers() → (EmbeddingProvider, LLMProvider)
```

依 `.env` 的 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 選擇實作，單例模式（全域快取）：

```
LLM_PROVIDER=ollama    → OllamaLLMProvider
             openai    → OpenAILLMProvider
             anthropic → AnthropicLLMProvider
             gemini    → GeminiLLMProvider
             grok      → GrokLLMProvider

EMBEDDING_PROVIDER=local   → LocalEmbeddingProvider (sentence-transformers)
                   openai  → OpenAIEmbeddingProvider
                   ollama  → OllamaEmbeddingProvider
```

---

## 文件生命週期

```
原始檔案（PDF / DOCX / PPTX / MP3 / MP4...）
   │ transcribe_file()
   ▼
暫存區（_staging/*.txt）
   │ classify_document() / auto_cluster_kgs()
   ▼
KG 工作區（workspace/kg_xxx/_text/*.txt）
   │ build_graph_for_kg()
   ▼
Neo4j Entity + RELATION（知識圖譜）
```

**File Watcher**：`services/file_watcher_service.py` 監控 `workspace/` 目錄，
新檔案自動建立 `Document` 節點並更新路由層概念（不自動觸發 SVO 提取）。

---

## 關鍵設定常數

| 常數 | 預設值 | 說明 |
|------|--------|------|
| `KG_ROUTE_THRESHOLD` | 0.05 | 路由層最低分數門檻 |
| `MAX_KG_PER_QUERY` | 5 | 每次問答最多路由到幾個 KG |
| `SCORE_THRESHOLD` | 0.70 | 文件向量搜尋相似度門檻 |
| `CONCEPT_EXTRACTION_MAX` | 8 | 每次提取最多概念數 |
| `CONCEPT_COARSE_TOP_K` | 100 | 兩階段檢索 Stage-1 向量粗篩候選數 |
| `_SVO_CONCURRENCY` | 2 | SVO 提取並行 chunk 數 |
| `_SVO_SPARSE_FACT_THRESHOLD` | 3 | Graph-CoT 簡化版：BFS 事實數低於此值時加深一跳重查 |
| `INTEREST_INIT` | 1.0 | ConceptNode 初始 interest 分數 |
| `PROFESSIONAL_INIT` | 1.0 | ConceptNode 初始 professional 分數 |
| BFS hops | 1-2（最深 3） | 圖遍歷跳數（根據 query 長度動態選擇，證據稀疏時加深至 3）|
| BFS cache | TTL 300s，LRU 上限 1000 筆 | BFS 結果快取（超過上限淘汰最舊項目） |
| `MAX_UPLOAD_SIZE_MB` | 50 | 上傳檔案大小上限（`.env`） |
| `CHAT_RATE_LIMIT_PER_MINUTE` | 20 | 問答端點每來源每分鐘請求上限（`.env`，0=停用） |
