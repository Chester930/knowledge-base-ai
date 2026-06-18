# 多場景知識圖譜系統 — 架構計畫書

> 版本：v0.6  
> 目標：將現有單一知識庫擴展為「多場景 KG + 兩層路由 + SVO 深度查詢 + 自動文件分配 + 多 Provider 支援」的協作世界知識系統，並以完整文件發布至 GitHub 供公開使用

---

## 一、願景

任何人都能建立自己的**場景知識圖譜（KnowledgeGraph，簡稱 KG）**。
每個 KG 對應一個資料夾，資料夾裡存放由各種原始格式轉譯出來的文字檔。
使用者觸發「轉換成圖譜」後，系統將文字拆解為 SVO 三元組寫入 Neo4j；
後續資料夾有新檔案或檔案更新，系統自動偵測並合併。

Agent 提問時先用 ConceptNode 路由到相關 KG，再從 SVO graph 做深度查詢整合回答。

---

## 二、雙層知識結構

系統並存兩套結構，各有職責：

| 層 | 結構 | 職責 | 建立時機 |
|----|------|------|----------|
| **路由層** | `ConceptNode` + `[:EFFECTIVE]` | 快速找到相關 KG | 文字檔寫入後自動 |
| **知識層** | `Entity` + 語義關係邊 | 深度查詢、推理、回答 | 使用者手動觸發 |

### 路由層（現有，不動）

```
(:KnowledgeGraph)-[:EFFECTIVE {interest, professional}]->(:ConceptNode {q_vector})
(:Document)-[:EFFECTIVE {interest, professional}]->(:ConceptNode {q_vector})
```

用途：`compute_match_score(query_concepts, kg_concepts)` 決定召回哪些 KG。

### 知識層（新增）

```
(:Entity {id, name, type, kg_id})
    -[:RELATION {verb, source_doc_id, confidence}]->
(:Entity {id, name, type, kg_id})
```

範例：
```
(:Entity{name:"Jetson Orin"})-[:支援]->(:Entity{name:"CUDA 加速"})
(:Entity{name:"強化學習"})-[:需要]->(:Entity{name:"大量運算資源"})
(:Entity{name:"Jetson Orin"})-[:屬於]->(:Entity{name:"NVIDIA 邊緣 AI 平台"})
```

---

## 三、檔案兩層結構

每份文件在**檔案系統**只存兩層，SVO 提取後直接進 **Neo4j**，無中間本地 KG 檔案：

```
Layer 1  原始檔案    slide_01.pdf
              ↓ [轉譯器]
Layer 2  文字檔      slide_01.txt    ← 存在 _staging/（暫存）或 KG _text/（已分配）
              ↓ [SVO 提取 → 記憶體暫存 → 比對現有 KG]
              ↓ 相似度 ≥ 門檻 → MERGE 進匹配的 DB KG
              ↓ 相似度 < 門檻 → 文字檔留在 _staging/ 等待人工處理
Database  Neo4j KnowledgeGraph     ← Entity + RELATION 直接寫入
```

> **設計原則**：SVO 三元組只在提取過程中暫存於記憶體，不落地為本地檔案；
> 一旦找到匹配的 KG 即直接 MERGE 進 Neo4j，找不到則回暫存區。

### 各層說明

| 層 | 格式 | 位置 | 說明 |
|----|------|------|------|
| Layer 1 | 原始格式 | `_source/` | 備份，不直接處理 |
| Layer 2 | `.txt` | `_staging/` 或 KG `_text/` | 轉譯後的純文字，唯一的本地中間檔 |
| Database | Neo4j 節點 | Neo4j | SVO 直接合併後的 Entity + RELATION |

### 工作區資料夾結構

```
<workspace>/
├── _staging/                ← 暫存區：已轉譯、待分配 或 SVO 無匹配 KG
│   ├── report.txt           （等待使用者手動指定 KG）
│   └── lecture.txt
├── kg_edge_ai/              ← KG「邊緣 AI 硬體」
│   ├── _source/             原始檔備份
│   │   └── slide_01.pdf
│   └── _text/               已分配文字檔（分配後即觸發 SVO 提取）
│       └── slide_01.txt
└── kg_rl_theory/
    ├── _source/
    └── _text/
```

### 支援的原始格式與轉譯方式

| 格式 | 轉譯方式 | 備註 |
|------|----------|------|
| `.pdf` | pypdf → pdfminer → OCR（三層備援） | 現有 |
| `.pptx` / `.ppt` | python-pptx / COM | 現有 |
| `.docx` / `.doc` | python-docx / COM | 現有 |
| `.mp4` / `.mp3` | Whisper（語音轉文字） | 後續加入 |
| `.txt` / `.md` | 直接使用 | 現有 |

---

## 四、文件生命週期

```
[原始檔案放入 _source/ 或直接上傳]
    │
    ▼
【轉譯器】掃描資料夾 或 偵測新檔案
    │  PDF / PPTX / DOCX / TXT / MD → .txt
    ▼
[_staging/*.txt]  ← 暫存區
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
【分配器：自動】                         【分配器：手動】
概念比對所有現有 KG                      使用者查看候選清單
    │                                          │
    ├─ 分數 ≥ 自動門檻                         └─ 使用者選擇 KG
    │     → 自動移動
    └─ 分數 < 最低門檻
          → 留在 _staging/（等待手動）
    │
    ▼
[kg_xxx/_text/*.txt]  文字檔進入 KG 資料夾
    │  同步：建立 Neo4j Document + 路由層 ConceptNode
    │
    ▼
【SVO 提取器】（使用者觸發，或分配後自動執行）
    │  LLM 拆解 SVO → 記憶體暫存
    │  比對此 KG 的現有 Entity 節點
    ▼
Neo4j KnowledgeGraph  Entity + RELATION MERGE 寫入
```

### 各步驟觸發方式

| 步驟 | 自動觸發 | 手動觸發 |
|------|---------|---------|
| 轉譯（原始→ .txt）| File Watcher 偵測新檔 | UI 上傳 / API |
| 分配（Staging → KG）| 分數 ≥ 自動門檻 | 使用者確認候選清單 |
| SVO 提取（.txt → DB）| 分配完成後可選擇自動 | 使用者點選「建立知識圖譜」|
| 無匹配 KG 時 | — | 文字檔留 _staging/，使用者手動指定 |

---

## 五、SVO 三元組提取

SVO 提取在記憶體中執行，結果直接寫入 Neo4j，不產生本地中間檔。

### LLM Prompt 策略

對每個文字檔分段（每段約 1000 字）呼叫 LLM：

```
請從以下文字中，提取所有主要知識點，以「主詞 | 動詞 | 受詞」格式輸出。
每行一組，動詞用精簡動詞（支援、屬於、需要、包含、導致、比較…）。
只輸出三元組，不加說明。

文字：
{chunk}
```

### 合併策略（MERGE）

提取後直接以 MERGE 寫入目標 KG，相同三元組不重複建節點，只累加 `confidence`：

```cypher
MERGE (s:Entity {name: $subject, kg_id: $kg_id})
MERGE (o:Entity {name: $object, kg_id: $kg_id})
MERGE (s)-[r:RELATION {verb: $verb, source_doc_id: $doc_id}]->(o)
ON CREATE SET r.confidence = 1, r.created_at = datetime()
ON MATCH SET  r.confidence = r.confidence + 1, r.updated_at = datetime()
```

### 無匹配 KG 的處理

SVO 提取完成後，若所有 KG 相似度均低於門檻：
- SVO 結果**不寫入 Neo4j**（丟棄記憶體暫存）
- 文字檔**留在 `_staging/`**，標記狀態為 `unmatched`
- 使用者可在暫存區查看 `unmatched` 文件，手動指定 KG 後重新觸發 SVO 提取

---

## 六、文件分配機制

當一份文件不知道要放進哪個 KG 時，可透過**概念比對**自動找出最適合的 KG。
此機制與 Agent 路由對稱：路由是「問題找 KG」，分配是「文件找 KG」，同樣使用 `compute_match_score`。

### 6.1 Inbox 概念

系統設有一個**全域收件匣（Inbox）**資料夾，未指定 KG 的文件先放這裡：

```
<workspace>/
├── _inbox/              未分類的原始檔案 / 文字檔
│   ├── report.pdf
│   └── notes.txt
├── kg_edge_ai/          KG「邊緣 AI 硬體」
│   ├── _source/
│   └── _text/
└── kg_rl_theory/        KG「強化學習理論」
    ├── _source/
    └── _text/
```

Inbox 中的文件已完成轉譯並建立 `Document` 節點，但 **尚未** 與任何 KG 建立 `[:CONTAINS]` 關係。

### 6.2 分配流程

```
[Inbox 文件]
    │
    ├─ 1. extract_and_init_document_concepts(doc)
    │       → 取得 doc_concepts（路由層概念）
    │
    ├─ 2. 對所有公開 KG 計算 compute_match_score(doc_concepts, kg_concepts)
    │       → 按分數排序，產出候選清單
    │
    ├─ 3a. [auto_assign=True 且 top_score ≥ threshold]
    │       → 自動移動檔案到 KG 的 _text/ 資料夾
    │       → 建立 (:KnowledgeGraph)-[:CONTAINS]->(:Document)
    │       → 回傳 ClassifyResult{auto_assigned: True}
    │
    ├─ 3b. [auto_assign=False 或 top_score < threshold]
    │       → 回傳候選 KG 排名（top-N）供使用者選擇
    │       → 使用者確認後手動觸發移動
    │
    └─ 3c. [所有 KG 分數 < min_threshold]
            → 建議使用者建立新 KG
            → 或保留在 Inbox 等待人工處理
```

### 6.3 分配 vs 路由的差異

| | 文件分配 | Agent 路由 |
|--|---------|-----------|
| 輸入 | 文件的 ConceptNode | 問題的 query_concepts |
| 比對對象 | 所有 KG 的 ConceptNode | 所有 KG 的 ConceptNode |
| 核心函式 | `compute_match_score` | `compute_match_score` |
| 輸出 | 候選 KG 排名 → 決定文件歸屬 | 候選 KG 排名 → 決定查詢範圍 |
| 觸發 | 文件上傳後（手動或自動）| 使用者提問時（即時）|

### 6.4 資料模型（已有骨架，補充說明）

```python
class ClassifyRequest(BaseModel):
    doc_id: UUID
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # 自動分配門檻
    auto_assign: bool = False                                # True = 達門檻直接分配
    owner_id: str = "default"                               # 新增：限定搜尋自己的 KG

class ClassifyResult(BaseModel):
    doc_id: UUID
    candidates: list[KGCandidate] = []                      # 新增：完整候選清單
    matched_graph_id: UUID | None = None                    # 自動分配時填入
    matched_graph_name: str | None = None
    score: float = 0.0
    auto_assigned: bool = False

class KGCandidate(BaseModel):                               # 新增
    kg_id: UUID
    kg_name: str
    score: float
    top_matched_concepts: list[str] = []

class MoveDocumentRequest(BaseModel):
    doc_id: UUID                                            # 補充 doc_id
    from_graph_id: UUID | None = None                      # None = 從 Inbox 移動
    to_graph_id: UUID
    move_file: bool = True                                  # 同時移動實體檔案
```

### 6.5 API 端點

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/inbox/ingest` | 上傳文件到 Inbox（轉譯 + 建 Document，不指定 KG）|
| `GET` | `/inbox/documents` | 列出 Inbox 中所有未分配文件 |
| `POST` | `/inbox/classify/{doc_id}` | 對單一文件執行分配（回傳候選或自動分配）|
| `POST` | `/inbox/classify-all` | 批次對 Inbox 所有文件執行分配 |
| `POST` | `/documents/{doc_id}/move` | 手動將文件從 Inbox / KG-A 移到 KG-B |

---

## 八、查詢流程（雙層）

```
POST /agent/chat  { question, owner_id? }
    │
    ├─ 1. build_query_concepts(question)
    │
    ├─ 2. [路由層] compute_match_score(query_concepts, KG 概念)
    │       → 選出 score > KG_ROUTE_THRESHOLD 的 KG（最多 MAX_KG_PER_QUERY 個）
    │       → SSE 事件 kg_route 回傳給前端
    │
    ├─ 3. [知識層] 對選中的 KG 做 SVO 圖遍歷
    │       → 從 query_concepts 中的名詞實體出發
    │       → 廣度優先遍歷 1-2 跳，收集相關 Entity + RELATION
    │       → 轉為自然語言「知識片段」
    │
    ├─ 4. [文件層] compute_match_score → top_k 文件原文片段（補充細節）
    │
    └─ 5. RAG prompt = 知識層片段 + 文件層片段 → Ollama 串流回答
```

---

## 九、資料模型（完整）

### Pydantic（`models/knowledge_graph.py`）

```python
class KnowledgeGraphCreate(BaseModel):
    name: str
    description: str
    folder_path: str
    owner_id: str = "default"
    is_public: bool = True

class KnowledgeGraph(BaseModel):
    id: UUID
    name: str
    description: str
    folder_path: str
    owner_id: str
    is_public: bool
    doc_count: int = 0
    entity_count: int = 0       # SVO 實體數
    relation_count: int = 0     # SVO 關係數
    created_at: datetime
    updated_at: datetime

class KnowledgeGraphDetail(KnowledgeGraph):
    top_concepts: list[str] = []    # 路由層前 10 概念
    top_entities: list[str] = []    # 知識層前 10 實體

class SVOTriple(BaseModel):
    subject: str
    verb: str
    object: str
    confidence: int = 1
    source_doc_id: UUID | None = None

class BuildGraphRequest(BaseModel):
    kg_id: UUID
    doc_ids: list[UUID] | None = None   # None = 全部文件
    force_rebuild: bool = False
```

---

## 十、API 設計

### KnowledgeGraph CRUD

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/knowledge-graphs` | 建立 KG |
| `GET` | `/knowledge-graphs` | 列出（?owner_id= 過濾私人）|
| `GET` | `/knowledge-graphs/{id}` | 詳情（含 top concepts / entities）|
| `PUT` | `/knowledge-graphs/{id}` | 更新 name / description / is_public |
| `DELETE` | `/knowledge-graphs/{id}` | 刪除 KG |

### 文件管理

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/knowledge-graphs/{id}/ingest` | 上傳原始檔案（轉譯 → Document → 路由層）|
| `POST` | `/knowledge-graphs/{id}/ingest-dir` | 批次匯入目錄 |
| `GET` | `/knowledge-graphs/{id}/documents` | 列出文件 |
| `DELETE` | `/knowledge-graphs/{id}/documents/{doc_id}` | 移除文件 |

### SVO 知識層

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/knowledge-graphs/{id}/build-graph` | 觸發 SVO 提取並寫入圖譜 |
| `GET` | `/knowledge-graphs/{id}/graph` | 取得 Entity + RELATION 列表 |
| `PUT` | `/knowledge-graphs/{id}/refresh` | 重算 KG 路由層概念 |

### Agent 查詢（更新）

```
POST /agent/chat
{
    "question":          "...",
    "owner_id":          "Alice",   // 選填
    "top_k":             5,
    "max_chars_per_doc": 3000
}
```

SSE 事件序列：
```
data: {"status": "searching"}
data: {"kg_route": [{"name": "邊緣 AI 硬體", "score": 0.42}, ...]}
data: {"svo_facts": ["Jetson Orin 支援 CUDA 加速", ...]}
data: {"sources": [...]}
data: {"status": "generating"}
data: {"token": "..."}   // 串流 token
data: {"done": true}
```

---

## 十一、新增 / 修改檔案清單

### 新增

```
repositories/
  knowledge_graph_repo.py    KG CRUD、Entity/RELATION 讀寫

services/
  knowledge_graph_service.py 建立KG、refresh概念、路由
  svo_service.py             SVO提取、寫入圖譜、圖遍歷
  classify_service.py        文件分配：概念比對 + 檔案移動
  file_watcher_service.py    監控 _text/ 目錄異動

routers/
  knowledge_graph.py         KG CRUD + SVO API
  inbox.py                   Inbox 管理 + 文件分配 API
```

### 修改

```
models/knowledge_graph.py    補齊 Pydantic 模型（KGCandidate、MoveDocumentRequest 等）
repositories/concept_repo.py + get_kg_concepts()
                             + init_kg_concept()
                             + sync_kg_effective()
                             + get_documents_concepts_in_kgs()
core/constants.py            + KG_ROUTE_THRESHOLD = 0.05
                             + MAX_KG_PER_QUERY = 5
                             + CLASSIFY_AUTO_THRESHOLD = 0.30
                             + CLASSIFY_MIN_THRESHOLD = 0.05
routers/agent.py             chat() 加入雙層路由
main.py                      掛載新路由、啟動 File Watcher
```

---

## 十二、實作順序

```
Phase 0 — Provider 抽象層（所有功能的基礎）
  [ ] core/providers/base.py          LLMProvider / EmbeddingProvider ABC
  [ ] core/providers/llm/ollama.py    現有邏輯重構
  [ ] core/providers/llm/openai.py
  [ ] core/providers/llm/anthropic.py
  [ ] core/providers/llm/gemini.py
  [ ] core/providers/llm/grok.py
  [ ] core/providers/embedding/local.py   現有邏輯重構
  [ ] core/providers/embedding/openai.py
  [ ] core/providers/embedding/ollama.py
  [ ] core/providers/factory.py
  [ ] core/config.py                  新增 provider 設定欄位
  [ ] .env.example                    完整設定範本
  [ ] 更新 concept_engine / embedding_service / agent router 使用 factory

Phase 1 — 轉譯器（Transcriber）
  [ ] services/transcribe_service.py
        transcribe_file(src_path) → txt_path
        transcribe_folder(folder_path) → list[txt_path]
        支援：PDF / PPTX / DOCX / TXT / MD（現有邏輯重構）
  [ ] services/file_watcher_service.py
        監控指定資料夾，新檔自動呼叫 transcribe_file → 放入 _staging/
  [ ] routers/transcribe.py
        POST /transcribe/file       上傳單一檔案
        POST /transcribe/folder     指定資料夾批次轉譯
  [ ] main.py                       掛載路由、啟動 Watcher

Phase 2 — 分配器（Distributor）
  [ ] routers/staging.py
        GET  /staging               列出暫存區所有 .txt
        POST /staging/{name}/classify   對單一文件執行分配（回傳候選）
        POST /staging/classify-all      批次自動分配
        POST /staging/{name}/assign     手動指定 KG 並移動
  [ ] services/classify_service.py
        classify_document(txt_path, threshold, auto_assign)
        move_to_kg(txt_path, kg_id)   移動檔案 + 建 Document + 路由層概念

Phase 3 — 資料層（KG 基礎）
  [ ] models/knowledge_graph.py        補齊所有 Pydantic 模型
  [ ] repositories/knowledge_graph_repo.py
  [ ] repositories/concept_repo.py     新增 KG 相關方法
  [ ] core/constants.py                新增路由 + 分配常數
  [ ] services/knowledge_graph_service.py
        create_kg / delete_kg / refresh_kg_concepts
  [ ] routers/knowledge_graph.py       KG CRUD
  [ ] main.py                          掛載路由

Phase 4 — 單文件 KG（Layer 3）
  [ ] services/svo_service.py
        extract_svo_triples(txt_path) → .kg.json
        merge_kg_to_db(kg_json_paths, kg_id)  合併進 Neo4j
  [ ] routers/knowledge_graph.py       新增 build-graph / merge API

Phase 5 — 雙層路由 Agent
  [ ] routers/agent.py                 整合路由層 + 知識層 + 文件層

Phase 6 — UI
  [ ] 轉譯器介面（上傳 / 資料夾監控狀態）
  [ ] 暫存區介面（列表 / 分類建議 / 手動指定）
  [ ] KG 管理介面
  [ ] 問答頁面顯示 kg_route + svo_facts
```

---

## 十三、Provider 抽象層

使用者只需修改 `.env` 即可切換 LLM 與 Embedding 來源，程式碼其餘部分不動。

### 13.1 支援的 Provider

| 類型 | Provider | 識別碼 | 說明 |
|------|----------|--------|------|
| LLM | Ollama（本地）| `ollama` | 完全離線，支援 qwen2.5、llama3、mistral 等 |
| LLM | OpenAI | `openai` | GPT-4o、GPT-4o-mini，需 API Key |
| LLM | Anthropic | `anthropic` | Claude Sonnet / Opus，需 API Key |
| LLM | Google Gemini | `gemini` | Gemini 1.5 Flash / Pro，需 API Key |
| LLM | xAI Grok | `grok` | Grok-2，需 API Key |
| Embedding | sentence-transformers（本地）| `local` | 完全離線，預設 paraphrase-multilingual-MiniLM |
| Embedding | OpenAI | `openai` | text-embedding-3-small / large，需 API Key |
| Embedding | Ollama | `ollama` | nomic-embed-text 等本地模型 |

### 13.2 抽象介面設計

```python
# core/providers/llm_provider.py
class LLMProvider(ABC):
    async def generate(self, prompt: str) -> str: ...
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...

# core/providers/embedding_provider.py
class EmbeddingProvider(ABC):
    @property
    def dim(self) -> int: ...
    def encode(self, text: str) -> list[float]: ...

# 實作類別（各自一個檔案）
OllamaLLMProvider      OpenAILLMProvider
AnthropicLLMProvider   GeminiLLMProvider   GrokLLMProvider

LocalEmbeddingProvider   OpenAIEmbeddingProvider   OllamaEmbeddingProvider
```

### 13.3 .env 設定範例

```dotenv
# ── 選擇 Provider ──────────────────────────────
LLM_PROVIDER=ollama           # ollama | openai | anthropic | gemini | grok
EMBEDDING_PROVIDER=local      # local | openai | ollama

# ── Neo4j ──────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# ── Ollama（本地）──────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ── OpenAI ─────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ── Anthropic ──────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# ── Google Gemini ───────────────────────────────
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-flash

# ── xAI Grok ───────────────────────────────────
GROK_API_KEY=xai-...
GROK_MODEL=grok-2

# ── 本地 Embedding（sentence-transformers）──────
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ── 系統設定 ────────────────────────────────────
CONCEPT_EXTRACTION_MAX=8
SCORE_THRESHOLD=0.70
WORKSPACE_DIR=./workspace
```

### 13.4 Provider 工廠

```python
# core/providers/factory.py
def get_llm_provider() -> LLMProvider:
    match settings.llm_provider:
        case "ollama":    return OllamaLLMProvider()
        case "openai":    return OpenAILLMProvider()
        case "anthropic": return AnthropicLLMProvider()
        case "gemini":    return GeminiLLMProvider()
        case "grok":      return GrokLLMProvider()

def get_embedding_provider() -> EmbeddingProvider:
    match settings.embedding_provider:
        case "local":  return LocalEmbeddingProvider()
        case "openai": return OpenAIEmbeddingProvider()
        case "ollama": return OllamaEmbeddingProvider()
```

### 13.5 新增檔案

```
core/
  providers/
    __init__.py
    base.py               LLMProvider / EmbeddingProvider ABC
    factory.py            get_llm_provider() / get_embedding_provider()
    llm/
      ollama.py
      openai.py
      anthropic.py
      gemini.py
      grok.py
    embedding/
      local.py            sentence-transformers
      openai.py
      ollama.py
```

**修改範圍：**
- `services/concept_engine.py`：`extract_concepts()` 改用 `get_llm_provider().generate()`
- `services/embedding_service.py`：改用 `get_embedding_provider()`
- `routers/agent.py`：LLM 串流改用 `get_llm_provider().stream()`
- `core/config.py`：新增 `llm_provider`、`embedding_provider` 及各 provider 設定欄位

---

## 十四、GitHub 開源準備

### 14.1 文件結構

```
/                            ← 專案根目錄
├── README.md                快速開始、功能介紹、架構圖
├── CONTRIBUTING.md          如何貢獻
├── LICENSE                  MIT
├── .env.example             完整設定範本（不含真實 key）
├── docker-compose.yml       Neo4j + API 一鍵啟動
├── Dockerfile
├── requirements.txt         所有相依套件（含 optional）
├── requirements-dev.txt     開發用套件
└── docs/
    ├── PLAN_multi_knowledge_graph.md   本計畫書
    ├── SETUP.md             詳細安裝指南（本地 / Docker）
    ├── PROVIDERS.md         各 Provider 設定教學
    ├── ARCHITECTURE.md      系統架構說明
    └── API.md               REST API 完整參考
```

### 14.2 README.md 內容大綱

```
# 智慧知識庫（World Knowledge Graph）

## 功能特色
- 多場景知識圖譜
- 自動文件分類（轉譯器 + 分配器）
- SVO 三元組知識結構
- 支援 Ollama / OpenAI / Claude / Gemini / Grok

## 快速開始（5 分鐘）
1. clone repo
2. cp .env.example .env  → 填入設定
3. docker-compose up     → 啟動 Neo4j
4. pip install -r requirements.txt
5. uvicorn main:app --reload

## Provider 選擇
### 本地（完全免費）
### 雲端（OpenAI / Claude / Gemini / Grok）

## 架構圖
## API 文件連結
## 貢獻指南
```

### 14.3 PROVIDERS.md 內容大綱

針對每個 Provider 提供：
- 前置需求（安裝 Ollama / 取得 API Key）
- `.env` 設定範例
- 推薦模型選擇
- 費用估算（雲端 provider）
- 常見問題

### 14.4 requirements.txt 結構

```
# 核心
fastapi
neo4j
pydantic-settings
httpx

# 文件轉譯
pypdf
pdfminer.six
python-docx
python-pptx
easyocr

# Embedding（本地）
sentence-transformers
torch

# LLM Provider（選裝）
openai          # OpenAI + Grok（Grok 相容 OpenAI SDK）
anthropic       # Claude
google-generativeai  # Gemini

# 可選功能
openai-whisper  # 影片/音訊轉文字
watchdog        # File Watcher
```

---

## 十五、待決策事項

| 項目 | 說明 | 目前預設 |
|------|------|----------|
| `owner_id` | 初期字串名稱，後期綁帳號系統 | 字串 |
| 文件跨 KG | 同一份文件可屬於多個 KG | 允許 |
| SVO 重建策略 | 全量重建 or 增量 merge | 增量 MERGE |
| 無相關 KG 回退（查詢）| 無 KG 超過門檻時 | 搜全庫 |
| 無相關 KG 回退（分配）| 所有 KG 分數 < min_threshold | 留 _staging/，標記 unmatched |
| SVO 無匹配 KG | 提取完成但找不到目標 KG | 丟棄 SVO，文字檔留 _staging/ |
| 分配後 SVO 時機 | 分配完成後立即 or 手動觸發 | 手動觸發（節省 LLM 用量）|
| Whisper 整合 | 影片/音訊轉文字 | Phase 6 後規劃 |
| Entity 去重 | 同義詞合併（Ontology alignment）| 不在本計畫 |

---

## 十六、不在範圍內（本計畫）

- 使用者帳號 / 認證系統
- KG 之間的連結 / 繼承
- Entity 同義詞合併（Ontology alignment）
- Whisper 語音轉文字（獨立 Phase 規劃）
- KG 版本控制
