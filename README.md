# 智慧知識庫

多場景知識圖譜 RAG 系統。將文件轉化為結構化 SVO 知識圖譜，透過雙層路由（ConceptNode + BFS 圖遍歷）提供精準問答，支援自我精煉迴圈與多 LLM Provider。

## 核心特性

| 特性 | 說明 |
|------|------|
| **多場景 KG** | 每個知識圖譜獨立管理，Neo4j Enterprise 下各有專屬資料庫 |
| **30 種語意關係** | SVO 三元組以精確語意類型儲存（IS_A / CAUSES / USES…），非模糊向量 |
| **雙層 RAG 路由** | ConceptNode embedding 路由 → BFS 圖遍歷 → 圖譜驅動文件選取 |
| **自我精煉迴圈** | 低信心時自動補充原文 chunk，最多 3 輪，停止門檻 0.65 |
| **Sentence-Aware Chunking** | 按句子邊界切分並持久化，SVO 節點記錄來源 chunk 座標 |
| **多 Provider** | Ollama / OpenAI / Anthropic / Gemini / Grok，本地雲端自由切換 |
| **自動分群建 KG** | LLM 分析暫存區文件，自動命名分群，確認後一鍵建立並觸發建圖 |
| **PDF OCR 三層備援** | pypdf → pdfminer → EasyOCR（繁中+英），無損轉換掃描版 PDF |

## 系統架構

```
使用者問題
  │
  ▼
[E1] 問題概念提取（LLM + embedding，LRU 快取）
  │
  ▼
[E2] KG 路由（ConceptNode cosine × alignment × magnitude 加權分）
  │
  ▼
[E3] BFS 圖遍歷（多 KG 並行，in-memory TTL=300s 快取）
  │   facts / source_doc_ids / chunk_ids
  ▼
[E4] 混合文件檢索
  ├── Graph-Driven：SVO 指向文件 → _pick_relevant_chunks()
  │     batch_embed + keyword boost + SVO entity boost + enum bonus
  └── Similarity Fallback：concept match 補充
  │
  ▼
[E5] 自我精煉迴圈（max 3 rounds，ChunkStore 補充原文）
  │
  ▼
[E6] LLM 串流輸出（SSE token stream）
```

### Neo4j 資料庫分層

```
主資料庫
├── KnowledgeGraph 節點（KG 元資料）
├── Document 節點（文件元資料）
└── ConceptNode 路由層（帶 embedding 向量）

每 KG 獨立資料庫（Enterprise）/ 主庫 kg_id 屬性區隔（Community）
└── Entity 節點 + 30 種語意關係邊（SVO 知識層）
```

## 技術棧

| 元件 | 說明 |
|------|------|
| FastAPI >= 0.115 | 後端 API，SSE 串流 |
| Neo4j Driver >= 5.24 | 主資料庫 + 每 KG 獨立資料庫（Enterprise） |
| sentence-transformers >= 3.0 | 本地 embedding（無需 API 金鑰） |
| Ollama / OpenAI / Anthropic / Gemini / Grok | 可選 LLM |
| EasyOCR / PaddleOCR | PDF OCR 備援 |
| faster-whisper | 音影片轉譯 |
| Vanilla JS | 前端，無框架依賴 |

## 快速啟動

### 1. 環境設定

```bash
cp .env.example .env
# 編輯 .env，至少填入：
# NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
# LLM_PROVIDER + 對應金鑰或 OLLAMA_BASE_URL
```

最小設定範例（Ollama 本地）：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b

EMBEDDING_PROVIDER=local
```

詳細 Provider 設定見 [`.env.example`](.env.example) 與 [`docs/PROVIDERS.md`](docs/PROVIDERS.md)。

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 啟動

**Docker（推薦）**

```bash
docker compose up -d --build
# API：http://localhost:8000
# Neo4j Browser：http://localhost:7474
```

**本地開發**

1. 啟動 Neo4j Desktop（確認 Bolt port 與 `.env` 一致）
2. 啟動 API：

```bash
python -m uvicorn main:app --reload --port 8000
```

> **Windows 一鍵啟動**：執行 `./start.ps1`，自動檢查 Docker / Ollama 並啟動服務。

## 使用流程

### 匯入文件並建立 KG

1. **直接匯入**：UI →「＋新增」→「直接匯入知識庫」，選擇資料夾
2. **暫存區 + 自動分群**：
   - 文件先進暫存區（`workspace/_staging/`）
   - UI →「📥 暫存區」→「🤖 自動分群建立 KG」
   - 預覽 LLM 建議的命名與分組，可編輯後確認
   - 確認後自動建立 KG 並在背景觸發 SVO 知識圖譜建構
3. **手動建圖**：「🗂️ 知識圖譜」→ 選 KG →「⚡ 建立知識圖譜」

### 問答

UI →「💬 問答」，直接輸入問題。系統會顯示：

- 路由到的 KG 清單與匹配分數
- 匹配的 SVO 知識事實（`[語意類別] 主詞 動詞 受詞`）
- 自我精煉輪次進度（低信心時）
- 串流回答

### 命令列工具

```bash
# 批次匯入文件（繞過 HTTP timeout，適合大量文件）
python run_ingest.py /path/to/docs
python run_ingest.py /path/to/docs --kg <kg_id>

# 建構 SVO 知識圖譜
python run_build_kg.py                    # 增量（只處理未建構的文件）
python run_build_kg.py --force            # 強制重建所有 KG
python run_build_kg.py --kg <kg_id>       # 只重建指定 KG

# 維護工具
python run_label_kg.py --kg <kg_id>                          # 套用 Entity 語意型別標籤
python run_reclassify_related_to.py --kg <kg_id> --dry-run   # 預覽邊重分類
python run_reclassify_related_to.py --kg <kg_id>             # 執行邊重分類
```

## 目錄結構

```
├── core/
│   ├── config.py           # 所有 .env 設定（唯一入口）
│   ├── database.py         # Neo4j AsyncDriver 連線管理
│   ├── constants.py        # 路由權重、門檻等常數
│   └── providers/          # LLM / Embedding Provider
│       ├── factory.py      # init_providers() 工廠
│       ├── llm/            # ollama, openai, anthropic, gemini, grok
│       └── embedding/      # local, openai, ollama
│
├── models/                 # Pydantic 資料模型
├── repositories/           # Neo4j CRUD（concept, document, kg）
├── routers/
│   ├── agent.py            # POST /agent/chat（SSE 問答，含自我精煉）
│   ├── knowledge_graph.py  # KG CRUD + build-graph SSE
│   ├── documents.py        # 文件 CRUD + 批次匯入
│   ├── staging.py          # 暫存區管理 + 自動分群確認
│   ├── search.py           # 向量搜尋
│   └── transcribe.py       # 音影片轉譯
│
├── services/
│   ├── svo_service.py              # SVO 提取、Neo4j MERGE、BFS 查詢
│   ├── chunk_store.py              # Chunk 持久化（chunk_store/{kg_id}/）
│   ├── knowledge_graph_service.py  # KG 建立、自動分群
│   ├── concept_engine.py           # ConceptNode 路由層（含 LRU 快取）
│   ├── classify_service.py         # 暫存區分類與分配
│   └── ingestion_service.py        # 文件解析（PDF/DOCX/PPTX/TXT/MD）
│
├── ui/templates/index.html # 前端單頁應用（深色模式支援）
├── tests/                  # pytest 測試套件（455 tests）
├── docs/                   # 詳細文件
│
├── main.py                 # FastAPI 應用進入點
├── docker-compose.yml      # Neo4j + FastAPI Docker 設定
├── .env.example            # 環境設定範本
└── start.ps1               # Windows 一鍵啟動腳本
```

## 30 種語意關係類型

| 群組 | 關係類型 |
|------|---------|
| 層級 / 組成 | `IS_A` `PART_OF` `CONTAINS` `INSTANCE_OF` |
| 因果 / 效應 | `CAUSES` `PREVENTS` `ENABLES` `IMPROVES` `INHIBITS` |
| 功能 / 操作 | `USES` `REQUIRES` `PRODUCES` `IMPLEMENTS` `REPLACES` `EXTENDS` |
| 比較 | `CONTRASTS` `SIMILAR_TO` `OUTPERFORMS` |
| 描述 / 定義 | `DEFINED_AS` `HAS_PROPERTY` `MEASURED_BY` `APPLIES_TO` |
| 時序 | `PRECEDES` `FOLLOWS` `CO_OCCURS` |
| 資料流 | `INPUTS` `TRANSFORMS` |
| 歸屬 / 解決 | `CREATED_BY` `SOLVES` |
| 其他 | `RELATED_TO`（模糊關係，目標使用率 < 5%） |

## 測試

```bash
pip install -r requirements-dev.txt
pytest
```

## 文件索引

| 文件 | 說明 |
|------|------|
| [docs/SETUP.md](docs/SETUP.md) | 詳細安裝指南（Docker / 本地 / Windows） |
| [docs/PROVIDERS.md](docs/PROVIDERS.md) | LLM / Embedding Provider 完整設定 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系統架構深度說明 |
| [docs/API.md](docs/API.md) | REST API 完整參考 |
| [docs/behavior_tree.md](docs/behavior_tree.md) | 完整 Behavior Tree 與優化清單 |
| [ROADMAP.md](ROADMAP.md) | 開發路線圖 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 貢獻指南 |

## 授權

MIT License — 詳見 [LICENSE](LICENSE)
