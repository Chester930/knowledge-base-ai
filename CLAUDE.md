# CLAUDE.md — 智慧知識庫 (Knowledge Base AI)

本檔案提供 Claude Code CLI Agent 理解此專案所需的完整背景。

## 專案概覽

**多場景知識圖譜 RAG 系統**：將文件轉化為結構化 SVO 知識圖譜，透過雙層路由（ConceptNode + BFS 圖遍歷）提供精準問答。

- **後端**：FastAPI + Neo4j + Ollama/OpenAI/Anthropic/Gemini/Grok
- **前端**：Vanilla JS 單頁應用（無框架）
- **知識圖譜**：30 種語意關係類型作為 Neo4j edge label
- **部署**：Docker Compose 或本地 Python + Neo4j Desktop

## 目錄結構

```
├── core/                    # 設定、DB 連線、Provider 工廠
│   ├── config.py            # 所有 .env 設定讀取（唯一入口）
│   ├── database.py          # Neo4j AsyncDriver 連線管理
│   ├── constants.py         # 向量維度、初始權重等常數
│   └── providers/           # LLM / Embedding Provider 實作
│       ├── factory.py       # init_providers() 工廠函式
│       ├── llm/             # ollama, openai, anthropic, gemini, grok
│       └── embedding/       # local (sentence-transformers), openai, ollama
│
├── models/                  # Pydantic 資料模型
├── repositories/            # Neo4j CRUD（concept_repo, document_repo, kg_repo）
├── routers/                 # FastAPI 路由
│   ├── agent.py             # POST /agent/chat（SSE 串流問答）
│   ├── knowledge_graph.py   # KG CRUD + build-graph SSE
│   ├── documents.py         # 文件 CRUD + 批次匯入
│   ├── staging.py           # 暫存區管理（自動分群）
│   ├── search.py            # 向量搜尋
│   └── transcribe.py        # OCR 轉譯
│
├── services/
│   ├── svo_service.py       # ★ 核心：SVO 提取、Neo4j MERGE、BFS 查詢
│   ├── knowledge_graph_service.py  # KG 建立、自動分群、路由層刷新
│   ├── concept_engine.py    # ConceptNode 路由層計算
│   ├── classify_service.py  # 暫存區文件分類
│   ├── ingestion_service.py # 文件解析（PDF/DOCX/PPTX/TXT/MD）
│   └── file_watcher_service.py     # workspace/ 目錄監聽
│
├── ui/templates/index.html  # 前端單頁應用（含深色模式）
├── tests/                   # pytest 測試套件
│
├── run_build_kg.py          # ★ 批次建構所有 KG 的 SVO 知識圖譜
├── run_ingest.py            # 批次匯入文件至 KG
├── run_label_kg.py          # 為 Entity 節點套用語意型別標籤
├── run_reclassify_related_to.py  # 將 RELATED_TO 邊重新分類為精確語意類型
├── main.py                  # FastAPI 應用進入點
├── docker-compose.yml       # Neo4j Enterprise + FastAPI
└── start.ps1                # Windows 一鍵啟動腳本
```

## 常用指令

### 啟動服務

```bash
# Docker（推薦）
docker compose up -d --build

# 本地開發
python -m uvicorn main:app --reload --port 8000
```

### 知識圖譜建構

```bash
# 建構所有 KG（增量，跳過已處理文件）
python run_build_kg.py

# 強制重建（清除後從頭提取）
python run_build_kg.py --force

# 只重建指定 KG
python run_build_kg.py --kg <kg_id>

# 只更新關係邊（保留 Entity 節點）
python run_build_kg.py --force --relations-only
```

### 文件匯入

```bash
# 匯入整個資料夾至指定 KG
python run_ingest.py --kg <kg_id> --dir /path/to/docs

# 匯入單一檔案
python run_ingest.py --kg <kg_id> --file /path/to/doc.pdf
```

### 維護腳本

```bash
# 套用 Entity 語意型別標籤（Concept / Algorithm / Tool 等）
python run_label_kg.py --kg <kg_id>

# 將模糊的 RELATED_TO 邊重新分類為精確語意類型
python run_reclassify_related_to.py --kg <kg_id> --dry-run
python run_reclassify_related_to.py --kg <kg_id>
```

### 測試

```bash
pip install -r requirements-dev.txt
pytest
```

## 架構核心概念

### 雙層 RAG 流程

```
使用者問題
  → [路由層] ConceptNode embedding 比對 → 選出最相關 KG
  → [SVO 層] BFS 圖遍歷（1-2 跳）→ 結構化知識事實
  → [文件層] 圖譜驅動取出原文片段（非純相似度）
  → RAG Prompt → LLM → SSE 串流回答
```

### SVO 三元組格式（6 欄）

```
subject | subject_type | rel_type | verb | object | object_type
```

- `rel_type` 必須是 30 種語意類型之一（`svo_service.py` 的 `_VALID_REL_TYPES`）
- 跨文件相同實體自動 MERGE（`MERGE (e:Entity {name, kg_id})`）
- 提取失敗自動重試 2 次（exponential backoff），最終失敗的文件保留 `svo_processed_at=null` 供下次增量補跑

### Neo4j 資料庫分層

- **主資料庫**：`KnowledgeGraph` 節點、`Document` 節點、`ConceptNode` 路由層
- **KG 專用資料庫**（Enterprise）：`Entity` 節點 + 30 種語意關係邊
- Community 版自動 fallback：所有 Entity 存主資料庫，以 `kg_id` 屬性區隔

### Provider 工廠

`core/providers/factory.py` 的 `init_providers()` 依 `.env` 的 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 選擇實作。新增 provider 只需在對應目錄新增一個類別並在 `factory.py` 註冊。

## 開發慣例

- **新增 KG**：透過 UI 或 `POST /knowledge-graphs`，不需改程式碼
- **新增關係類型**：修改 `svo_service.py` 的 `_VALID_REL_TYPES` 和 `_ALL_REL_PATTERN`，同步更新 `run_reclassify_related_to.py` 的 prompt
- **修改 SVO 提取 prompt**：`svo_service.py` 的 `_build_svo_prompt()`
- **調整路由權重**：`core/constants.py` 的 `INTEREST_INIT` / `PROFESSIONAL_INIT`
- **新增文件格式**：`services/ingestion_service.py` 的 `_read_text()`

## 環境設定

```bash
cp .env.example .env
# 編輯 .env，至少填入：
# NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
# LLM_PROVIDER + 對應的 API KEY 或 OLLAMA_BASE_URL
# EMBEDDING_PROVIDER（建議 local，無需額外設定）
```

詳細 Provider 設定參見 `.env.example`。
