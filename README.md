# 智慧知識庫

基於本體論的多資料庫知識圖譜系統，以 FastAPI + Neo4j + Ollama 構建，支援文件匯入、SVO 知識抽取、雙層路由 RAG 問答。

## 架構概覽

```
┌─────────────────────────────────────────────────────────┐
│                     neo4j（主資料庫）                     │
│  KnowledgeGraph 節點  ←→  Document 節點                  │
│  ConceptNode 路由層（帶 embedding 向量）                  │
└──────────────────────┬──────────────────────────────────┘
                       │ db_name
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   kgxxxxxx01    kgxxxxxx02    kgxxxxxx03     （Neo4j Enterprise 專用 DB）
   Entity 節點   Entity 節點   Entity 節點
   RELATION 邊   RELATION 邊   RELATION 邊
   （SVO 知識層）
```

### 雙層路由問答流程

```
使用者問題
  │
  ▼
[路由層] ConceptNode embedding 比對 → 選出最相關 KG
  │
  ▼
[SVO 知識層] BFS 圖遍歷 → 取出結構化知識事實 + 來源文件 ID
  │
  ▼
[文件層] 依圖譜指向取出原文片段（圖譜驅動，非相似度搜尋）
  │
  ▼
RAG Prompt → LLM → 串流回答
```

## 知識圖譜本體論結構

每條關係邊直接以語意類型作為 Neo4j relationship type（30 種）：

| 群組 | 類型 |
|------|------|
| 層級/組成 | `IS_A` `PART_OF` `CONTAINS` `INSTANCE_OF` |
| 因果/效應 | `CAUSES` `PREVENTS` `ENABLES` `IMPROVES` `INHIBITS` |
| 功能/操作 | `USES` `REQUIRES` `PRODUCES` `IMPLEMENTS` `REPLACES` `EXTENDS` |
| 比較 | `CONTRASTS` `SIMILAR_TO` `OUTPERFORMS` |
| 描述/定義 | `DEFINED_AS` `HAS_PROPERTY` `MEASURED_BY` `APPLIES_TO` |
| 時序 | `PRECEDES` `FOLLOWS` `CO_OCCURS` |
| 資料流 | `INPUTS` `TRANSFORMS` |
| 歸屬/解決 | `CREATED_BY` `SOLVES` |
| 其他 | `RELATED_TO`（最後手段，目標使用率 < 5%）|

## 功能

- **文件匯入**：PDF / DOCX / PPTX / TXT / MD，支援資料夾批次匯入
- **OCR 轉譯**：上傳原始檔至暫存區，LLM 轉為純文字
- **自動分群建立 KG**：LLM 分析暫存區文件，自動命名並分群（可預覽編輯後確認）
- **手動分配**：暫存區文件逐篇分類，或批次自動分配
- **SVO 知識抽取**：6 欄本體論格式抽取，跨文章相同實體自動融合
- **多資料庫隔離**：每個 KG 使用獨立 Neo4j 資料庫（Enterprise 功能），Community 版自動 fallback
- **雙層 RAG 問答**：ConceptNode 路由 + SVO 圖遍歷 + 圖譜驅動文件選取
- **知識圖譜視覺化**：Entity 節點 + RELATION 邊，含實體類型與語意分類 badge

## 技術棧

| 元件 | 版本 / 說明 |
|------|------------|
| FastAPI | 後端 API，SSE 串流 |
| Neo4j | 主資料庫 + 每 KG 獨立資料庫（Enterprise）；Community 版自動 fallback |
| LLM | Ollama（本地）/ OpenAI / Anthropic / Google Gemini / xAI Grok |
| Embedding | sentence-transformers（本地）/ OpenAI / Ollama |
| Vanilla JS | 前端單頁應用，無框架依賴 |

## 快速啟動

### 1. 環境設定

```bash
cp .env.example .env
# 編輯 .env，填入 Neo4j 連線資訊與 LLM 設定
```

`.env` 必要設定（完整範本見 `.env.example`）：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

LLM_PROVIDER=ollama          # ollama | openai | anthropic | gemini | grok
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b

EMBEDDING_PROVIDER=local     # local | openai | ollama
```

### 2. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 3. 啟動 Neo4j

**方法 A：Docker（推薦）**

```bash
docker compose up -d
# API 啟動後開啟 http://localhost:8000
```

**方法 B：本地 + Neo4j Desktop**

1. Neo4j Desktop → 啟動 DBMS（確認 Bolt port 與 .env 一致）
2. 啟動 API：

```bash
python -m uvicorn main:app --reload --port 8000
```

開啟瀏覽器：`http://localhost:8000`

> **Windows 一鍵啟動**：`start.ps1`（自動檢查 Docker / Ollama 並啟動服務）

## 使用流程

### 建立知識圖譜（自動分群）

1. **匯入文件**：「＋新增」→「直接匯入知識庫」，選擇資料夾路徑
2. **自動分群**：「📥 暫存區」→「🤖 自動分群建立 KG」
   - LLM 分析文件內容，提出命名方案
   - 預覽並可編輯 KG 名稱與描述
   - 確認後一次建立所有 KG
3. **建立知識圖譜**：「🗂️ 知識圖譜」→ 選 KG →「⚡ 建立知識圖譜」

### 問答

切換到「💬 問答」tab，直接輸入問題。系統會：
1. 路由到相關 KG
2. 顯示匹配的知識事實（`[語意類別] 主詞 動詞 受詞`）
3. 串流回答

## 資料庫查詢

開啟 Neo4j Browser：`http://localhost:7474`

```cypher
// 查所有 KG
MATCH (kg:KnowledgeGraph) RETURN kg.name, kg.db_name, kg.entity_count

// 查某 KG 的 Entity（在對應 db 執行）
MATCH (e:Entity) RETURN e.name, e.type LIMIT 50

// 查特定語意類型的關係（rel_type 即 Neo4j edge label）
MATCH (s:Entity)-[r:CAUSES]->(o:Entity)
RETURN s.name, r.verb, o.name

// IS_A 階層遍歷（1-3 跳）
MATCH path = (e:Entity)-[:IS_A*1..3]->(root:Entity)
RETURN path

// 查高信心度的知識事實
MATCH (s:Entity)-[r]->(o:Entity)
WHERE r.confidence >= 2
RETURN s.name, type(r), r.verb, o.name
ORDER BY r.confidence DESC LIMIT 20
```

## 專案結構

```
├── core/               # 設定、資料庫連線、LLM/Embedding providers
├── models/             # Pydantic 資料模型
├── repositories/       # Neo4j CRUD 操作
├── routers/            # FastAPI 路由（agent, knowledge_graph, staging, transcribe）
├── services/           # 業務邏輯
│   ├── svo_service.py          # SVO 抽取、圖譜建立、BFS 查詢
│   ├── knowledge_graph_service.py  # KG CRUD、自動分群
│   ├── classify_service.py     # 暫存區分類與分配
│   └── concept_engine.py       # ConceptNode 路由層
└── ui/templates/       # 前端單頁應用（Vanilla JS）
```
