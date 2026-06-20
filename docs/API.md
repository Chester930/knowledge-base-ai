# REST API 參考

基礎 URL：`http://localhost:8000`

互動式文件：`http://localhost:8000/docs`（Swagger UI）

---

## 系統

### GET /health

服務健康檢查。

**回應：**
```json
{ "status": "ok", "version": "1.0.0" }
```

### GET /agent/health

Agent 詳細狀態。

**回應：**
```json
{
  "status": "ok",
  "document_count": 20,
  "kg_count": 3,
  "entity_count": 3791,
  "llm_provider": "ollama",
  "embedding_provider": "local",
  "dual_layer_routing": true
}
```

---

## 知識庫（KnowledgeGraph）

### GET /knowledge-graphs

列出所有知識庫。

**回應：**
```json
[
  {
    "id": "92531e0b-...",
    "name": "CLIArchitecture",
    "description": "CLI 架構設計與實現",
    "folder_path": "workspace/kg_cliarchitecture",
    "doc_count": 18,
    "entity_count": 3508,
    "relation_count": 2553,
    "created_at": "2026-06-18T08:53:27Z",
    "updated_at": "2026-06-21T01:32:41Z"
  }
]
```

### POST /knowledge-graphs

建立新知識庫。

**請求：**
```json
{
  "name": "MyKG",
  "description": "我的知識庫",
  "is_public": true
}
```

**回應：** 同 GET 單筆格式

### GET /knowledge-graphs/{id}

取得知識庫詳情（含 top concepts / entities）。

### PUT /knowledge-graphs/{id}

更新知識庫名稱或描述。

**請求：**
```json
{ "name": "NewName", "description": "新描述" }
```

### DELETE /knowledge-graphs/{id}

刪除知識庫（含所有文件與圖譜資料）。

### GET /knowledge-graphs/{id}/documents

列出知識庫下所有文件。

### POST /knowledge-graphs/{id}/build-graph

觸發 SVO 知識圖譜建構（SSE 串流）。

**請求：**
```json
{
  "kg_id": "92531e0b-...",
  "force_rebuild": false
}
```

**SSE 事件：**
```
data: {"event": "chunk_start", "total_chunks": 12, "message": "[文件名] 並行處理 12 個段落…"}
data: {"event": "chunk_done", "chunk_idx": 1, "triples_merged": 15, "message": "[文件名] 第 1 段完成：15 組三元組"}
data: {"event": "error", "chunk_idx": 3, "message": "提取失敗：..."}
data: {"event": "done", "message": "知識圖譜建構完成，共合併 2483 組三元組"}
```

### GET /knowledge-graphs/{id}/graph

取得知識庫的 Entity 與 RELATION 資料（圖譜視覺化用）。

**Query 參數：**
- `limit`：最多回傳幾個 Entity（預設 100）
- `rel_type`：篩選特定關係類型

### PUT /knowledge-graphs/{id}/refresh

重新計算路由層概念（ConceptNode）。

---

## 文件（Document）

### POST /documents/ingest

上傳文件至暫存區（轉譯 + 建立 Document 節點）。

**Form Data：**
- `file`：上傳的原始檔案（PDF / DOCX / PPTX / TXT / MD / MP3 / MP4 等）

**回應：**
```json
{
  "doc_id": "uuid",
  "title": "文件名稱",
  "char_count": 12500,
  "staging": true
}
```

### POST /documents/ingest-dir

批次匯入目錄內所有支援格式的檔案。

**請求：**
```json
{ "folder_path": "/path/to/docs", "kg_id": "uuid-optional" }
```

### DELETE /documents/{doc_id}

刪除文件（含對應的 Entity 資料）。

---

## 暫存區（Staging）

### GET /staging

列出暫存區中所有未分配文件。

**回應：**
```json
{
  "documents": [
    {
      "id": "uuid",
      "title": "report.txt",
      "char_count": 8200,
      "status": "pending"
    }
  ]
}
```

### POST /staging/{doc_id}/classify

對單一文件執行 KG 分配建議。

**請求：**
```json
{ "threshold": 0.3, "auto_assign": false }
```

**回應：**
```json
{
  "doc_id": "uuid",
  "candidates": [
    { "kg_id": "uuid", "kg_name": "CLIArchitecture", "score": 0.72 }
  ],
  "auto_assigned": false
}
```

### POST /staging/{doc_id}/assign

手動將暫存區文件分配到指定 KG。

**請求：**
```json
{ "kg_id": "uuid" }
```

### POST /staging/auto-cluster

LLM 自動分析暫存區文件，提出 KG 分群方案（預覽）。

**回應：**
```json
{
  "cluster_id": "uuid",
  "proposed_kgs": [
    {
      "name": "CLIArchitecture",
      "description": "CLI 架構設計...",
      "documents": ["doc1.txt", "doc2.txt"]
    }
  ]
}
```

### POST /staging/confirm-cluster

確認自動分群方案，一次建立所有 KG 並分配文件。

**請求：**
```json
{ "cluster_id": "uuid" }
```

---

## 問答（Agent）

### POST /agent/chat

雙層路由問答（SSE 串流）。

**請求：**
```json
{
  "question": "Claude Code 的 fork 代理是什麼？",
  "session_id": "optional-session-id"
}
```

**SSE 事件序列：**

```
data: {"status": "searching"}

data: {"kg_route": [
  {"id": "uuid", "name": "CLIArchitecture", "score": 0.195, "matched_concepts": []}
]}

data: {"svo_facts": [
  "子代理(概念) -[組成:部分于]→ Fork代理(其他)",
  "[推理鏈] AgentTool(工具) -[延伸]→ 子代理 → 協調者"
]}

data: {"sources": [
  {"title": "第 9 章：分叉代理", "source": "graph"},
  {"title": "第 8 章：產生子代理", "source": "graph"},
  {"title": "第 10 章：任務", "score": 0.82, "source": "similarity"}
]}

data: {"status": "generating"}
data: {"token": "根據"}
data: {"token": "知識圖譜"}
...
data: {"done": true}
```

---

## 搜尋（Search）

### POST /search

向量相似度搜尋。

**請求：**
```json
{
  "query": "強化學習算法比較",
  "kg_id": "uuid-optional",
  "top_k": 5
}
```

**回應：**
```json
{
  "results": [
    {
      "doc_id": "uuid",
      "title": "文件標題",
      "score": 0.87,
      "excerpt": "文件摘要..."
    }
  ]
}
```

---

## 轉譯（Transcribe）

### POST /transcribe/file

上傳原始檔案並轉譯為純文字（不存入 KG）。

**Form Data：** `file`（原始檔案）

**回應：**
```json
{
  "text": "轉譯後的文字內容...",
  "char_count": 5200,
  "elapsed_seconds": 3.2
}
```

---

## 錯誤格式

所有錯誤使用標準 HTTP 狀態碼：

```json
{
  "detail": "錯誤描述"
}
```

| 狀態碼 | 說明 |
|--------|------|
| 400 | 請求格式錯誤 |
| 404 | 資源不存在 |
| 422 | 資料驗證失敗 |
| 500 | 伺服器內部錯誤 |
