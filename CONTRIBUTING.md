# 貢獻與客製化指南

## 快速客製化

### 1. 更換 LLM Provider

修改 `.env`：

```env
# 本地 Ollama（預設）
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:14b

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o

# Anthropic Claude
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# Google Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-pro

# xAI Grok
LLM_PROVIDER=grok
GROK_API_KEY=xai-...
GROK_MODEL=grok-2
```

重啟服務即生效，無需改程式碼。

---

### 2. 更換 Embedding Provider

```env
# 本地 sentence-transformers（預設，無需 API Key）
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# OpenAI
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

> ⚠️ 換 Embedding Provider 後需重新建構所有 KG（`python run_build_kg.py --force`），因為向量維度可能不同。

---

### 3. 新增知識庫（KG）

**方法 A：UI 操作**
1. 開啟 `http://localhost:8000`
2. 「📥 暫存區」→ 上傳文件 → 「🤖 自動分群建立 KG」
3. 確認分群後，逐一「⚡ 建立知識圖譜」

**方法 B：命令列**
```bash
# 1. 透過 API 建立 KG
curl -X POST http://localhost:8000/knowledge-graphs \
  -H "Content-Type: application/json" \
  -d '{"name": "MyKG", "description": "我的知識庫"}'

# 2. 匯入文件
python run_ingest.py --kg <kg_id> --dir ./my_docs

# 3. 建構知識圖譜
python run_build_kg.py --kg <kg_id>
```

---

### 4. 新增語意關係類型

編輯 `services/svo_service.py`：

```python
# 1. 在 _VALID_REL_TYPES 加入新類型
_VALID_REL_TYPES = {
    ...
    "MY_NEW_TYPE",   # 加在對應群組
}

# 2. 在 _ALL_REL_PATTERN 加入（供 Cypher MATCH 使用）
_ALL_REL_PATTERN = (
    "IS_A|PART_OF|...|MY_NEW_TYPE"
)

# 3. 在 REL_TYPE_LABELS 加入中文顯示名稱（UI 用）
REL_TYPE_LABELS = {
    ...
    "MY_NEW_TYPE": "自訂",
}
```

加完後重新建構 KG（`--force`）讓新類型生效。

---

### 5. 新增 LLM Provider

在 `core/providers/llm/` 新增 `myprovider.py`：

```python
from core.providers.base import LLMProvider

class MyProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = ...  # 初始化 SDK

    def generate(self, prompt: str) -> str:
        ...

    def generate_json(self, prompt: str) -> dict:
        ...

    def stream(self, prompt: str):
        ...  # yield token strings
```

在 `core/providers/factory.py` 的 `init_providers()` 加入：

```python
elif settings.llm_provider == "myprovider":
    from core.providers.llm.myprovider import MyProvider
    llm = MyProvider(model=..., api_key=...)
```

`.env` 設定 `LLM_PROVIDER=myprovider` 即可切換。

---

### 6. 調整 SVO 提取 Prompt

`services/svo_service.py` 的 `_build_svo_prompt()` 函式控制提取行為。常見調整：

- **提高精確度**：在 prompt 加入「只提取有明確語意依據的關係，寧缺勿濫」
- **調整輸出語言**：prompt 中指定輸出語言（目前為繁體中文）
- **限制關係數量**：調整 `每篇文章最多提取 N 組三元組` 的指示

---

## 開發環境設定

```bash
# 安裝含開發依賴
pip install -r requirements.txt -r requirements-dev.txt

# 執行測試
pytest

# 啟動開發伺服器（熱重載）
python -m uvicorn main:app --reload --port 8000
```

## 資料庫查詢參考

```cypher
// 查所有 KG 及統計
MATCH (kg:KnowledgeGraph)
RETURN kg.name, kg.entity_count, kg.relation_count

// 在 KG 專用 DB 中查 Entity（Neo4j Browser 切換 DB 後執行）
MATCH (e:Entity) RETURN e.name, e.type LIMIT 50

// 查特定語意關係
MATCH (s:Entity)-[r:CAUSES]->(o:Entity)
RETURN s.name, r.verb, o.name

// 查高信心度三元組
MATCH (s:Entity)-[r]->(o:Entity)
WHERE r.confidence >= 2
RETURN s.name, type(r), r.verb, o.name
ORDER BY r.confidence DESC LIMIT 20
```

## 提交規範

使用 Conventional Commits：

```
feat(svo): 新增 OPPOSES 關係類型
fix(router): 修正 SSE 連線中斷問題
docs: 更新 Provider 設定說明
perf(kg): 降低 SVO 並行數以減少記憶體用量
```
