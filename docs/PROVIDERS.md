# Provider 設定指南

系統支援多種 LLM 和 Embedding Provider，透過 `.env` 切換，無需修改程式碼。

---

## LLM Provider

### Ollama（本地，免費）

**前置需求：**
1. 安裝 [Ollama](https://ollama.com/download)
2. 下載模型：`ollama pull qwen2.5:7b`
3. 啟動服務：`ollama serve`（通常自動啟動）

**.env 設定：**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
```

**推薦模型（中文效果）：**

| 模型 | 大小 | 速度 | 品質 | 適用場景 |
|------|------|------|------|---------|
| `qwen2.5:7b` | 4.7GB | 快 | 良好 | 日常使用（推薦） |
| `qwen2.5:14b` | 9GB | 中 | 佳 | 更高品質需求 |
| `qwen2.5:32b` | 20GB | 慢 | 優 | 高端 GPU 用戶 |
| `llama3.1:8b` | 4.9GB | 快 | 良好 | 英文為主 |

**常見問題：**
- `connection refused`：執行 `ollama serve` 確保服務運行
- 回應慢：降低並行數 `OLLAMA_NUM_PARALLEL=1`

---

### OpenAI

**.env 設定：**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
```

**推薦模型：**

| 模型 | 費用（輸入/輸出）| 適用場景 |
|------|----------------|---------|
| `gpt-4o-mini` | $0.15 / $0.60 per 1M tokens | 日常使用（推薦，高 CP 值）|
| `gpt-4o` | $2.50 / $10.00 per 1M tokens | 高品質需求 |

**估算費用：**
- 每份文件（約 10K 字）SVO 提取：約 $0.001（使用 gpt-4o-mini）
- 每次問答：約 $0.0005

**取得 API Key：** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

### Anthropic Claude

**.env 設定：**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

**推薦模型：**

| 模型 | 說明 |
|------|------|
| `claude-haiku-4-5-20251001` | 快速、低成本，適合大量 SVO 提取 |
| `claude-sonnet-4-6` | 平衡品質與速度（推薦）|
| `claude-opus-4-8` | 最高品質，適合複雜推理 |

**取得 API Key：** [console.anthropic.com](https://console.anthropic.com)

---

### Google Gemini

**.env 設定：**
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-flash
```

**推薦模型：**

| 模型 | 說明 |
|------|------|
| `gemini-1.5-flash` | 快速，有免費額度（推薦入門）|
| `gemini-1.5-pro` | 高品質，適合複雜任務 |
| `gemini-2.0-flash` | 最新版 Flash |

**取得 API Key：** [aistudio.google.com](https://aistudio.google.com)

---

### xAI Grok

**.env 設定：**
```env
LLM_PROVIDER=grok
GROK_API_KEY=xai-...
GROK_MODEL=grok-2
```

**取得 API Key：** [console.x.ai](https://console.x.ai)

---

## Embedding Provider

### Local（sentence-transformers，預設，免費）

無需額外設定，首次啟動自動從 HuggingFace 下載模型（約 500MB）。

**.env 設定：**
```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**其他可用模型：**

| 模型 | 維度 | 說明 |
|------|------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 預設，多語言，輕量 |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 更高維度，品質較佳 |
| `BAAI/bge-m3` | 1024 | 中文優化，需較多記憶體 |

> ⚠️ 更換 Embedding 模型後，因向量維度改變，需執行 `python run_build_kg.py --force` 重建所有 KG。

---

### OpenAI Embedding

**.env 設定：**
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

| 模型 | 維度 | 費用 per 1M tokens |
|------|------|-------------------|
| `text-embedding-3-small` | 1536 | $0.02（推薦）|
| `text-embedding-3-large` | 3072 | $0.13 |

---

### Ollama Embedding

**.env 設定：**
```env
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

```bash
# 下載 Embedding 模型
ollama pull nomic-embed-text
```

---

## 組合建議

| 使用場景 | LLM | Embedding | 說明 |
|---------|-----|-----------|------|
| **完全本地（隱私優先）** | Ollama qwen2.5:7b | local | 零費用，資料不出境 |
| **高品質 + 本地 Embedding** | OpenAI gpt-4o-mini | local | 低成本，本地向量 |
| **全雲端** | Anthropic Claude Sonnet | OpenAI text-embedding-3-small | 最高品質 |
| **免費入門** | Gemini 1.5 Flash | local | Gemini 有免費額度 |

---

## 混用多個 Provider

LLM 和 Embedding 可以分別使用不同 Provider：

```env
# Ollama 做 LLM，OpenAI 做 Embedding（或反過來）
LLM_PROVIDER=ollama
OLLAMA_LLM_MODEL=qwen2.5:14b

EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
