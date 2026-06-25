# 安裝指南

## 前置需求

| 軟體 | 版本 | 說明 |
|------|------|------|
| Python | 3.10+ | 建議 3.11 |
| Neo4j | 5.x | Desktop 或 Docker |
| Ollama | 最新版 | 本地 LLM（可換成雲端 Provider）|
| Docker | 可選 | 最快的部署方式 |

---

## 方法 A：Docker（推薦，5 分鐘）

### 1. 下載專案

```bash
git clone <your-repo-url>
cd knowledge-base-ai
```

### 2. 設定環境變數

```bash
cp .env.example .env
```

用文字編輯器開啟 `.env`，最少填入：

```env
NEO4J_PASSWORD=your_password_here

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_LLM_MODEL=qwen2.5:7b

EMBEDDING_PROVIDER=local
```

> **注意**：`OLLAMA_BASE_URL` 在 Docker 容器內需用 `host.docker.internal` 才能存取宿主機的 Ollama。

### 3. 啟動 Ollama 並下載模型

```bash
# 宿主機上執行（非 Docker）
ollama pull qwen2.5:7b
ollama serve
```

### 4. 啟動服務

```bash
docker compose up -d --build
```

首次執行會下載 Neo4j 映像和安裝 Python 套件，約需 3-5 分鐘。

### 5. 開啟瀏覽器

```
http://localhost:8000
```

Neo4j Browser（查詢圖譜用）：`http://localhost:7475`

### Docker 常用指令

```bash
# 查看日誌
docker compose logs -f api

# 停止
docker compose down

# 重建（更新程式碼後）
docker compose up -d --build

# 進入容器
docker exec -it kg-api bash
```

---

## 方法 B：本地開發

### 1. 下載專案

```bash
git clone <your-repo-url>
cd knowledge-base-ai
```

### 2. 建立虛擬環境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. 安裝套件

```bash
pip install -r requirements.txt
```

> **GPU 加速**（可選）：`sentence-transformers` 會自動使用 CUDA。
> 若要啟用 OCR 和 Whisper 的 GPU 支援，需安裝 CUDA 版 PyTorch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. 啟動 Neo4j

**選項 1：Neo4j Desktop**（建議新手）
1. 下載 [Neo4j Desktop](https://neo4j.com/download/)
2. 建立新 DBMS（版本 5.x）
3. 啟動 DBMS，記錄連線資訊（預設 Bolt port: 7687）

**選項 2：Docker 只啟 Neo4j**
```bash
docker compose up -d neo4j
```

### 5. 設定環境變數

```bash
cp .env.example .env
```

填入 Neo4j 連線資訊：
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 6. 啟動 API

```bash
python -m uvicorn main:app --reload --port 8000
```

---

## Windows 一鍵啟動（PowerShell）

```powershell
.\start.ps1
```

腳本會自動：
1. 檢查 Docker Desktop 是否運行
2. 偵測 Neo4j Desktop 的 port 衝突（7687）
3. 確認 Ollama 和指定模型已就緒
4. 執行 `docker compose up -d --build`
5. 輪詢 `/health` 直到服務就緒

---

## 建構知識圖譜

服務啟動後，透過 UI 或命令列建構知識圖譜：

### UI 操作

1. 開啟 `http://localhost:8000`
2. 「**📥 暫存區**」→ 上傳文件
3. 「**🤖 自動分群建立 KG**」→ LLM 自動命名分群
4. 確認後 → 「**⚡ 建立知識圖譜**」

### 命令列操作

```bash
# 批次匯入文件
python run_ingest.py --kg <kg_id> --dir ./my_docs

# 建構所有 KG 的知識圖譜
python run_build_kg.py

# 強制重建（清除現有資料）
python run_build_kg.py --force
```

---

## 常見問題

### Neo4j 連線失敗

```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**解決方式**：
- 確認 Neo4j 已啟動（Neo4j Desktop 或 `docker compose up neo4j`）
- 確認 `.env` 的 `NEO4J_URI` 和密碼正確
- Docker 內連接 Neo4j：URI 應為 `bolt://neo4j:7687`（使用 service name）

### Ollama 模型下載失敗

```bash
# 手動下載
ollama pull qwen2.5:7b

# 查看已安裝的模型
ollama list
```

### Embedding 模型下載慢

首次啟動時 `sentence-transformers` 會從 HuggingFace 下載模型（約 500MB）。
下載後會快取到 `~/.cache/huggingface/`，之後啟動不需再下載。

若網路受限，可預先下載後設定：
```env
LOCAL_EMBEDDING_MODEL=/path/to/local/model
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

### OCR 初始化慢

首次使用 OCR（掃描型 PDF）時會下載 EasyOCR 模型（繁中 + 英文，約 200MB）。
之後從快取載入只需幾秒。

### Docker 容器記憶體不足

Embedding 模型 + LLM 可能佔用大量記憶體。建議 Docker 分配至少 **4GB RAM**。
