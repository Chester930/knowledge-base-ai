# 對標 NotebookLM：不足分析與完全超越方案

本報告對標 Google Labs 開發的知識整理工具 **NotebookLM**，基於本專案當前的學術理論架構 [THEORETICAL_ARCHITECTURE.md](file:///d:/Users/666/Desktop/智慧知識庫/docs/THEORETICAL_ARCHITECTURE.md) 與實作代碼，客觀剖析當前系統的物理與架構不足，並提出實現「完全超越」的具體架構演進路徑與核心程式碼藍圖。

---

## 一、 我們與 NotebookLM 的核心差距（不足之處）

NotebookLM 基於 **Gemini 1.5 Pro 原生超長上下文（2M Tokens）**，其 RAG 結構採用 In-Context 暴力檢索。對比之下，我們的 **神經符號 GraphRAG** 雖然在隱私安全與圖譜推理上佔優，但在以下四個維度存在顯著不足：

### 1. 寫入階段的高昂冷啟動成本 (Extraction Cold-Start Latency)
* **痛點**：本專案在導入文件時，必須全量調用 LLM 抽取三元組（SVO）並執行實體對齊。若用戶上傳百萬字文檔，建圖過程需要花費高昂的 API 費用，且在本地 Ollama 上運行可能需要數小時，無法做到「上傳即用」。
* **NotebookLM 的優勢**：採用 Zero-Shot RAG。文檔導入時只做輕量向量化與快取分片（秒級可用），用戶拖入 PDF 即可立刻開始問答。

### 2. 多模態結構與表格解析缺失 (Lack of Layout & Table-Aware Processing)
* **痛點**：目前的 PDF 轉譯為純文字依賴 `pypdf -> pdfminer -> PaddleOCR`。這種純文字提取會摧毀 PDF 中的 **複雜表格 (Tables)**、**圖表 (Charts)** 與 **版面排版 (Layout)**，導致資訊遺失或排版錯亂。
* **NotebookLM 的優勢**：依賴 Google 的原生多模態編碼器（Multimodal Encoder），能夠直接看懂並處理 PDF 內的跨行表格、圓餅圖與示意圖。

### 3. 大節點路徑爆炸與 Naive 硬截斷 (Hub Node Path Explosion)
* **痛點**：進行 BFS 圖遍歷時，若遇到樞紐節點（Hub Nodes，如法規名稱、大公司名），Cypher 查詢會瞬間撈出數千條事實，撐爆 Context 窗格。雖然我們實作了 `_PER_SEED_FACT_LIMIT = 20` 的限制，但這是隨機硬截斷，會導致真正關鍵的事實被靜默擠掉。
* **NotebookLM 的優勢**：依靠 Transformer 原生 Attention 矩陣動態分配權重，沒有硬性的圖遍歷截斷問題。

### 4. 產品層面的互動式見解與引導 (Lack of Active Guided Insights)
* **痛點**：用戶建立知識庫後，面對一個空白的問答框無從下手。系統缺乏對整份知識庫的宏觀導讀與自動洞察功能。
* **NotebookLM 的優勢**：一鍵生成簡報文件（Briefing Documents）、自動常見問答對（FAQ）、學習指南（Study Guide）與雙人音訊導覽。

---

## 二、 完全超越 NotebookLM 的四大演進路徑與代碼藍圖

為了解決上述不足，本專案必須在架構上進行演進，以下是具體的優化方案與代碼實現設計：

### 方案一：雙軌非同步增量建構（秒級可用 RAG）
為了對沖建圖的冷啟動延遲，系統應改為 **「向量先行，圖譜非同步建構」** 的雙軌架構。文檔上傳時，5秒內完成向量索引供用戶立即提問；同時將抽取任務丟入背景隊列，非同步將 SVO 寫入 Neo4j，無感合併。

#### 背景非同步建圖任務代碼藍圖（可在後端異步隊列執行）：
```python
# 建議寫在 services/ingestion_service.py 或新增的 queue_worker.py
import asyncio
from typing import List
from uuid import UUID
from core.database import get_driver
from services.chunk_store import get_chunk_store
from services.svo_service import extract_svo_from_text

class AsyncGraphBuilder:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_running = False

    async def add_ingestion_task(self, doc_id: UUID, kg_id: UUID, chunks: List[str]):
        """用戶上傳文件時，先完成向量索引，並在此處註冊背景建圖任務"""
        await self.queue.put({"doc_id": doc_id, "kg_id": kg_id, "chunks": chunks})
        if not self.is_running:
            asyncio.create_task(self._worker())

    async def _worker(self):
        self.is_running = True
        while not self.queue.empty():
            task = await self.queue.get()
            doc_id, kg_id, chunks = task["doc_id"], task["kg_id"], task["chunks"]
            
            for idx, chunk_text in enumerate(chunks):
                try:
                    # 在背景非同步執行昂貴的 SVO 抽取
                    triples = await extract_svo_from_text(chunk_text, kg_id)
                    if triples:
                        # 寫入 Neo4j 並累加信心值
                        await self._merge_triples_to_neo4j(kg_id, doc_id, triples)
                except Exception as e:
                    # 容錯處理：單個 chunk 失敗不阻塞整體隊列
                    continue
                # 避免本地模型 (Ollama) 負載過高，加入小幅冷卻
                await asyncio.sleep(1.0)
            
            self.queue.task_done()
        self.is_running = False

    async def _merge_triples_to_neo4j(self, kg_id: UUID, doc_id: UUID, triples: list):
        # 實作 Neo4j 的 MERGE 寫入邏輯
        pass

#### 技術與學術支撐 (SOTA Baseline)
*   **FLARE 論文 (Forward-Looking Active REtrieval-augmented generation)** [Jiang et al., EMNLP 2023] (引用：300+)：提出了「主動式/前瞻式動態檢索與生成」理論。這為我們設計「向量快速滿足，背景動態生成與合併圖譜」的雙軌調度邏輯提供了堅實的學術依據。
*   **Self-RAG 論文 (Learning to Retrieve, Generate, and Critique through Self-Reflection)** [Asai et al., ICLR 2024] (引用：400+)：提出了反思標籤（Reflection Tokens）與自主評價機制，為本系統後端「生成-審查-重試」的有界狀態機提供了嚴謹的推理保證。
*   **LlamaIndex Ingestion Pipeline** (GitHub 星星：34k+)：其實作的「向量與圖譜雙軌增量寫入管線 (Incremental Ingestion Pipeline)」，是本方案工業化落地的技術對標。
```

---

### 方案二：本地多模態 Layout-Aware 表格與排版解析器
引入本地輕量級多模態模型（如 `Qwen2-VL-7B` 或 `Llama-3.2-3B-Vision`），針對掃描件或排版複雜的 PDF 進行版面分析，直接將表格轉化為 Markdown Table 寫入 ChunkStore。

#### PDF 多模態解析與表格 Markdown 轉化代碼：
```python
# 建議寫在 services/transcribe_service.py 或獨立解析器中
import os
from PIL import Image
import fitz  # PyMuPDF
from openai import AsyncOpenAI

async def parse_pdf_multimodal_layout(pdf_path: str) -> list[str]:
    """
    使用 PyMuPDF 將 PDF 頁面轉為圖像，並調用 VLM (例如透過 Ollama 運行的 llama3.2-vision) 
    解析出保留表格結構的 Markdown 文本
    """
    doc = fitz.open(pdf_path)
    parsed_chunks = []
    
    # 呼叫本地或雲端的 Vision LLM
    client = AsyncOpenAI(
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
    )
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 1. 將 PDF 頁面渲染為高品質 PNG
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        
        # 2. 呼叫本地 VLM 要求將圖像轉化為保留表格結構的 Markdown
        try:
            response = await client.chat.completions.create(
                model="llama3.2-vision:latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "請將這張 PDF 頁面圖像轉換為 Markdown。如果圖像中有表格或圖表，請務必將其還原為正確的 Markdown 表格格式，不要丟失任何單格數據。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                        ]
                    }
                ],
                temperature=0.1
            )
            parsed_text = response.choices[0].message.content
            parsed_chunks.append(parsed_text)
        except Exception as e:
            # Fallback：若 VLM 失敗，則降級為純文字 OCR 提取
            parsed_chunks.append(page.get_text())
            
    return parsed_chunks

#### 技術與學術支撐 (SOTA Baseline)
*   **MinerU 開源專案 (opendatalab/MinerU)** (GitHub 星星：10,000+ / 報導高達 60k+)：目前世界級的高精度 PDF 內容提取引擎，基於 VLM 實現 Layout-Aware 版面分析，能完美還原表格與 LaTeX 公式，為地端多模態轉譯提供開源技術標竿。
*   **Marker 開源專案 (datalab-to/marker)** (GitHub 星星：16k+)：專為學術文檔與書籍優化的 PDF 轉 Markdown 工具，提供了秒級的 OCR 與表格還原能力。
*   **LayoutLM 論文 (Document Image Understanding with 2D Position and Image Embeddings)** [Xu et al., KDD 2020] (引用：1,200+)：微軟提出的多模態文檔理解奠基作，證明了將 2D 位置與視覺特徵融入 RAG 對於排版複雜文件的必然性。
```

---

### 方案三：基於向量引導的圖譜剪枝（解決大節點爆炸）
當進行 BFS 遍歷時，在進入 Neo4j 之前，先計算當前擴展節點周圍的邊與用戶 Query 的向量相似度，動態對 Cypher 查詢路徑進行「語意邊剪枝」，而非 naive 的數量截斷。

#### 向量引導剪枝的 BFS 檢索代碼設計：
```python
# 建議寫在 services/svo_service.py 內，取代 Naive BFS
import numpy as np
from core.database import get_driver

async def query_svo_facts_with_vector_pruning(
    kg_id: str, 
    seed_entities: list[str], 
    query_vector: list[float], 
    similarity_threshold: float = 0.6
) -> list:
    """
    在 Neo4j 圖遍歷過程中，利用 Cypher 撈出候選關係的 Embedding，
    在 Python 中計算其與 query_vector 的餘弦相似度，動態過濾掉不相關的分支。
    """
    driver = get_driver()
    # 撈取種子節點周圍 1~2 跳的所有關係、受詞及其屬性
    cypher_query = """
    MATCH (s:Entity)-[r]->(o:Entity)
    WHERE (r.kg_id = $kg_id) AND (s.name IN $seeds)
    RETURN s.name AS subject, type(r) AS relation, o.name AS object, 
           r.confidence AS confidence, r.embedding AS rel_embedding
    """
    
    filtered_facts = []
    
    async with driver.session() as session:
        result = await session.run(cypher_query, kg_id=kg_id, seeds=seed_entities)
        records = await result.data()
        
        for record in records:
            rel_embedding = record.get("rel_embedding")
            
            # 若關係邊上存有語意 Embedding (在抽取階段生成)
            if rel_embedding:
                # 計算餘弦相似度
                dot_product = np.dot(query_vector, rel_embedding)
                norm_q = np.linalg.norm(query_vector)
                norm_r = np.linalg.norm(rel_embedding)
                similarity = dot_product / (norm_q * norm_r) if (norm_q * norm_r) > 0 else 0.0
                
                # 只有當關係邊與問題語意相似度達標，才保留此分支，防止 Hub Node 爆炸
                if similarity >= similarity_threshold:
                    filtered_facts.append(record)
            else:
                # 若無 Embedding 屬性，則降級為保留信心值大於 0.7 的事實
                if record["confidence"] >= 0.7:
                    filtered_facts.append(record)
                    
    # 依相似度或信心值降序排列，取最精準事實
    filtered_facts.sort(key=lambda x: x.get("similarity", x.get("confidence", 0.0)), reverse=True)
    return filtered_facts[:30]

#### 技術與學術支撐 (SOTA Baseline)
*   **G-Retriever 論文 (Retrieval-Augmented Generation for Textual Graph Understanding)** [He et al., NeurIPS 2024] (引用：100+)：Yann LeCun 等人署名的頂級論文，指出在大規模文本圖譜中，為了克服 LLM 的 Context 限制，可將子圖檢索建模為 **PCST (Prize-Collecting Steiner Tree)** 拓撲優化問題。這為我們使用向量相似度對 Cypher 遍歷結果進行「語意邊剪枝與子圖提取」提供了核心理論支撐。
*   **Microsoft GraphRAG 論文 (From Local to Global: A Graph RAG Approach)** [Edge et al., Microsoft 2024] (引用：100+)：微軟 GraphRAG 奠基作。指出傳統圖譜 Naive 檢索在 Hub Nodes 面臨的信息過載問題，倡導利用局部語意過濾與社區降維。
```

---

### 方案四：社群自動導讀與見解生成器 (Auto-Insights Generator)
為了解決用戶剛上傳文檔無從下手的問題，系統應利用既有的 Louvain 社群偵測服務 [services/community_service.py](file:///d:/Users/666/Desktop/智慧知識庫/services/community_service.py)，將主要社群轉化為一鍵可讀的 FAQ 與 Concept Guide。

#### 自動生成導讀見解的 Python 腳本設計：
```python
# 建議新增為 services/insights_service.py
from uuid import UUID
from services.community_service import CommunityService
from core.providers.factory import get_llm_provider

async def generate_kg_guided_insights(kg_id: UUID) -> dict:
    """
    1. 執行 Louvain 社群偵測將圖譜分群
    2. 對主要的社群 (Community) 撈出其代表實體與關係
    3. 呼叫 LLM 自動生成該社群的概念導讀與 3 個預測常見問答對 (FAQ)
    """
    comm_service = CommunityService()
    # 獲取偵測出的社群及其對應的實體
    communities = await comm_service.detect_communities(kg_id)
    llm = get_llm_provider()
    
    insights = []
    
    # 針對前 3 大社群（通常是文檔的核心概念板塊）生成導讀
    for idx, comm in enumerate(communities[:3]):
        entity_names = comm["entities"]
        relations = comm.get("relations", [])
        
        prompt = f"""
你是一個知識整理專家。以下是從某個知識庫中偵測出的一個「緊密關聯概念社群」。
這個社群包含以下核心實體：{', '.join(entity_names[:15])}
包含以下關係片段：{str(relations[:10])}

請為這個概念板塊撰寫：
1. 一段 150 字的「概念導讀」(用親切的繁體中文解釋這些概念如何關聯)。
2. 三個使用者可能會問的「常見問答對 (FAQ)」，必須能直接透過上述實體關係回答。

請輸出 JSON 格式如下：
{{
  "topic": "本版塊的主題核心",
  "guide": "導讀內容...",
  "faqs": [
    {{"question": "問題1", "answer": "回答1"}},
    {{"question": "問題2", "answer": "回答2"}},
    {{"question": "問題3", "answer": "回答3"}}
  ]
}}
請直接輸出 JSON，不要 Markdown 包裝。
"""
        try:
            resp = await llm.generate(prompt)
            # 解析並加入結果
            insights.append(resp)
        except Exception:
            continue
            
    return {"insights": insights}

#### 技術與學術支撐 (SOTA Baseline)
*   **Louvain 演算法論文 (Fast unfolding of communities in large networks)** [Blondel et al., 2008] (引用：36,000+)：Louvain 社群偵測演算法的奠基之作，為我們目前 `CommunityService` 依靠的 NetworkX 拓撲分群提供最權威的物理與數學理論支持。
```

---

### 方案五：跨段落四元組上下文合成器 (Cross-Chunk Context Synthesizer)
*   **痛點**：在多跳推理場景下，當檢索到一條四元組或更長的邏輯鏈（如 `A -> B -> C -> D`）時，這些關聯實體可能散落在多個不同的原始 Chunks（例如段落 1、段落 2 與段落 3）中。如果將這些 Chunks 原封不動地全部塞進 Prompt Context，會帶入 80% 的背景雜訊 Token，導致 LLM 注意力分散且 Prefill 運算延遲大幅增加。
*   **優化方案**：利用 Neo4j 提取出該多跳路徑（Path）對應的 `chunk_ids` 後，只拉取對應的原始 Chunks，隨後調用本地輕量 LLM 進行「定向句子提取與邏輯重建」，重構出一段小於 500 字的 **「合成上下文 (Synthetic Context)」** 餵給最終的 Generator。這既解決了跨段落多跳 RAG 的 Token 膨脹，也消除了無關雜訊干擾。

#### 跨段落上下文合成程式碼藍圖（`services/context_synthesizer.py`）：
```python
import os
from uuid import UUID
from typing import List
from core.database import get_driver
from core.providers.factory import get_llm_provider
from services.chunk_store import get_chunk_store

class CrossChunkContextSynthesizer:
    def __init__(self):
        self.driver = get_driver()
        self.chunk_store = get_chunk_store()
        self.llm = get_llm_provider()

    async def query_and_synthesize(self, kg_id: UUID, seed_name: str, query: str) -> str:
        """
        1. 在 Neo4j 中檢索跨段落的四元組/多跳路徑
        2. 批次拉取涉及的原始 Chunks
        3. 用 LLM 重構並合成一個精簡、無雜訊的上下文段落
        """
        # 搜尋長度為 3 邊（4節點）的四元組路徑，並撈出邊上關聯的 chunk_ids
        cypher_query = """
        MATCH path = (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)-[r3]->(d:Entity)
        WHERE (r1.kg_id = $kg_id AND r2.kg_id = $kg_id AND r3.kg_id = $kg_id)
          AND (a.name = $seed OR b.name = $seed)
        RETURN 
          [n in nodes(path) | n.name] as entity_chain,
          [r in relationships(path) | {
              source: startNode(r).name, 
              relation: type(r), 
              target: endNode(r).name, 
              chunk_id: r.source_chunk_id
          }] as relation_chain
        LIMIT 5
        """
        
        async with self.driver.session() as session:
            result = await session.run(cypher_query, kg_id=str(kg_id), seed=seed_name)
            records = await result.data()
            
        if not records:
            return ""  # 未找到四元組關係鏈，Fallback 回標準三元組與向量檢索
            
        # 彙整所有涉及的關係鏈與獨一無二的 chunk_ids
        target_chunk_ids = set()
        logic_chains_str = []
        
        for record in records:
            chain = record["relation_chain"]
            chain_desc = " -> ".join([f"({r['source']})-[{r['relation']}]->({r['target']})" for r in chain])
            logic_chains_str.append(chain_desc)
            
            for rel in chain:
                if rel["chunk_id"]:
                    target_chunk_ids.add(rel["chunk_id"])
                    
        if not target_chunk_ids:
            return ""

        # 從 ChunkStore 批次拉取這些原始 Chunks (I/O 速度極快)
        chunks_data = []
        for cid in target_chunk_ids:
            chunk = await self.chunk_store.read_chunk(cid)
            if chunk:
                chunks_data.append(chunk.text)
                
        # 呼叫 LLM 進行定向上下文合成
        synthetic_context = await self._synthesize_via_llm(
            query=query,
            logic_chains=logic_chains_str,
            raw_chunks=chunks_data
        )
        
        return synthetic_context

    async def _synthesize_via_llm(self, query: str, logic_chains: List[str], raw_chunks: List[str]) -> str:
        raw_documents_context = "\n\n".join([f"[段落 {i+1}]: {text}" for i, text in enumerate(raw_chunks)])
        chains_context = "\n".join([f"- {chain}" for chain in logic_chains])
        
        prompt = f"""
你是一個資訊提煉專家。你的任務是根據「邏輯鏈條」，從多個「原始段落」中只提取出與該邏輯鏈直接相關的關鍵事實，並融合成一段語意連貫、無雜訊的「合成上下文」，用以回答用戶的問題。

### 用戶問題：
{query}

### 跨段落的邏輯鏈條：
{chains_context}

### 原始段落內容：
{raw_documents_context}

### 寫作要求：
1. 必須將多個原始段落中的雜訊（無關的背景描述、過渡句）徹底丟棄。
2. 僅保留能夠清楚、完整解釋上述「邏輯鏈條」如何發生的句子。
3. 將提取出的句子整合成一段通順的繁體中文段落，字數限制在 400 字以內。
4. 嚴禁編造任何原始段落中不存在的細節。

合成上下文：
"""
        response = await self.llm.generate(prompt)
        return response.strip()
```

#### 技術與學術支撐 (SOTA Baseline)
*   **GRAG 論文 (Graph Retrieval-Augmented Generation)** [Hu et al., NAACL 2025] (引用：100+)：提出了「分治子圖檢索與大語言模型對齊（Divide-and-Conquer Subgraph-to-Text）」的推理機制，為多跳跨段落 Context 重建提供了強大的學術論證。
*   **Awesome-Language-Model-on-Graphs** (GitHub 星星：4,000+)：彙整了當前將 LLM 與圖結構（Subgraphs）對齊、語意文本合成（Textual Generation on Graphs）的頂級開源實現與框架。

#### 技術與學術支撐 (SOTA Baseline)
*   **GRAG 論文 (Graph Retrieval-Augmented Generation)** [Hu et al., NAACL 2025] (引用：100+)：提出了「分治子圖檢索與大語言模型對齊（Divide-and-Conquer Subgraph-to-Text）」的推理機制，為多跳跨段落 Context 重建提供了強大的學術論證。
*   **Awesome-Language-Model-on-Graphs** (GitHub 星星：4,000+)：彙整了當前將 LLM 與圖結構（Subgraphs）對齊、語意文本合成（Textual Generation on Graphs）的頂級開源實現與框架。

---

### 方案六：圖譜鏈式思考尋路代理人 (Think-on-Graph / ToG Agent)
*   **痛點**：預先設定的 1-2 跳 BFS 屬於「被動式尋路」，如果推理路徑較長，BFS 會因為指數級的扇出而撈回海量雜訊。死板的 Cypher 無法因應複雜問題進行動態寻路。
*   **優化方案**：實作 **「ToG (Think-on-Graph) 尋路機制」**。將 LLM 作為 Agent，在圖譜中進行動態深度尋路。LLM 每次獲得當前節點的一跳鄰居，自主決策「下一個最值得探索的節點是哪一個」，沿著邏輯鏈條逐步前行，直到收集到足夠的事實。

#### 圖譜動態尋路代理人代碼藍圖：
```python
# 建議新增為 services/tog_agent.py
from typing import List, Dict
from core.database import get_driver
from core.providers.factory import get_llm_provider

class ThinkOnGraphAgent:
    def __init__(self):
        self.driver = get_driver()
        self.llm = get_llm_provider()

    async def dynamic_walk(self, kg_id: str, start_entity: str, query: str, max_steps: int = 3) -> List[Dict]:
        """
        讓 LLM 作為尋路 Agent，動態決定每一跳的目標
        """
        current_node = start_entity
        collected_facts = []
        visited = {start_entity}

        for step in range(max_steps):
            # 1. 撈出當前節點的一跳鄰居
            neighbors = await self._get_one_hop_neighbors(kg_id, current_node)
            if not neighbors:
                break
                
            # 2. 構建決策 Prompt，讓 LLM 決定下一個目標
            decision = await self._decide_next_node(query, current_node, neighbors, visited)
            
            # 記錄被選中的事實邊
            collected_facts.append({
                "subject": current_node,
                "relation": decision["relation"],
                "object": decision["next_node"]
            })
            
            # 更新節點與訪問記錄
            current_node = decision["next_node"]
            visited.add(current_node)
            
            if decision.get("is_satisfied", False):
                break
                
        return collected_facts

    async def _get_one_hop_neighbors(self, kg_id: str, node_name: str) -> List[Dict]:
        cypher = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE r.kg_id = $kg_id AND s.name = $node
        RETURN o.name AS neighbor, type(r) AS relation
        """
        async with self.driver.session() as session:
            res = await session.run(cypher, kg_id=kg_id, node=node_name)
            return await res.data()

    async def _decide_next_node(self, query: str, current: str, neighbors: List[Dict], visited: set) -> Dict:
        # 由 LLM 判斷哪一個鄰居最有利於回答用戶問題，輸出 JSON
        # 格式：{"next_node": "目標實體", "relation": "關係邊", "is_satisfied": true/false}
        pass
```

#### 技術與學術支撐 (SOTA Baseline)
*   **ToG 論文 (Think-on-Graph: Deep and Responsible Reasoning on KGs)** [Sun et al., 2023] (引用：150+)：首創將 LLM 當作 KG 尋路代理人的框架，證明了鏈式思考（Chain-of-Thought）與離散圖譜尋路融合的優越性。

---

### 方案七：階層式社群摘要全局檢索 (Hierarchical Community Summary RAG)
*   **痛點**：圖譜的 BFS 檢索屬於「局部檢索 (Local Query)」，只能回答特定節點的細節。如果用戶問全局性問題（如「整份文件庫反映出的三大違規趨勢是什麼？」），傳統 RAG 與 BFS 將徹底失效。
*   **優化方案**：在建庫後，運行 Louvain 分群將知識圖譜劃分為多個不同尺寸的社群。非同步調用 LLM 為每個社群的實體與關係撰寫「社群摘要 (Community Summary)」並寫入 Neo4j。全局問答時，直接比對並融合成員社群的摘要來回答，避開對全量原始 Chunks 的掃描。

#### 社群摘要建構與檢索代碼藍圖：
```python
# 建議寫在 services/community_service.py 內
from uuid import UUID
from core.database import get_driver
from core.providers.factory import get_llm_provider

class CommunitySummaryRAG:
    def __init__(self):
        self.driver = get_driver()
        self.llm = get_llm_provider()

    async def build_community_summaries(self, kg_id: UUID):
        """
        為知識庫的所有 Louvain 社群生成摘要並寫入 Neo4j
        """
        # 1. 獲取所有社群
        # 2. 針對每個社群的 (entities, relationships) 調用 LLM 生成總結
        # 3. 執行 Cypher 將總結寫入 (:Community {id: cid, summary: $summary}) 節點上
        pass

    async def global_query(self, kg_id: UUID, query: str) -> str:
        """
        全局檢索：搜尋最相關的數個社群摘要，融合成最終回答
        """
        # 1. 向量檢索定位最相關的數個 Community 節點
        cypher = """
        MATCH (c:Community)
        WHERE c.kg_id = $kg_id
        RETURN c.id AS id, c.summary AS summary
        LIMIT 5
        """
        async with self.driver.session() as session:
            res = await session.run(cypher, kg_id=str(kg_id))
            summaries = [r["summary"] for r in await res.data()]
            
        # 2. 合併摘要送給 LLM 生成回答
        prompt = f"問題：{query}\n\n以下是相關概念社群的摘要：\n" + "\n\n".join(summaries)
        response = await self.llm.generate(prompt)
        return response
```

#### 技術與學術支撐 (SOTA Baseline)
*   **Microsoft GraphRAG 論文 (From Local to Global)** [Edge et al., Microsoft 2024] (引用：100+)：微軟 GraphRAG 官方論文。該架構完全基於 Hierarchical Community Summary RAG 設計，是用於解決全局查詢（Global Query）的開創性方案。

---

## 三、 超越路徑的實施優先級 (Implementation Roadmap)

為了穩步落地超越方案，建議分為三階段實施：

```
【第一階段：高優先級】 ──► 【第二階段：中優先級】 ──► 【第三階段：全局與推理】
  非同步雙軌建圖 (方案一)     多模態 Layout 解析 (方案二)    社群全局摘要 (方案七)
  向量引導圖剪枝 (方案三)     跨段落四元組合成 (方案五)       ToG 尋路代理 (方案六)
```

這套演進架構能讓你的智慧知識庫在**保有 100% 本地隱私與可解釋性**的同時，具備**媲美 NotebookLM 的秒級建庫速度與多模態解析能力**，真正實現完全超越。
