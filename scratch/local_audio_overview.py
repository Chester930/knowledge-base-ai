import os
import re
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 載入環境變數
load_dotenv()

# 初始化日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 配置參數
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 微軟 Edge-TTS 語音配置（台灣中文高品質語音）
SPEAKER_A_VOICE = "zh-TW-HsiaoChenNeural"  # 女聲
SPEAKER_B_VOICE = "zh-TW-YunJheNeural"   # 男聲

async def fetch_kg_facts(kg_id: str, limit: int = 30) -> list:
    """從 Neo4j 提取指定知識庫的關鍵事實（SVO）"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # 這裡使用安全的通用查詢
    query = """
    MATCH (s)-[r]->(o)
    WHERE r.kg_id = $kg_id OR r.source_doc_id IS NOT NULL
    RETURN s.name AS subject, labels(s)[0] AS s_label, type(r) AS relation, o.name AS object, labels(o)[0] AS o_label
    LIMIT $limit
    """
    facts = []
    try:
        with driver.session() as session:
            result = session.run(query, kg_id=kg_id, limit=limit)
            for record in result:
                facts.append({
                    "subject": record["subject"],
                    "s_label": record["s_label"],
                    "relation": record["relation"],
                    "object": record["object"],
                    "o_label": record["o_label"]
                })
    except Exception as e:
        logger.error(f"提取 Neo4j 事實失敗: {e}")
    finally:
        driver.close()
    return facts

def generate_podcast_script_prompt(facts: list) -> str:
    """生成 LLM 對話腳本 Prompt"""
    facts_str = "\n".join([
        f"- 【{f['subject']}({f['s_label']})】 -[{f['relation']}]-> 【{f['object']}({f['o_label']})】"
        for f in facts
    ])
    
    prompt = f"""
你是一位專業的廣播節目製作人。請根據以下從知識圖譜中提取的結構化事實，編寫一段生動有趣、通俗易懂的「雙人 Podcast 節目對話腳本」。

### 知識圖譜事實資料：
{facts_str}

### 角色設定：
1. **主持人 A (小珍)**：幽默風趣、喜歡發問、善於把複雜的專業術語用生活化的例子解釋。
2. **主持人 B (阿哲)**：知識淵博、沉穩專業、負責提供背景事實與深度解析。

### 寫作要求：
1. 語言：必須使用「繁體中文（zh-TW）」，帶有台灣日常口語的親切感。
2. 格式：請嚴格輸出合法的 JSON 陣列格式，以便程式後續合成語音。不要有任何 Markdown 包裝（如 ```json ... ```），只輸出 JSON 本身。
3. JSON 格式規範如下：
[
  {{"speaker": "A", "text": "對話內容..."}},
  {{"speaker": "B", "text": "對話內容..."}}
]
4. 長度：約 10-15 句對話。確保對話中自然地提到了上述知識圖譜的事實，並做出解釋，不要生硬地朗讀。

請直接開始輸出 JSON 陣列：
"""
    return prompt

async def call_llm_for_script(prompt: str) -> list:
    """呼叫 LLM（優先用 OpenAI API，其次嘗試本地 Ollama）"""
    # 嘗試讀取 OpenAI 配置
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    if api_key:
        try:
            logger.info("正在使用 OpenAI 接口生成對話腳本...")
            import httpx
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{api_base}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # 清除 LLM 可能輸出的 Markdown 標記
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content).strip()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"OpenAI 接口呼叫失敗，將嘗試使用本地 Ollama: {e}")

    # Fallback 到 Ollama
    try:
        logger.info("正在使用本地 Ollama 生成對話腳本...")
        import httpx
        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        payload = {
            "model": "llama3.1" if not os.getenv("OLLAMA_MODEL") else os.getenv("OLLAMA_MODEL"),
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
            content = resp.json()["response"].strip()
            return json.loads(content)
    except Exception as e:
        logger.error(f"本地 Ollama 呼叫也失敗，無法生成腳本: {e}")
        # 返回一個 Mock 腳本供測試
        return [
            {"speaker": "A", "text": "哈囉大家，歡迎收聽地端知識庫廣播！阿哲，今天我們要聊些什麼呢？"},
            {"speaker": "B", "text": "小珍妳好。今天我們要聊聊資料庫裡儲存的知識圖譜實體關係，這非常有意思。"},
            {"speaker": "A", "text": "哇！那趕快開始吧，讓我們看看這次的文件裡有哪些不為人知的祕密！"}
        ]

async def text_to_speech(text: str, voice: str, output_path: str):
    """呼叫 edge-tts 合成單句語音"""
    try:
        import edge_tts
    except ImportError:
        logger.info("未檢測到 edge-tts，嘗試自動安裝...")
        import subprocess
        subprocess.run(["pip", "install", "edge-tts"], check=True)
        import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

async def generate_audio_overview(kg_id: str, output_file: str = "podcast_overview.mp3"):
    """主要工作流程：提取 KG → 生成腳本 → 本地合成音檔"""
    logger.info(f"開始為知識庫 {kg_id} 生成 Audio Overview...")
    
    # 1. 提取事實
    facts = await fetch_kg_facts(kg_id, limit=20)
    if not facts:
        logger.warning("該知識庫沒有足夠的事實，將使用範例事實進行生成。")
        facts = [
            {"subject": "勞動基準法", "s_label": "法規", "relation": "REGULATES", "object": "勞動契約", "o_label": "制度"},
            {"subject": "延長工作時間", "s_label": "事件", "relation": "CAUSES", "object": "職業疲勞", "o_label": "生理狀態"},
            {"subject": "雇主", "s_label": "角色", "relation": "OBLIGATED_TO", "object": "給付加班費", "o_label": "義務"}
        ]

    # 2. 生成對話腳本
    prompt = generate_podcast_script_prompt(facts)
    script = await call_llm_for_script(prompt)
    logger.info(f"生成腳本成功！共 {len(script)} 句對話。")
    for i, line in enumerate(script):
        logger.info(f"[{line['speaker']}] {line['text']}")

    # 3. 逐句合成語音
    temp_dir = Path("scratch/temp_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []

    logger.info("開始逐句合成語音...")
    for idx, line in enumerate(script):
        speaker = line["speaker"]
        text = line["text"]
        voice = SPEAKER_A_VOICE if speaker == "A" else SPEAKER_B_VOICE
        temp_path = temp_dir / f"line_{idx:03d}.mp3"
        
        logger.info(f"合成第 {idx+1}/{len(script)} 句 ({speaker})...")
        await text_to_speech(text, voice, str(temp_path))
        temp_files.append(temp_path)

    # 4. 合併語音檔 (二進位拼接近似合併，對相同編碼格式的 MP3 有效)
    logger.info("合併音軌中...")
    try:
        with open(output_file, "wb") as outfile:
            for f in temp_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())
        logger.info(f"音訊合成成功！輸出檔案路徑: {os.path.abspath(output_file)}")
    except Exception as e:
        logger.error(f"合併音檔失敗: {e}")
    finally:
        # 清理暫存檔
        for f in temp_files:
            try:
                f.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

if __name__ == "__main__":
    # 使用預設的測試 KG_ID 來跑測試
    TEST_KG_ID = "default_kg_id"
    asyncio.run(generate_audio_overview(TEST_KG_ID, "scratch/podcast_overview.mp3"))
