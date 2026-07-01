import asyncio
import os
import sys
import shutil
from pathlib import Path
from uuid import UUID

# 設定 PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from core.database import connect, disconnect, get_driver
from core.providers.factory import init_providers
from services.ingestion_service import _read_pdf
from services.classify_service import assign_document_to_kg
from services.svo_service import build_graph_for_kg, apply_type_labels
from services.kb_skill_service import sync_public_kgs
from repositories.knowledge_graph_repo import KnowledgeGraphRepository

PDF_PATH = Path("workspace/downloads_import/養龍蝦：OpenClaw從入門到精通.pdf")
KG_ID = UUID("f7c7662b-1c24-4e6c-b74a-1aa65fbc456d")  # 軟體架構與專案開發

async def main():
    if not PDF_PATH.exists():
        print(f"錯誤：找不到 PDF 檔案 {PDF_PATH}")
        sys.exit(1)
        
    print("--- 1. 連線資料庫與初始化 Provider ---")
    await connect()
    init_providers()
    driver = get_driver()
    
    print("\n--- 2. 解析 PDF 檔案 ---")
    print(f"正在解析 {PDF_PATH.name}...")
    try:
        text = _read_pdf(PDF_PATH)
        print(f"解析成功！長度：{len(text)} 字元")
    except Exception as e:
        print(f"解析 PDF 失敗：{e}")
        await disconnect()
        sys.exit(1)
        
    print("\n--- 3. 寫入暫存區 staging ---")
    staging_dir = Path("workspace/_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    txt_filename = "養龍蝦：OpenClaw從入門到精通.txt"
    txt_path = staging_dir / txt_filename
    txt_path.write_text(text, encoding="utf-8")
    print(f"已寫入暫存區：{txt_path.resolve()}")
    
    print("\n--- 4. 分配文件至『軟體架構與專案開發』KG ---")
    try:
        await assign_document_to_kg(txt_filename, KG_ID)
        print("已成功分配文件並完成路由層初始化！")
    except Exception as e:
        print(f"分配文件失敗：{e}")
        await disconnect()
        sys.exit(1)
        
    print("\n--- 5. 提取 SVO 關係三元組 (同步進行，非背景 Task) ---")
    # 查找剛才寫入的 Document ID
    kg_repo = KnowledgeGraphRepository(driver)
    docs = await kg_repo.get_documents(KG_ID)
    doc_id = None
    for d in docs:
        if "養龍蝦" in d.get("title", ""):
            doc_id = d["id"]
            break
            
    if doc_id:
        print(f"找到對應 Document ID: {doc_id}，開始提取 SVOFact...")
        try:
            # 同步跑 build_graph_for_kg 直到完成
            async for progress in build_graph_for_kg(KG_ID, doc_ids=[doc_id], force_rebuild=True):
                print(f"  [Progress] {progress}")
            await apply_type_labels(KG_ID)
            print("SVO 關係三元組提取與標籤標記完成！")
        except Exception as e:
            print(f"SVO 提取失敗：{e}")
    else:
        print("警告：未在 KG 中找到對應的 Document ID")
        
    print("\n--- 6. 刷新公開知識庫 Registry (sync_public_kgs) ---")
    res_sync = await sync_public_kgs(driver)
    print(f"同步結果: {res_sync}")
    
    # 也順便將 registry.json 複製到 Hub 專案中，確保 Hub 能即時同步
    try:
        shutil.copy("registry.json", "../world-knowledge-hub/registry_local.json")
        print("已將最新 registry.json 同步至 world-knowledge-hub 專案")
    except Exception as se:
        print(f"複製 registry.json 到 Hub 失敗: {se}")

    await disconnect()
    print("\n🎉 養龍蝦 PDF 導入與圖譜建置完全完成！")

if __name__ == "__main__":
    asyncio.run(main())
