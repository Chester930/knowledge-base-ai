"""
批次搬移並匯入文件的獨立腳本（繞過 HTTP timeout）。
直接在 Neo4j 環境中跑，可追蹤進度。
用法：python run_ingest.py
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SOURCE_DIR = r"C:\Users\666\Downloads"
TARGET_DIR = r"D:\666\Downloads"


async def main():
    from core.database import connect, disconnect, get_driver
    from repositories.concept_repo import ConceptRepository
    from services.embedding_service import init_embedding_service
    from services.ingestion_service import move_and_ingest, SUPPORTED_EXTENSIONS
    from core.config import settings

    logger.info("=== 智慧知識庫批次匯入開始 ===")

    # 統計來源
    src = Path(SOURCE_DIR)
    files = [f for f in src.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()]
    logger.info(f"來源目錄：{SOURCE_DIR}")
    logger.info(f"目標目錄：{TARGET_DIR}")
    logger.info(f"找到 {len(files)} 個支援格式的文件")

    # 連線
    logger.info("連線 Neo4j…")
    await connect()
    svc = init_embedding_service(settings.embedding_model)
    await ConceptRepository(get_driver()).create_vector_index(svc.dim)
    logger.info("資料庫連線完成")

    start = time.time()
    success, errors = await move_and_ingest(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        delete_on_success=True,
    )
    elapsed = time.time() - start

    logger.info("=== 匯入完成 ===")
    logger.info(f"✅ 成功：{len(success)} 個")
    logger.info(f"❌ 失敗：{len(errors)} 個")
    logger.info(f"⏱  耗時：{elapsed/60:.1f} 分鐘")

    if errors:
        logger.info("失敗清單：")
        for e in errors:
            logger.info(f"  {e}")

    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
