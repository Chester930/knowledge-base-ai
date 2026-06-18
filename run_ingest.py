"""
批次匯入文件的獨立腳本（繞過 HTTP timeout）。
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

INGEST_DIR = r"D:\666\Downloads"


async def main():
    from core.database import connect, disconnect, get_driver
    from repositories.concept_repo import ConceptRepository
    from core.providers.factory import init_providers
    from services.ingestion_service import ingest_directory, SUPPORTED_EXTENSIONS
    from core.config import settings

    logger.info("=== 智慧知識庫批次匯入開始 ===")

    src = Path(INGEST_DIR)
    files = [f for f in src.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()]
    logger.info(f"目錄：{INGEST_DIR}")
    logger.info(f"找到 {len(files)} 個支援格式的文件")
    logger.info(f"LLM Provider：{settings.llm_provider}  Embedding：{settings.embedding_provider}")

    logger.info("連線 Neo4j…")
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    logger.info("資料庫連線完成，開始匯入…")

    start = time.time()
    success, errors = await ingest_directory(INGEST_DIR)
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
