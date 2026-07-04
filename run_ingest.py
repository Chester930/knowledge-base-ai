"""
批次匯入文件的獨立腳本（繞過 HTTP timeout）。
直接在 Neo4j 環境中跑，可追蹤進度。

用法：
  python run_ingest.py /path/to/docs
  python run_ingest.py /path/to/docs --kg <kg_id>
"""
import argparse
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


async def main(ingest_dir: str, kg_id: str | None):
    from core.database import connect, disconnect, get_driver
    from repositories.concept_repo import ConceptRepository
    from core.providers.factory import init_providers
    from services.ingestion_service import ingest_directory, SUPPORTED_EXTENSIONS
    from services.svo_service import create_entity_index
    from core.config import settings

    logger.info("=== 智慧知識庫批次匯入開始 ===")

    src = Path(ingest_dir)
    if not src.is_dir():
        logger.error(f"目錄不存在：{ingest_dir}")
        sys.exit(1)

    files = [f for f in src.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()]
    logger.info(f"目錄：{ingest_dir}")
    logger.info(f"找到 {len(files)} 個支援格式的文件")
    logger.info(f"LLM Provider：{settings.llm_provider}  Embedding：{settings.embedding_provider}")
    if kg_id:
        logger.info(f"目標 KG：{kg_id}")

    logger.info("連線 Neo4j…")
    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()
    logger.info("資料庫連線完成，開始匯入…")

    start = time.time()
    success, errors = await ingest_directory(ingest_dir, kg_id=kg_id)
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
    parser = argparse.ArgumentParser(description="批次匯入文件至智慧知識庫")
    parser.add_argument("dir", help="要匯入的文件目錄路徑")
    parser.add_argument("--kg", metavar="KG_ID", default=None, help="目標知識圖譜 ID（可選）")
    args = parser.parse_args()
    asyncio.run(main(args.dir, args.kg))
