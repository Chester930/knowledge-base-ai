import asyncio
import logging
import sys
from uuid import UUID
from core.database import connect, disconnect, get_driver
from core.config import settings
from services.embedding_service import init_embedding_service
from repositories.concept_repo import ConceptRepository
from services.concept_engine import extract_and_init_document_concepts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

async def fix_missing_concepts():
    logger.info("連線資料庫並初始化服務…")
    await connect()
    driver = get_driver()
    
    # 初始化 embedding
    svc = init_embedding_service(settings.embedding_model)
    await ConceptRepository(driver).create_vector_index(svc.dim)
    
    # 查詢沒有 EFFECTIVE 概念關聯的所有文件
    logger.info("查詢缺少概念的文件…")
    result = await driver.execute_query(
        """
        MATCH (d:Document)
        WHERE NOT (d)-[:EFFECTIVE]->(:ConceptNode)
        RETURN d.id AS id, d.title AS title, d.content AS content
        """
    )
    
    records = result.records
    logger.info(f"找到 {len(records)} 個文件缺少概念關聯。")
    
    for i, r in enumerate(records, 1):
        doc_id = UUID(r["id"])
        title = r["title"]
        content = r["content"]
        
        logger.info(f"[{i}/{len(records)}] 正在為「{title}」({doc_id}) 提取概念…")
        try:
            await extract_and_init_document_concepts(doc_id, content)
            logger.info(f"  -> 成功！")
        except Exception as e:
            logger.error(f"  -> 失敗：{e}")
            
    await disconnect()
    logger.info("全部處理完成。")

if __name__ == "__main__":
    asyncio.run(fix_missing_concepts())
