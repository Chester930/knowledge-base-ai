"""
run_sync_public_kgs.py — 批次同步所有公開 KG 到 registry.json

用法：
    python run_sync_public_kgs.py
"""
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    from core.database import connect, disconnect, get_driver
    from core.providers.factory import init_providers
    from repositories.concept_repo import ConceptRepository
    from services.kb_skill_service import sync_public_kgs

    await connect()
    init_providers()

    # 確保向量索引存在
    from core.providers.factory import get_embedding_provider
    embedding = get_embedding_provider()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)

    logger.info("開始同步所有公開 KG → registry.json …")
    result = await sync_public_kgs(get_driver())

    logger.info(
        f"同步完成：✅ {result['synced']} 個 KG，"
        f"🗑️ 移除 {result['removed']} 個，"
        f"❌ {len(result['errors'])} 個錯誤"
    )
    if result["errors"]:
        for err in result["errors"]:
            logger.warning(f"  - {err}")

    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
