"""
為所有（或指定）KG 的 Entity 節點套用型別語義標籤。
依 Entity.type 屬性附加對應的 Neo4j 標籤（不移除 :Entity 主標籤）：
  概念→Concept  算法→Algorithm  技術→Technology  方法→Method
  工具→Tool     框架→Framework  模型→Model       系統→System
  人物→Person   組織→Organization 資料集→Dataset 指標→Metric  其他→Other

用法：
  python run_label_kg.py              # 對所有 KG 套標籤
  python run_label_kg.py --kg <id>    # 只跑指定 KG
"""
import argparse
import asyncio
import logging
import sys
from uuid import UUID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("label_kg_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def main(target_kg_id: str | None):
    from core.database import connect, disconnect, get_driver
    from core.providers.factory import init_providers
    from repositories.concept_repo import ConceptRepository
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from services.svo_service import apply_type_labels, create_entity_index

    logger.info("=== KG 型別標籤套用開始 ===")
    logger.info(f"target={target_kg_id or '全部'}")

    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()

    kg_repo = KnowledgeGraphRepository(get_driver())

    if target_kg_id:
        kg = await kg_repo.get_by_id(UUID(target_kg_id))
        if kg is None:
            logger.error(f"找不到 KG：{target_kg_id}")
            await disconnect()
            return
        kgs = [kg]
    else:
        kgs = await kg_repo.list_all(include_private=True)

    if not kgs:
        logger.warning("沒有任何 KG")
        await disconnect()
        return

    logger.info(f"共 {len(kgs)} 個 KG 待處理")

    total_nodes = 0
    for i, kg in enumerate(kgs, 1):
        logger.info(f"\n[{i}/{len(kgs)}] {kg.name}（entity_count={kg.entity_count}）")
        if kg.entity_count == 0:
            logger.info("  ⏭  無 Entity，跳過")
            continue
        try:
            stats = await apply_type_labels(kg.id, db_name=kg.db_name)
            n = sum(stats.values())
            total_nodes += n
            logger.info(f"  ✅ 套用完成：{n} 個節點")
            for label, count in sorted(stats.items(), key=lambda x: -x[1]):
                logger.info(f"     :{label} → {count} 個")
        except Exception as e:
            logger.exception(f"  ❌ 標籤失敗：{kg.name} — {e}")

    logger.info(f"\n=== 完成，共 {total_nodes} 個節點獲得型別標籤 ===")
    await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Entity 型別標籤套用")
    parser.add_argument("--kg", type=str, default=None, help="只處理指定 kg_id")
    args = parser.parse_args()

    asyncio.run(main(target_kg_id=args.kg))
