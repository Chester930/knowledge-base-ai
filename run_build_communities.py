"""
為所有（或指定）KG 建立社群摘要（THEORETICAL_ARCHITECTURE.md 第9節⑤）。
對 Entity 關係圖做社群偵測（networkx Louvain），為每個社群生成 LLM 摘要並存回 Neo4j，
供 /agent/chat 的全域性查詢（例如「總結這個知識庫的技術演進」）路由使用。

用法：
  python run_build_communities.py                    # 對所有 KG 建立社群摘要
  python run_build_communities.py --kg <id>           # 只跑指定 KG
  python run_build_communities.py --min-size 5        # 調整社群最小規模門檻（預設 3）
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
        logging.FileHandler("build_communities_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def main(target_kg_id: str | None, min_size: int):
    from core.database import connect, disconnect, get_driver
    from core.providers.factory import init_providers
    from repositories.concept_repo import ConceptRepository
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from services.community_service import build_communities_for_kg

    logger.info("=== KG 社群摘要建立開始 ===")
    logger.info(f"target={target_kg_id or '全部'}, min_size={min_size}")

    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)

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

    total_communities = 0
    for i, kg in enumerate(kgs, 1):
        logger.info(f"\n[{i}/{len(kgs)}] {kg.name}（entity_count={kg.entity_count}）")
        if kg.entity_count == 0:
            logger.info("  ⏭  無 Entity，跳過")
            continue
        try:
            n = await build_communities_for_kg(kg.id, db_name=kg.db_name, min_size=min_size)
            total_communities += n
            logger.info(f"  ✅ 建立完成：{n} 個社群摘要")
        except Exception as e:
            logger.exception(f"  ❌ 社群摘要失敗：{kg.name} — {e}")

    logger.info(f"\n=== 完成，共建立 {total_communities} 個社群摘要 ===")
    await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG 社群摘要建立")
    parser.add_argument("--kg", type=str, default=None, help="只處理指定 kg_id")
    parser.add_argument("--min-size", type=int, default=3, help="社群最小規模門檻（預設 3）")
    args = parser.parse_args()

    asyncio.run(main(target_kg_id=args.kg, min_size=args.min_size))
