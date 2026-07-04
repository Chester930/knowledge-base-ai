"""
對所有現有 KG 執行完整的 SVO 知識圖譜建構。
步驟：
  1. 列出所有 KG
  2. 對每個 KG 執行 build_graph_for_kg（SVO 提取 + MERGE 進 Neo4j）
  3. 執行 refresh_kg_concepts（刷新路由層 EFFECTIVE 概念）

用法：
  python run_build_kg.py                    # 所有 KG，force_rebuild=False（增量）
  python run_build_kg.py --force            # 先清除再重建
  python run_build_kg.py --kg <kg_id>       # 只跑指定 KG
  python run_build_kg.py --kg <kg_id> --force
"""
import argparse
import asyncio
import logging
import sys
import time
from uuid import UUID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_kg_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def build_one_kg(kg, force_rebuild: bool, relations_only: bool = False) -> dict:
    """對單一 KG 執行 SVO 建構、刷新概念，並套用型別標籤，回傳結果摘要。"""
    from services.svo_service import build_graph_for_kg, apply_type_labels
    from services.knowledge_graph_service import refresh_kg_concepts

    kg_id = kg.id
    name = kg.name
    logger.info(f"  → KG: {name}（{kg_id}）db={kg.db_name or '主資料庫'}")

    total_triples = 0
    chunk_count = 0
    errors = []

    t0 = time.time()
    async for progress in build_graph_for_kg(
        kg_id,
        force_rebuild=force_rebuild,
        rebuild_relations_only=relations_only,
    ):
        if progress.event == "chunk_start":
            chunk_count += 1
        elif progress.event == "chunk_done":
            total_triples += progress.triples_merged
            logger.info(f"    [{progress.chunk_idx}/{progress.total_chunks}] {progress.message}")
        elif progress.event == "error":
            errors.append(progress.message)
            logger.warning(f"    ⚠️  {progress.message}")
        elif progress.event == "done":
            logger.info(f"    ✅ {progress.message}")

    elapsed = time.time() - t0

    # 刷新路由層概念
    try:
        await refresh_kg_concepts(kg_id)
        logger.info(f"    🔄 路由層概念刷新完成")
    except Exception as e:
        errors.append(f"refresh_concepts: {e}")
        logger.warning(f"    ⚠️  路由層刷新失敗：{e}")

    # 套用型別標籤
    label_stats: dict = {}
    try:
        label_stats = await apply_type_labels(kg_id, db_name=kg.db_name)
        total_labeled = sum(label_stats.values())
        logger.info(f"    🏷  型別標籤完成：{total_labeled} 個節點 → {label_stats}")
    except Exception as e:
        errors.append(f"apply_type_labels: {e}")
        logger.warning(f"    ⚠️  標籤套用失敗：{e}")

    return {
        "name": name,
        "kg_id": str(kg_id),
        "chunks": chunk_count,
        "triples": total_triples,
        "elapsed": elapsed,
        "label_stats": label_stats,
        "errors": errors,
    }


async def main(target_kg_id: str | None, force_rebuild: bool, relations_only: bool = False):
    from core.database import connect, disconnect, get_driver
    from core.providers.factory import init_providers
    from repositories.concept_repo import ConceptRepository
    from repositories.knowledge_graph_repo import KnowledgeGraphRepository
    from services.svo_service import create_entity_index

    logger.info("=== 知識圖譜 SVO 批次建構開始 ===")
    logger.info(f"force_rebuild={force_rebuild}  target={target_kg_id or '全部'}")

    await connect()
    embedding = init_providers()
    await ConceptRepository(get_driver()).create_vector_index(embedding.dim)
    await create_entity_index()
    logger.info("資料庫連線完成")

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
        logger.warning("沒有任何 KG，請先建立 KG 並匯入文件")
        await disconnect()
        return

    logger.info(f"共 {len(kgs)} 個 KG 待處理")
    total_start = time.time()

    results = []
    for i, kg in enumerate(kgs, 1):
        logger.info(f"\n[{i}/{len(kgs)}] 處理：{kg.name}（doc_count={kg.doc_count}）")
        if kg.doc_count == 0:
            logger.info("  ⏭  無文件，跳過")
            results.append({"name": kg.name, "kg_id": str(kg.id), "skipped": True})
            continue
        try:
            result = await build_one_kg(kg, force_rebuild=force_rebuild, relations_only=relations_only)
            results.append(result)
        except Exception as e:
            logger.exception(f"  ❌ KG 建構異常：{kg.name}")
            results.append({"name": kg.name, "kg_id": str(kg.id), "error": str(e)})

    total_elapsed = time.time() - total_start

    # 最終摘要
    logger.info("\n" + "=" * 50)
    logger.info("=== 建構完成 ===")
    logger.info(f"⏱  總耗時：{total_elapsed/60:.1f} 分鐘")
    ok = [r for r in results if "triples" in r]
    skip = [r for r in results if r.get("skipped")]
    fail = [r for r in results if "error" in r]

    logger.info(f"✅ 成功：{len(ok)} 個 KG")
    logger.info(f"⏭  跳過：{len(skip)} 個 KG（無文件）")
    logger.info(f"❌ 失敗：{len(fail)} 個 KG")

    for r in ok:
        err_hint = f"（{len(r['errors'])} 個 chunk 錯誤）" if r.get("errors") else ""
        labeled = sum(r.get("label_stats", {}).values())
        logger.info(
            f"  {r['name']}：{r['triples']} triples，"
            f"{r['chunks']} chunks，{labeled} 個節點標籤，{r['elapsed']:.1f}s {err_hint}"
        )
    for r in fail:
        logger.info(f"  ❌ {r['name']}：{r['error']}")

    await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG SVO 批次建構")
    parser.add_argument("--force", action="store_true", help="先清除 Entity 再重建（force_rebuild）")
    parser.add_argument("--relations-only", action="store_true",
                        help="搭配 --force：只清除關係邊、保留 Entity 節點，只更新關係")
    parser.add_argument("--kg", type=str, default=None, help="只處理指定 kg_id")
    args = parser.parse_args()

    asyncio.run(main(
        target_kg_id=args.kg,
        force_rebuild=args.force,
        relations_only=args.relations_only,
    ))
