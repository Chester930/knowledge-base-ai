"""
對 ConceptNode 的 embedding 模型做對比學習微調（THEORETICAL_ARCHITECTURE.md 第9節⑦）。
需要本機有可用 GPU（或至少能接受較慢的 CPU 訓練時間）。

用法：
  python run_finetune_embeddings.py                      # 用預設參數微調
  python run_finetune_embeddings.py --epochs 3            # 調整訓練輪數
  python run_finetune_embeddings.py --output ./models/x   # 自訂輸出路徑
  python run_finetune_embeddings.py --max-pairs 2000       # 限制樣本數（加快測試）

完成後，若要套用微調後的模型，需手動編輯 .env：
  LOCAL_EMBEDDING_MODEL=<輸出路徑，例如 ./models/finetuned_concept_embedder/final>
並重啟服務。**套用新模型後務必對既有 ConceptNode 重新產生向量**（例如重跑
run_build_kg.py --force），否則新舊向量維度模型不一致會讓路由分數失真。
"""
import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_embeddings_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def main(output_dir: str, epochs: int, batch_size: int, max_pairs: int):
    from core.config import settings
    from core.database import connect, disconnect, get_driver
    from services.contrastive_training_service import (
        _MIN_PAIRS_TO_TRAIN,
        finetune_embedding_model,
        generate_training_pairs,
    )

    logger.info("=== ConceptNode Embedding 對比學習微調開始 ===")
    logger.info(f"base_model={settings.local_embedding_model}, epochs={epochs}, max_pairs={max_pairs}")

    await connect()
    pairs = await generate_training_pairs(get_driver(), max_pairs=max_pairs)
    await disconnect()

    logger.info(f"共產生 {len(pairs)} 組正樣本對")
    if len(pairs) < _MIN_PAIRS_TO_TRAIN:
        logger.error(
            f"樣本數（{len(pairs)}）低於最低門檻（{_MIN_PAIRS_TO_TRAIN}），"
            "訓練意義不大甚至可能讓模型過擬合，已中止。請先匯入更多文件/建立更多 KG。"
        )
        return

    save_path = finetune_embedding_model(
        pairs=pairs,
        base_model_name=settings.local_embedding_model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
    )

    logger.info("=== 微調完成 ===")
    logger.info(f"模型已存至：{save_path}")
    logger.info(
        f"若要套用，請於 .env 設定 LOCAL_EMBEDDING_MODEL={save_path}，"
        "重啟服務後，務必對既有 ConceptNode 重新產生向量（例如重跑 run_build_kg.py --force）。"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConceptNode Embedding 對比學習微調")
    parser.add_argument("--output", type=str, default="./models/finetuned_concept_embedder",
                        help="微調後模型輸出目錄")
    parser.add_argument("--epochs", type=int, default=1, help="訓練輪數（預設 1）")
    parser.add_argument("--batch-size", type=int, default=16, help="訓練批次大小（預設 16）")
    parser.add_argument("--max-pairs", type=int, default=5000, help="最多使用幾組正樣本對（預設 5000）")
    args = parser.parse_args()

    asyncio.run(main(
        output_dir=args.output, epochs=args.epochs,
        batch_size=args.batch_size, max_pairs=args.max_pairs,
    ))
