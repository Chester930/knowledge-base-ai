"""
為 ConceptNode 建立圖拓撲感知共嵌入向量（THEORETICAL_ARCHITECTURE.md 第9節①）。
對 (Document|KnowledgeGraph)-[:EFFECTIVE]->(ConceptNode) 二分圖跑 node2vec，
產生的 `q_vector_graph` 會在路由查詢時與既有文字向量 `q_vector` 加權融合
（見 core/constants.py 的 GRAPH_EMBEDDING_ALPHA），不覆蓋原始文字向量。

用法：
  python run_build_graph_embeddings.py                # 用預設參數跑全部
  python run_build_graph_embeddings.py --epochs 10     # 調整 skip-gram 訓練輪數
  python run_build_graph_embeddings.py --p 0.5 --q 2.0  # 調整 node2vec 遊走傾向
                                                          # （p<1 傾向回頭/BFS，q<1 傾向遠離/DFS）

node2vec 是 transductive：新增 Document/KG/ConceptNode 後，需要重跑本腳本才能讓新節點
獲得圖結構向量；重跑前，新節點的路由分數會自動退回純文字向量（向後相容，不會出錯）。
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
        logging.FileHandler("build_graph_embeddings_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def main(num_walks: int, walk_length: int, p: float, q: float, epochs: int):
    from core.database import connect, disconnect, get_driver
    from services.graph_embedding_service import build_graph_embeddings

    logger.info("=== ConceptNode 圖拓撲共嵌入建立開始 ===")
    logger.info(f"num_walks={num_walks}, walk_length={walk_length}, p={p}, q={q}, epochs={epochs}")

    await connect()
    try:
        n = await build_graph_embeddings(
            get_driver(), num_walks=num_walks, walk_length=walk_length,
            p=p, q=q, epochs=epochs,
        )
        logger.info(f"=== 完成，共更新 {n} 個 ConceptNode 的圖結構向量 ===")
    finally:
        await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConceptNode 圖拓撲共嵌入建立")
    parser.add_argument("--num-walks", type=int, default=10, help="每個節點的隨機遊走次數（預設 10）")
    parser.add_argument("--walk-length", type=int, default=40, help="單次遊走長度（預設 40）")
    parser.add_argument("--p", type=float, default=1.0, help="node2vec 回頭參數（預設 1.0）")
    parser.add_argument("--q", type=float, default=1.0, help="node2vec 遠離參數（預設 1.0）")
    parser.add_argument("--epochs", type=int, default=5, help="skip-gram 訓練輪數（預設 5）")
    args = parser.parse_args()

    asyncio.run(main(
        num_walks=args.num_walks, walk_length=args.walk_length,
        p=args.p, q=args.q, epochs=args.epochs,
    ))
