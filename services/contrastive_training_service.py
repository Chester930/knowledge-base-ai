"""
Contrastive Concept Learning — 對比自我監督概念學習 (THEORETICAL_ARCHITECTURE.md 第9節⑦)

對 ConceptNode 的文字 embedding 模型做對比學習微調：把「同一份文件/同一個 KG 下共現的
概念」當正樣本對，用 in-batch negatives（`MultipleNegativesRankingLoss`，即 InfoNCE 的
標準實作）微調，讓 Gating Router 的餘弦相似度計算更貼合本專案實際的知識領域分佈。

技術選型：直接使用 `sentence-transformers`（本專案既有相依套件）內建的
`SentenceTransformerTrainer` 微調 API，而非自建訓練框架——這是既有函式庫的一等公民功能，
不是新增的訓練基礎設施。從既有預訓練模型（`local_embedding_model`）微調，而非從零訓練。
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_PAIRS = 5000
_MIN_PAIRS_TO_TRAIN = 20  # 樣本數過少時訓練沒有意義，甚至可能讓模型過擬合到極少數概念


async def generate_training_pairs(driver, max_pairs: int = _DEFAULT_MAX_PAIRS) -> list[tuple[str, str]]:
    """
    產生對比學習的正樣本對：同一份 Document 下共現的 ConceptNode 名稱配對。

    若 Document 層級的共現配對數量不足 `_MIN_PAIRS_TO_TRAIN`，退而求其次改用
    KnowledgeGraph 層級的共現（同一個 KG 底下的概念，訊號較弱但涵蓋更廣）補足。
    """
    result = await driver.execute_query(
        """
        MATCH (d:Document)-[:EFFECTIVE]->(c1:ConceptNode)
        MATCH (d)-[:EFFECTIVE]->(c2:ConceptNode)
        WHERE c1.name < c2.name
        RETURN DISTINCT c1.name AS a, c2.name AS b
        LIMIT $limit
        """,
        limit=max_pairs,
    )
    pairs = [(r["a"], r["b"]) for r in result.records]

    if len(pairs) < _MIN_PAIRS_TO_TRAIN:
        logger.info(f"Document 層級共現配對僅 {len(pairs)} 組，補充 KnowledgeGraph 層級配對")
        remaining = max_pairs - len(pairs)
        kg_result = await driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph)-[:EFFECTIVE]->(c1:ConceptNode)
            MATCH (kg)-[:EFFECTIVE]->(c2:ConceptNode)
            WHERE c1.name < c2.name
            RETURN DISTINCT c1.name AS a, c2.name AS b
            LIMIT $limit
            """,
            limit=remaining,
        )
        seen = set(pairs)
        for r in kg_result.records:
            pair = (r["a"], r["b"])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    return pairs


def finetune_embedding_model(
    pairs: list[tuple[str, str]],
    base_model_name: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 16,
) -> str:
    """
    用對比學習（in-batch negatives）微調 sentence-transformers 模型，儲存至 `output_dir`。
    回傳實際存檔路徑。呼叫端需自行確保 `pairs` 數量足夠（見 `_MIN_PAIRS_TO_TRAIN`）。
    """
    if not pairs:
        raise ValueError("訓練樣本為空，無法微調")

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    logger.info(f"載入基礎模型：{base_model_name}")
    model = SentenceTransformer(base_model_name)

    train_dataset = Dataset.from_dict({
        "anchor": [p[0] for p in pairs],
        "positive": [p[1] for p in pairs],
    })
    loss = MultipleNegativesRankingLoss(model)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=min(batch_size, len(pairs)),
        report_to=[],
        logging_steps=max(1, len(pairs) // batch_size),
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss,
    )
    trainer.train()

    save_path = str(Path(output_dir) / "final")
    model.save(save_path)
    logger.info(f"微調完成，模型已存至：{save_path}")
    return save_path
