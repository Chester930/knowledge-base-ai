"""
Contrastive Training Service 測試（THEORETICAL_ARCHITECTURE.md 第9節⑦：對比自我監督概念學習）

測試：
- generate_training_pairs : Document/KG 層級共現正樣本對產生與 fallback 邏輯
- finetune_embedding_model: 對比學習微調的真實端到端 smoke test（用本機已快取的小模型，
  非 mock——證明訓練流程真的能跑通並存出可用模型，而不只是接口正確）
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.contrastive_training_service import (
    _MIN_PAIRS_TO_TRAIN,
    finetune_embedding_model,
    generate_training_pairs,
)


def _rec(**kwargs):
    r = MagicMock()
    r.__getitem__ = lambda self, k: kwargs.get(k)
    return r


def _result(records):
    res = MagicMock()
    res.records = records
    return res


# ── generate_training_pairs ─────────────────────────────────────────────────

class TestGenerateTrainingPairs:
    async def test_uses_document_level_pairs_when_sufficient(self):
        doc_pairs = [_rec(a=f"a{i}", b=f"b{i}") for i in range(_MIN_PAIRS_TO_TRAIN + 5)]
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result(doc_pairs))

        result = await generate_training_pairs(driver, max_pairs=100)

        assert len(result) == _MIN_PAIRS_TO_TRAIN + 5
        # 只呼叫一次（Document 層級查詢），因為樣本已足夠，不需要補 KG 層級
        assert driver.execute_query.call_count == 1

    async def test_falls_back_to_kg_level_when_insufficient(self):
        doc_pairs = [_rec(a="a1", b="b1"), _rec(a="a2", b="b2")]  # 只有 2 組，不足門檻
        kg_pairs = [_rec(a=f"x{i}", b=f"y{i}") for i in range(30)]
        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=[_result(doc_pairs), _result(kg_pairs)])

        result = await generate_training_pairs(driver, max_pairs=100)

        assert driver.execute_query.call_count == 2
        assert len(result) == 2 + 30

    async def test_dedups_pairs_already_present_from_document_level(self):
        doc_pairs = [_rec(a="a1", b="b1")]
        kg_pairs = [_rec(a="a1", b="b1"), _rec(a="a2", b="b2")]  # 第一組與 document 層級重複
        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=[_result(doc_pairs), _result(kg_pairs)])

        result = await generate_training_pairs(driver, max_pairs=100)

        assert result.count(("a1", "b1")) == 1
        assert ("a2", "b2") in result

    async def test_empty_result_when_no_cooccurrence(self):
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=_result([]))

        result = await generate_training_pairs(driver, max_pairs=100)

        assert result == []


# ── finetune_embedding_model ────────────────────────────────────────────────

class TestFinetuneEmbeddingModel:
    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError):
            finetune_embedding_model([], "sentence-transformers/all-MiniLM-L6-v2", "/tmp/x")

    def test_real_finetune_smoke_test_produces_loadable_model(self):
        """
        真實端到端測試（非 mock）：用本機已快取的小模型跑 1 epoch 的對比學習微調，
        驗證存出的模型檔案完整、且能重新載入並產生 embedding。
        """
        from sentence_transformers import SentenceTransformer

        pairs = [
            ("機器學習", "深度學習"),
            ("資料庫", "SQL"),
            ("神經網路", "反向傳播"),
            ("知識圖譜", "實體關係"),
        ]
        tmp_dir = tempfile.mkdtemp()
        try:
            save_path = finetune_embedding_model(
                pairs=pairs,
                base_model_name="sentence-transformers/all-MiniLM-L6-v2",
                output_dir=tmp_dir,
                epochs=1,
                batch_size=2,
            )

            assert Path(save_path).exists()
            assert (Path(save_path) / "config.json").exists()

            # 微調後的模型應能正常重新載入並產生向量
            reloaded = SentenceTransformer(save_path)
            vec = reloaded.encode("機器學習")
            assert len(vec) == reloaded.get_sentence_embedding_dimension()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
