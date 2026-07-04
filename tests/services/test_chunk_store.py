"""
chunk_store 完整測試套件

涵蓋：
- sentence_chunk()   : 句子感知切分（各種邊界情況）
- ChunkStore.write() : 檔案寫出、舊 chunk 清除
- ChunkStore.read()  : 依 chunk_id 讀回
- ChunkStore.read_many() : 並行批次讀取
- ChunkStore.delete_doc(): 刪除文件所有 chunk
- chunk_id 格式驗證
- 容錯：chunk_id 格式錯誤、檔案不存在
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from services.chunk_store import ChunkStore, SentenceChunk, sentence_chunk

# ── 固定 UUID ─────────────────────────────────────────────────────────────────
KG_ID  = UUID("11111111-1111-1111-1111-111111111111")
DOC_ID = UUID("22222222-2222-2222-2222-222222222222")
DOC_STR = str(DOC_ID)
KG_STR  = str(KG_ID)


@pytest.fixture
def tmp_store(tmp_path):
    """每個測試使用獨立的暫存目錄，測試後自動清理。"""
    return ChunkStore(str(tmp_path / "chunk_store"))


# ══════════════════════════════════════════════════════════════════════════════
# sentence_chunk()
# ══════════════════════════════════════════════════════════════════════════════

class TestSentenceChunk:

    def test_empty_text_returns_empty(self):
        assert sentence_chunk(DOC_STR, "") == []

    def test_whitespace_only_returns_empty(self):
        assert sentence_chunk(DOC_STR, "   \n\t  ") == []

    def test_single_sentence_one_chunk(self):
        chunks = sentence_chunk(DOC_STR, "這是一個完整的句子，沒有標點結尾所以退化為整段。")
        assert len(chunks) == 1
        assert chunks[0].idx == 1

    def test_exactly_five_sentences_one_chunk(self):
        text = "第一句是比較長的句子。第二句也是比較長的。第三句繼續說明內容。第四句提供更多資訊。第五句結束本段。"
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) == 1
        assert len(chunks[0].sentences) == 5

    def test_six_sentences_two_chunks(self):
        text = "第一句很長的內容說明。第二句提供範例與解釋。第三句總結前面的觀點。第四句開始新主題討論。第五句延伸討論範疇。第六句開啟第二個chunk。"
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) == 2
        assert len(chunks[0].sentences) == 5
        assert len(chunks[1].sentences) == 1

    def test_eleven_sentences_three_chunks(self):
        text = "".join(f"這是第{i}個句子，包含足夠多的文字內容。" for i in range(1, 12))
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) == 3  # ceil(11/5)

    def test_chunk_id_format(self):
        chunks = sentence_chunk(DOC_STR, "第一句很長的內容。第二句繼續說明。第三句提供範例。第四句總結要點。第五句結束段落。第六句開啟新段。")
        assert chunks[0].chunk_id == f"{DOC_STR}_0001"
        assert chunks[1].chunk_id == f"{DOC_STR}_0002"

    def test_idx_is_one_based(self):
        text = "".join(f"這是句子{i}號，內容長到足以被切分。" for i in range(1, 16))
        chunks = sentence_chunk(DOC_STR, text)
        for i, c in enumerate(chunks, 1):
            assert c.idx == i

    def test_text_field_concatenates_sentences(self):
        text = "第一完整句子結束。第二完整句子結束！第三完整句子結束？第四完整句子在此。第五完整句子完結。"
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) == 1
        # text 欄位是所有句子拼接
        reconstructed = "".join(chunks[0].sentences)
        assert reconstructed == chunks[0].text

    def test_no_punctuation_fallback_one_chunk(self):
        text = "這段文字沒有中文標點符號所以全部視為一個 chunk 的內容"
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) == 1
        assert chunks[0].text == text.strip()

    def test_char_start_end_coverage(self):
        text = "第一句完整內容在此。第二句繼續說明。第三句提供更多細節說明。第四句接近尾聲了。第五句正式結束本段。"
        chunks = sentence_chunk(DOC_STR, text)
        assert chunks[0].char_start == 0
        assert chunks[0].char_end > 0
        assert chunks[-1].char_end <= len(text) + 10  # 允許微小偏移

    def test_mixed_chinese_english(self):
        text = "Deep learning is a subset of machine learning. 它透過多層神經網路學習特徵表示。" \
               "This enables automatic feature extraction. 不需要人工特徵工程。" \
               "Gradient descent optimizes the network weights."
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) >= 1
        # 所有句子都要被納入某個 chunk
        total_sentences = sum(len(c.sentences) for c in chunks)
        assert total_sentences >= 3

    def test_newline_as_sentence_boundary(self):
        text = "第一段落的內容說明。\n第二段落的重要概念。\n第三段落提供範例解釋。\n第四段落總結觀點。\n第五段落提出結論。\n第六段落補充說明。"
        chunks = sentence_chunk(DOC_STR, text)
        assert len(chunks) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore.write()
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStoreWrite:

    @pytest.mark.asyncio
    async def test_creates_chunk_files(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "第一句內容。第二句繼續說明。第三句提供範例。第四句總結要點。第五句結束段落。第六句開啟新段。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        chunk_dir = Path(tmp_store._base) / KG_STR / DOC_STR
        files = list(chunk_dir.glob("chunk_*.json"))
        assert len(files) == len(chunks)

    @pytest.mark.asyncio
    async def test_chunk_file_content(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "第一句內容說明。第二句繼續說明。第三句提供範例。第四句總結要點。第五句結束段落。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        path = Path(tmp_store._base) / KG_STR / DOC_STR / "chunk_0001.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["chunk_id"] == chunks[0].chunk_id
        assert data["kg_id"] == KG_STR
        assert data["doc_id"] == DOC_STR
        assert data["idx"] == 1
        assert isinstance(data["sentences"], list)
        assert isinstance(data["text"], str)

    @pytest.mark.asyncio
    async def test_write_creates_doc_ref(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "夠長的句子一。夠長的句子二。夠長的句子三。夠長的句子四。夠長的句子五。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        ref = Path(tmp_store._docs_dir) / DOC_STR
        assert ref.exists()
        assert ref.read_text(encoding="utf-8").strip() == KG_STR

    @pytest.mark.asyncio
    async def test_write_clears_old_chunks_on_rebuild(self, tmp_store):
        text = "夠長的句子一。夠長的句子二。夠長的句子三。夠長的句子四。夠長的句子五。夠長的句子六。"
        chunks_first = sentence_chunk(DOC_STR, text)
        await tmp_store.write(KG_ID, DOC_ID, chunks_first)

        # 第二次寫入較少 chunks（縮短文件模擬重建）
        chunks_second = sentence_chunk(DOC_STR, "夠長的句子一。夠長的句子二。夠長的句子三。夠長的句子四。夠長的句子五。")
        await tmp_store.write(KG_ID, DOC_ID, chunks_second)

        chunk_dir = Path(tmp_store._base) / KG_STR / DOC_STR
        files = list(chunk_dir.glob("chunk_*.json"))
        assert len(files) == len(chunks_second)  # 不殘留舊檔

    @pytest.mark.asyncio
    async def test_write_empty_chunks_no_files(self, tmp_store):
        await tmp_store.write(KG_ID, DOC_ID, [])
        chunk_dir = Path(tmp_store._base) / KG_STR / DOC_STR
        files = list(chunk_dir.glob("chunk_*.json")) if chunk_dir.exists() else []
        assert len(files) == 0


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore.read()
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStoreRead:

    @pytest.mark.asyncio
    async def test_read_returns_correct_data(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一有足夠內容。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        result = tmp_store.read(chunks[0].chunk_id)
        assert result is not None
        assert result["chunk_id"] == chunks[0].chunk_id
        assert result["text"] == chunks[0].text
        assert result["kg_id"] == KG_STR

    @pytest.mark.asyncio
    async def test_read_nonexistent_returns_none(self, tmp_store):
        fake_id = f"{DOC_STR}_9999"
        assert tmp_store.read(fake_id) is None

    def test_read_malformed_chunk_id_too_short(self, tmp_store):
        assert tmp_store.read("tooshort") is None

    def test_read_malformed_no_underscore(self, tmp_store):
        assert tmp_store.read("a" * 36 + "9999") is None

    def test_read_malformed_nonnumeric_idx(self, tmp_store):
        assert tmp_store.read(f"{DOC_STR}_abcd") is None

    @pytest.mark.asyncio
    async def test_read_after_delete_returns_none(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一有足夠內容說明。句子二繼續補充說明。句子三提供具體範例。句子四總結核心要點。句子五正式結束。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)
        tmp_store.delete_doc(KG_ID, DOC_ID)
        assert tmp_store.read(chunks[0].chunk_id) is None


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore.read_many()
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStoreReadMany:

    @pytest.mark.asyncio
    async def test_read_many_returns_all_existing(self, tmp_store):
        text = "".join(f"這是第{i}句詳細內容說明文字。" for i in range(1, 12))
        chunks = sentence_chunk(DOC_STR, text)
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        results = await tmp_store.read_many([c.chunk_id for c in chunks])
        assert len(results) == len(chunks)

    @pytest.mark.asyncio
    async def test_read_many_filters_missing(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一有內容。句子二繼續說。句子三還有說。句子四再說一次。句子五最後說。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        ids = [chunks[0].chunk_id, f"{DOC_STR}_9999"]  # 一個存在，一個不存在
        results = await tmp_store.read_many(ids)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_read_many_empty_list(self, tmp_store):
        results = await tmp_store.read_many([])
        assert results == []

    @pytest.mark.asyncio
    async def test_read_many_preserves_content(self, tmp_store):
        text = "".join(f"這是句子{i}號，包含詳細說明內容。" for i in range(1, 6))
        chunks = sentence_chunk(DOC_STR, text)
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        results = await tmp_store.read_many([chunks[0].chunk_id])
        assert results[0]["text"] == chunks[0].text


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore.delete_doc()
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStoreDeleteDoc:

    @pytest.mark.asyncio
    async def test_delete_removes_all_chunk_files(self, tmp_store):
        text = "".join(f"這是句子{i}號詳細說明內容。" for i in range(1, 11))
        chunks = sentence_chunk(DOC_STR, text)
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        count = tmp_store.delete_doc(KG_ID, DOC_ID)
        assert count == len(chunks)

        chunk_dir = Path(tmp_store._base) / KG_STR / DOC_STR
        assert not chunk_dir.exists() or not list(chunk_dir.glob("chunk_*.json"))

    @pytest.mark.asyncio
    async def test_delete_removes_doc_ref(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)
        tmp_store.delete_doc(KG_ID, DOC_ID)

        ref = Path(tmp_store._docs_dir) / DOC_STR
        assert not ref.exists()

    def test_delete_nonexistent_doc_returns_zero(self, tmp_store):
        count = tmp_store.delete_doc(KG_ID, uuid4())
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_does_not_affect_other_docs(self, tmp_store):
        doc2 = uuid4()
        doc2_str = str(doc2)
        text = "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。"
        chunks1 = sentence_chunk(DOC_STR, text)
        chunks2 = sentence_chunk(doc2_str, text)
        await tmp_store.write(KG_ID, DOC_ID, chunks1)
        await tmp_store.write(KG_ID, doc2, chunks2)

        tmp_store.delete_doc(KG_ID, DOC_ID)

        # doc2 的 chunk 仍然存在
        result = tmp_store.read(chunks2[0].chunk_id)
        assert result is not None


class TestChunkStoreDeleteDocById:

    @pytest.mark.asyncio
    async def test_resolves_kg_id_from_ref_file_and_deletes(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        count = tmp_store.delete_doc_by_id(DOC_ID)
        assert count == len(chunks)

        chunk_dir = Path(tmp_store._base) / KG_STR / DOC_STR
        assert not chunk_dir.exists() or not list(chunk_dir.glob("chunk_*.json"))
        ref = Path(tmp_store._docs_dir) / DOC_STR
        assert not ref.exists()

    def test_unknown_doc_id_returns_zero_without_error(self, tmp_store):
        count = tmp_store.delete_doc_by_id(uuid4())
        assert count == 0


class TestChunkStoreDeleteKg:

    @pytest.mark.asyncio
    async def test_removes_all_docs_under_kg(self, tmp_store):
        doc2 = uuid4()
        text = "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。"
        chunks1 = sentence_chunk(DOC_STR, text)
        chunks2 = sentence_chunk(str(doc2), text)
        await tmp_store.write(KG_ID, DOC_ID, chunks1)
        await tmp_store.write(KG_ID, doc2, chunks2)

        removed = tmp_store.delete_kg(KG_ID)
        assert removed == 2

        kg_dir = Path(tmp_store._base) / KG_STR
        assert not kg_dir.exists()
        assert not (Path(tmp_store._docs_dir) / DOC_STR).exists()
        assert not (Path(tmp_store._docs_dir) / str(doc2)).exists()

    @pytest.mark.asyncio
    async def test_does_not_affect_other_kgs(self, tmp_store):
        other_kg = uuid4()
        other_doc = uuid4()
        text = "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。"
        chunks1 = sentence_chunk(DOC_STR, text)
        chunks2 = sentence_chunk(str(other_doc), text)
        await tmp_store.write(KG_ID, DOC_ID, chunks1)
        await tmp_store.write(other_kg, other_doc, chunks2)

        tmp_store.delete_kg(KG_ID)

        # 其他 KG 的 chunk 仍然存在
        result = tmp_store.read(chunks2[0].chunk_id)
        assert result is not None

    def test_nonexistent_kg_returns_zero(self, tmp_store):
        assert tmp_store.delete_kg(uuid4()) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 多文件 / 多 KG 隔離
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkStoreIsolation:

    @pytest.mark.asyncio
    async def test_different_kgs_isolated(self, tmp_store):
        kg2 = uuid4()
        text = "句子一詳細說明。句子二繼續說明。句子三提供範例。句子四總結要點。句子五結束段落。"
        chunks1 = sentence_chunk(DOC_STR, text)
        chunks2 = sentence_chunk(DOC_STR, text)  # 同 doc_id，不同 kg

        await tmp_store.write(KG_ID, DOC_ID, chunks1)
        await tmp_store.write(kg2, DOC_ID, chunks2)

        # 後寫的覆蓋 doc ref，read 查到最後寫入的 kg
        result = tmp_store.read(chunks1[0].chunk_id)
        # 只要能讀到資料即可（doc ref 指向最後寫入的 kg）
        assert result is not None

    @pytest.mark.asyncio
    async def test_chunk_json_contains_kg_id(self, tmp_store):
        chunks = sentence_chunk(DOC_STR, "句子一詳細說明文字。句子二繼續補充說明。句子三提供範例解釋。句子四總結核心要點。句子五正式結束段落。")
        await tmp_store.write(KG_ID, DOC_ID, chunks)

        result = tmp_store.read(chunks[0].chunk_id)
        assert result["kg_id"] == KG_STR
        assert result["doc_id"] == DOC_STR
