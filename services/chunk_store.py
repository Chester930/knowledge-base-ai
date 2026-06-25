from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)

_SENTENCES_PER_CHUNK = 5
_SENT_SPLIT_RE = re.compile(r'(?<=[。！？!?\n])')
_MIN_SENT_LEN = 5


# ── 資料結構 ──────────────────────────────────────────────────────────────────

@dataclass
class SentenceChunk:
    chunk_id: str          # "{doc_id}_{idx:04d}"
    idx: int               # 1-based
    sentences: list[str]   # 原始句子列表
    text: str              # 拼接後段落全文
    char_start: int        # 在原文的起始字元位置（近似值）
    char_end: int          # 在原文的結束字元位置（近似值）


# ── 句子感知切分 ──────────────────────────────────────────────────────────────

def sentence_chunk(doc_id: str, text: str) -> list[SentenceChunk]:
    """
    將文字切分為句子感知的 SentenceChunk 列表。
    - 先按中英文句尾標點（。！？!?\\n）切句
    - 每 _SENTENCES_PER_CHUNK 句組成一個 Chunk
    - 無法切分時退化為整段一個 Chunk
    """
    if not text or not text.strip():
        return []

    raw_parts = _SENT_SPLIT_RE.split(text)

    sentences: list[tuple[str, int, int]] = []  # (text, char_start, char_end)
    pos = 0
    for part in raw_parts:
        stripped = part.strip()
        if len(stripped) >= _MIN_SENT_LEN:
            sentences.append((stripped, pos, pos + len(part)))
        pos += len(part)

    # 最末尾未被標點結束的殘餘文字
    if pos < len(text):
        tail = text[pos:].strip()
        if len(tail) >= _MIN_SENT_LEN:
            sentences.append((tail, pos, len(text)))

    if not sentences:
        return [SentenceChunk(
            chunk_id=f"{doc_id}_0001", idx=1,
            sentences=[text.strip()], text=text.strip(),
            char_start=0, char_end=len(text),
        )]

    chunks: list[SentenceChunk] = []
    for i in range(0, len(sentences), _SENTENCES_PER_CHUNK):
        group = sentences[i:i + _SENTENCES_PER_CHUNK]
        idx = len(chunks) + 1
        chunks.append(SentenceChunk(
            chunk_id=f"{doc_id}_{idx:04d}",
            idx=idx,
            sentences=[s[0] for s in group],
            text="".join(s[0] for s in group),
            char_start=group[0][1],
            char_end=group[-1][2],
        ))

    return chunks


# ── ChunkStore ────────────────────────────────────────────────────────────────

class ChunkStore:
    """
    將每份文件的 SentenceChunk 持久化為 JSON 檔案。

    目錄結構：
        {base_dir}/{kg_id}/{doc_id}/chunk_{idx:04d}.json
        {base_dir}/_docs/{doc_id}          ← 記錄 doc 屬於哪個 kg_id（一行文字）
    """

    def __init__(self, base_dir: str):
        self._base = Path(base_dir)
        self._docs_dir = self._base / "_docs"

    def _chunk_path(self, kg_id: str, doc_id: str, idx: int) -> Path:
        return self._base / kg_id / doc_id / f"chunk_{idx:04d}.json"

    def _doc_ref_path(self, doc_id: str) -> Path:
        return self._docs_dir / doc_id

    # ── 寫入 ──────────────────────────────────────────────────────────────────

    async def write(
        self,
        kg_id: UUID,
        doc_id: UUID,
        chunks: list[SentenceChunk],
        vectors: list[list[float]] | None = None,
    ) -> None:
        """
        持久化 SentenceChunk 列表。
        vectors: 與 chunks 等長的 embedding 向量列表（☆6 優化）；None 則不儲存向量。
        """
        kg_str = str(kg_id)
        doc_str = str(doc_id)
        chunk_dir = self._base / kg_str / doc_str
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for old in chunk_dir.glob("chunk_*.json"):
            old.unlink(missing_ok=True)

        for i, sc in enumerate(chunks):
            data = {
                "chunk_id": sc.chunk_id,
                "idx": sc.idx,
                "doc_id": doc_str,
                "kg_id": kg_str,
                "sentences": sc.sentences,
                "text": sc.text,
                "char_start": sc.char_start,
                "char_end": sc.char_end,
                "vector": vectors[i] if vectors and i < len(vectors) else None,
            }
            self._chunk_path(kg_str, doc_str, sc.idx).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._doc_ref_path(doc_str).write_text(kg_str, encoding="utf-8")

        logger.debug(f"ChunkStore: 已儲存 {len(chunks)} 個 chunks（doc={doc_str}, vectors={'有' if vectors else '無'}）")

    # ── 讀取 ──────────────────────────────────────────────────────────────────

    def _resolve_kg(self, doc_id: str) -> str | None:
        ref = self._doc_ref_path(doc_id)
        if not ref.exists():
            return None
        return ref.read_text(encoding="utf-8").strip()

    def read(self, chunk_id: str) -> dict | None:
        """
        chunk_id 格式：{UUID-36chars}_{idx:04d}
        從 _docs/{doc_id} 取得 kg_id，再定位檔案。
        """
        if len(chunk_id) < 41 or chunk_id[36] != "_":
            return None
        doc_id = chunk_id[:36]
        try:
            idx = int(chunk_id[37:])
        except ValueError:
            return None
        kg_id = self._resolve_kg(doc_id)
        if kg_id is None:
            return None
        path = self._chunk_path(kg_id, doc_id, idx)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"ChunkStore.read 失敗 [{chunk_id}]: {e}")
            return None

    async def read_many(self, chunk_ids: list[str]) -> list[dict]:
        """並行讀取多個 Chunk，過濾讀取失敗的項目。"""
        if not chunk_ids:
            return []
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(
            *[loop.run_in_executor(None, self.read, cid) for cid in chunk_ids]
        )
        return [r for r in results if r is not None]

    async def read_ranked(
        self, chunk_ids: list[str], query_vector: list[float]
    ) -> list[dict]:
        """
        讀取 chunks 並依 cosine 相似度排序（☆6 優化）。
        有向量的 chunk 按語意相似度排前面；無向量的 chunk 排後面保底。
        """
        chunks = await self.read_many(chunk_ids)
        if not chunks or not query_vector:
            return chunks

        def _cos(v1: list[float], v2: list[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = sum(a * a for a in v1) ** 0.5
            n2 = sum(b * b for b in v2) ** 0.5
            if n1 < 1e-9 or n2 < 1e-9:
                return 0.0
            return dot / (n1 * n2)

        scored = []
        for c in chunks:
            vec = c.get("vector")
            score = _cos(vec, query_vector) if vec else -1.0
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    # ── 刪除 ──────────────────────────────────────────────────────────────────

    def delete_doc(self, kg_id: UUID, doc_id: UUID) -> int:
        """刪除某份文件的所有 Chunk 檔案，回傳刪除數量。"""
        doc_str = str(doc_id)
        kg_str = str(kg_id)
        chunk_dir = self._base / kg_str / doc_str
        count = 0
        if chunk_dir.exists():
            for f in chunk_dir.glob("chunk_*.json"):
                f.unlink(missing_ok=True)
                count += 1
            try:
                chunk_dir.rmdir()
            except OSError:
                pass
        ref = self._doc_ref_path(doc_str)
        if ref.exists():
            ref.unlink(missing_ok=True)
        if count:
            logger.debug(f"ChunkStore: 已刪除 {count} 個 chunks（doc={doc_str}）")
        return count


# ── 模組層級 singleton ────────────────────────────────────────────────────────

_store: ChunkStore | None = None


def get_chunk_store() -> ChunkStore:
    global _store
    if _store is None:
        from core.config import settings
        _store = ChunkStore(settings.chunk_store_dir)
    return _store
