from __future__ import annotations
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from models.document import Document
from services.ingestion_service import (
    SUPPORTED_EXTENSIONS,
    _read_pdf,
    _read_text,
    _read_docx,
    _read_pptx,
    ingest_file,
    ingest_directory,
    move_and_ingest,
)


def _make_doc(**kwargs) -> Document:
    defaults = dict(
        id=uuid4(), title="test", content="內容",
        file_type="txt", file_path=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Document(**defaults)


# ── _read_text ───────────────────────────────────────────────────────────────

class TestReadText:
    def test_utf8_content(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("你好世界", encoding="utf-8")
        assert _read_text(f) == "你好世界"

    def test_big5_encoding(self, tmp_path):
        f = tmp_path / "b.txt"
        f.write_bytes("測試文件".encode("big5"))
        assert "測試文件" in _read_text(f)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert _read_text(f) == ""

    def test_utf8_bom(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes("BOM文件".encode("utf-8-sig"))
        assert "BOM文件" in _read_text(f)


# ── _read_pdf ────────────────────────────────────────────────────────────────

class TestReadPdf:
    def test_single_page(self, tmp_path):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF 第一頁"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader), \
             patch("services.ingestion_service._ocr_pdf", return_value=""):
            result = _read_pdf(tmp_path / "a.pdf")

        assert "PDF 第一頁" in result

    def test_multiple_pages_joined(self, tmp_path):
        pages = []
        for i in range(3):
            p = MagicMock()
            p.extract_text.return_value = f"第{i+1}頁內容"
            pages.append(p)
        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("pypdf.PdfReader", return_value=mock_reader), \
             patch("services.ingestion_service._ocr_pdf", return_value=""):
            result = _read_pdf(tmp_path / "multi.pdf")

        for i in range(3):
            assert f"第{i+1}頁內容" in result

    def test_none_page_text_becomes_empty_string(self, tmp_path):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader), \
             patch("services.ingestion_service._ocr_pdf", return_value=""):
            result = _read_pdf(tmp_path / "null.pdf")

        assert result == ""

    def test_empty_pdf_returns_empty(self, tmp_path):
        mock_reader = MagicMock()
        mock_reader.pages = []

        with patch("pypdf.PdfReader", return_value=mock_reader), \
             patch("services.ingestion_service._ocr_pdf", return_value=""):
            result = _read_pdf(tmp_path / "empty.pdf")

        assert result == ""


# ── _read_docx ───────────────────────────────────────────────────────────────

class TestReadDocx:
    def test_paragraphs_extracted(self, tmp_path):
        p1, p2, p3 = MagicMock(), MagicMock(), MagicMock()
        p1.text, p2.text, p3.text = "段落一", "", "段落三"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [p1, p2, p3]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            result = _read_docx(tmp_path / "test.docx")

        assert "段落一" in result
        assert "段落三" in result

    def test_empty_paragraphs_skipped(self, tmp_path):
        p = MagicMock()
        p.text = "   "
        mock_doc = MagicMock()
        mock_doc.paragraphs = [p]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            result = _read_docx(tmp_path / "ws.docx")

        assert result.strip() == ""

    def test_table_cells_extracted(self, tmp_path):
        cell = MagicMock()
        cell.text = "儲存格A"
        row = MagicMock()
        row.cells = [cell]
        table = MagicMock()
        table.rows = [row]
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = [table]

        with patch("docx.Document", return_value=mock_doc):
            result = _read_docx(tmp_path / "tbl.docx")

        assert "儲存格A" in result


# ── _read_pptx ───────────────────────────────────────────────────────────────

class TestReadPptx:
    def test_slide_content_extracted(self, tmp_path):
        shape = MagicMock()
        shape.text = "投影片標題"
        slide = MagicMock()
        slide.shapes = [shape]
        prs = MagicMock()
        prs.slides = [slide]

        with patch("pptx.Presentation", return_value=prs):
            result = _read_pptx(tmp_path / "test.pptx")

        assert "投影片標題" in result
        assert "第 1 頁" in result

    def test_multiple_slides(self, tmp_path):
        slides = []
        for i in range(3):
            shape = MagicMock()
            shape.text = f"第{i+1}頁文字"
            slide = MagicMock()
            slide.shapes = [shape]
            slides.append(slide)
        prs = MagicMock()
        prs.slides = slides

        with patch("pptx.Presentation", return_value=prs):
            result = _read_pptx(tmp_path / "multi.pptx")

        for i in range(3):
            assert f"第{i+1}頁文字" in result

    def test_empty_shapes_skipped(self, tmp_path):
        shape = MagicMock()
        shape.text = ""
        slide = MagicMock()
        slide.shapes = [shape]
        prs = MagicMock()
        prs.slides = [slide]

        with patch("pptx.Presentation", return_value=prs):
            result = _read_pptx(tmp_path / "empty.pptx")

        assert result == ""


# ── SUPPORTED_EXTENSIONS ─────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_includes_all_expected_formats(self):
        expected = {".md", ".txt", ".pdf", ".docx", ".pptx", ".doc", ".ppt"}
        assert expected == SUPPORTED_EXTENSIONS


# ── ingest_file ───────────────────────────────────────────────────────────────

class TestIngestFile:
    async def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await ingest_file(str(tmp_path / "ghost.txt"))

    async def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="不支援的檔案格式"):
            await ingest_file(str(f))

    async def test_empty_content_raises(self, tmp_path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n\t  ")
        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository"):
            with pytest.raises(ValueError, match="內容為空"):
                await ingest_file(str(f))

    async def test_txt_file_ingested_successfully(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("這是一份有效的測試文件，內容不為空。", encoding="utf-8")

        doc = _make_doc(title="note", content="這是一份有效的測試文件，內容不為空。")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo), \
             patch("services.ingestion_service.extract_and_init_document_concepts",
                   new=AsyncMock()):
            result = await ingest_file(str(f))

        assert result.title == "note"
        mock_repo.create.assert_called_once()

    async def test_md_file_ingested_as_md_type(self, tmp_path):
        f = tmp_path / "wiki.md"
        f.write_text("# 標題\n\n這是 Markdown 內容。", encoding="utf-8")

        doc = _make_doc(title="wiki", file_type="md")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo), \
             patch("services.ingestion_service.extract_and_init_document_concepts",
                   new=AsyncMock()):
            result = await ingest_file(str(f))

        assert result.file_type == "md"


# ── ingest_directory ─────────────────────────────────────────────────────────

class TestIngestDirectory:
    async def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            await ingest_directory("/does/not/exist")

    async def test_empty_dir_returns_empty_lists(self, tmp_path):
        success, errors = await ingest_directory(str(tmp_path))
        assert success == []
        assert errors == []

    async def test_unsupported_files_silently_skipped(self, tmp_path):
        (tmp_path / "ignore.exe").write_bytes(b"binary")
        (tmp_path / "ignore.csv").write_text("a,b,c")
        success, errors = await ingest_directory(str(tmp_path))
        assert success == []
        assert errors == []

    async def test_supported_file_ingested(self, tmp_path):
        (tmp_path / "doc.txt").write_text("有效的文件內容", encoding="utf-8")

        doc = _make_doc(title="doc")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo), \
             patch("services.ingestion_service.extract_and_init_document_concepts",
                   new=AsyncMock()):
            success, errors = await ingest_directory(str(tmp_path))

        assert len(success) == 1
        assert errors == []

    async def test_failed_file_reported_in_errors(self, tmp_path):
        (tmp_path / "bad.txt").write_text("   ")  # empty content → error

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository"):
            success, errors = await ingest_directory(str(tmp_path))

        assert success == []
        assert len(errors) == 1
        assert "bad.txt" in errors[0]


# ── move_and_ingest ──────────────────────────────────────────────────────────

class TestMoveAndIngest:
    async def test_creates_target_dir_if_missing(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst" / "nested"
        src.mkdir()
        (src / "note.txt").write_text("有效內容", encoding="utf-8")

        doc = _make_doc(title="note")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo), \
             patch("services.ingestion_service.extract_and_init_document_concepts",
                   new=AsyncMock()):
            success, errors = await move_and_ingest(str(src), str(dst))

        assert dst.exists()
        assert len(success) == 1

    async def test_duplicate_name_gets_counter_suffix(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "file.txt").write_text("內容A", encoding="utf-8")
        (dst / "file.txt").write_text("已存在", encoding="utf-8")  # collision

        doc = _make_doc(title="file_1")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo), \
             patch("services.ingestion_service.extract_and_init_document_concepts",
                   new=AsyncMock()):
            await move_and_ingest(str(src), str(dst))

        # Original file in dst should be untouched
        assert (dst / "file.txt").read_text() == "已存在"
        assert (dst / "file_1.txt").exists()

    async def test_ingest_failure_restores_file_to_source(self, tmp_path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "problem.txt").write_text("內容", encoding="utf-8")

        mock_repo = AsyncMock()
        mock_repo.create.side_effect = RuntimeError("DB 掛掉了")

        with patch("services.ingestion_service.get_driver"), \
             patch("services.ingestion_service.DocumentRepository", return_value=mock_repo):
            success, errors = await move_and_ingest(str(src), str(dst))

        assert len(success) == 0
        assert len(errors) >= 1
        # File should be restored to source
        assert (src / "problem.txt").exists()
