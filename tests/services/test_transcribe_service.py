from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch

from services.transcribe_service import (
    SUPPORTED_EXTENSIONS,
    AUDIO_EXTENSIONS,
    TranscribeResult,
    _unique_txt_path,
    _sanitize,
    _read_text,
    _extract_text,
    transcribe_file,
    transcribe_folder,
)


# ── SUPPORTED_EXTENSIONS / AUDIO_EXTENSIONS ─────────────────────────────────

class TestSupportedExtensions:
    def test_includes_document_and_audio_formats(self):
        expected = {
            ".md", ".txt", ".pdf", ".docx", ".pptx", ".doc", ".ppt",
            ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv", ".avi",
        }
        assert expected == SUPPORTED_EXTENSIONS

    def test_audio_extensions_subset_of_supported(self):
        assert AUDIO_EXTENSIONS <= SUPPORTED_EXTENSIONS


# ── _unique_txt_path ─────────────────────────────────────────────────────────

class TestUniqueTxtPath:
    def test_no_collision_returns_stem_path(self, tmp_path):
        p = _unique_txt_path(tmp_path, "note", overwrite=False)
        assert p == tmp_path / "note.txt"

    def test_overwrite_true_ignores_existing(self, tmp_path):
        (tmp_path / "note.txt").write_text("existing")
        p = _unique_txt_path(tmp_path, "note", overwrite=True)
        assert p == tmp_path / "note.txt"

    def test_collision_gets_counter_suffix(self, tmp_path):
        (tmp_path / "note.txt").write_text("existing")
        p = _unique_txt_path(tmp_path, "note", overwrite=False)
        assert p == tmp_path / "note_1.txt"

    def test_multiple_collisions_increment_counter(self, tmp_path):
        (tmp_path / "note.txt").write_text("a")
        (tmp_path / "note_1.txt").write_text("b")
        p = _unique_txt_path(tmp_path, "note", overwrite=False)
        assert p == tmp_path / "note_2.txt"


# ── _sanitize ────────────────────────────────────────────────────────────────

class TestSanitize:
    def test_strips_control_characters(self):
        assert _sanitize("a\x00b\x07c") == "abc"

    def test_keeps_newline_tab_carriage_return(self):
        assert _sanitize("a\nb\tc") == "a\nb\tc"

    def test_collapses_triple_blank_lines(self):
        assert _sanitize("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_trailing_whitespace_per_line(self):
        assert _sanitize("a   \nb  ") == "a\nb"

    def test_strips_leading_trailing_whitespace_overall(self):
        assert _sanitize("  \n  a  \n  ") == "a"


# ── _read_text ───────────────────────────────────────────────────────────────

class TestReadText:
    def test_utf8_content(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("你好世界", encoding="utf-8")
        assert _read_text(f) == "你好世界"

    def test_big5_encoding_fallback(self, tmp_path):
        f = tmp_path / "b.txt"
        f.write_bytes("測試文件".encode("big5"))
        assert "測試文件" in _read_text(f)

    def test_utf8_bom(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes("BOM文件".encode("utf-8-sig"))
        assert "BOM文件" in _read_text(f)


# ── _extract_text dispatch ───────────────────────────────────────────────────

class TestExtractTextDispatch:
    def test_txt_dispatches_to_read_text(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("內容", encoding="utf-8")
        with patch("services.transcribe_service._read_text", return_value="mocked") as m:
            assert _extract_text(f) == "mocked"
            m.assert_called_once_with(f)

    def test_pdf_dispatches_to_read_pdf(self, tmp_path):
        f = tmp_path / "a.pdf"
        with patch("services.transcribe_service._read_pdf", return_value="pdf-text") as m:
            assert _extract_text(f) == "pdf-text"
            m.assert_called_once_with(f)

    def test_audio_dispatches_to_read_audio(self, tmp_path):
        f = tmp_path / "a.mp3"
        with patch("services.transcribe_service._read_audio", return_value="audio-text") as m:
            assert _extract_text(f) == "audio-text"
            m.assert_called_once_with(f)

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "a.xyz"
        with pytest.raises(ValueError, match="不支援的格式"):
            _extract_text(f)


# ── transcribe_file ───────────────────────────────────────────────────────────

class TestTranscribeFile:
    def test_missing_file_returns_error_result(self, tmp_path):
        result = transcribe_file(tmp_path / "ghost.txt", staging_dir=tmp_path)
        assert isinstance(result, TranscribeResult)
        assert result.success is False
        assert "找不到檔案" in result.error

    def test_unsupported_extension_returns_error_result(self, tmp_path):
        f = tmp_path / "a.xyz"
        f.write_text("content")
        result = transcribe_file(f, staging_dir=tmp_path)
        assert result.success is False
        assert "不支援的格式" in result.error

    def test_extraction_exception_captured_not_raised(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        with patch("services.transcribe_service._extract_text", side_effect=RuntimeError("boom")):
            result = transcribe_file(f, staging_dir=tmp_path)
        assert result.success is False
        assert "boom" in result.error

    def test_empty_extracted_text_returns_error(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        with patch("services.transcribe_service._extract_text", return_value="   "):
            result = transcribe_file(f, staging_dir=tmp_path)
        assert result.success is False
        assert "提取內容為空" in result.error

    def test_successful_transcription_writes_txt_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("original")
        with patch("services.transcribe_service._extract_text", return_value="轉譯後的內容"):
            result = transcribe_file(f, staging_dir=tmp_path)
        assert result.success is True
        assert result.error is None
        assert result.char_count == len("轉譯後的內容")
        assert Path(result.txt_path).read_text(encoding="utf-8") == "轉譯後的內容"

    def test_default_staging_dir_used_when_none_given(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content")
        fake_staging = tmp_path / "_staging"
        with patch("services.transcribe_service._default_staging_dir", return_value=fake_staging), \
             patch("services.transcribe_service._extract_text", return_value="文字"):
            result = transcribe_file(f)
        assert result.success is True
        assert str(fake_staging) in result.txt_path


# ── transcribe_folder ─────────────────────────────────────────────────────────

class TestTranscribeFolder:
    def test_nonexistent_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            transcribe_folder(tmp_path / "ghost")

    def test_empty_folder_returns_empty_list(self, tmp_path):
        results = transcribe_folder(tmp_path, staging_dir=tmp_path / "out")
        assert results == []

    def test_unsupported_files_ignored(self, tmp_path):
        (tmp_path / "ignore.exe").write_bytes(b"binary")
        (tmp_path / "ignore.csv").write_text("a,b,c")
        results = transcribe_folder(tmp_path, staging_dir=tmp_path / "out")
        assert results == []

    def test_supported_files_transcribed(self, tmp_path):
        (tmp_path / "a.txt").write_text("內容A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("內容B", encoding="utf-8")
        out = tmp_path / "out"
        results = transcribe_folder(tmp_path, staging_dir=out)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_non_recursive_by_default(self, tmp_path):
        (tmp_path / "top.txt").write_text("頂層", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("巢狀", encoding="utf-8")
        out = tmp_path / "out"
        results = transcribe_folder(tmp_path, staging_dir=out, recursive=False)
        assert len(results) == 1
        assert "top" in results[0].src_path

    def test_recursive_finds_nested_files(self, tmp_path):
        (tmp_path / "top.txt").write_text("頂層", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("巢狀", encoding="utf-8")
        out = tmp_path / "out"
        results = transcribe_folder(tmp_path, staging_dir=out, recursive=True)
        assert len(results) == 2
