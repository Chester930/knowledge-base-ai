"""
Transcribe Router 測試

POST /transcribe/file            — 上傳單一檔案並轉譯
POST /transcribe/folder          — 批次轉譯資料夾
GET  /transcribe/staging         — 列出暫存區 .txt
GET  /transcribe/watcher/status  — File Watcher 狀態
GET  /transcribe/supported-formats — 支援格式清單
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from services.transcribe_service import TranscribeResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok_result(src="__upload__test.pdf", txt="test.txt", chars=500) -> TranscribeResult:
    return TranscribeResult(
        src_path=src, txt_path=txt,
        success=True, error=None,
        char_count=chars, elapsed_seconds=0.5,
    )


def _fail_result(src="bad.pdf", error="轉譯失敗") -> TranscribeResult:
    return TranscribeResult(
        src_path=src, txt_path=None,
        success=False, error=error,
        char_count=0, elapsed_seconds=0.1,
    )


def _upload_file(filename="doc.pdf", content=b"%PDF-1.4 fake"):
    return {"file": (filename, io.BytesIO(content), "application/pdf")}


# ── GET /transcribe/supported-formats ────────────────────────────────────────

class TestSupportedFormats:
    async def test_returns_format_list(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/transcribe/supported-formats")

        assert res.status_code == 200
        data = res.json()
        assert "formats" in data
        assert isinstance(data["formats"], list)
        assert len(data["formats"]) > 0

    async def test_includes_common_formats(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/transcribe/supported-formats")

        formats = res.json()["formats"]
        assert ".pdf" in formats
        assert ".txt" in formats
        assert ".docx" in formats

    async def test_formats_are_sorted(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/transcribe/supported-formats")

        formats = res.json()["formats"]
        assert formats == sorted(formats)


# ── GET /transcribe/staging ───────────────────────────────────────────────────

class TestTranscribeListStaging:
    async def test_empty_dir_returns_zero(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms:
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/transcribe/staging")
                assert res.status_code == 200
                data = res.json()
                assert data["count"] == 0
                assert data["files"] == []

    async def test_lists_txt_files(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "a.txt").write_text("內容A", encoding="utf-8")
            (staging / "b.txt").write_text("內容B", encoding="utf-8")
            (staging / "skip.pdf").write_bytes(b"PDF")

            with patch("routers.transcribe.settings") as ms:
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/transcribe/staging")
                assert res.status_code == 200
                data = res.json()
                assert data["count"] == 2
                names = [f["name"] for f in data["files"]]
                assert "a.txt" in names
                assert "b.txt" in names
                assert "skip.pdf" not in names

    async def test_file_has_required_fields(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "report.txt").write_text("測試內容", encoding="utf-8")

            with patch("routers.transcribe.settings") as ms:
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/transcribe/staging")
                f = res.json()["files"][0]
                assert "name" in f
                assert "size_bytes" in f
                assert "modified" in f


# ── GET /transcribe/watcher/status ───────────────────────────────────────────

class TestWatcherStatus:
    async def test_returns_status_dict(self, test_app):
        with patch("routers.transcribe.watcher_status",
                   return_value={"running": False, "watched_dirs": []}):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/transcribe/watcher/status")

        assert res.status_code == 200
        data = res.json()
        assert "running" in data
        assert "watched_dirs" in data

    async def test_returns_running_true(self, test_app):
        with patch("routers.transcribe.watcher_status",
                   return_value={"running": True, "watched_dirs": ["/some/path"]}):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/transcribe/watcher/status")

        assert res.json()["running"] is True


# ── POST /transcribe/file ─────────────────────────────────────────────────────

class TestTranscribeFile:
    async def test_pdf_upload_success(self, test_app):
        result = _ok_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", return_value=result):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/file", files=_upload_file("doc.pdf"))

        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["char_count"] == 500
        assert "elapsed_seconds" in data

    async def test_unsupported_format_returns_415(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/transcribe/file",
                               files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")})

        assert res.status_code == 415
        assert "不支援" in res.json()["detail"]

    async def test_txt_file_accepted(self, test_app):
        result = _ok_result(src="note.txt", txt="note.txt", chars=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", return_value=result):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/file",
                                      files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")})

        assert res.status_code == 200

    async def test_transcribe_failure_returns_422(self, test_app):
        result = _fail_result(error="無法解析此 PDF")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", return_value=result):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/file", files=_upload_file("broken.pdf"))

        assert res.status_code == 422
        assert "無法解析" in res.json()["detail"]

    async def test_staging_subdir_form_field(self, test_app):
        result = _ok_result()
        calls = []

        def _spy(upload_tmp, staging_dir=None, overwrite=False):
            calls.append({"staging_dir": str(staging_dir)})
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", side_effect=_spy):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post(
                        "/transcribe/file",
                        files=_upload_file("doc.pdf"),
                        data={"staging_subdir": "project_x"},
                    )

        assert res.status_code == 200
        assert "project_x" in calls[0]["staging_dir"]

    async def test_overwrite_form_field_passed(self, test_app):
        result = _ok_result()
        calls = []

        def _spy(upload_tmp, staging_dir=None, overwrite=False):
            calls.append({"overwrite": overwrite})
            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", side_effect=_spy):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    await c.post(
                        "/transcribe/file",
                        files=_upload_file("doc.pdf"),
                        data={"overwrite": "true"},
                    )

        assert calls[0]["overwrite"] is True

    async def test_response_fields_complete(self, test_app):
        result = _ok_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_file", return_value=result):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/file", files=_upload_file("doc.pdf"))

        data = res.json()
        for field in ("src_path", "txt_path", "success", "error", "char_count", "elapsed_seconds"):
            assert field in data

    async def test_no_file_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/transcribe/file")
        assert res.status_code == 422


# ── POST /transcribe/folder ───────────────────────────────────────────────────

class TestTranscribeFolder:
    async def test_folder_not_found_returns_404(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/transcribe/folder",
                               json={"folder_path": "/nonexistent/path/xyz"})
        assert res.status_code == 404

    async def test_path_is_file_not_dir_returns_400(self, test_app):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            file_path = f.name

        try:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/transcribe/folder", json={"folder_path": file_path})
            assert res.status_code == 400
        finally:
            Path(file_path).unlink(missing_ok=True)

    async def test_success_returns_summary(self, test_app):
        results = [_ok_result("a.pdf"), _ok_result("b.pdf"), _fail_result("c.pdf")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_folder", return_value=results):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/folder", json={"folder_path": tmpdir})

        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 3
        assert data["success"] == 2
        assert data["failed"] == 1
        assert len(data["results"]) == 3

    async def test_empty_folder_returns_zero(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_folder", return_value=[]):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.post("/transcribe/folder", json={"folder_path": tmpdir})

        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["success"] == 0
        assert data["failed"] == 0

    async def test_recursive_param_passed(self, test_app):
        calls = []

        def _spy(folder, staging_dir=None, recursive=False, overwrite=False):
            calls.append({"recursive": recursive})
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_folder", side_effect=_spy):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    await c.post("/transcribe/folder",
                                 json={"folder_path": tmpdir, "recursive": True})

        assert calls[0]["recursive"] is True

    async def test_overwrite_param_passed(self, test_app):
        calls = []

        def _spy(folder, staging_dir=None, recursive=False, overwrite=False):
            calls.append({"overwrite": overwrite})
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.transcribe.settings") as ms, \
                 patch("routers.transcribe.transcribe_folder", side_effect=_spy):
                ms.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    await c.post("/transcribe/folder",
                                 json={"folder_path": tmpdir, "overwrite": True})

        assert calls[0]["overwrite"] is True

    async def test_missing_folder_path_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/transcribe/folder", json={})
        assert res.status_code == 422
