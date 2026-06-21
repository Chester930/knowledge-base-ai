"""
Staging Router 測試

GET    /staging                        — 列出暫存區 .txt
POST   /staging/{filename}/classify    — 分析單一文件
POST   /staging/classify-all           — 批次分析所有文件
POST   /staging/{filename}/assign      — 手動指定 KG
DELETE /staging/{filename}             — 刪除暫存文件
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from models.knowledge_graph import ClassifyResult, KGCandidate


# ── helpers ───────────────────────────────────────────────────────────────────

def _result(filename="doc.txt", status="pending", score=0.0, matched_kg_id=None) -> ClassifyResult:
    return ClassifyResult(
        txt_filename=filename,
        candidates=[],
        matched_kg_id=matched_kg_id,
        score=score,
        status=status,
    )


def _result_with_match(filename="doc.txt", score=0.85) -> ClassifyResult:
    kg_id = uuid4()
    return ClassifyResult(
        txt_filename=filename,
        candidates=[KGCandidate(kg_id=kg_id, kg_name="測試KG", score=score)],
        matched_kg_id=kg_id,
        matched_kg_name="測試KG",
        score=score,
        status="assigned",
        auto_assigned=True,
    )


# ── GET /staging ──────────────────────────────────────────────────────────────

class TestListStaging:
    async def test_empty_staging_dir(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/staging")

        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 0
        assert data["files"] == []
        assert "staging_dir" in data

    async def test_lists_txt_files(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "doc_a.txt").write_text("內容A", encoding="utf-8")
            (staging / "doc_b.txt").write_text("內容B長一點", encoding="utf-8")
            (staging / "not_txt.pdf").write_text("PDF")  # 不應被列出

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/staging")

        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 2
        names = [f["name"] for f in data["files"]]
        assert "doc_a.txt" in names
        assert "doc_b.txt" in names
        assert "not_txt.pdf" not in names

    async def test_file_metadata_fields(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "test.txt").write_text("測試內容", encoding="utf-8")

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/staging")

        f = res.json()["files"][0]
        assert "name" in f
        assert "size_bytes" in f
        assert "size_chars" in f
        assert f["size_bytes"] > 0
        assert f["size_chars"] > 0

    async def test_creates_staging_dir_if_missing(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            # _staging 不存在
            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/staging")
                # assert 在 tmpdir 還存在時執行
                assert res.status_code == 200
                assert Path(tmpdir, "_staging").exists()

    async def test_files_returned_in_sorted_order(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            for name in ["c.txt", "a.txt", "b.txt"]:
                (staging / name).write_text("x", encoding="utf-8")

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.get("/staging")

        names = [f["name"] for f in res.json()["files"]]
        assert names == sorted(names)


# ── POST /staging/{filename}/classify ─────────────────────────────────────────

class TestClassifyOne:
    async def test_classify_returns_result(self, test_app):
        result = _result_with_match("report.txt", score=0.92)
        with patch("routers.staging.classify_document", new=AsyncMock(return_value=result)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/report.txt/classify", json={})

        assert res.status_code == 200
        data = res.json()
        assert data["txt_filename"] == "report.txt"
        assert data["score"] == pytest.approx(0.92)
        assert data["status"] == "assigned"

    async def test_file_not_found_returns_404(self, test_app):
        with patch("routers.staging.classify_document",
                   new=AsyncMock(side_effect=FileNotFoundError("找不到 missing.txt"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/missing.txt/classify", json={})

        assert res.status_code == 404

    async def test_classify_passes_threshold(self, test_app):
        result = _result("doc.txt")
        with patch("routers.staging.classify_document",
                   new=AsyncMock(return_value=result)) as mock_cls:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/doc.txt/classify", json={"threshold": 0.6})

        _, kwargs = mock_cls.call_args
        assert kwargs.get("threshold") == pytest.approx(0.6)

    async def test_classify_auto_assign_true(self, test_app):
        result = _result_with_match()
        with patch("routers.staging.classify_document",
                   new=AsyncMock(return_value=result)) as mock_cls:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/doc.txt/classify", json={"auto_assign": True})

        _, kwargs = mock_cls.call_args
        assert kwargs.get("auto_assign") is True

    async def test_classify_owner_id_passed(self, test_app):
        result = _result()
        with patch("routers.staging.classify_document",
                   new=AsyncMock(return_value=result)) as mock_cls:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/doc.txt/classify", json={"owner_id": "alice"})

        _, kwargs = mock_cls.call_args
        assert kwargs.get("owner_id") == "alice"

    async def test_service_error_returns_500(self, test_app):
        with patch("routers.staging.classify_document",
                   new=AsyncMock(side_effect=RuntimeError("DB 連線失敗"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/doc.txt/classify", json={})

        assert res.status_code == 500

    async def test_default_body_accepted(self, test_app):
        result = _result("doc.txt")
        with patch("routers.staging.classify_document", new=AsyncMock(return_value=result)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/doc.txt/classify")

        assert res.status_code == 200


# ── POST /staging/classify-all ────────────────────────────────────────────────

class TestClassifyAll:
    async def test_returns_list_of_results(self, test_app):
        results = [_result("a.txt"), _result("b.txt", score=0.7, status="unmatched")]
        with patch("routers.staging.classify_all", new=AsyncMock(return_value=results)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/classify-all")

        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_empty_staging_returns_empty_list(self, test_app):
        with patch("routers.staging.classify_all", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/staging/classify-all")

        assert res.status_code == 200
        assert res.json() == []

    async def test_threshold_param_passed(self, test_app):
        with patch("routers.staging.classify_all",
                   new=AsyncMock(return_value=[])) as mock_all:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/classify-all?threshold=0.8")

        mock_all.assert_called_once_with(threshold=pytest.approx(0.8), auto_assign=False)

    async def test_auto_assign_param_passed(self, test_app):
        with patch("routers.staging.classify_all",
                   new=AsyncMock(return_value=[])) as mock_all:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/classify-all?auto_assign=true")

        mock_all.assert_called_once_with(threshold=pytest.approx(0.3), auto_assign=True)

    async def test_invalid_threshold_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/staging/classify-all?threshold=1.5")
        assert res.status_code == 422


# ── POST /staging/{filename}/assign ───────────────────────────────────────────

class TestAssignOne:
    async def test_assign_returns_ok(self, test_app):
        kg_id = uuid4()
        with patch("routers.staging.assign_document_to_kg", new=AsyncMock()):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/staging/report.txt/assign", json={"kg_id": str(kg_id)})

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "assigned"
        assert data["filename"] == "report.txt"
        assert data["kg_id"] == str(kg_id)

    async def test_file_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        with patch("routers.staging.assign_document_to_kg",
                   new=AsyncMock(side_effect=FileNotFoundError("找不到"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/staging/missing.txt/assign", json={"kg_id": str(kg_id)})

        assert res.status_code == 404

    async def test_kg_not_found_returns_404(self, test_app):
        kg_id = uuid4()
        with patch("routers.staging.assign_document_to_kg",
                   new=AsyncMock(side_effect=ValueError("KG 不存在"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/staging/doc.txt/assign", json={"kg_id": str(kg_id)})

        assert res.status_code == 404

    async def test_missing_kg_id_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/staging/doc.txt/assign", json={})
        assert res.status_code == 422

    async def test_invalid_uuid_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/staging/doc.txt/assign", json={"kg_id": "not-a-uuid"})
        assert res.status_code == 422

    async def test_service_error_returns_500(self, test_app):
        kg_id = uuid4()
        with patch("routers.staging.assign_document_to_kg",
                   new=AsyncMock(side_effect=RuntimeError("Neo4j timeout"))):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/staging/doc.txt/assign", json={"kg_id": str(kg_id)})

        assert res.status_code == 500

    async def test_assign_calls_service_with_correct_args(self, test_app):
        kg_id = uuid4()
        with patch("routers.staging.assign_document_to_kg",
                   new=AsyncMock()) as mock_assign:
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/staging/report.txt/assign", json={"kg_id": str(kg_id)})

        mock_assign.assert_called_once_with("report.txt", kg_id)


# ── DELETE /staging/{filename} ────────────────────────────────────────────────

class TestDeleteStaging:
    async def test_delete_existing_file_returns_204(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "old.txt").write_text("舊文件", encoding="utf-8")

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.delete("/staging/old.txt")

        assert res.status_code == 204

    async def test_delete_removes_file_from_disk(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            target = staging / "to_delete.txt"
            target.write_text("要刪的", encoding="utf-8")
            assert target.exists()

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    await c.delete("/staging/to_delete.txt")

            assert not target.exists()

    async def test_delete_nonexistent_returns_404(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "_staging").mkdir()
            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    res = await c.delete("/staging/ghost.txt")

        assert res.status_code == 404

    async def test_delete_does_not_affect_other_files(self, test_app):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "_staging"
            staging.mkdir()
            (staging / "keep.txt").write_text("保留", encoding="utf-8")
            (staging / "delete_me.txt").write_text("刪我", encoding="utf-8")

            with patch("routers.staging.settings") as mock_settings:
                mock_settings.workspace_dir = tmpdir
                async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                    await c.delete("/staging/delete_me.txt")

            assert (staging / "keep.txt").exists()
            assert not (staging / "delete_me.txt").exists()
