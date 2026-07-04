from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from repositories.document_repo import DocumentRepository


class _FakeTemporal:
    """模擬 neo4j.time.DateTime，提供 .to_native() 回傳原生 python datetime。"""
    def __init__(self, dt: datetime):
        self._dt = dt

    def to_native(self) -> datetime:
        return self._dt


class _FakeResult:
    def __init__(self, records):
        self.records = records


def _fake_doc_node(**overrides):
    now = _FakeTemporal(datetime(2026, 1, 1, 12, 0, 0))
    node = {
        "id": str(uuid4()),
        "title": "測試文件",
        "content": "文件內容",
        "file_path": "/path/to/file.txt",
        "file_type": "txt",
        "created_at": now,
        "updated_at": now,
    }
    node.update(overrides)
    return node


def _repo_with_driver():
    driver = AsyncMock()
    return DocumentRepository(driver), driver


class TestCreate:
    async def test_returns_document_from_returned_node(self):
        repo, driver = _repo_with_driver()
        node = _fake_doc_node(title="新文件")
        driver.execute_query.return_value = _FakeResult([{"d": node}])

        doc = await repo.create("新文件", "內容", "/a/b.txt", "txt")

        assert doc.title == "新文件"
        assert doc.file_type == "txt"

    async def test_passes_params_to_merge_query(self):
        repo, driver = _repo_with_driver()
        node = _fake_doc_node()
        driver.execute_query.return_value = _FakeResult([{"d": node}])

        await repo.create("標題", "內容", "/path.txt", "md")

        _, kwargs = driver.execute_query.call_args
        assert kwargs["title"] == "標題"
        assert kwargs["content"] == "內容"
        assert kwargs["file_path"] == "/path.txt"
        assert kwargs["file_type"] == "md"

    async def test_none_file_path_passed_through(self):
        repo, driver = _repo_with_driver()
        node = _fake_doc_node(file_path=None)
        driver.execute_query.return_value = _FakeResult([{"d": node}])

        doc = await repo.create("標題", "內容", None, "manual")

        assert doc.file_path is None
        _, kwargs = driver.execute_query.call_args
        assert kwargs["file_path"] is None


class TestGetById:
    async def test_returns_document_when_found(self):
        repo, driver = _repo_with_driver()
        doc_id = uuid4()
        node = _fake_doc_node(id=str(doc_id))
        driver.execute_query.return_value = _FakeResult([{"d": node}])

        doc = await repo.get_by_id(doc_id)

        assert doc.id == doc_id

    async def test_returns_none_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        doc = await repo.get_by_id(uuid4())

        assert doc is None


class TestListAll:
    async def test_returns_all_documents(self):
        repo, driver = _repo_with_driver()
        nodes = [_fake_doc_node(title=f"文件{i}") for i in range(3)]
        driver.execute_query.return_value = _FakeResult([{"d": n} for n in nodes])

        docs = await repo.list_all()

        assert len(docs) == 3
        assert docs[0].title == "文件0"

    async def test_empty_result_returns_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        docs = await repo.list_all()

        assert docs == []

    async def test_passes_limit_and_offset(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.list_all(limit=10, offset=20)

        _, kwargs = driver.execute_query.call_args
        assert kwargs == {"offset": 20, "limit": 10}


class TestDelete:
    async def test_returns_true_when_deleted(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"cnt": 1}])

        assert await repo.delete(uuid4()) is True

    async def test_returns_false_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"cnt": 0}])

        assert await repo.delete(uuid4()) is False


class TestSearchByTitle:
    async def test_returns_matching_documents(self):
        repo, driver = _repo_with_driver()
        nodes = [_fake_doc_node(title="AI 筆記")]
        driver.execute_query.return_value = _FakeResult([{"d": n} for n in nodes])

        docs = await repo.search_by_title("AI")

        assert len(docs) == 1
        assert docs[0].title == "AI 筆記"

    async def test_passes_query_param(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.search_by_title("關鍵字")

        _, kwargs = driver.execute_query.call_args
        assert kwargs["q"] == "關鍵字"


class TestGetCount:
    async def test_returns_count(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"cnt": 42}])

        assert await repo.get_count() == 42


class TestGetOrphanDocuments:
    async def test_returns_preview_with_default_length(self):
        repo, driver = _repo_with_driver()
        long_content = "x" * 500
        driver.execute_query.return_value = _FakeResult([
            {"id": "doc-1", "title": "標題", "content": long_content}
        ])

        docs = await repo.get_orphan_documents()

        assert len(docs[0]["preview"]) == 300

    async def test_replaces_newlines_with_spaces(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "doc-1", "title": "標題", "content": "第一行\n第二行"}
        ])

        docs = await repo.get_orphan_documents()

        assert "\n" not in docs[0]["preview"]
        assert "第一行 第二行" in docs[0]["preview"]

    async def test_none_content_becomes_empty_preview(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "doc-1", "title": "標題", "content": None}
        ])

        docs = await repo.get_orphan_documents()

        assert docs[0]["preview"] == ""

    async def test_custom_preview_chars_respected(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "doc-1", "title": "標題", "content": "x" * 100}
        ])

        docs = await repo.get_orphan_documents(preview_chars=10)

        assert len(docs[0]["preview"]) == 10

    async def test_empty_result_returns_empty_list(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        docs = await repo.get_orphan_documents()

        assert docs == []
