from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import core.database as database


class _FakeResult:
    def __init__(self, records):
        self.records = records


@pytest.fixture(autouse=True)
def _reset_driver():
    """每個測試前後重置模組級 _driver 全域變數，避免測試互相汙染。"""
    database._driver = None
    yield
    database._driver = None


class TestConnect:
    async def test_creates_driver_and_verifies_connectivity(self):
        mock_driver = AsyncMock()
        mock_settings = MagicMock(neo4j_uri="bolt://x", neo4j_user="neo4j", neo4j_password="pw")

        with patch("core.database.settings", mock_settings), \
             patch("core.database.AsyncGraphDatabase") as mock_graphdb:
            mock_graphdb.driver.return_value = mock_driver
            await database.connect()

        mock_graphdb.driver.assert_called_once_with("bolt://x", auth=("neo4j", "pw"))
        mock_driver.verify_connectivity.assert_called_once()
        assert database._driver is mock_driver

    async def test_connectivity_failure_propagates(self):
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = RuntimeError("連線失敗")

        with patch("core.database.AsyncGraphDatabase") as mock_graphdb:
            mock_graphdb.driver.return_value = mock_driver
            with pytest.raises(RuntimeError, match="連線失敗"):
                await database.connect()


class TestDisconnect:
    async def test_closes_driver_and_resets_to_none(self):
        mock_driver = AsyncMock()
        database._driver = mock_driver

        await database.disconnect()

        mock_driver.close.assert_called_once()
        assert database._driver is None

    async def test_noop_when_no_driver(self):
        database._driver = None
        await database.disconnect()  # 不應拋出例外
        assert database._driver is None


class TestGetDriver:
    def test_raises_when_not_connected(self):
        database._driver = None
        with pytest.raises(RuntimeError, match="資料庫未連線"):
            database.get_driver()

    def test_returns_driver_when_connected(self):
        mock_driver = MagicMock()
        database._driver = mock_driver
        assert database.get_driver() is mock_driver


class TestCreateKgDatabase:
    async def test_succeeds_immediately_when_database_online(self):
        mock_driver = AsyncMock()
        database._driver = mock_driver

        await database.create_kg_database("kgtest123")

        create_call = mock_driver.execute_query.call_args_list[0]
        assert "CREATE DATABASE" in create_call.args[0]
        assert "kgtest123" in create_call.args[0]
        assert create_call.kwargs["database_"] == "system"

        ready_call = mock_driver.execute_query.call_args_list[1]
        assert ready_call.args[0] == "RETURN 1"
        assert ready_call.kwargs["database_"] == "kgtest123"
        assert mock_driver.execute_query.call_count == 2

    async def test_retries_until_database_comes_online(self):
        mock_driver = AsyncMock()
        # 第一次 CREATE 成功；接著 RETURN 1 失敗兩次、第三次成功
        mock_driver.execute_query.side_effect = [
            _FakeResult([]),               # CREATE DATABASE
            RuntimeError("尚未上線"),        # RETURN 1 attempt 1
            RuntimeError("尚未上線"),        # RETURN 1 attempt 2
            _FakeResult([]),                # RETURN 1 attempt 3 — 成功
        ]
        database._driver = mock_driver

        with patch("core.database.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await database.create_kg_database("kgslow")

        assert mock_driver.execute_query.call_count == 4
        assert mock_sleep.call_count == 2

    async def test_gives_up_after_max_attempts_without_raising(self):
        mock_driver = AsyncMock()
        mock_driver.execute_query.side_effect = [
            _FakeResult([]),  # CREATE DATABASE
        ] + [RuntimeError("一直失敗")] * 40  # RETURN 1 一直失敗
        database._driver = mock_driver

        with patch("core.database.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await database.create_kg_database("kgnever")  # 不應拋出例外

        assert mock_sleep.call_count == 40


class TestDropKgDatabase:
    async def test_issues_drop_database_query(self):
        mock_driver = AsyncMock()
        database._driver = mock_driver

        await database.drop_kg_database("kgtoremove")

        call = mock_driver.execute_query.call_args
        assert "DROP DATABASE" in call.args[0]
        assert "kgtoremove" in call.args[0]
        assert "DESTROY DATA" in call.args[0]
        assert call.kwargs["database_"] == "system"


class TestListKgDatabases:
    async def test_returns_only_online_kg_prefixed_databases(self):
        mock_driver = AsyncMock()
        mock_driver.execute_query.return_value = _FakeResult([
            {"name": "kgabc123", "currentStatus": "online"},
            {"name": "kgoffline", "currentStatus": "offline"},
            {"name": "neo4j", "currentStatus": "online"},
            {"name": "system", "currentStatus": "online"},
        ])
        database._driver = mock_driver

        result = await database.list_kg_databases()

        assert result == ["kgabc123"]

    async def test_empty_when_no_databases(self):
        mock_driver = AsyncMock()
        mock_driver.execute_query.return_value = _FakeResult([])
        database._driver = mock_driver

        assert await database.list_kg_databases() == []

    async def test_queries_system_database(self):
        mock_driver = AsyncMock()
        mock_driver.execute_query.return_value = _FakeResult([])
        database._driver = mock_driver

        await database.list_kg_databases()

        _, kwargs = mock_driver.execute_query.call_args
        assert kwargs["database_"] == "system"
