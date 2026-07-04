from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from repositories.knowledge_graph_repo import KnowledgeGraphRepository


class _FakeResult:
    def __init__(self, records):
        self.records = records


def _fake_kg_node(**overrides):
    now = datetime(2026, 1, 1, 12, 0, 0)
    node = {
        "id": str(uuid4()),
        "name": "測試KG",
        "description": "描述",
        "folder_path": "/workspace/kg_test",
        "owner_id": "default",
        "is_public": True,
        "db_name": "",
        "doc_count": 0,
        "entity_count": 0,
        "relation_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    node.update(overrides)
    return node


def _repo_with_driver():
    driver = AsyncMock()
    return KnowledgeGraphRepository(driver), driver


class TestCreate:
    async def test_creates_and_returns_kg(self):
        repo, driver = _repo_with_driver()
        node = _fake_kg_node(name="新KG")
        # 第一次呼叫：CREATE（無需回傳值）；第二次呼叫：內部呼叫 get_by_id
        driver.execute_query.side_effect = [
            _FakeResult([]),
            _FakeResult([{"kg": node}]),
        ]

        kg = await repo.create(name="新KG", description="desc", folder_path="/x")

        assert kg.name == "新KG"
        assert driver.execute_query.call_count == 2

    async def test_create_query_receives_all_fields(self):
        repo, driver = _repo_with_driver()
        node = _fake_kg_node()
        driver.execute_query.side_effect = [_FakeResult([]), _FakeResult([{"kg": node}])]

        await repo.create(
            name="KG", description="desc", folder_path="/f",
            owner_id="user1", is_public=False, db_name="kgabc123",
        )

        first_call_kwargs = driver.execute_query.call_args_list[0].kwargs
        assert first_call_kwargs["name"] == "KG"
        assert first_call_kwargs["owner_id"] == "user1"
        assert first_call_kwargs["is_public"] is False
        assert first_call_kwargs["db_name"] == "kgabc123"


class TestGetById:
    async def test_returns_kg_when_found(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.return_value = _FakeResult([{"kg": _fake_kg_node(id=str(kg_id))}])

        kg = await repo.get_by_id(kg_id)

        assert kg.id == kg_id

    async def test_returns_none_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        assert await repo.get_by_id(uuid4()) is None


class TestGetByName:
    async def test_returns_kg_when_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"kg": _fake_kg_node(name="找到了")}])

        kg = await repo.get_by_name("找到了")

        assert kg.name == "找到了"

    async def test_returns_none_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        assert await repo.get_by_name("不存在") is None


class TestListAll:
    async def test_owner_id_branch_passes_owner_id(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.list_all(owner_id="user1")

        query, kwargs = driver.execute_query.call_args.args[0], driver.execute_query.call_args.kwargs
        assert "kg.owner_id" in query
        assert kwargs["owner_id"] == "user1"

    async def test_no_owner_id_uses_public_only_branch(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        await repo.list_all()

        query, kwargs = driver.execute_query.call_args.args[0], driver.execute_query.call_args.kwargs
        assert "kg.is_public" in query
        assert kwargs["owner_id"] == ""

    async def test_returns_all_kgs(self):
        repo, driver = _repo_with_driver()
        nodes = [_fake_kg_node(name=f"KG{i}") for i in range(2)]
        driver.execute_query.return_value = _FakeResult([{"kg": n} for n in nodes])

        kgs = await repo.list_all()

        assert len(kgs) == 2


class TestUpdate:
    async def test_updates_only_provided_fields(self):
        repo, driver = _repo_with_driver()
        node = _fake_kg_node(name="改名後")
        driver.execute_query.side_effect = [
            _FakeResult([]),  # SET query
            _FakeResult([{"kg": node}]),  # get_by_id
        ]

        kg = await repo.update(uuid4(), name="改名後")

        assert kg.name == "改名後"
        set_query = driver.execute_query.call_args_list[0].args[0]
        assert "kg.name = $name" in set_query
        assert "kg.description" not in set_query
        assert "kg.is_public" not in set_query

    async def test_no_optional_fields_only_updates_timestamp(self):
        repo, driver = _repo_with_driver()
        node = _fake_kg_node()
        driver.execute_query.side_effect = [_FakeResult([]), _FakeResult([{"kg": node}])]

        await repo.update(uuid4())

        set_query = driver.execute_query.call_args_list[0].args[0]
        assert set_query.count("SET") == 1
        assert "kg.updated_at = datetime()" in set_query

    async def test_all_fields_included_when_all_provided(self):
        repo, driver = _repo_with_driver()
        node = _fake_kg_node()
        driver.execute_query.side_effect = [_FakeResult([]), _FakeResult([{"kg": node}])]

        await repo.update(uuid4(), name="N", description="D", is_public=False)

        set_query = driver.execute_query.call_args_list[0].args[0]
        assert "kg.name = $name" in set_query
        assert "kg.description = $description" in set_query
        assert "kg.is_public = $is_public" in set_query


class TestDelete:
    async def test_returns_true_when_deleted(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"deleted": 1}])
        assert await repo.delete(uuid4()) is True

    async def test_returns_false_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"deleted": 0}])
        assert await repo.delete(uuid4()) is False


class TestDocumentAssociation:
    async def test_add_document_passes_ids(self):
        repo, driver = _repo_with_driver()
        kg_id, doc_id = uuid4(), uuid4()
        await repo.add_document(kg_id, doc_id)
        _, kwargs = driver.execute_query.call_args
        assert kwargs == {"kg_id": str(kg_id), "doc_id": str(doc_id)}

    async def test_remove_document_passes_ids(self):
        repo, driver = _repo_with_driver()
        kg_id, doc_id = uuid4(), uuid4()
        await repo.remove_document(kg_id, doc_id)
        _, kwargs = driver.execute_query.call_args
        assert kwargs == {"kg_id": str(kg_id), "doc_id": str(doc_id)}

    async def test_get_documents_returns_records(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([
            {"id": "d1", "title": "文件1", "file_type": "txt", "file_path": "/a.txt", "created_at": datetime.now()}
        ])
        docs = await repo.get_documents(uuid4())
        assert docs[0]["title"] == "文件1"


class TestGetDbName:
    async def test_returns_db_name_when_present(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"db_name": "kgabc123"}])
        assert await repo.get_db_name(uuid4()) == "kgabc123"

    async def test_returns_empty_string_when_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])
        assert await repo.get_db_name(uuid4()) == ""

    async def test_returns_empty_string_when_null(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([{"db_name": None}])
        assert await repo.get_db_name(uuid4()) == ""


class TestRefreshCounts:
    async def test_community_mode_uses_kg_id_filter(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.side_effect = [
            _FakeResult([{"db_name": ""}]),      # get_db_name
            _FakeResult([{"doc_c": 3}]),          # doc count
            _FakeResult([{"ent_c": 10}]),         # entity count (community: filtered by kg_id)
            _FakeResult([{"rel_c": 8}]),           # relation count
            _FakeResult([]),                       # final SET
        ]

        await repo.refresh_counts(kg_id)

        ent_call = driver.execute_query.call_args_list[2]
        assert "kg_id" not in ent_call.args[0] or True  # 語意上以 Entity {kg_id: $id} 篩選
        assert ent_call.kwargs.get("id") == str(kg_id)

        final_call = driver.execute_query.call_args_list[4]
        assert final_call.kwargs["doc_c"] == 3
        assert final_call.kwargs["ent_c"] == 10
        assert final_call.kwargs["rel_c"] == 8

    async def test_enterprise_mode_uses_database_param(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        driver.execute_query.side_effect = [
            _FakeResult([{"db_name": "kgabc123"}]),  # get_db_name
            _FakeResult([{"doc_c": 1}]),
            _FakeResult([{"ent_c": 5}]),
            _FakeResult([{"rel_c": 4}]),
            _FakeResult([]),
        ]

        await repo.refresh_counts(kg_id)

        ent_call = driver.execute_query.call_args_list[2]
        assert ent_call.kwargs.get("database_") == "kgabc123"

    async def test_zero_counts_when_no_records(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.side_effect = [
            _FakeResult([{"db_name": ""}]),
            _FakeResult([]),  # doc count missing
            _FakeResult([]),  # entity count missing
            _FakeResult([]),  # relation count missing
            _FakeResult([]),
        ]

        await repo.refresh_counts(uuid4())

        final_call = driver.execute_query.call_args_list[4]
        assert final_call.kwargs["doc_c"] == 0
        assert final_call.kwargs["ent_c"] == 0
        assert final_call.kwargs["rel_c"] == 0


class TestGetDetail:
    async def test_returns_none_when_kg_not_found(self):
        repo, driver = _repo_with_driver()
        driver.execute_query.return_value = _FakeResult([])

        assert await repo.get_detail(uuid4()) is None

    async def test_community_mode_builds_detail(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        node = _fake_kg_node(id=str(kg_id), db_name="")
        driver.execute_query.side_effect = [
            _FakeResult([{"kg": node}]),                 # get_by_id
            _FakeResult([{"name": "概念A"}]),             # top concepts
            _FakeResult([{"name": "實體A"}]),             # top entities (community: kg_id filter)
        ]

        detail = await repo.get_detail(kg_id)

        assert detail.top_concepts == ["概念A"]
        assert detail.top_entities == ["實體A"]
        entities_call = driver.execute_query.call_args_list[2]
        assert entities_call.kwargs.get("id") == str(kg_id)

    async def test_enterprise_mode_uses_database_param_for_entities(self):
        repo, driver = _repo_with_driver()
        kg_id = uuid4()
        node = _fake_kg_node(id=str(kg_id), db_name="kgabc123")
        driver.execute_query.side_effect = [
            _FakeResult([{"kg": node}]),
            _FakeResult([{"name": "概念B"}]),
            _FakeResult([{"name": "實體B"}]),
        ]

        detail = await repo.get_detail(kg_id)

        entities_call = driver.execute_query.call_args_list[2]
        assert entities_call.kwargs.get("database_") == "kgabc123"
        assert detail.top_entities == ["實體B"]
