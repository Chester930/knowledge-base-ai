"""
Ontology Service 測試 — 本體動態擴充機制

- get_extra_entity_types / get_extra_rel_types：讀取合併結果（global + 該 KG）
- add_extension：kg/global 兩種 scope 的隔離性、去重、數量上限
- get_effective_rel_pattern：組出 BFS 用的 Cypher relationship pattern
- 持久化：寫入後重新載入（模擬程序重啟）仍可讀回
"""
from __future__ import annotations
import json
import pytest

import services.ontology_service as ontology_service


@pytest.fixture(autouse=True)
def _isolated_ext_file(tmp_path, monkeypatch):
    """每個測試使用獨立的暫存檔案，避免污染專案根目錄的 ontology_extensions.json，
    並在測試前後重置記憶體快取。"""
    ext_file = tmp_path / "ontology_extensions.json"
    monkeypatch.setattr(ontology_service, "_EXT_FILE", ext_file)
    ontology_service._reset_cache_for_tests()
    yield
    ontology_service._reset_cache_for_tests()


class TestReadEmptyState:
    def test_no_file_returns_empty_entity_types(self):
        assert ontology_service.get_extra_entity_types("kg-1") == []

    def test_no_file_returns_empty_rel_types(self):
        assert ontology_service.get_extra_rel_types("kg-1") == []

    def test_effective_rel_pattern_unchanged_when_no_extensions(self):
        base = "IS_A|PART_OF|CAUSES"
        assert ontology_service.get_effective_rel_pattern("kg-1", base) == base


class TestAddExtensionKgScope:
    async def test_kg_scoped_type_visible_to_that_kg(self):
        await ontology_service.add_extension("kg-1", ["新類型"], ["NEW_REL"], scope="kg")
        assert "新類型" in ontology_service.get_extra_entity_types("kg-1")
        assert "NEW_REL" in ontology_service.get_extra_rel_types("kg-1")

    async def test_kg_scoped_type_not_visible_to_other_kg(self):
        await ontology_service.add_extension("kg-1", ["新類型"], ["NEW_REL"], scope="kg")
        assert ontology_service.get_extra_entity_types("kg-2") == []
        assert ontology_service.get_extra_rel_types("kg-2") == []

    async def test_duplicate_addition_does_not_duplicate_entries(self):
        await ontology_service.add_extension("kg-1", ["新類型"], ["NEW_REL"], scope="kg")
        await ontology_service.add_extension("kg-1", ["新類型"], ["NEW_REL"], scope="kg")
        assert ontology_service.get_extra_entity_types("kg-1").count("新類型") == 1
        assert ontology_service.get_extra_rel_types("kg-1").count("NEW_REL") == 1

    async def test_rel_type_normalized_to_uppercase(self):
        await ontology_service.add_extension("kg-1", [], ["new_rel"], scope="kg")
        assert "NEW_REL" in ontology_service.get_extra_rel_types("kg-1")


class TestAddExtensionGlobalScope:
    async def test_global_scoped_type_visible_to_all_kgs(self):
        await ontology_service.add_extension("kg-1", ["全域類型"], ["GLOBAL_REL"], scope="global")
        assert "全域類型" in ontology_service.get_extra_entity_types("kg-2")
        assert "GLOBAL_REL" in ontology_service.get_extra_rel_types("kg-999")

    async def test_kg_and_global_types_both_returned_for_same_kg(self):
        await ontology_service.add_extension("kg-1", ["全域類型"], [], scope="global")
        await ontology_service.add_extension("kg-1", ["專屬類型"], [], scope="kg")
        types = ontology_service.get_extra_entity_types("kg-1")
        assert "全域類型" in types
        assert "專屬類型" in types


class TestAddExtensionCapAndDefaults:
    async def test_invalid_scope_defaults_to_kg(self):
        result = await ontology_service.add_extension("kg-1", ["X"], [], scope="not-a-real-scope")
        assert result["scope"] == "kg"
        assert ontology_service.get_extra_entity_types("kg-2") == []

    async def test_new_types_capped_per_call(self):
        many = [f"類型{i}" for i in range(10)]
        result = await ontology_service.add_extension("kg-1", many, [], scope="kg")
        assert len(result["entity_types"]) <= ontology_service._MAX_NEW_TYPES_PER_CALL

    async def test_blank_and_empty_strings_ignored(self):
        result = await ontology_service.add_extension("kg-1", ["", "  ", "有效類型"], [], scope="kg")
        assert result["entity_types"] == ["有效類型"]


class TestEffectiveRelPattern:
    async def test_includes_custom_types_after_addition(self):
        base = "IS_A|PART_OF"
        await ontology_service.add_extension("kg-1", [], ["REGULATES"], scope="kg")
        pattern = ontology_service.get_effective_rel_pattern("kg-1", base)
        assert pattern == "IS_A|PART_OF|REGULATES"

    async def test_other_kg_pattern_unaffected(self):
        base = "IS_A|PART_OF"
        await ontology_service.add_extension("kg-1", [], ["REGULATES"], scope="kg")
        assert ontology_service.get_effective_rel_pattern("kg-2", base) == base


class TestPersistenceAcrossReload:
    async def test_survives_cache_reset(self):
        await ontology_service.add_extension("kg-1", ["持久化類型"], ["PERSIST_REL"], scope="kg")
        ontology_service._reset_cache_for_tests()  # 模擬重啟：清空記憶體快取
        assert "持久化類型" in ontology_service.get_extra_entity_types("kg-1")
        assert "PERSIST_REL" in ontology_service.get_extra_rel_types("kg-1")

    async def test_file_contains_expected_structure(self):
        await ontology_service.add_extension("kg-1", ["X"], ["Y"], scope="kg")
        with open(ontology_service._EXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "global" in data and "kg" in data
        assert data["kg"]["kg-1"]["entity_types"] == ["X"]
        assert data["kg"]["kg-1"]["rel_types"] == ["Y"]
