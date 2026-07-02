from __future__ import annotations
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from models.kb_skill import KBRegistry, KBSkill
from services.kb_skill_service import load_registry, save_registry, upsert_skill, remove_skill


def _skill(kb_id="kb-1", name="測試KG") -> KBSkill:
    return KBSkill(
        instance_id="local", kb_id=kb_id, name=name, last_sync="2026-01-01T00:00:00Z",
    )


@pytest.fixture
def tmp_registry_path(tmp_path):
    path = tmp_path / "registry.json"
    with patch("services.kb_skill_service.settings") as mock_settings:
        mock_settings.registry_path = str(path)
        yield path


class TestSaveLoadRegistry:
    def test_save_then_load_roundtrip(self, tmp_registry_path):
        registry = KBRegistry(updated_at="x", skills=[_skill()])
        save_registry(registry)

        loaded = load_registry()
        assert len(loaded.skills) == 1
        assert loaded.skills[0].kb_id == "kb-1"

    def test_save_creates_no_leftover_tmp_files(self, tmp_registry_path):
        save_registry(KBRegistry(updated_at="x", skills=[_skill()]))
        siblings = list(tmp_registry_path.parent.iterdir())
        assert siblings == [tmp_registry_path], f"應只留下 registry.json，實際：{siblings}"

    def test_save_is_atomic_original_untouched_on_failure(self, tmp_registry_path):
        """寫入失敗（模擬崩潰）時，既有 registry.json 內容不應被破壞或截斷。"""
        save_registry(KBRegistry(updated_at="x", skills=[_skill(name="原始資料")]))
        original_content = tmp_registry_path.read_text(encoding="utf-8")

        with patch("os.replace", side_effect=OSError("模擬寫入中斷")):
            with pytest.raises(OSError):
                save_registry(KBRegistry(updated_at="y", skills=[_skill(name="新資料")]))

        assert tmp_registry_path.read_text(encoding="utf-8") == original_content
        # 失敗時暫存檔也應被清除，不留垃圾檔案
        siblings = list(tmp_registry_path.parent.iterdir())
        assert siblings == [tmp_registry_path]

    def test_load_missing_file_returns_empty_registry(self, tmp_registry_path):
        registry = load_registry()
        assert registry.skills == []

    def test_load_corrupt_file_returns_empty_registry_not_crash(self, tmp_registry_path):
        tmp_registry_path.write_text("{not valid json", encoding="utf-8")
        registry = load_registry()
        assert registry.skills == []


class TestUpsertRemoveSkill:
    def test_upsert_adds_new_skill(self, tmp_registry_path):
        upsert_skill(_skill(kb_id="a"))
        registry = load_registry()
        assert [s.kb_id for s in registry.skills] == ["a"]

    def test_upsert_replaces_existing_by_kb_id(self, tmp_registry_path):
        upsert_skill(_skill(kb_id="a", name="舊名稱"))
        upsert_skill(_skill(kb_id="a", name="新名稱"))
        registry = load_registry()
        assert len(registry.skills) == 1
        assert registry.skills[0].name == "新名稱"

    def test_remove_skill(self, tmp_registry_path):
        upsert_skill(_skill(kb_id="a"))
        upsert_skill(_skill(kb_id="b"))
        remove_skill("a")
        registry = load_registry()
        assert [s.kb_id for s in registry.skills] == ["b"]
