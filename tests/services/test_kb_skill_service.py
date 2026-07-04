from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from models.kb_skill import KBRegistry, KBSkill
from services.kb_skill_service import (
    load_registry,
    save_registry,
    upsert_skill,
    remove_skill,
    generate_skill,
    sync_public_kgs,
)


def _make_skill(kb_id="kb-1"):
    return KBSkill(
        instance_id="local",
        kb_id=kb_id,
        name="測試KG",
        last_sync="2026-01-01T00:00:00+00:00",
    )


# ── load_registry / save_registry ─────────────────────────────────────────────

class TestLoadRegistry:
    def test_returns_empty_registry_when_file_missing(self, tmp_path):
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(tmp_path / "ghost.json")
            registry = load_registry()
        assert registry.skills == []

    def test_loads_existing_registry_file(self, tmp_path):
        p = tmp_path / "registry.json"
        p.write_text(
            KBRegistry(updated_at="now", skills=[_make_skill()]).model_dump_json(),
            encoding="utf-8",
        )
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            registry = load_registry()
        assert len(registry.skills) == 1
        assert registry.skills[0].kb_id == "kb-1"

    def test_corrupted_file_falls_back_to_empty_registry(self, tmp_path):
        p = tmp_path / "registry.json"
        p.write_text("not valid json{{{", encoding="utf-8")
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            registry = load_registry()
        assert registry.skills == []


class TestSaveRegistry:
    def test_writes_registry_to_file(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            save_registry(KBRegistry(updated_at="old", skills=[_make_skill()]))
        assert p.exists()
        assert "kb-1" in p.read_text(encoding="utf-8")

    def test_updates_timestamp_on_save(self, tmp_path):
        p = tmp_path / "registry.json"
        registry = KBRegistry(updated_at="old", skills=[])
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            save_registry(registry)
        assert registry.updated_at != "old"


# ── upsert_skill / remove_skill ───────────────────────────────────────────────

class TestUpsertSkill:
    def test_adds_new_skill(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            upsert_skill(_make_skill(kb_id="new-1"))
            registry = load_registry()
        assert [s.kb_id for s in registry.skills] == ["new-1"]

    def test_replaces_existing_skill_with_same_kb_id(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            upsert_skill(_make_skill(kb_id="dup"))
            skill2 = _make_skill(kb_id="dup")
            skill2.name = "更新後的名稱"
            upsert_skill(skill2)
            registry = load_registry()
        assert len(registry.skills) == 1
        assert registry.skills[0].name == "更新後的名稱"


class TestRemoveSkill:
    def test_removes_matching_skill(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            upsert_skill(_make_skill(kb_id="to-remove"))
            remove_skill("to-remove")
            registry = load_registry()
        assert registry.skills == []

    def test_noop_when_kb_id_not_found(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            upsert_skill(_make_skill(kb_id="keep"))
            remove_skill("ghost-id")
            registry = load_registry()
        assert [s.kb_id for s in registry.skills] == ["keep"]


# ── generate_skill ─────────────────────────────────────────────────────────────

class TestGenerateSkill:
    async def test_raises_when_kg_not_found(self):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = None
        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.instance_id = "local"
            with pytest.raises(ValueError, match="KG 不存在"):
                await generate_skill(uuid4(), MagicMock())

    async def test_builds_skill_with_top_concepts_and_fingerprint(self):
        kg = MagicMock(
            description="desc", db_name="", entity_count=10,
            relation_count=5, doc_count=2,
        )
        kg.name = "測試KG"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kg_concepts.return_value = [
            {"name": "AI", "interest_score": 0.8, "professional_score": 0.6, "q_vector": [1.0, 0.0]},
            {"name": "機器學習", "interest_score": 0.4, "professional_score": 0.4, "q_vector": [0.0, 1.0]},
        ]

        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.instance_id = "local"
            skill = await generate_skill(uuid4(), MagicMock())

        assert skill.name == "測試KG"
        assert skill.top_concepts[0].name == "AI"  # 分數較高排前面
        assert skill.fingerprint_vector == [0.5, 0.5]
        assert skill.entity_count == 10

    async def test_handles_numpy_like_vectors_via_tolist(self):
        kg = MagicMock(description="", db_name="", entity_count=0, relation_count=0, doc_count=0)
        kg.name = "K"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        fake_vector = MagicMock()
        fake_vector.tolist.return_value = [0.2, 0.4]

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kg_concepts.return_value = [
            {"name": "X", "interest_score": 0.5, "professional_score": 0.5, "q_vector": fake_vector},
        ]

        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.instance_id = "local"
            skill = await generate_skill(uuid4(), MagicMock())

        assert skill.fingerprint_vector == [0.2, 0.4]

    async def test_empty_concepts_produce_empty_fingerprint(self):
        kg = MagicMock(description="", db_name="", entity_count=0, relation_count=0, doc_count=0)
        kg.name = "K"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_kg_concepts.return_value = []

        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.instance_id = "local"
            skill = await generate_skill(uuid4(), MagicMock())

        assert skill.fingerprint_vector == []
        assert skill.top_concepts == []


# ── sync_public_kgs ────────────────────────────────────────────────────────────

class TestSyncPublicKgs:
    async def test_syncs_only_public_kgs(self, tmp_path):
        public_kg = MagicMock(id=uuid4(), is_public=True)
        public_kg.name = "公開KG"
        private_kg = MagicMock(id=uuid4(), is_public=False)
        private_kg.name = "私有KG"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.list_all.return_value = [public_kg, private_kg]

        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.settings") as mock_settings, \
             patch("services.kb_skill_service.generate_skill",
                   new=AsyncMock(return_value=_make_skill(kb_id=str(public_kg.id)))):
            mock_settings.registry_path = str(p)
            result = await sync_public_kgs(MagicMock())

        assert result["synced"] == 1
        assert result["errors"] == []

    async def test_generation_failure_recorded_as_error(self, tmp_path):
        public_kg = MagicMock(id=uuid4(), is_public=True)
        public_kg.name = "壞掉的KG"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.list_all.return_value = [public_kg]

        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.kb_skill_service.settings") as mock_settings, \
             patch("services.kb_skill_service.generate_skill",
                   new=AsyncMock(side_effect=RuntimeError("生成失敗"))):
            mock_settings.registry_path = str(p)
            result = await sync_public_kgs(MagicMock())

        assert result["synced"] == 0
        assert len(result["errors"]) == 1
        assert "壞掉的KG" in result["errors"][0]

    async def test_removes_local_skills_no_longer_public(self, tmp_path):
        p = tmp_path / "registry.json"
        with patch("services.kb_skill_service.settings") as mock_settings:
            mock_settings.registry_path = str(p)
            # 預先寫入一個本機 skill，代表曾經公開過、現在已私有化
            stale = _make_skill(kb_id="now-private")
            stale.is_local = True
            upsert_skill(stale)

            mock_kg_repo = AsyncMock()
            mock_kg_repo.list_all.return_value = []  # 目前沒有任何公開 KG

            with patch("services.kb_skill_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
                result = await sync_public_kgs(MagicMock())

            registry = load_registry()

        assert result["removed"] == 1
        assert registry.skills == []
