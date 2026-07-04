from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.knowledge_graph_service import (
    _kg_folder,
    _make_db_name,
    create_kg,
    delete_kg,
    auto_cluster_kgs,
    confirm_auto_cluster,
    refresh_kg_concepts,
)


# ── _kg_folder ────────────────────────────────────────────────────────────────

class TestKgFolder:
    def test_spaces_replaced_with_underscore(self):
        with patch("services.knowledge_graph_service.settings") as mock_settings:
            mock_settings.workspace_dir = "workspace"
            folder = _kg_folder("my kg name")
        assert folder.name == "kg_my_kg_name"

    def test_slashes_replaced_with_underscore(self):
        with patch("services.knowledge_graph_service.settings") as mock_settings:
            mock_settings.workspace_dir = "workspace"
            folder = _kg_folder("a/b")
        assert folder.name == "kg_a_b"

    def test_lowercased(self):
        with patch("services.knowledge_graph_service.settings") as mock_settings:
            mock_settings.workspace_dir = "workspace"
            folder = _kg_folder("MyKG")
        assert folder.name == "kg_mykg"


# ── _make_db_name ─────────────────────────────────────────────────────────────

class TestMakeDbName:
    def test_starts_with_kg_prefix(self):
        assert _make_db_name("測試知識庫").startswith("kg")

    def test_strips_non_alphanumeric_and_uses_ascii_only(self):
        name = _make_db_name("軟體架構")
        # 中文字元被 [^a-z0-9] 全部濾除，只剩 'kg' 前綴 + 8 碼 uuid 後綴
        assert name.startswith("kg")
        assert len(name) == len("kg") + 8

    def test_ascii_name_truncated_to_eight_chars(self):
        name = _make_db_name("ABCDEFGHIJKL")
        # safe = 前 8 碼 lowercase ascii, suffix = 8 碼 uuid hex
        assert name.startswith("kgabcdefgh")
        assert len(name) == len("kg") + 8 + 8

    def test_unique_across_calls(self):
        n1 = _make_db_name("重複名稱")
        n2 = _make_db_name("重複名稱")
        assert n1 != n2


# ── create_kg ─────────────────────────────────────────────────────────────────

class TestCreateKg:
    async def test_raises_if_name_exists(self, tmp_path):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_name.return_value = MagicMock()

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
            with pytest.raises(ValueError, match="已存在"):
                await create_kg(name="重複KG")

    async def test_creates_folders_and_kg_node(self, tmp_path):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_name.return_value = None
        created_kg = MagicMock(id=uuid4(), db_name="kgabc12345")
        mock_kg_repo.create.return_value = created_kg

        mock_driver = AsyncMock()
        mock_driver.execute_query = AsyncMock()

        with patch("services.knowledge_graph_service.settings") as mock_settings, \
             patch("services.knowledge_graph_service.get_driver", return_value=mock_driver), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.create_kg_database", new=AsyncMock()), \
             patch("services.knowledge_graph_service.add_watch_dir") as mock_watch:
            mock_settings.workspace_dir = str(tmp_path)
            result = await create_kg(name="新知識庫", description="desc")

        assert result is created_kg
        mock_kg_repo.create.assert_called_once()
        mock_watch.assert_called_once()
        folder = tmp_path / "kg_新知識庫"
        assert (folder / "_source").exists()
        assert (folder / "_text").exists()

    async def test_falls_back_to_main_db_when_enterprise_db_creation_fails(self, tmp_path):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_name.return_value = None
        created_kg = MagicMock(id=uuid4(), db_name="")
        mock_kg_repo.create.return_value = created_kg

        with patch("services.knowledge_graph_service.settings") as mock_settings, \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.create_kg_database",
                   new=AsyncMock(side_effect=RuntimeError("非 Enterprise 版"))), \
             patch("services.knowledge_graph_service.add_watch_dir"):
            mock_settings.workspace_dir = str(tmp_path)
            await create_kg(name="社群版KG")

        # db_name 應該以空字串（主資料庫模式）呼叫 kg_repo.create
        _, kwargs = mock_kg_repo.create.call_args
        assert kwargs["db_name"] == ""


# ── delete_kg ─────────────────────────────────────────────────────────────────

class TestDeleteKg:
    async def test_returns_false_when_kg_not_found(self):
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = None

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
            result = await delete_kg(uuid4())

        assert result is False

    async def test_drops_neo4j_database_when_db_name_set(self):
        kg = MagicMock(db_name="kgabc123", folder_path="/tmp/kg_x")
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg
        mock_kg_repo.delete.return_value = True
        mock_drop = AsyncMock()

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.drop_kg_database", new=mock_drop):
            result = await delete_kg(uuid4())

        mock_drop.assert_called_once_with("kgabc123")
        assert result is True

    async def test_delete_files_true_removes_folder(self, tmp_path):
        folder = tmp_path / "kg_to_delete"
        folder.mkdir()
        (folder / "note.txt").write_text("x")

        kg = MagicMock(db_name="", folder_path=str(folder))
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg
        mock_kg_repo.delete.return_value = True

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
            result = await delete_kg(uuid4(), delete_files=True)

        assert result is True
        assert not folder.exists()

    async def test_delete_files_false_keeps_folder(self, tmp_path):
        folder = tmp_path / "kg_keep"
        folder.mkdir()

        kg = MagicMock(db_name="", folder_path=str(folder))
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg
        mock_kg_repo.delete.return_value = True

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
            await delete_kg(uuid4(), delete_files=False)

        assert folder.exists()


# ── auto_cluster_kgs ──────────────────────────────────────────────────────────

class TestAutoClusterKgs:
    async def test_raises_when_below_min_docs(self, tmp_path):
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_orphan_documents.return_value = []

        with patch("services.knowledge_graph_service.settings") as mock_settings, \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.document_repo.DocumentRepository", return_value=mock_doc_repo):
            mock_settings.workspace_dir = str(tmp_path)
            with pytest.raises(ValueError, match="至少需要"):
                await auto_cluster_kgs(min_docs=2)

    async def test_clusters_staging_and_orphan_docs(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("關於 AI 的文章", encoding="utf-8")
        (staging / "b.txt").write_text("關於音樂的文章", encoding="utf-8")

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_orphan_documents.return_value = []

        async def fake_stream(prompt):
            for tok in ['[{"name": "AI群組", "description": "desc", "ids": ["d1"]}]']:
                yield tok

        mock_llm = MagicMock()
        mock_llm.stream = fake_stream

        with patch("services.knowledge_graph_service.settings") as mock_settings, \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.document_repo.DocumentRepository", return_value=mock_doc_repo), \
             patch("core.providers.factory.get_llm_provider", return_value=mock_llm):
            mock_settings.workspace_dir = str(tmp_path)
            result = await auto_cluster_kgs(min_docs=2)

        names = [c["name"] for c in result]
        assert "AI群組" in names
        # 未被 LLM 分配到任何群組的 d2 應歸入「其他」
        assert "其他" in names

    async def test_invalid_llm_json_raises(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")
        (staging / "b.txt").write_text("內容", encoding="utf-8")

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_orphan_documents.return_value = []

        async def fake_stream(prompt):
            for tok in ["不是 JSON 的回應"]:
                yield tok

        mock_llm = MagicMock()
        mock_llm.stream = fake_stream

        with patch("services.knowledge_graph_service.settings") as mock_settings, \
             patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.document_repo.DocumentRepository", return_value=mock_doc_repo), \
             patch("core.providers.factory.get_llm_provider", return_value=mock_llm):
            mock_settings.workspace_dir = str(tmp_path)
            with pytest.raises(ValueError, match="未回傳有效 JSON"):
                await auto_cluster_kgs(min_docs=2)


# ── confirm_auto_cluster ──────────────────────────────────────────────────────

class TestConfirmAutoCluster:
    # confirm_auto_cluster 內部有 `from repositories.knowledge_graph_repo import
    # KnowledgeGraphRepository` 的區域 import，會覆蓋模組層級的同名匯入，
    # 因此這裡必須 patch 定義它的模組，patch services.knowledge_graph_service
    # 裡的名稱不會生效。

    async def test_skips_cluster_with_no_name_or_files(self):
        clusters = [{"name": "", "files": [], "doc_ids": []}]
        with patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.knowledge_graph_repo.KnowledgeGraphRepository"):
            result = await confirm_auto_cluster(clusters)
        assert result == []

    async def test_creates_kg_and_assigns_staging_files(self):
        clusters = [{"name": "新群組", "description": "d", "files": ["a.txt"], "doc_ids": []}]
        created_kg = MagicMock(id=uuid4(), db_name="kgxyz")
        mock_kg_repo = AsyncMock()

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.knowledge_graph_repo.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.create_kg", new=AsyncMock(return_value=created_kg)), \
             patch("services.classify_service.assign_document_to_kg", new=AsyncMock()) as mock_assign:
            result = await confirm_auto_cluster(clusters)

        assert len(result) == 1
        assert result[0]["assigned"] == 1
        assert result[0]["errors"] == []
        mock_assign.assert_called_once_with("a.txt", created_kg.id)

    async def test_records_error_when_kg_name_conflict(self):
        clusters = [{"name": "衝突KG", "files": ["a.txt"], "doc_ids": []}]

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.knowledge_graph_repo.KnowledgeGraphRepository"), \
             patch("services.knowledge_graph_service.create_kg",
                   new=AsyncMock(side_effect=ValueError("KG 名稱已存在：衝突KG"))):
            result = await confirm_auto_cluster(clusters)

        assert result[0]["error"] == "KG 名稱已存在：衝突KG"

    async def test_missing_staging_file_recorded_as_error(self):
        clusters = [{"name": "新群組", "files": ["ghost.txt"], "doc_ids": []}]
        created_kg = MagicMock(id=uuid4(), db_name="kgxyz")

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.knowledge_graph_repo.KnowledgeGraphRepository"), \
             patch("services.knowledge_graph_service.create_kg", new=AsyncMock(return_value=created_kg)), \
             patch("services.classify_service.assign_document_to_kg",
                   new=AsyncMock(side_effect=FileNotFoundError())):
            result = await confirm_auto_cluster(clusters)

        assert result[0]["assigned"] == 0
        assert "找不到檔案" in result[0]["errors"][0]

    async def test_orphan_doc_ids_trigger_concept_refresh(self):
        doc_id = str(uuid4())
        clusters = [{"name": "新群組", "files": [], "doc_ids": [doc_id]}]
        created_kg = MagicMock(id=uuid4(), db_name="kgxyz")
        mock_kg_repo = AsyncMock()

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("repositories.knowledge_graph_repo.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.knowledge_graph_service.create_kg", new=AsyncMock(return_value=created_kg)), \
             patch("services.knowledge_graph_service.refresh_kg_concepts", new=AsyncMock()) as mock_refresh:
            result = await confirm_auto_cluster(clusters)

        assert result[0]["assigned"] == 1
        mock_kg_repo.add_document.assert_called_once()
        mock_refresh.assert_called_once_with(created_kg.id)


# ── refresh_kg_concepts ───────────────────────────────────────────────────────

class TestRefreshKgConcepts:
    async def test_uses_document_concepts_when_available(self):
        kg_id = uuid4()
        doc_id = str(uuid4())

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {
            doc_id: [{"name": "機器學習"}]
        }
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_documents.return_value = [{"id": doc_id}]

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [0.1, 0.2]

        with patch("services.knowledge_graph_service.get_driver"), \
             patch("services.knowledge_graph_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("core.providers.factory.get_embedding_provider", return_value=mock_embedding):
            await refresh_kg_concepts(kg_id)

        mock_concept_repo.get_or_create.assert_any_call("機器學習", "general", [0.1, 0.2])
        mock_concept_repo.sync_kg_effective.assert_called_once_with(kg_id)
        mock_kg_repo.refresh_counts.assert_called_once_with(kg_id)

    async def test_falls_back_to_entity_frequency_when_no_doc_concepts(self):
        kg_id = uuid4()

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_documents.return_value = []
        mock_kg_repo.get_db_name.return_value = ""

        mock_driver = AsyncMock()
        mock_record = {"name": "深度學習"}
        mock_result = MagicMock()
        mock_result.records = [mock_record]
        mock_driver.execute_query = AsyncMock(return_value=mock_result)

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [0.5]

        with patch("services.knowledge_graph_service.get_driver", return_value=mock_driver), \
             patch("services.knowledge_graph_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("core.providers.factory.get_embedding_provider", return_value=mock_embedding):
            await refresh_kg_concepts(kg_id)

        mock_concept_repo.get_or_create.assert_any_call("深度學習", "general", [0.5])

    async def test_text_param_adds_llm_extracted_concepts(self):
        kg_id = uuid4()

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {}
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_documents.return_value = []
        mock_kg_repo.get_db_name.return_value = ""

        mock_driver = AsyncMock()
        mock_result = MagicMock()
        mock_result.records = []
        mock_driver.execute_query = AsyncMock(return_value=mock_result)

        mock_embedding = MagicMock()
        mock_embedding.encode.return_value = [0.3]

        with patch("services.knowledge_graph_service.get_driver", return_value=mock_driver), \
             patch("services.knowledge_graph_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.knowledge_graph_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("core.providers.factory.get_embedding_provider", return_value=mock_embedding), \
             patch("services.concept_engine.extract_concepts", new=AsyncMock(return_value={"自訂概念"})):
            await refresh_kg_concepts(kg_id, text="一些文字")

        mock_concept_repo.get_or_create.assert_any_call("自訂概念", "general", [0.3])
