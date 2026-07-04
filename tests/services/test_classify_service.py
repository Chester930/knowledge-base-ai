from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.classify_service import (
    classify_document,
    assign_document_to_kg,
    classify_all,
)


def _kg_candidate_settings(tmp_path):
    """回傳一個已 patch 好 settings.workspace_dir 的 context helper。"""
    return patch("services.classify_service.settings", workspace_dir=str(tmp_path))


def _close_instead_of_schedule(coro):
    """
    取代 asyncio.create_task：只關閉背景協程、不真正排程執行，
    避免 assign_document_to_kg 的 _auto_svo() 產生 'coroutine was never awaited' 警告。
    """
    coro.close()
    return MagicMock()


# ── classify_document ─────────────────────────────────────────────────────────

class TestClassifyDocument:
    async def test_missing_staging_file_raises(self, tmp_path):
        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"):
            with pytest.raises(FileNotFoundError):
                await classify_document("ghost.txt")

    async def test_no_matching_kg_returns_unmatched(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        mock_concept_repo = AsyncMock()

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.classify_service.KnowledgeGraphRepository"), \
             patch("services.concept_engine.build_query_concepts", new=AsyncMock(return_value={})), \
             patch("services.concept_engine.route_kgs", new=AsyncMock(return_value={})):
            result = await classify_document("a.txt")

        assert result.status == "unmatched"
        assert result.candidates == []

    async def test_low_score_kg_excluded_from_candidates(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        kg_id = uuid4()
        mock_concept_repo = AsyncMock()
        mock_kg_repo = AsyncMock()

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.concept_engine.build_query_concepts", new=AsyncMock(return_value={})), \
             patch("services.concept_engine.route_kgs", new=AsyncMock(return_value={kg_id: {"x": 1}})), \
             patch("services.concept_engine.compute_match_score", return_value=(0.01, [])):
            result = await classify_document("a.txt")

        assert result.status == "unmatched"
        mock_kg_repo.get_by_id.assert_not_called()

    async def test_pending_when_score_below_auto_threshold(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        kg_id = uuid4()
        mock_concept_repo = AsyncMock()
        mock_kg_repo = AsyncMock()
        matched_kg = MagicMock()
        matched_kg.name = "軟體架構"
        mock_kg_repo.get_by_id.return_value = matched_kg

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.concept_engine.build_query_concepts", new=AsyncMock(return_value={})), \
             patch("services.concept_engine.route_kgs", new=AsyncMock(return_value={kg_id: {"x": 1}})), \
             patch("services.concept_engine.compute_match_score", return_value=(0.15, ["a"])):
            result = await classify_document("a.txt", threshold=0.30, auto_assign=True)

        assert result.status == "pending"
        assert result.auto_assigned is False

    async def test_auto_assigned_when_score_above_threshold(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        kg_id = uuid4()
        mock_concept_repo = AsyncMock()
        mock_kg_repo = AsyncMock()
        matched_kg = MagicMock()
        matched_kg.name = "軟體架構"
        mock_kg_repo.get_by_id.return_value = matched_kg

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.ConceptRepository", return_value=mock_concept_repo), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.concept_engine.build_query_concepts", new=AsyncMock(return_value={})), \
             patch("services.concept_engine.route_kgs", new=AsyncMock(return_value={kg_id: {"x": 1}})), \
             patch("services.concept_engine.compute_match_score", return_value=(0.9, ["a"])), \
             patch("services.classify_service.assign_document_to_kg", new=AsyncMock()) as mock_assign:
            result = await classify_document("a.txt", threshold=0.30, auto_assign=True)

        assert result.status == "assigned"
        assert result.auto_assigned is True
        mock_assign.assert_called_once_with("a.txt", kg_id)


# ── assign_document_to_kg ──────────────────────────────────────────────────────

class TestAssignDocumentToKg:
    async def test_missing_staging_file_raises(self, tmp_path):
        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"):
            with pytest.raises(FileNotFoundError):
                await assign_document_to_kg("ghost.txt", uuid4())

    async def test_kg_not_found_raises(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = None

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo):
            with pytest.raises(ValueError, match="KG 不存在"):
                await assign_document_to_kg("a.txt", uuid4())

    async def test_successful_assignment_moves_file_and_creates_document(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("文件內容", encoding="utf-8")

        kg_folder = tmp_path / "kg_test"
        kg = MagicMock(folder_path=str(kg_folder), db_name="")
        kg.name = "測試KG"
        kg_id = uuid4()
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        doc = MagicMock(id=uuid4())
        mock_doc_repo = AsyncMock()
        mock_doc_repo.create.return_value = doc

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.classify_service.DocumentRepository", return_value=mock_doc_repo), \
             patch("services.concept_engine.extract_and_init_document_concepts", new=AsyncMock()), \
             patch("services.knowledge_graph_service.refresh_kg_concepts", new=AsyncMock()), \
             patch("services.classify_service.asyncio.create_task", side_effect=_close_instead_of_schedule):
            await assign_document_to_kg("a.txt", kg_id)

        assert not (staging / "a.txt").exists()
        assert (kg_folder / "_text" / "a.txt").exists()
        mock_kg_repo.add_document.assert_called_once_with(kg_id, doc.id)

    async def test_duplicate_filename_at_destination_gets_counter_suffix(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("新內容", encoding="utf-8")

        kg_folder = tmp_path / "kg_test"
        text_dir = kg_folder / "_text"
        text_dir.mkdir(parents=True)
        (text_dir / "a.txt").write_text("已存在", encoding="utf-8")

        kg = MagicMock(folder_path=str(kg_folder), db_name="")
        kg.name = "測試KG"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        doc = MagicMock(id=uuid4())
        mock_doc_repo = AsyncMock()
        mock_doc_repo.create.return_value = doc

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.classify_service.DocumentRepository", return_value=mock_doc_repo), \
             patch("services.concept_engine.extract_and_init_document_concepts", new=AsyncMock()), \
             patch("services.knowledge_graph_service.refresh_kg_concepts", new=AsyncMock()), \
             patch("services.classify_service.asyncio.create_task", side_effect=_close_instead_of_schedule):
            await assign_document_to_kg("a.txt", uuid4())

        assert (text_dir / "a.txt").read_text(encoding="utf-8") == "已存在"
        assert (text_dir / "a_1.txt").exists()

    async def test_failure_restores_file_to_staging_and_reraises(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容", encoding="utf-8")

        kg_folder = tmp_path / "kg_test"
        kg = MagicMock(folder_path=str(kg_folder), db_name="")
        kg.name = "測試KG"
        mock_kg_repo = AsyncMock()
        mock_kg_repo.get_by_id.return_value = kg

        mock_doc_repo = AsyncMock()
        mock_doc_repo.create.side_effect = RuntimeError("DB 掛掉了")

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.get_driver"), \
             patch("services.classify_service.KnowledgeGraphRepository", return_value=mock_kg_repo), \
             patch("services.classify_service.DocumentRepository", return_value=mock_doc_repo):
            with pytest.raises(RuntimeError, match="DB 掛掉了"):
                await assign_document_to_kg("a.txt", uuid4())

        assert (staging / "a.txt").exists()
        assert not (kg_folder / "_text" / "a.txt").exists()


# ── classify_all ──────────────────────────────────────────────────────────────

class TestClassifyAll:
    async def test_no_staging_files_returns_empty_list(self, tmp_path):
        with _kg_candidate_settings(tmp_path):
            result = await classify_all()
        assert result == []

    async def test_batch_classifies_all_files(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "a.txt").write_text("內容A", encoding="utf-8")
        (staging / "b.txt").write_text("內容B", encoding="utf-8")

        async def fake_classify(filename, **kwargs):
            return MagicMock(status="assigned", txt_filename=filename)

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.classify_document", new=fake_classify):
            result = await classify_all()

        assert len(result) == 2

    async def test_per_file_error_captured_not_raised(self, tmp_path):
        staging = tmp_path / "_staging"
        staging.mkdir()
        (staging / "bad.txt").write_text("內容", encoding="utf-8")

        with _kg_candidate_settings(tmp_path), \
             patch("services.classify_service.classify_document",
                   new=AsyncMock(side_effect=RuntimeError("分類失敗"))):
            result = await classify_all()

        assert len(result) == 1
        assert result[0].status == "error"
        assert result[0].txt_filename == "bad.txt"
