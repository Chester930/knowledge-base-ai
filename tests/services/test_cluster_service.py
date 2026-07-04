from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from services.cluster_service import (
    _cosine,
    _connected_components,
    _intra_cluster_avg,
    cluster_staging_files,
)


# ── _cosine ─────────────────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── _connected_components ───────────────────────────────────────────────────

class TestConnectedComponents:
    def test_simple_grouping(self):
        nodes = ["A", "B", "C", "D"]
        # A-B 有關聯，C-D 有關聯
        sim_matrix = {
            ("A", "B"): 0.8,
            ("C", "D"): 0.7,
        }
        # 標準化 key 為 min, max
        matrix = {}
        for (u, v), val in sim_matrix.items():
            matrix[(min(u, v), max(u, v))] = val

        components = _connected_components(nodes, matrix, 0.5)
        # 應該分出兩個 components: ["A", "B"] 與 ["C", "D"]
        assert len(components) == 2
        groups = [set(c) for c in components]
        assert {"A", "B"} in groups
        assert {"C", "D"} in groups

    def test_all_disconnected(self):
        nodes = ["A", "B", "C"]
        components = _connected_components(nodes, {}, 0.5)
        assert len(components) == 3

    def test_large_chain_does_not_hit_recursion_limit(self):
        """
        迭代式 DFS 不應受 Python 遞迴深度限制影響：建立一條長度超過
        sys.getrecursionlimit() 的鏈狀關聯圖，舊版遞迴實作在此規模下會拋出
        RecursionError，改為顯式堆疊後應能正常處理並回傳單一 component。
        """
        import sys
        n = sys.getrecursionlimit() + 500
        # 補零到固定寬度，避免字典序排序造成 min/max 與數字順序不一致
        nodes = [f"doc_{i:06d}" for i in range(n)]
        # 鏈狀關聯：doc_0-doc_1-doc_2-...-doc_{n-1}，形成單一 connected component
        sim_matrix = {
            (min(nodes[i], nodes[i + 1]), max(nodes[i], nodes[i + 1])): 1.0
            for i in range(n - 1)
        }

        components = _connected_components(nodes, sim_matrix, 0.5)

        assert len(components) == 1
        assert len(components[0]) == n


# ── _intra_cluster_avg ───────────────────────────────────────────────────────

class TestIntraClusterAvg:
    def test_single_member_returns_one(self):
        assert _intra_cluster_avg(["A"], {}) == 1.0

    def test_pair_average(self):
        sim = {("A", "B"): 0.6}
        assert _intra_cluster_avg(["A", "B"], sim) == pytest.approx(0.6)

    def test_triplet_average(self):
        sim = {
            ("A", "B"): 0.6,
            ("A", "C"): 0.8,
            ("B", "C"): 0.4,
        }
        # avg = (0.6 + 0.8 + 0.4) / 3 = 0.6
        assert _intra_cluster_avg(["A", "B", "C"], sim) == pytest.approx(0.6)


# ── cluster_staging_files ────────────────────────────────────────────────────

class TestClusterStagingFiles:
    @pytest.mark.anyio
    async def test_empty_staging_returns_empty(self):
        with patch("services.cluster_service.Path.glob", return_value=[]), \
             patch("services.cluster_service.settings") as mock_settings:
            mock_settings.workspace_dir = "mock_workspace"
            res = await cluster_staging_files()
            assert res == []

    @pytest.mark.anyio
    async def test_staging_clustering_flow(self):
        # 模擬暫存區有 3 個 unmatched 文件
        mock_files = [Path("file1.txt"), Path("file2.txt"), Path("file3.txt")]
        
        # 模擬 classify_document 回傳 unmatched
        mock_classify_result = MagicMock()
        mock_classify_result.status = "unmatched"
        mock_classify_result.score = 0.1

        # 模擬 _get_doc_embedding：file1 與 file2 高度相似，file3 孤立
        async def mock_embedding(path):
            if "file1" in path.name:
                return [1.0, 0.0], ["AI", "機器學習"]
            elif "file2" in path.name:
                return [0.9, 0.1], ["AI", "深度學習"]
            else:
                return [0.0, 1.0], ["音樂", "搖滾"]

        # 模擬 LLM 建議名稱
        mock_suggest = AsyncMock(return_value=("AI技術", "關於人工智慧的研究"))

        with patch("services.cluster_service.Path.glob", return_value=mock_files), \
             patch("services.cluster_service.Path.mkdir"), \
             patch("services.cluster_service.settings") as mock_settings, \
             patch("services.classify_service.classify_document", new_callable=AsyncMock, return_value=mock_classify_result), \
             patch("services.cluster_service._get_doc_embedding", new=mock_embedding), \
             patch("services.cluster_service._suggest_kg_name", new=mock_suggest):
            
            mock_settings.workspace_dir = "mock_workspace"
            
            suggestions = await cluster_staging_files()
            
            # 應該有 1 個推薦群組（包含 file1.txt 和 file2.txt，file3.txt 相似度不夠被剔除）
            assert len(suggestions) == 1
            s = suggestions[0]
            assert s["suggested_name"] == "AI技術"
            assert s["suggested_description"] == "關於人工智慧的研究"
            assert set(s["files"]) == {"file1.txt", "file2.txt"}
            assert s["intra_similarity"] > 0.8
