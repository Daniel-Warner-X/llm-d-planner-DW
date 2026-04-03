"""Unit tests for cluster GPU detection."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestGPUProductMap:
    """Validate GPU_PRODUCT_MAP values are known to ModelCatalog."""

    def test_all_map_values_are_in_model_catalog(self):
        from planner.cluster.gpu_detector import GPU_PRODUCT_MAP
        from planner.knowledge_base.model_catalog import ModelCatalog

        catalog = ModelCatalog()
        for label, canonical in GPU_PRODUCT_MAP.items():
            assert (
                catalog.get_gpu_type(canonical) is not None
            ), f"GPU_PRODUCT_MAP['{label}'] = '{canonical}' is not in ModelCatalog"

    def test_map_covers_all_nvidia_catalog_gpu_types(self):
        """Regression: Every NVIDIA GPU in catalog should be in GPU_PRODUCT_MAP."""
        from planner.cluster.gpu_detector import GPU_PRODUCT_MAP
        from planner.knowledge_base.model_catalog import ModelCatalog

        catalog = ModelCatalog()
        # Get all GPU types from catalog
        all_gpu_types = catalog.get_all_gpu_types()
        catalog_gpu_types = {gpu.gpu_type for gpu in all_gpu_types}

        # Remove MI300X (uses amd.com/gpu label, not covered by this map)
        nvidia_gpu_types = catalog_gpu_types - {"MI300X"}

        # Get unique GPU types in the map
        map_gpu_types = set(GPU_PRODUCT_MAP.values())

        # Every NVIDIA GPU should be represented in the map
        missing = nvidia_gpu_types - map_gpu_types
        assert not missing, f"GPUs in catalog but not in GPU_PRODUCT_MAP: {missing}"


def _make_node(name: str, gpu_label: str | None = None) -> MagicMock:
    """Helper: create a mock K8s V1Node with optional GPU label."""
    node = MagicMock()
    node.metadata.name = name
    labels = {}
    if gpu_label:
        labels["nvidia.com/gpu.product"] = gpu_label
    node.metadata.labels = labels
    return node


@pytest.mark.unit
class TestDetectClusterGPUs:
    """Test detect_cluster_gpus() with mocked kubernetes client."""

    def setup_method(self):
        from planner.cluster.gpu_detector import reset_gpu_cache

        reset_gpu_cache()

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", False)
    def test_returns_empty_when_kubernetes_not_installed(self):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch.dict(os.environ, {"PLANNER_DETECT_CLUSTER_GPUS": "false"})
    def test_returns_empty_when_disabled_via_env(self):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_single_gpu_type(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "NVIDIA-H100-80GB-HBM3")]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_multiple_gpu_types(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-A100-SXM4-80GB"),
            _make_node("node2", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node3", "NVIDIA-L4"),
        ]
        result = detect_cluster_gpus()
        assert result == ["A100-80", "H100", "L4"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_deduplicates_same_gpu_on_multiple_nodes(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node2", "NVIDIA-H100-SXM5-80GB"),
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_skips_nodes_without_gpu_labels(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("cpu-node1"),
            _make_node("gpu-node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("cpu-node2"),
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_skips_unknown_gpu_labels(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-H100-80GB-HBM3"),
            _make_node("node2", "Tesla-V100-SXM2-32GB"),  # Unknown
        ]
        result = detect_cluster_gpus()
        assert result == ["H100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_returns_empty_when_no_nodes(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = []
        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_returns_empty_when_no_gpu_labels_on_any_node(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("cpu1"), _make_node("cpu2")]
        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config", side_effect=Exception("no config"))
    def test_returns_empty_on_config_error(self, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes", side_effect=Exception("API error"))
    def test_returns_empty_on_api_error(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        result = detect_cluster_gpus()
        assert result == []

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_case_insensitive_label_matching(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-l4")]
        result = detect_cluster_gpus()
        assert result == ["L4"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_a100_40gb_variants(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [
            _make_node("node1", "NVIDIA-A100-SXM4-40GB"),
            _make_node("node2", "NVIDIA-A100-40GB-PCIe"),
        ]
        result = detect_cluster_gpus()
        assert result == ["A100-40"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_cached_result_avoids_repeated_api_calls(self, mock_list, mock_config):
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "NVIDIA-L4")]
        detect_cluster_gpus()
        detect_cluster_gpus()
        # Second call should use cache, not call list_nodes again
        assert mock_list.call_count == 1

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_a10g(self, mock_list, mock_config):
        """Detect A10G GPU from K8s node label."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-a10g")]
        result = detect_cluster_gpus()
        assert result == ["A10G"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_l40(self, mock_list, mock_config):
        """Detect L40 GPU from K8s node label."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-l40")]
        result = detect_cluster_gpus()
        assert result == ["L40"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_l20(self, mock_list, mock_config):
        """Detect L20 GPU from K8s node label."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-l20")]
        result = detect_cluster_gpus()
        assert result == ["L20"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_b100(self, mock_list, mock_config):
        """Detect B100 GPU from K8s node label."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-b100")]
        result = detect_cluster_gpus()
        assert result == ["B100"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_h200_141gb(self, mock_list, mock_config):
        """Detect H200 GPU from K8s node label (nvidia-h200-141gb)."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-h200-141gb")]
        result = detect_cluster_gpus()
        assert result == ["H200"]

    @patch("planner.cluster.gpu_detector._HAS_KUBERNETES", True)
    @patch("planner.cluster.gpu_detector._load_k8s_config")
    @patch("planner.cluster.gpu_detector._list_nodes")
    def test_detects_h200_141gb_hbm3(self, mock_list, mock_config):
        """Detect H200 GPU from K8s node label (nvidia-h200-141gb-hbm3)."""
        from planner.cluster.gpu_detector import detect_cluster_gpus

        mock_list.return_value = [_make_node("node1", "nvidia-h200-141gb-hbm3")]
        result = detect_cluster_gpus()
        assert result == ["H200"]
