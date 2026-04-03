"""Unit tests for GPU type normalization (gpu_normalizer.py).

Tests cover existing alias/expansion behavior and new fuzzy resolution fallback.
"""

import pytest


@pytest.mark.unit
class TestNormalizeGpuTypesExisting:
    """Test existing normalization behavior (no regression)."""

    def setup_method(self):
        """Reset catalog singleton before each test."""
        import planner.shared.utils.gpu_normalizer as mod

        mod._catalog_instance = None

    def test_empty_input(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types([]) == []

    def test_exact_canonical_name(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["H100"]) == ["H100"]

    def test_case_insensitive(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["h100"]) == ["H100"]

    def test_alias_match(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-H100"]) == ["H100"]

    def test_a100_expansion(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        result = normalize_gpu_types(["A100"])
        assert result == ["A100-40", "A100-80"]

    def test_deduplication(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        result = normalize_gpu_types(["H100", "h100", "NVIDIA-H100"])
        assert result == ["H100"]

    def test_any_gpu_skipped(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["any gpu"]) == []

    def test_unknown_gpu_skipped(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["TOTALLY_FAKE_GPU"]) == []


@pytest.mark.unit
class TestFuzzyResolution:
    """Test new fuzzy resolution fallback for pattern-based GPU names."""

    def setup_method(self):
        """Reset catalog singleton before each test."""
        import planner.shared.utils.gpu_normalizer as mod

        mod._catalog_instance = None

    def test_nvidia_h100_sxm5_80gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-H100-SXM5-80GB"]) == ["H100"]

    def test_nvidia_a100_sxm4_80gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-A100-SXM4-80GB"]) == ["A100-80"]

    def test_nvidia_a100_sxm4_40gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-A100-SXM4-40GB"]) == ["A100-40"]

    def test_nvidia_a100_pcie_80gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-A100-PCIe-80GB"]) == ["A100-80"]

    def test_nvidia_a100_pcie_40gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-A100-40GB-PCIe"]) == ["A100-40"]

    def test_nvidia_a100_no_disambig(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        # A100 with no memory hint should return both variants
        result = normalize_gpu_types(["NVIDIA-A100"])
        assert result == ["A100-40", "A100-80"]

    def test_tesla_t4(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        # T4 is not in the model catalog, so it should be skipped
        assert normalize_gpu_types(["Tesla-T4"]) == []

    def test_nvidia_l40_48gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        result = normalize_gpu_types(["NVIDIA-L40-48GB"])
        assert result == ["L40"]
        # Ensure it's not matched as L4
        assert "L4" not in result

    def test_nvidia_b200_192gb(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["NVIDIA-B200-192GB"]) == ["B200"]

    def test_lowercase_k8s_label(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        assert normalize_gpu_types(["nvidia-h100-sxm5-80gb"]) == ["H100"]

    def test_mixed_known_and_fuzzy(self):
        from planner.shared.utils.gpu_normalizer import normalize_gpu_types

        result = normalize_gpu_types(["H100", "NVIDIA-A100-SXM4-80GB"])
        assert result == ["A100-80", "H100"]
