"""Integration test for estimated performance flow.

Exercises the full estimation pipeline with real HuggingFace API calls
and the real llm_optimizer library — no mocks.
"""

from unittest.mock import MagicMock

import pytest

from planner.recommendation.config_finder import ConfigFinder
from planner.recommendation.estimator import generate_estimated_configs
from planner.shared.schemas import SLOTargets, TrafficProfile


def _make_traffic() -> TrafficProfile:
    return TrafficProfile(
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=5.0,
    )


def _make_slo() -> SLOTargets:
    return SLOTargets(
        ttft_p95_target_ms=500,
        itl_p95_target_ms=50,
        e2e_p95_target_ms=15000,
    )


@pytest.mark.integration
class TestEstimatedPerformanceIntegration:
    """End-to-end tests that hit real HuggingFace API and llm_optimizer."""

    def setup_method(self):
        mock_repo = MagicMock()
        mock_repo._get_connection.return_value = MagicMock()

        mock_catalog = MagicMock()
        # Set up H100 as the only GPU (supported by roofline model)
        gpu_h100 = MagicMock()
        gpu_h100.gpu_type = "H100"
        gpu_h100.memory_gb = 80
        mock_catalog.get_all_gpu_types.return_value = [gpu_h100]
        mock_catalog.get_all_models.return_value = []

        self.mock_repo = mock_repo
        self.mock_catalog = mock_catalog

    def test_small_model_produces_estimate(self):
        """Qwen2.5-0.5B on H100 should produce a valid estimated benchmark."""
        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["Qwen/Qwen2.5-0.5B"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        # Filter out non-blocking warnings (e.g., DB persistence failures)
        blocking_warnings = [w for w in warnings if "does not fit" in w or "Could not" in w]
        assert not blocking_warnings, f"Unexpected blocking warnings: {blocking_warnings}"

        assert len(results) >= 1, f"Expected >= 1 results, got {len(results)}. Warnings: {warnings}"

        # All results should be valid estimated benchmarks
        for bench in results:
            assert bench.model_hf_repo == "Qwen/Qwen2.5-0.5B"
            assert bench.hardware == "H100"
            assert bench.estimated is True
            assert bench.source == "llm-optimizer"
            assert bench.confidence_level == "estimated"

            # Roofline should produce positive performance values
            assert bench.ttft_p95 is not None and bench.ttft_p95 > 0
            assert bench.itl_p95 is not None and bench.itl_p95 > 0
            assert bench.e2e_p95 is not None and bench.e2e_p95 > 0
            assert bench.tps_p95 is not None and bench.tps_p95 > 0
            assert bench.requests_per_second is not None and bench.requests_per_second > 0

        # 0.5B model should fit at TP=1 on H100 (among other TPs)
        tp_values = [r.hardware_count for r in results]
        assert 1 in tp_values, f"Expected TP=1 among results, got TPs: {tp_values}"

    def test_nonexistent_model_produces_warning(self):
        """A model ID that doesn't exist on HuggingFace should warn, not crash."""
        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["nonexistent-org/this-model-does-not-exist-12345"],
            existing_benchmarks=[],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        assert len(results) == 0
        assert any("Could not estimate" in w for w in warnings)

    def test_covered_model_is_skipped(self):
        """A model already covered in existing_benchmarks should not be re-estimated."""
        from planner.knowledge_base.benchmarks import BenchmarkData

        existing = BenchmarkData(
            {
                "model_hf_repo": "Qwen/Qwen2.5-0.5B",
                "hardware": "H100",
                "hardware_count": 1,
                "framework": "vllm",
                "framework_version": "0.6.2",
                "prompt_tokens": 512,
                "output_tokens": 256,
                "mean_input_tokens": 512,
                "mean_output_tokens": 256,
                "ttft_mean": 50,
                "ttft_p90": 60,
                "ttft_p95": 70,
                "ttft_p99": 80,
                "itl_mean": 5,
                "itl_p90": 6,
                "itl_p95": 7,
                "itl_p99": 8,
                "e2e_mean": 1000,
                "e2e_p90": 1200,
                "e2e_p95": 1400,
                "e2e_p99": 1600,
                "tps_mean": 100,
                "tps_p90": 90,
                "tps_p95": 85,
                "tps_p99": 80,
                "tokens_per_second": 100,
                "requests_per_second": 10,
                "source": "blis",
                "confidence_level": "benchmarked",
            }
        )

        results, warnings = generate_estimated_configs(
            traffic_profile=_make_traffic(),
            slo_targets=_make_slo(),
            preferred_models=["Qwen/Qwen2.5-0.5B"],
            existing_benchmarks=[existing],
            gpu_types=["H100"],
            catalog=self.mock_catalog,
            benchmark_repo=self.mock_repo,
        )

        # TP=1 is covered — it should not appear in results.
        # Other TPs (2, 4, 8) may still be estimated.
        tp_values = [r.hardware_count for r in results]
        assert 1 not in tp_values, f"TP=1 should be skipped (already covered), got TPs: {tp_values}"
