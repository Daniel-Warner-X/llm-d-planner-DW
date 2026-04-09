"""Unit tests for BenchmarkRepository.save_benchmarks()."""

from unittest.mock import MagicMock, call, patch

import pytest

from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository


def _make_benchmark(
    model: str = "test/model",
    hardware: str = "H100",
    prompt_tokens: int = 512,
    output_tokens: int = 256,
) -> BenchmarkData:
    """Create a minimal BenchmarkData for testing."""
    return BenchmarkData(
        {
            "model_hf_repo": model,
            "hardware": hardware,
            "hardware_count": 1,
            "framework": "vllm",
            "framework_version": "0.6.2",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "mean_input_tokens": prompt_tokens,
            "mean_output_tokens": output_tokens,
            "ttft_mean": 100,
            "ttft_p90": 120,
            "ttft_p95": 130,
            "ttft_p99": 150,
            "itl_mean": 10,
            "itl_p90": 12,
            "itl_p95": 14,
            "itl_p99": 18,
            "e2e_mean": 3000,
            "e2e_p90": 3500,
            "e2e_p95": 4000,
            "e2e_p99": 5000,
            "tps_mean": 50.0,
            "tps_p90": 45.0,
            "tps_p95": 42.0,
            "tps_p99": 38.0,
            "tokens_per_second": 50.0,
            "requests_per_second": 10.0,
        }
    )


@pytest.fixture
def repo():
    """Create a BenchmarkRepository with mocked DB connection."""
    with patch.object(BenchmarkRepository, "_test_connection"):
        return BenchmarkRepository(database_url="postgresql://fake")


@pytest.mark.unit
class TestSaveBenchmarksRollback:
    """A6: save_benchmarks() rolls back on insert failure."""

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_rollback_called_on_insert_failure(self, mock_insert, repo):
        """Connection should be rolled back when insert_benchmarks raises."""
        mock_conn = MagicMock()
        with patch.object(repo, "_get_connection", return_value=mock_conn):
            mock_insert.side_effect = RuntimeError("DB write failed")

            with pytest.raises(RuntimeError, match="DB write failed"):
                repo.save_benchmarks([_make_benchmark()])

            mock_conn.rollback.assert_called_once()
            mock_conn.close.assert_called_once()

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_no_rollback_on_success(self, mock_insert, repo):
        """Connection should NOT be rolled back on successful insert."""
        mock_conn = MagicMock()
        with patch.object(repo, "_get_connection", return_value=mock_conn):
            repo.save_benchmarks([_make_benchmark()])

            mock_conn.rollback.assert_not_called()
            mock_conn.close.assert_called_once()

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_connection_closed_even_on_failure(self, mock_insert, repo):
        """Connection must always be closed, even after rollback."""
        mock_conn = MagicMock()
        with patch.object(repo, "_get_connection", return_value=mock_conn):
            mock_insert.side_effect = Exception("unexpected")

            with pytest.raises(Exception, match="unexpected"):
                repo.save_benchmarks([_make_benchmark()])

            mock_conn.close.assert_called_once()


@pytest.mark.unit
class TestSaveBenchmarksValidationBeforeConnection:
    """A5: Data preparation happens before DB connection is opened."""

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_to_dict_failure_does_not_open_connection(self, mock_insert, repo):
        """If to_dict() raises, _get_connection() should never be called."""
        bad_bench = MagicMock(spec=BenchmarkData)
        bad_bench.to_dict.side_effect = AttributeError("broken")

        with patch.object(repo, "_get_connection") as mock_get_conn:
            with pytest.raises(AttributeError, match="broken"):
                repo.save_benchmarks([bad_bench])

            mock_get_conn.assert_not_called()
            mock_insert.assert_not_called()

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_data_prepared_before_connection(self, mock_insert, repo):
        """Verify insert receives prepared dicts with prompt_tokens/output_tokens filled."""
        bench = _make_benchmark(prompt_tokens=1024, output_tokens=512)
        mock_conn = MagicMock()

        with patch.object(repo, "_get_connection", return_value=mock_conn):
            repo.save_benchmarks([bench], source="test-src", confidence_level="benchmarked")

        # insert_benchmarks should have been called with prepared dicts
        mock_insert.assert_called_once()
        args = mock_insert.call_args
        benchmark_dicts = args[0][1]  # second positional arg
        assert len(benchmark_dicts) == 1
        assert benchmark_dicts[0]["prompt_tokens"] == 1024
        assert benchmark_dicts[0]["output_tokens"] == 512
        assert args[1]["source"] == "test-src"
        assert args[1]["confidence_level"] == "benchmarked"

    @patch("planner.knowledge_base.benchmarks.insert_benchmarks")
    def test_setdefault_fills_missing_prompt_output_tokens(self, mock_insert, repo):
        """setdefault should fill prompt_tokens/output_tokens from mean_input/output_tokens."""
        bench = _make_benchmark()
        # Simulate a dict where prompt_tokens/output_tokens are missing
        original_to_dict = bench.to_dict

        def to_dict_without_tokens():
            d = original_to_dict()
            del d["prompt_tokens"]
            del d["output_tokens"]
            return d

        mock_conn = MagicMock()

        with (
            patch.object(bench, "to_dict", to_dict_without_tokens),
            patch.object(repo, "_get_connection", return_value=mock_conn),
        ):
            repo.save_benchmarks([bench])

        benchmark_dicts = mock_insert.call_args[0][1]
        # Should have been filled from mean_input_tokens / mean_output_tokens
        assert benchmark_dicts[0]["prompt_tokens"] == 512
        assert benchmark_dicts[0]["output_tokens"] == 256
