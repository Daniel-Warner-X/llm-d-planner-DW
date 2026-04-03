"""Unit tests for CostManager — validates ModelCatalog-backed implementation."""

from unittest.mock import MagicMock

import pytest


def _make_gpu(gpu_type: str, cost: float, aliases: list[str] | None = None) -> MagicMock:
    """Create a mock GPUType."""
    gpu = MagicMock()
    gpu.gpu_type = gpu_type
    gpu.cost_per_hour_usd = cost
    gpu.aliases = aliases or [gpu_type]
    return gpu


def _make_catalog(*gpus) -> MagicMock:
    """Create a mock ModelCatalog with the given GPU entries."""
    catalog = MagicMock()
    gpu_map = {}
    for gpu in gpus:
        gpu_map[gpu.gpu_type.upper()] = gpu
        for alias in gpu.aliases:
            gpu_map[alias.upper()] = gpu

    def get_gpu_type(name: str):
        return gpu_map.get(name.upper())

    catalog.get_gpu_type = get_gpu_type
    catalog.get_all_gpu_types.return_value = list(gpus)
    return catalog


@pytest.mark.unit
class TestCostManager:
    def test_get_cost_from_catalog(self):
        """Returns cost from catalog when no custom override."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.get_cost("H100") == 3.0

    def test_get_cost_scales_by_num_gpus(self):
        """Cost is multiplied by num_gpus."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.get_cost("H100", num_gpus=4) == 12.0

    def test_custom_cost_overrides_catalog(self):
        """Custom cost takes precedence over catalog."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(custom_costs={"H100": 1.5}, catalog=catalog)
        assert cm.get_cost("H100") == 1.5

    def test_get_cost_returns_none_for_unknown_gpu(self):
        """Returns None when GPU is not in catalog and not in custom costs."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.get_cost("V100") is None

    def test_get_all_costs_returns_catalog_costs(self):
        """get_all_costs returns all GPU types from catalog."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(
            _make_gpu("H100", 3.0),
            _make_gpu("L4", 0.8),
        )
        cm = CostManager(catalog=catalog)
        costs = cm.get_all_costs()
        assert costs == {"H100": 3.0, "L4": 0.8}

    def test_get_all_costs_custom_overrides_catalog(self):
        """Custom costs override catalog values in get_all_costs."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(custom_costs={"H100": 1.5}, catalog=catalog)
        costs = cm.get_all_costs()
        assert costs["H100"] == 1.5

    def test_has_cost_true_for_catalog_gpu(self):
        """has_cost returns True for GPU in catalog."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.has_cost("H100") is True

    def test_has_cost_false_for_unknown_gpu(self):
        """has_cost returns False for GPU not in catalog or custom costs."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.has_cost("V100") is False

    def test_custom_gpu_not_in_catalog_has_cost(self):
        """Custom cost for a GPU not in catalog is still accessible."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(custom_costs={"V100": 2.0}, catalog=catalog)
        assert cm.has_cost("V100") is True
        assert cm.get_cost("V100") == 2.0

    def test_is_using_custom_costs_false_by_default(self):
        """is_using_custom_costs is False when no custom costs provided."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(catalog=catalog)
        assert cm.is_using_custom_costs() is False

    def test_is_using_custom_costs_true_when_provided(self):
        """is_using_custom_costs is True when custom costs with non-None values."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        cm = CostManager(custom_costs={"H100": 1.5}, catalog=catalog)
        assert cm.is_using_custom_costs() is True

    def test_invalid_negative_cost_raises_value_error(self):
        """Negative custom cost raises ValueError."""
        from planner.gpu_recommender import CostManager

        catalog = _make_catalog(_make_gpu("H100", 3.0))
        with pytest.raises(ValueError):
            CostManager(custom_costs={"H100": -1.0}, catalog=catalog)

    def test_alias_resolution_via_catalog(self):
        """Alias lookup works when ModelCatalog resolves the alias."""
        from planner.gpu_recommender import CostManager

        gpu = _make_gpu("A100-40", 2.5, aliases=["NVIDIA-A100-40GB", "A100-40", "A100-40GB"])
        catalog = _make_catalog(gpu)
        cm = CostManager(catalog=catalog)
        # Catalog resolves alias to the canonical GPU
        assert cm.get_cost("A100-40GB") == 2.5
        assert cm.get_cost("NVIDIA-A100-40GB") == 2.5
