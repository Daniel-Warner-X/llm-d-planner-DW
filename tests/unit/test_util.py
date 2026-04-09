"""Unit tests for ui/util.py Scenario dataclass."""

import sys
from pathlib import Path

# ui/ is not a package; add it to sys.path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ui"))

import pytest


@pytest.mark.unit
def test_scenario_has_no_model_config():
    """model_config and text_config were dead state after PR #156 and must be removed."""
    from util import Scenario

    s = Scenario()
    assert not hasattr(s, "model_config"), "model_config must be removed from Scenario"
    assert not hasattr(s, "text_config"), "text_config must be removed from Scenario"


@pytest.mark.unit
def test_scenario_can_show_mem_util_chart_is_gone():
    """can_show_mem_util_chart() was dead code referencing model_config; must be removed."""
    from util import Scenario

    assert not hasattr(Scenario, "can_show_mem_util_chart"), (
        "can_show_mem_util_chart must be removed — it was dead code"
    )


@pytest.mark.unit
def test_scenario_reset_does_not_reference_model_config():
    """reset() must not reference the removed model_config field."""
    from util import Scenario

    s = Scenario()
    s.model_name = "some/model"
    s.reset()
    assert s.model_name == "Qwen/Qwen2.5-7B-Instruct"
