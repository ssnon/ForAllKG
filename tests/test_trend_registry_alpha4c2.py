from __future__ import annotations

from dac_her.domains.registry import get_domain_profile
from domains.sers.trend_alpha4c21 import (
    SERS_AU_AG_TREND_ADAPTER as HISTORICAL_ALPHA4C21_ADAPTER,
)
from dac_her.domains.trend_registry import (
    available_trend_adapters,
    get_trend_adapter,
)


def test_alpha4c21_historical_adapter_remains_importable():
    assert (
        HISTORICAL_ALPHA4C21_ADAPTER.semantics_id
        == "sers_au_ag_trend_v2_alpha4c21"
    )


def test_active_sers_trend_adapter_advances_to_alpha4c211():
    profile = get_domain_profile("sers_au_ag")
    assert profile.trend_adapter_id == "sers_au_ag"
    assert "sers_au_ag" in available_trend_adapters()
    adapter = get_trend_adapter(profile)
    assert adapter.semantics_id == "sers_au_ag_trend_v5_alpha4c2121"
    assert adapter.required_inputs == frozenset({
        "canonical_graph",
        "measurement_result_identity",
        "method_context",
        "comparison_context",
    })
