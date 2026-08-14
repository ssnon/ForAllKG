from __future__ import annotations

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_registry import (
    available_trend_adapters,
    get_trend_adapter,
)


def test_sers_trend_adapter_is_activated_in_alpha4c2():
    profile = get_domain_profile("sers_au_ag")
    assert profile.trend_adapter_id == "sers_au_ag"
    assert "sers_au_ag" in available_trend_adapters()
    adapter = get_trend_adapter(profile)
    assert adapter.semantics_id == "sers_au_ag_trend_v1_alpha4c2"
    assert adapter.required_inputs == frozenset({
        "canonical_graph",
        "measurement_result_identity",
        "method_context",
        "comparison_context",
    })
