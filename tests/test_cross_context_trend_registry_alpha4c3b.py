from __future__ import annotations

from dac_her.domains.cross_context_trend_registry import (
    available_cross_context_trend_adapters,
    get_cross_context_trend_adapter,
)


def test_alpha4c3b_registers_sers_context_projection_adapter():
    assert available_cross_context_trend_adapters() == (
        "sers_au_ag",
    )
    adapter = get_cross_context_trend_adapter(
        "sers_au_ag"
    )
    assert adapter.adapter_id == "sers_au_ag"
    assert adapter.domain_profile_id == "sers_au_ag"
    assert adapter.context_semantics_id == (
        "sers_au_ag_trend_context_v1_alpha4c3b"
    )
    assert "paper_local_trend_results" in (
        adapter.required_inputs
    )
    assert "comparison_context" in adapter.required_inputs
    assert "method_context" in adapter.required_inputs
