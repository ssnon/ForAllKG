from __future__ import annotations

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_precision_registry import (
    available_trend_precision_adapters,
    get_trend_precision_adapter,
)


def test_active_sers_precision_adapter_is_alpha4c211():
    profile = get_domain_profile("sers_au_ag")
    assert available_trend_precision_adapters() == ("sers_au_ag",)
    adapter = get_trend_precision_adapter(profile)
    assert adapter.trend_semantics_id == "sers_au_ag_trend_v5_alpha4c2121"
    assert (
        adapter.precision_semantics_id
        == "sers_au_ag_trend_precision_v5_alpha4c21211"
    )
