from __future__ import annotations

import pytest

from dac_her.domains.comparison_registry import (
    ComparisonAdapterUnavailableError,
    available_comparison_adapters,
    get_comparison_adapter,
)
from dac_her.domains.dac_her import DAC_HER_PROFILE
from domains.sers.profile import SERS_AU_AG_PROFILE


def test_alpha4b3b_only_sers_comparison_adapter_is_implemented():
    assert available_comparison_adapters() == ("sers_au_ag",)
    adapter = get_comparison_adapter(SERS_AU_AG_PROFILE)
    assert adapter.adapter_id == "sers_au_ag"
    assert adapter.domain_profile_id == "sers_au_ag"
    assert adapter.semantics_id == "sers_au_ag_comparison_v7_alpha4b3b321"


def test_alpha4b3b_unimplemented_domain_fails_closed():
    with pytest.raises(
        ComparisonAdapterUnavailableError,
        match="No comparison adapter",
    ):
        get_comparison_adapter(DAC_HER_PROFILE)
