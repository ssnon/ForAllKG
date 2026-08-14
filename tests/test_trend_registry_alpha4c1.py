from __future__ import annotations

import pytest

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_registry import (
    TrendAdapterUnavailableError,
    available_trend_adapters,
    get_trend_adapter,
)


def test_alpha4c1_does_not_activate_sers_extraction_policy():
    profile = get_domain_profile("sers_au_ag")
    assert profile.trend_adapter_id is None
    assert available_trend_adapters() == ()
    with pytest.raises(TrendAdapterUnavailableError):
        get_trend_adapter(profile)
