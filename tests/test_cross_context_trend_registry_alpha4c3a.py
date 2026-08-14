from __future__ import annotations

from dac_her.domains.cross_context_trend_registry import (
    available_cross_context_trend_adapters,
)


def test_alpha4c3a_registry_api_remains_phase_extensible():
    # alpha4c.3a established an empty registry skeleton. Later phases may
    # register domain adapters without changing the generic registry API.
    adapters = available_cross_context_trend_adapters()
    assert len(adapters) == len(set(adapters))
    assert all(value.strip() for value in adapters)
