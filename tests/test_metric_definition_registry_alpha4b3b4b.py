from __future__ import annotations

import pytest

from dac_her.domains.metric_definition_registry import (
    MetricDefinitionAdapterUnavailableError,
    available_metric_definition_adapters,
    get_metric_definition_adapter,
)
from dac_her.domains.registry import get_domain_profile


def test_only_sers_metric_definition_adapter_is_registered():
    assert available_metric_definition_adapters() == ("sers_au_ag",)


def test_sers_profile_resolves_metric_definition_adapter():
    profile = get_domain_profile("sers_au_ag")
    adapter = get_metric_definition_adapter(profile)
    assert adapter.adapter_id == "sers_au_ag"
    assert adapter.domain_profile_id == "sers_au_ag"
    assert adapter.semantics_id == "sers_au_ag_metric_definition_v3_alpha4c4c1"


def test_unimplemented_domain_fails_closed():
    profile = get_domain_profile("dac_her")
    with pytest.raises(MetricDefinitionAdapterUnavailableError):
        get_metric_definition_adapter(profile)
