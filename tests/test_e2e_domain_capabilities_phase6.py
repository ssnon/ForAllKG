from __future__ import annotations

from dataclasses import replace

import pytest

from domains.registry import get_domain_profile
from scripts.run_dac_discovery_e2e import _resolve_feasibility_capability


def test_dac_her_declares_and_resolves_feasibility_capability():
    profile = get_domain_profile("dac_her")
    adapter = _resolve_feasibility_capability(profile)
    assert adapter is not None
    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == profile.profile_id


def test_sers_without_feasibility_adapter_is_valid_core_pipeline_capability_state():
    profile = get_domain_profile("sers_au_ag")
    assert profile.feasibility_adapter_id is None
    assert _resolve_feasibility_capability(profile) is None


def test_explicit_but_unknown_feasibility_adapter_remains_fail_closed():
    profile = replace(
        get_domain_profile("sers_au_ag"),
        feasibility_adapter_id="missing_adapter",
    )
    with pytest.raises(ValueError, match="Unknown feasibility adapter"):
        _resolve_feasibility_capability(profile)


def test_explicit_cross_domain_adapter_is_not_silently_reused():
    profile = replace(
        get_domain_profile("sers_au_ag"),
        feasibility_adapter_id="dac_her",
    )
    with pytest.raises(ValueError, match="mismatch"):
        _resolve_feasibility_capability(profile)
