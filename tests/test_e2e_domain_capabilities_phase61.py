from __future__ import annotations

from dataclasses import replace

import pytest

from domains.registry import get_domain_profile
from scripts.run_dac_discovery_e2e import _resolve_feasibility_capability


def test_capability_resolver_uses_the_supplied_profile_object():
    base = get_domain_profile("sers_au_ag")

    unknown = replace(base, feasibility_adapter_id="missing_adapter")
    with pytest.raises(ValueError, match="Unknown feasibility adapter"):
        _resolve_feasibility_capability(unknown)

    cross_domain = replace(base, feasibility_adapter_id="dac_her")
    with pytest.raises(ValueError, match="mismatch"):
        _resolve_feasibility_capability(cross_domain)


def test_none_capability_still_skips_cleanly():
    profile = get_domain_profile("sers_au_ag")
    assert profile.feasibility_adapter_id is None
    assert _resolve_feasibility_capability(profile) is None


def test_dac_adapter_still_resolves():
    profile = get_domain_profile("dac_her")
    adapter = _resolve_feasibility_capability(profile)
    assert adapter is not None
    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == "dac_her"
