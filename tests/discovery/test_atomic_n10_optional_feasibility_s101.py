from dataclasses import replace

import pytest

from domains.feasibility_registry import resolve_optional_feasibility_adapter
from domains.registry import get_domain_profile


def test_optional_feasibility_allows_domain_without_declared_adapter():
    profile = get_domain_profile("sers_au_ag")
    assert profile.feasibility_adapter_id is None
    assert resolve_optional_feasibility_adapter(profile) is None


def test_optional_feasibility_preserves_strict_declared_adapter_resolution():
    profile = get_domain_profile("dac_her")
    adapter = resolve_optional_feasibility_adapter(profile)
    assert adapter is not None
    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == "dac_her"


def test_optional_feasibility_still_fails_unknown_declared_adapter():
    base = get_domain_profile("sers_au_ag")
    unknown = replace(base, feasibility_adapter_id="missing_adapter")
    with pytest.raises(ValueError, match="Unknown feasibility adapter"):
        resolve_optional_feasibility_adapter(unknown)


def test_optional_feasibility_still_fails_cross_domain_adapter():
    base = get_domain_profile("sers_au_ag")
    cross_domain = replace(base, feasibility_adapter_id="dac_her")
    with pytest.raises(ValueError, match="mismatch"):
        resolve_optional_feasibility_adapter(cross_domain)
