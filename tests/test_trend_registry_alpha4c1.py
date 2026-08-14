from __future__ import annotations

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_registry import available_trend_adapters, get_trend_adapter
from dac_her.trend_domain import TREND_EVIDENCE_CONTRACT_SEMANTICS_ID


def test_alpha4c1_contract_remains_frozen_after_domain_activation():
    assert TREND_EVIDENCE_CONTRACT_SEMANTICS_ID == "trend_evidence_contract_v1_alpha4c1"
    profile = get_domain_profile("sers_au_ag")
    assert profile.trend_adapter_id == "sers_au_ag"
    assert "sers_au_ag" in available_trend_adapters()
    assert get_trend_adapter(profile).domain_profile_id == "sers_au_ag"
