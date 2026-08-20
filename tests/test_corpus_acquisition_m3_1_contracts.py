from __future__ import annotations

from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryDiscovery, SupplementaryDiscoveryPolicy


def test_policy_is_conservative_by_default():
    policy = SupplementaryDiscoveryPolicy(policy_id="p")
    assert policy.auto_download_high_confidence_direct_files is True
    assert policy.allow_medium_confidence_direct_files is False


def test_discovery_contract_forbids_guessing_and_paywall_bypass():
    row = SupplementaryDiscovery(
        work_id="w",
        status="unresolved",
    )
    assert row.publisher_specific_url_guessing_performed is False
    assert row.paywall_bypass_attempted is False
