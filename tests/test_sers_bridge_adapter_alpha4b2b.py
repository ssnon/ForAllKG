import pytest

from dac_her.domains.bridge_registry import (
    available_bridge_adapters,
    get_bridge_adapter,
)


def test_alpha4b2b_sers_bridge_adapter_is_registered():
    assert available_bridge_adapters() == ('dac_her', 'sers_au_ag')
    adapter = get_bridge_adapter('sers_au_ag')
    assert adapter.adapter_id == 'sers_au_ag'
    assert adapter.domain_profile_id == 'sers_au_ag'
    assert adapter.prompt_version.startswith('sers-au-ag-bridge-v1-')
    assert adapter.policy_version.startswith('sers-au-ag-bridge-policy-v1-')
    assert 'SERS' in adapter.system_prompt
    assert 'hydrogen evolution' not in adapter.system_prompt.casefold()


def test_alpha4b2b_unimplemented_broad_catalysis_remains_fail_closed():
    with pytest.raises(ValueError, match='no Bridge adapter'):
        get_bridge_adapter('catalysis_mechanism')


def test_alpha4b2b_sers_fingerprint_files_are_domain_owned():
    adapter = get_bridge_adapter('sers_au_ag')
    extraction_names = {
        __import__('pathlib').Path(path).name
        for path in adapter.implementation_files.extraction
    }
    policy_names = {
        __import__('pathlib').Path(path).name
        for path in adapter.implementation_files.policy
    }
    assert 'sers_bridge_prompts.py' in extraction_names
    assert 'sers_bridge_recovery_prompts.py' in extraction_names
    assert 'sers_bridge_signatures.py' in extraction_names
    assert 'sers_bridge_policy.py' in policy_names
    assert 'bridge_policy_runtime.py' in policy_names
