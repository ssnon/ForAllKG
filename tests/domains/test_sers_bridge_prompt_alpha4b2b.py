from domains.sers.bridge_prompts import SERS_BRIDGE_SYSTEM_PROMPT
from domains.sers.bridge_recovery_prompts import SERS_BRIDGE_RECOVERY_SYSTEM_PROMPT


def test_alpha4b2b_prompt_is_sers_specific_and_zero_quota():
    prompt = SERS_BRIDGE_SYSTEM_PROMPT.casefold()
    assert 'surface-enhanced raman' in prompt
    assert 'returning zero concepts is valid' in prompt
    assert 'enhancement factor' in prompt
    assert 'nanogap' in prompt
    assert 'hydrogen evolution' not in prompt


def test_alpha4b2b_recovery_is_candidate_local_and_noncreative():
    prompt = SERS_BRIDGE_RECOVERY_SYSTEM_PROMPT.casefold()
    assert 'exactly one' in prompt
    assert 'do not broaden' in prompt
    assert 'repairable=false' in prompt
