from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def test_novelty_decomposition_requires_atomic_branch_splitting():
    prompt = _DECOMPOSE_SYSTEM

    assert (
        "CLAIM ATOMICITY / BRANCH-SPLITTING CONTRACT:"
        in prompt
    )
    assert (
        '"excitation wavelength or laser power"'
        in prompt
    )
    assert (
        "emit one claim for wavelength moderation "
        "and one claim for power moderation"
        in prompt
    )
    assert (
        "could have different prior-art status"
        in prompt
    )
    assert (
        "presence versus absence of one moderator"
        in prompt
    )
