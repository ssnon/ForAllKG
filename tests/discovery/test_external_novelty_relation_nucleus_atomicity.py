from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def test_claim_text_is_one_relation_nucleus():
    assert (
        "The claim text itself must contain ONE relation nucleus."
        in _DECOMPOSE_SYSTEM
    )

    assert (
        'Do not append a second mechanistic assertion using "because"'
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "emit it as a separate mediator or mechanistic_link claim"
        in _DECOMPOSE_SYSTEM
    )
