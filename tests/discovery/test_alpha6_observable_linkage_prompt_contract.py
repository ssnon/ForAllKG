from __future__ import annotations

from pathlib import Path


REQUIRED = (
    "observable MUST copy verbatim one observable "
    "from predicted_observations"
)

OUTCOME_RULE = (
    "Put the contrary or falsifying result "
    "in falsifying_outcome"
)


def _source(name: str) -> str:
    return Path(
        "pipeline_core/discovery"
    ).joinpath(
        name
    ).read_text(
        encoding="utf-8"
    )


def test_refinement_prompt_declares_observable_linkage_contract():
    text = _source(
        "novelty_refinement_prompt.py"
    )

    assert REQUIRED in text
    assert OUTCOME_RULE in text


def test_reaxis_prompt_declares_observable_linkage_contract():
    text = _source(
        "novelty_reaxis_prompt.py"
    )

    assert REQUIRED in text
    assert OUTCOME_RULE in text


def test_contract_does_not_relax_deterministic_validator():
    validator = _source(
        "hypothesis_validation.py"
    )

    assert (
        "FALSIFIER_OBSERVABLE_NOT_PREDICTED"
        in validator
    )

    assert (
        "_observable_matches("
        in validator
    )
