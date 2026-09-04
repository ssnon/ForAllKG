from pathlib import Path

from pipeline_core.discovery.novelty_claim_decomposition import (
    _clean_branch_specific_specification,
)


PROMPT = Path(
    "pipeline_core/discovery/"
    "external_novelty_llm.py"
)


def test_decomposer_requires_verbatim_identity_label_in_specification():
    text = PROMPT.read_text(
        encoding="utf-8"
    )

    assert (
        "LEXICAL IDENTITY ALIGNMENT"
        in text
    )

    assert (
        "MUST include at least one "
        "prior_art_identity_terms entry verbatim"
        in text
    )


def test_prompt_forbids_weakening_identity_to_satisfy_contract():
    text = PROMPT.read_text(
        encoding="utf-8"
    )

    assert (
        "Do not satisfy this rule by changing "
        "prior_art_identity_terms"
        in text
    )

    assert (
        "weaker, broader, or more convenient phrase"
        in text
    )


def test_prompt_forbids_new_scientific_content_from_identity_alignment():
    text = PROMPT.read_text(
        encoding="utf-8"
    )

    required = [
        "does not authorize adding synonyms",
        "mechanisms",
        "variables",
        "directions",
        "thresholds",
        "regimes",
        "scientific propositions",
    ]

    for needle in required:
        assert needle in text


def test_existing_sanitizer_accepts_exact_generic_identity_label():
    identity = [
        "surface environment"
    ]

    field = (
        "For surface environment, "
        "response Y differs under otherwise "
        "matched comparison conditions."
    )

    assert (
        _clean_branch_specific_specification(
            field,
            identity,
        )
        == field
    )


def test_existing_sanitizer_remains_fail_closed_without_identity_label():
    identity = [
        "surface environment"
    ]

    field = (
        "Response Y differs between condition A "
        "and condition B under otherwise matched "
        "comparison conditions."
    )

    assert (
        _clean_branch_specific_specification(
            field,
            identity,
        )
        == ""
    )


def test_sanitizer_source_is_not_modified_by_prompt_contract():
    source = Path(
        "pipeline_core/discovery/"
        "novelty_claim_decomposition.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "LEXICAL IDENTITY ALIGNMENT"
        not in source
    )
