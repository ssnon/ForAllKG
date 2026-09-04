import pytest

from pipeline_core.discovery.nonobviousness_post_generation import (
    _validate_n10_gate,
)


def _v2_gate(
    *,
    selection="CONDITIONAL",
    positive=False,
    fallback=False,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v2",

        "production_authority":
            True,

        "authority_scope":
            "alpha6_post_generation_candidate",

        "authority_source":
            "n10_role_aware_nonobviousness_v2",

        "positive_authority_requires":
            "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS",

        "conditional_is_positive":
            False,

        "absence_is_novelty":
            False,

        "candidate_semantics_preserved":
            True,

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:test",

                "selection_class":
                    selection,

                "action":
                    "REFINE_NOVELTY_BEARING_SPECIFICATION",

                "base_aggregation_action":
                    "REFINE_NOVELTY_BEARING_SPECIFICATION",

                "positive_nonobviousness_authority":
                    positive,

                "fallback_allowed":
                    fallback,
            }
        ],
    }


def test_post_generation_validator_accepts_conditional_v2():
    row = _validate_n10_gate(
        candidate_id="hypothesis:test",
        gate=_v2_gate(),
    )

    assert (
        row["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        row["positive_nonobviousness_authority"]
        is False
    )

    assert (
        row["fallback_allowed"]
        is False
    )


def test_post_generation_validator_accepts_eligible_positive_v2():
    row = _validate_n10_gate(
        candidate_id="hypothesis:test",
        gate=_v2_gate(
            selection="ELIGIBLE",
            positive=True,
            fallback=True,
        ),
    )

    assert row["selection_class"] == "ELIGIBLE"


def test_wrong_post_generation_scope_fails_closed():
    gate = _v2_gate()

    gate["authority_scope"] = (
        "alpha6_original_fallback"
    )

    with pytest.raises(
        ValueError,
        match="authority scope",
    ):
        _validate_n10_gate(
            candidate_id="hypothesis:test",
            gate=gate,
        )


def test_noneligible_positive_authority_fails_closed():
    with pytest.raises(
        ValueError,
        match="cannot carry positive authority",
    ):
        _validate_n10_gate(
            candidate_id="hypothesis:test",
            gate=_v2_gate(
                selection="CONDITIONAL",
                positive=True,
                fallback=False,
            ),
        )


def test_entrypoint_builds_role_aware_post_generation_gate():
    text = (
        __import__(
            "pathlib"
        )
        .Path(
            "scripts/discovery/"
            "enforce_alpha6_nonobviousness.py"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "build_nonobviousness_production_gate_v2_candidate"
        in text
    )

    assert (
        "build_nonobviousness_post_generation_production_gate_v2"
        in text
    )

    old_exact = (
        '"build_nonobviousness_production_gate",'
    )

    assert old_exact not in text
