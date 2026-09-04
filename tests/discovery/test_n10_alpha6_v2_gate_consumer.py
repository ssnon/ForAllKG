from types import SimpleNamespace

import pytest

from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)


def _portfolio(*ids: str):
    return SimpleNamespace(
        hypotheses=[
            SimpleNamespace(
                hypothesis_id=value
            )
            for value in ids
        ]
    )


def _v1(
    *,
    selection: str,
    fallback_allowed: bool,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v1",

        "production_authority":
            True,

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:1",

                "selection_class":
                    selection,

                "fallback_allowed":
                    fallback_allowed,
            }
        ],
    }


def _v2(
    *,
    selection: str,
    positive: bool,
    fallback_allowed: bool,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v2",

        "production_authority":
            True,

        "authority_scope":
            "alpha6_original_fallback",

        "conditional_is_positive":
            False,

        "absence_is_novelty":
            False,

        "candidate_semantics_preserved":
            True,

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:1",

                "selection_class":
                    selection,

                "fallback_allowed":
                    fallback_allowed,

                "positive_nonobviousness_authority":
                    positive,
            }
        ],
    }


def _validate(gate):
    return (
        TargetedNoveltyRefinementRuntime
        ._validate_scientific_novelty_gate(
            gate,
            _portfolio(
                "hypothesis:1"
            ),
        )
    )


def test_v1_conditional_fallback_true_is_preserved():
    gate = _v1(
        selection="CONDITIONAL",
        fallback_allowed=True,
    )

    by_id = _validate(gate)

    assert (
        by_id[
            "hypothesis:1"
        ][
            "fallback_allowed"
        ]
        is True
    )


def test_v1_conditional_false_remains_invalid():
    gate = _v1(
        selection="CONDITIONAL",
        fallback_allowed=False,
    )

    with pytest.raises(
        RuntimeError,
        match="internally inconsistent",
    ):
        _validate(gate)


def test_v1_eligible_true_remains_valid():
    gate = _v1(
        selection="ELIGIBLE",
        fallback_allowed=True,
    )

    assert _validate(gate) is not None


def test_v1_ineligible_false_remains_valid():
    gate = _v1(
        selection="INELIGIBLE",
        fallback_allowed=False,
    )

    assert _validate(gate) is not None


def test_v2_eligible_positive_allows_fallback():
    gate = _v2(
        selection="ELIGIBLE",
        positive=True,
        fallback_allowed=True,
    )

    by_id = _validate(gate)

    assert (
        by_id[
            "hypothesis:1"
        ][
            "fallback_allowed"
        ]
        is True
    )


def test_v2_conditional_is_fallback_negative():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    by_id = _validate(gate)

    assert (
        by_id[
            "hypothesis:1"
        ][
            "fallback_allowed"
        ]
        is False
    )


def test_v2_ineligible_is_fallback_negative():
    gate = _v2(
        selection="INELIGIBLE",
        positive=False,
        fallback_allowed=False,
    )

    assert _validate(gate) is not None


def test_v2_conditional_true_fails_closed():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=True,
    )

    with pytest.raises(
        RuntimeError,
        match="internally inconsistent",
    ):
        _validate(gate)


def test_v2_eligible_without_positive_authority_fails():
    gate = _v2(
        selection="ELIGIBLE",
        positive=False,
        fallback_allowed=False,
    )

    with pytest.raises(
        RuntimeError,
        match="lacks positive authority",
    ):
        _validate(gate)


def test_v2_noneligible_cannot_claim_positive_authority():
    gate = _v2(
        selection="CONDITIONAL",
        positive=True,
        fallback_allowed=False,
    )

    with pytest.raises(
        RuntimeError,
        match="non-ELIGIBLE",
    ):
        _validate(gate)


def test_v2_requires_production_authority():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "production_authority"
    ] = False

    with pytest.raises(
        RuntimeError,
        match="lacks production authority",
    ):
        _validate(gate)


def test_v2_requires_conditional_nonpositive_contract():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "conditional_is_positive"
    ] = True

    with pytest.raises(
        RuntimeError,
        match="CONDITIONAL non-positive",
    ):
        _validate(gate)


def test_v2_rejects_absence_as_novelty_contract():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "absence_is_novelty"
    ] = True

    with pytest.raises(
        RuntimeError,
        match="absence as novelty",
    ):
        _validate(gate)


def test_v2_requires_candidate_semantics_preservation():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "candidate_semantics_preserved"
    ] = False

    with pytest.raises(
        RuntimeError,
        match="preserve candidate semantics",
    ):
        _validate(gate)


def test_unknown_gate_schema_remains_rejected():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "schema_version"
    ] = "unsupported"

    with pytest.raises(
        RuntimeError,
        match="Unexpected scientific novelty",
    ):
        _validate(gate)


def test_v2_hypothesis_set_must_match_portfolio():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    with pytest.raises(
        RuntimeError,
        match="does not match Alpha6 source portfolio",
    ):
        (
            TargetedNoveltyRefinementRuntime
            ._validate_scientific_novelty_gate(
                gate,
                _portfolio(
                    "hypothesis:other"
                ),
            )
        )


def test_v2_duplicate_hypothesis_rows_fail():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    gate[
        "gates"
    ].append(
        dict(
            gate[
                "gates"
            ][0]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Duplicate",
    ):
        _validate(gate)


def test_v2_validated_boolean_drives_original_fallback():
    gate = _v2(
        selection="CONDITIONAL",
        positive=False,
        fallback_allowed=False,
    )

    by_id = _validate(gate)

    allowed = (
        TargetedNoveltyRefinementRuntime
        ._original_fallback_allowed(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            hypothesis_id="hypothesis:1",
            scientific_gate_by_id=by_id,
        )
    )

    assert allowed is False


def test_v1_validated_boolean_still_drives_original_fallback():
    gate = _v1(
        selection="CONDITIONAL",
        fallback_allowed=True,
    )

    by_id = _validate(gate)

    allowed = (
        TargetedNoveltyRefinementRuntime
        ._original_fallback_allowed(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            hypothesis_id="hypothesis:1",
            scientific_gate_by_id=by_id,
        )
    )

    assert allowed is True
