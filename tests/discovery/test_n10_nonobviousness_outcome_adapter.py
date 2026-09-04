import pytest

from pipeline_core.discovery.nonobviousness_outcome_adapter import (
    _resolve_atomic_outcome_from_n9,
    build_atomic_outcomes_from_n9,
)
from pipeline_core.discovery.nonobviousness_production_gate import (
    _claim_outcome as production_v1_claim_outcome,
)


@pytest.mark.parametrize(
    (
        "shadow_state",
        "claim_id",
        "full_by_claim",
        "specification",
    ),
    [
        (
            "SATURATED_PRIOR_ART",
            "claim:1",
            {},
            {},
        ),
        (
            "UNRESOLVED_PARTIAL",
            "claim:1",
            {},
            {},
        ),
        (
            "NEEDS_REFINEMENT",
            "claim:1",
            {},
            {
                "reason_codes": [
                    "missing_bridge",
                ],
                "missing_fields": [
                    "predicted_observation",
                ],
            },
        ),
        (
            "UNRESOLVED",
            "claim:1",
            {},
            {},
        ),
        (
            "UNKNOWN_STATE",
            "claim:1",
            {},
            {},
        ),
        (
            "READY_FOR_CLOSURE",
            "claim:1",
            {},
            {},
        ),
        (
            "READY_FOR_CLOSURE",
            "claim:1",
            {
                "claim:1": {
                    "claim_id": "claim:1",
                    "final_verdict":
                        "POTENTIALLY_NON_OBVIOUS",
                    "final_reason_codes": [
                        "synthetic_positive",
                    ],
                }
            },
            {},
        ),
        (
            "READY_FOR_CLOSURE",
            "claim:1",
            {
                "claim:1": {
                    "claim_id": "claim:1",
                    "final_verdict":
                        "ROUTINE_FROM_PRIOR_ART",
                    "final_reason_codes": [],
                }
            },
            {},
        ),
        (
            "READY_FOR_CLOSURE",
            "claim:1",
            {
                "claim:1": {
                    "claim_id": "claim:1",
                    "final_verdict":
                        "INSUFFICIENT_FOR_JUDGMENT",
                    "final_reason_codes": [],
                }
            },
            {},
        ),
        (
            "READY_FOR_CLOSURE",
            "claim:1",
            {
                "claim:1": {
                    "claim_id": "claim:1",
                    "final_verdict":
                        "BOGUS_VERDICT",
                }
            },
            {},
        ),
    ],
)
def test_adapter_atomic_mapping_has_exact_production_v1_parity(
    shadow_state,
    claim_id,
    full_by_claim,
    specification,
):
    expected = production_v1_claim_outcome(
        shadow_state=shadow_state,
        claim_id=claim_id,
        full_by_claim=full_by_claim,
        specification=specification,
    )

    actual = _resolve_atomic_outcome_from_n9(
        shadow_state=shadow_state,
        claim_id=claim_id,
        full_by_claim=full_by_claim,
        specification=specification,
    )

    assert actual == expected


def _intake(
    *,
    source_portfolio_id="portfolio:1",
):
    return {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            source_portfolio_id,
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:1",
                "claims": [
                    {
                        "claim": {
                            "claim_id":
                                "claim:saturated",
                        },
                        "shadow_state":
                            "SATURATED_PRIOR_ART",
                        "specification": {},
                    },
                    {
                        "claim": {
                            "claim_id":
                                "claim:ready",
                        },
                        "shadow_state":
                            "READY_FOR_CLOSURE",
                        "specification": {},
                    },
                ],
            }
        ],
    }


def _full(
    *,
    source_portfolio_id="portfolio:1",
):
    return {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            source_portfolio_id,
        "claims": [
            {
                "claim_id":
                    "claim:ready",
                "final_verdict":
                    "POTENTIALLY_NON_OBVIOUS",
                "final_reason_codes": [
                    "synthetic_positive",
                ],
            }
        ],
    }


def test_adapter_builds_explicit_claim_outcomes_only():
    result = build_atomic_outcomes_from_n9(
        intake_shadow=_intake(),
        full_shadow=_full(),
    )

    assert set(result) == {
        "claim:saturated",
        "claim:ready",
    }

    assert (
        result["claim:saturated"][
            "nonobviousness_outcome"
        ]
        == "SATURATED_PRIOR_ART"
    )

    assert (
        result["claim:ready"][
            "nonobviousness_outcome"
        ]
        == "POTENTIALLY_NON_OBVIOUS"
    )

    assert all(
        row["adapter_only"] is True
        and row["production_authority"] is False
        for row in result.values()
    )


def test_adapter_rejects_source_portfolio_mismatch():
    with pytest.raises(
        ValueError,
        match="source portfolio mismatch",
    ):
        build_atomic_outcomes_from_n9(
            intake_shadow=_intake(
                source_portfolio_id="portfolio:a",
            ),
            full_shadow=_full(
                source_portfolio_id="portfolio:b",
            ),
        )


def test_adapter_rejects_non_shadow_artifact():
    intake = _intake()
    intake["shadow_only"] = False

    with pytest.raises(
        ValueError,
        match="requires shadow artifacts",
    ):
        build_atomic_outcomes_from_n9(
            intake_shadow=intake,
            full_shadow=_full(),
        )


def test_adapter_does_not_require_importance_or_role():
    intake = _intake()

    # Atomic outcome translation is deliberately orthogonal to
    # hypothesis selection semantics.
    for hypothesis in intake["hypotheses"]:
        for decision in hypothesis["claims"]:
            decision["claim"].pop(
                "importance",
                None,
            )
            decision["claim"].pop(
                "novelty_selection_role",
                None,
            )

    result = build_atomic_outcomes_from_n9(
        intake_shadow=intake,
        full_shadow=_full(),
    )

    assert len(result) == 2


def test_ready_missing_full_remains_insufficient():
    intake = _intake()

    result = build_atomic_outcomes_from_n9(
        intake_shadow=intake,
        full_shadow={
            "schema_version":
                "nonobviousness-full-shadow-v1",
            "shadow_only": True,
            "source_portfolio_id":
                "portfolio:1",
            "claims": [],
        },
    )

    assert (
        result["claim:ready"][
            "nonobviousness_outcome"
        ]
        == "INSUFFICIENT_FOR_JUDGMENT"
    )


def test_adapter_never_maps_absence_label_to_positive():
    intake = _intake()

    intake["hypotheses"][0]["claims"][1][
        "shadow_state"
    ] = "NO_DIRECT_MATCH_FOUND"

    result = build_atomic_outcomes_from_n9(
        intake_shadow=intake,
        full_shadow=_full(),
    )

    assert (
        result["claim:ready"][
            "nonobviousness_outcome"
        ]
        == "INSUFFICIENT_FOR_JUDGMENT"
    )
