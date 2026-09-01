from pipeline_core.discovery.nonobviousness_production_gate import (
    build_nonobviousness_fallback_gate,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)


def _claim(
    claim_id,
    *,
    importance="core",
    state="READY_FOR_CLOSURE",
):
    return {
        "claim": {
            "claim_id": claim_id,
            "importance": importance,
        },
        "shadow_state": state,
    }


def _intake(
    claims,
):
    return {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only":
            True,
        "source_portfolio_id":
            "portfolio:p",
        "source_external_report_id":
            "report:r",
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:h",
                "external_status":
                    "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                "claims":
                    claims,
            }
        ],
    }


def _full(
    rows,
):
    return {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only":
            True,
        "source_portfolio_id":
            "portfolio:p",
        "claims":
            rows,
    }


def _verdict(
    claim_id,
    verdict,
):
    return {
        "hypothesis_id":
            "hypothesis:h",
        "claim_id":
            claim_id,
        "final_verdict":
            verdict,
        "final_reason_codes": [],
    }


def _gate(
    intake,
    full,
):
    result = (
        build_nonobviousness_fallback_gate(
            intake_shadow=intake,
            full_shadow=full,
        )
    )

    return (
        result,
        result["gates"][0],
    )


def test_potentially_nonobvious_core_allows_original_fallback():
    result, gate = _gate(
        _intake(
            [_claim("claim:c")]
        ),
        _full(
            [
                _verdict(
                    "claim:c",
                    "POTENTIALLY_NON_OBVIOUS",
                )
            ]
        ),
    )

    assert gate["fallback_allowed"] is True
    assert gate["selection_class"] == "ELIGIBLE"

    assert (
        result["authority_source"]
        == "n10_nonobviousness"
    )

    # Existing Alpha6 consumer accepts the N10-produced gate
    # without a second authority schema.
    validated = (
        TargetedNoveltyRefinementRuntime
        ._validate_scientific_novelty_gate(
            result,
            type(
                "_Portfolio",
                (),
                {
                    "hypotheses": [
                        type(
                            "_Card",
                            (),
                            {
                                "hypothesis_id":
                                    "hypothesis:h"
                            },
                        )()
                    ]
                },
            )(),
        )
    )

    assert (
        validated[
            "hypothesis:h"
        ]["fallback_allowed"]
        is True
    )


def test_insufficient_core_blocks_positive_fallback():
    _, gate = _gate(
        _intake(
            [_claim("claim:c")]
        ),
        _full(
            [
                _verdict(
                    "claim:c",
                    "INSUFFICIENT_FOR_JUDGMENT",
                )
            ]
        ),
    )

    assert gate["fallback_allowed"] is False
    assert gate["selection_class"] == "INELIGIBLE"

    # Epistemically insufficient is not relabeled routine.
    assert (
        gate["atomic_claims"][0][
            "nonobviousness_outcome"
        ]
        == "INSUFFICIENT_FOR_JUDGMENT"
    )


def test_saturated_core_blocks_original_fallback():
    _, gate = _gate(
        _intake(
            [
                _claim(
                    "claim:c",
                    state="SATURATED_PRIOR_ART",
                )
            ]
        ),
        _full([]),
    )

    assert gate["fallback_allowed"] is False

    assert (
        gate["action"]
        == "REMOVE_OR_REAXIS_ROUTINE_CORE_BRANCH"
    )


def test_supporting_saturated_claim_does_not_kill_positive_core():
    _, gate = _gate(
        _intake(
            [
                _claim(
                    "claim:core",
                    importance="core",
                ),
                _claim(
                    "claim:support",
                    importance="supporting",
                    state="SATURATED_PRIOR_ART",
                ),
            ]
        ),
        _full(
            [
                _verdict(
                    "claim:core",
                    "POTENTIALLY_NON_OBVIOUS",
                )
            ]
        ),
    )

    assert gate["fallback_allowed"] is True


def test_one_positive_core_cannot_hide_saturated_core_branch():
    _, gate = _gate(
        _intake(
            [
                _claim(
                    "claim:novel",
                    importance="core",
                ),
                _claim(
                    "claim:known",
                    importance="core",
                    state="SATURATED_PRIOR_ART",
                ),
            ]
        ),
        _full(
            [
                _verdict(
                    "claim:novel",
                    "POTENTIALLY_NON_OBVIOUS",
                )
            ]
        ),
    )

    assert gate["fallback_allowed"] is False

    assert any(
        "claim:known:SATURATED_PRIOR_ART"
        in reason
        for reason
        in gate["reason_codes"]
    )


def test_ready_claim_missing_full_result_fails_closed():
    _, gate = _gate(
        _intake(
            [_claim("claim:c")]
        ),
        _full([]),
    )

    assert gate["fallback_allowed"] is False

    assert (
        gate["atomic_claims"][0][
            "nonobviousness_outcome"
        ]
        == "INSUFFICIENT_FOR_JUDGMENT"
    )


def test_needs_refinement_core_blocks_and_exposes_action():
    _, gate = _gate(
        _intake(
            [
                _claim(
                    "claim:c",
                    state="NEEDS_REFINEMENT",
                )
            ]
        ),
        _full([]),
    )

    assert gate["fallback_allowed"] is False

    assert (
        gate["action"]
        == "REFINE_ATOMIC_NONOBVIOUSNESS_SPECIFICATION"
    )
