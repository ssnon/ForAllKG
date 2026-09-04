from copy import deepcopy

import pytest

from pipeline_core.discovery.nonobviousness_production_gate_v2 import (
    build_nonobviousness_production_gate_v2,
)


def _candidate(
    *,
    selection="CONDITIONAL",
    positive=False,
    allowed=False,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-"
            "gate-v2-candidate",

        "candidate_only":
            True,

        "production_authority":
            False,

        "alpha6_original_fallback_authority":
            False,

        "authority_policy":
            "none_candidate_only",

        "candidate_policy":
            "role-aware-v2-eligible-and-positive-only",

        "source_portfolio_id":
            "portfolio:1",

        "source_query_plan_id":
            "plan:1",

        "gate_count":
            1,

        "candidate_fallback_allowed_count":
            1 if allowed else 0,

        "candidate_fallback_blocked_count":
            0 if allowed else 1,

        "selection_counts": {
            "ELIGIBLE":
                1 if selection == "ELIGIBLE" else 0,

            "CONDITIONAL":
                1 if selection == "CONDITIONAL" else 0,

            "INELIGIBLE":
                1 if selection == "INELIGIBLE" else 0,
        },

        "policy": {
            "conditional_is_positive":
                False,

            "absence_is_novelty":
                False,
        },

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:1",

                "selection_class":
                    selection,

                "candidate_fallback_allowed":
                    allowed,

                "candidate_positive_nonobviousness_authority":
                    positive,

                "action":
                    (
                        "KEEP"
                        if selection == "ELIGIBLE"
                        else "RESOLVE"
                    ),

                "base_aggregation_action":
                    None,

                "blocking_claim_ids":
                    [],

                "unresolved_claim_ids":
                    (
                        []
                        if selection == "ELIGIBLE"
                        else ["claim:1"]
                    ),

                "unresolved_selection_role_claim_ids":
                    [],

                "structurally_unresolved_claim_ids":
                    [],

                "resolution_requirements":
                    [],

                "reason_codes":
                    ["synthetic"],

                "production_authority":
                    False,
            }
        ],
    }


def test_eligible_positive_promotes_exact_candidate_boolean():
    candidate = _candidate(
        selection="ELIGIBLE",
        positive=True,
        allowed=True,
    )

    result = (
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    assert (
        result["schema_version"]
        == "scientific-novelty-fallback-gate-v2"
    )

    assert (
        result["production_authority"]
        is True
    )

    assert (
        result["gates"][0][
            "fallback_allowed"
        ]
        is True
    )

    assert (
        result["fallback_allowed_count"]
        == 1
    )


def test_conditional_remains_fallback_negative():
    candidate = _candidate(
        selection="CONDITIONAL",
        positive=False,
        allowed=False,
    )

    result = (
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    assert (
        result["gates"][0][
            "selection_class"
        ]
        == "CONDITIONAL"
    )

    assert (
        result["gates"][0][
            "fallback_allowed"
        ]
        is False
    )


def test_ineligible_remains_fallback_negative():
    candidate = _candidate(
        selection="INELIGIBLE",
        positive=False,
        allowed=False,
    )

    result = (
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    assert (
        result["gates"][0][
            "fallback_allowed"
        ]
        is False
    )


def test_eligible_without_positive_authority_fails_closed():
    candidate = _candidate(
        selection="ELIGIBLE",
        positive=False,
        allowed=False,
    )

    with pytest.raises(
        ValueError,
        match="lacks positive",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_noneligible_positive_authority_fails_closed():
    candidate = _candidate(
        selection="CONDITIONAL",
        positive=True,
        allowed=False,
    )

    with pytest.raises(
        ValueError,
        match="non-ELIGIBLE",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_candidate_boolean_cannot_be_reinterpreted():
    candidate = _candidate(
        selection="ELIGIBLE",
        positive=True,
        allowed=True,
    )

    candidate[
        "gates"
    ][0][
        "candidate_fallback_allowed"
    ] = False

    candidate[
        "candidate_fallback_allowed_count"
    ] = 0

    candidate[
        "candidate_fallback_blocked_count"
    ] = 1

    with pytest.raises(
        ValueError,
        match="internally inconsistent",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_already_authoritative_candidate_is_rejected():
    candidate = _candidate()

    candidate[
        "production_authority"
    ] = True

    with pytest.raises(
        ValueError,
        match="already has production authority",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_wrong_candidate_schema_is_rejected():
    candidate = _candidate()

    candidate[
        "schema_version"
    ] = "wrong"

    with pytest.raises(
        ValueError,
        match="unexpected v2 candidate schema",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_duplicate_hypothesis_ids_fail_closed():
    candidate = _candidate()

    second = deepcopy(
        candidate["gates"][0]
    )

    candidate[
        "gates"
    ].append(
        second
    )

    candidate[
        "gate_count"
    ] = 2

    candidate[
        "candidate_fallback_blocked_count"
    ] = 2

    candidate[
        "selection_counts"
    ][
        "CONDITIONAL"
    ] = 2

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_declared_counts_are_verified():
    candidate = _candidate()

    candidate[
        "gate_count"
    ] = 999

    with pytest.raises(
        ValueError,
        match="gate_count mismatch",
    ):
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )


def test_resolution_diagnostics_are_preserved():
    candidate = _candidate()

    candidate[
        "gates"
    ][0][
        "resolution_requirements"
    ] = [
        {
            "claim_id":
                "claim:1",

            "action":
                "RESOLVE_NOVELTY_BEARING_EVIDENCE",
        }
    ]

    result = (
        build_nonobviousness_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    assert (
        result["gates"][0][
            "resolution_requirements"
        ]
        == candidate["gates"][0][
            "resolution_requirements"
        ]
    )

    assert (
        result["candidate_semantics_preserved"]
        is True
    )


def test_compiler_does_not_mutate_candidate():
    candidate = _candidate()
    before = deepcopy(candidate)

    build_nonobviousness_production_gate_v2(
        candidate_gate=candidate,
    )

    assert candidate == before
