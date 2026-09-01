from __future__ import annotations

from typing import Any


_GATE_SCHEMA = "scientific-novelty-fallback-gate-v1"
_INTAKE_SCHEMA = "nonobviousness-shadow-v1"
_FULL_SCHEMA = "nonobviousness-full-shadow-v1"


def _claim_outcome(
    *,
    shadow_state: str,
    claim_id: str,
    full_by_claim: dict[str, dict[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """Resolve one atomic claim into N10 production disposition."""

    if shadow_state == "SATURATED_PRIOR_ART":
        return (
            "SATURATED_PRIOR_ART",
            ("atomic_claim_saturated_by_prior_art",),
        )

    if shadow_state == "UNRESOLVED_PARTIAL":
        return (
            "NEEDS_REFINEMENT",
            ("partial_prior_art_requires_resolution",),
        )

    if shadow_state == "NEEDS_REFINEMENT":
        return (
            "NEEDS_REFINEMENT",
            ("atomic_specification_incomplete",),
        )

    if shadow_state == "UNRESOLVED":
        return (
            "INSUFFICIENT_FOR_JUDGMENT",
            ("atomic_claim_unresolved",),
        )

    if shadow_state != "READY_FOR_CLOSURE":
        return (
            "INSUFFICIENT_FOR_JUDGMENT",
            ("unsupported_shadow_state_fail_closed",),
        )

    full = full_by_claim.get(
        claim_id
    )

    if full is None:
        return (
            "INSUFFICIENT_FOR_JUDGMENT",
            ("ready_claim_missing_full_adjudication",),
        )

    verdict = str(
        full.get("final_verdict")
        or ""
    ).strip()

    if verdict == "POTENTIALLY_NON_OBVIOUS":
        return (
            verdict,
            tuple(
                full.get("final_reason_codes")
                or ()
            ),
        )

    if verdict == "ROUTINE_FROM_PRIOR_ART":
        return (
            verdict,
            tuple(
                full.get("final_reason_codes")
                or (
                    "routine_from_prior_art",
                )
            ),
        )

    if verdict == "INSUFFICIENT_FOR_JUDGMENT":
        return (
            verdict,
            tuple(
                full.get("final_reason_codes")
                or (
                    "nonobviousness_evidence_insufficient",
                )
            ),
        )

    # A READY claim may be deferred or malformed. Never turn that
    # operational absence into positive non-obviousness authority.
    return (
        "INSUFFICIENT_FOR_JUDGMENT",
        (
            "missing_or_unsupported_final_nonobviousness_verdict",
        ),
    )


def _action_for_outcomes(
    outcomes: tuple[str, ...],
) -> str:
    """Human/action-layer guidance; fallback authority remains boolean."""

    if any(
        value == "NEEDS_REFINEMENT"
        for value in outcomes
    ):
        return "REFINE_ATOMIC_NONOBVIOUSNESS_SPECIFICATION"

    if any(
        value in {
            "SATURATED_PRIOR_ART",
            "ROUTINE_FROM_PRIOR_ART",
        }
        for value in outcomes
    ):
        return "REMOVE_OR_REAXIS_ROUTINE_CORE_BRANCH"

    if any(
        value == "INSUFFICIENT_FOR_JUDGMENT"
        for value in outcomes
    ):
        return "RESOLVE_NONOBVIOUSNESS_EVIDENCE"

    if (
        outcomes
        and all(
            value == "POTENTIALLY_NON_OBVIOUS"
            for value in outcomes
        )
    ):
        return "KEEP_NONOBVIOUS_CANDIDATE"

    return "REFINE_OR_REASSESS_NONOBVIOUSNESS"


def build_nonobviousness_fallback_gate(
    *,
    intake_shadow: dict[str, Any],
    full_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Compile N10 atomic adjudication into Alpha6 fallback authority.

    Scientific policy:
    - POTENTIALLY_NON_OBVIOUS is the only positive N10 authority.
    - ROUTINE, SATURATED, NEEDS_REFINEMENT, and INSUFFICIENT do not
      permit original fallback.
    - Supporting claims do not automatically block a hypothesis.
    - Every CORE atomic claim must independently receive positive
      adjudication. One positive branch cannot hide a routine,
      under-specified, or unassessed core branch.
    - If a hypothesis has no claims explicitly marked core, all atomic
      claims are conservatively treated as selection-relevant.
    """

    if (
        intake_shadow.get("schema_version")
        != _INTAKE_SCHEMA
    ):
        raise ValueError(
            "unexpected N10 intake shadow schema"
        )

    if (
        full_shadow.get("schema_version")
        != _FULL_SCHEMA
    ):
        raise ValueError(
            "unexpected N10 full shadow schema"
        )

    if (
        intake_shadow.get("shadow_only")
        is not True
        or full_shadow.get("shadow_only")
        is not True
    ):
        raise ValueError(
            "N10 production gate requires shadow artifacts"
        )

    if (
        intake_shadow.get("source_portfolio_id")
        != full_shadow.get("source_portfolio_id")
    ):
        raise ValueError(
            "N10 intake/full source portfolio mismatch"
        )

    full_rows = full_shadow.get(
        "claims",
        []
    )

    if not isinstance(
        full_rows,
        list,
    ):
        raise ValueError(
            "N10 full shadow claims must be a list"
        )

    full_by_claim: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in full_rows:
        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "N10 full shadow claim row must be an object"
            )

        claim_id = str(
            row.get("claim_id")
            or ""
        ).strip()

        if not claim_id:
            raise ValueError(
                "N10 full shadow claim missing claim_id"
            )

        if claim_id in full_by_claim:
            raise ValueError(
                "duplicate N10 full-shadow claim_id: "
                + claim_id
            )

        full_by_claim[
            claim_id
        ] = row

    hypothesis_rows = (
        intake_shadow.get(
            "hypotheses",
            []
        )
    )

    if not isinstance(
        hypothesis_rows,
        list,
    ):
        raise ValueError(
            "N10 intake hypotheses must be a list"
        )

    gates: list[
        dict[str, Any]
    ] = []

    seen_hypotheses: set[
        str
    ] = set()

    for hypothesis in hypothesis_rows:
        if not isinstance(
            hypothesis,
            dict,
        ):
            raise ValueError(
                "N10 intake hypothesis row must be an object"
            )

        hypothesis_id = str(
            hypothesis.get(
                "hypothesis_id"
            )
            or ""
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "N10 intake hypothesis missing hypothesis_id"
            )

        if (
            hypothesis_id
            in seen_hypotheses
        ):
            raise ValueError(
                "duplicate N10 intake hypothesis_id: "
                + hypothesis_id
            )

        claims = hypothesis.get(
            "claims",
            []
        )

        if not isinstance(
            claims,
            list,
        ):
            raise ValueError(
                "N10 intake claims must be a list"
            )

        compiled_claims: list[
            dict[str, Any]
        ] = []

        for decision in claims:
            if not isinstance(
                decision,
                dict,
            ):
                raise ValueError(
                    "N10 intake claim decision must be an object"
                )

            claim = decision.get(
                "claim"
            )

            if not isinstance(
                claim,
                dict,
            ):
                raise ValueError(
                    "N10 intake decision missing atomic claim"
                )

            claim_id = str(
                claim.get("claim_id")
                or ""
            ).strip()

            if not claim_id:
                raise ValueError(
                    "N10 atomic claim missing claim_id"
                )

            importance = str(
                claim.get("importance")
                or "supporting"
            ).strip()

            if importance not in {
                "core",
                "supporting",
            }:
                raise ValueError(
                    "unsupported N10 claim importance: "
                    + importance
                )

            outcome, reasons = (
                _claim_outcome(
                    shadow_state=str(
                        decision.get(
                            "shadow_state"
                        )
                        or ""
                    ),
                    claim_id=claim_id,
                    full_by_claim=(
                        full_by_claim
                    ),
                )
            )

            compiled_claims.append(
                {
                    "claim_id":
                        claim_id,
                    "importance":
                        importance,
                    "shadow_state":
                        str(
                            decision.get(
                                "shadow_state"
                            )
                            or ""
                        ),
                    "nonobviousness_outcome":
                        outcome,
                    "reason_codes":
                        list(reasons),
                }
            )

        core = [
            row
            for row
            in compiled_claims
            if row["importance"]
            == "core"
        ]

        relevant = (
            core
            if core
            else compiled_claims
        )

        relevant_outcomes = tuple(
            str(
                row[
                    "nonobviousness_outcome"
                ]
            )
            for row in relevant
        )

        fallback_allowed = bool(
            relevant_outcomes
            and all(
                outcome
                == "POTENTIALLY_NON_OBVIOUS"
                for outcome
                in relevant_outcomes
            )
        )

        selection_class = (
            "ELIGIBLE"
            if fallback_allowed
            else "INELIGIBLE"
        )

        reason_codes: list[
            str
        ] = []

        if not relevant:
            reason_codes.append(
                "no_selection_relevant_atomic_claims"
            )

        if fallback_allowed:
            reason_codes.append(
                "all_core_atomic_claims_potentially_nonobvious"
            )
        else:
            reason_codes.append(
                "core_atomic_nonobviousness_not_fully_established"
            )

        for row in relevant:
            if (
                row[
                    "nonobviousness_outcome"
                ]
                != "POTENTIALLY_NON_OBVIOUS"
            ):
                reason_codes.append(
                    "core_claim_blocked:"
                    + row["claim_id"]
                    + ":"
                    + row[
                        "nonobviousness_outcome"
                    ]
                )

        gates.append(
            {
                "hypothesis_id":
                    hypothesis_id,
                "fallback_allowed":
                    fallback_allowed,
                "selection_class":
                    selection_class,
                "action":
                    _action_for_outcomes(
                        relevant_outcomes
                    ),
                "reason_codes":
                    list(
                        dict.fromkeys(
                            reason_codes
                        )
                    ),
                "atomic_claims":
                    compiled_claims,
                "selection_relevant_claim_ids":
                    [
                        row["claim_id"]
                        for row in relevant
                    ],
                "authority_source":
                    "n10_nonobviousness",
                "external_status":
                    hypothesis.get(
                        "external_status"
                    ),
            }
        )

        seen_hypotheses.add(
            hypothesis_id
        )

    return {
        "schema_version":
            _GATE_SCHEMA,

        "source_action_batch_schema":
            None,

        "source_nonobviousness_intake_schema":
            _INTAKE_SCHEMA,

        "source_nonobviousness_full_schema":
            _FULL_SCHEMA,

        "source_portfolio_id":
            intake_shadow.get(
                "source_portfolio_id"
            ),

        "source_external_report_id":
            intake_shadow.get(
                "source_external_report_id"
            ),

        "gate_count":
            len(gates),

        "gates":
            gates,

        "production_authority":
            True,

        "authority_scope":
            "alpha6_original_fallback_only",

        "authority_source":
            "n10_nonobviousness",

        "positive_authority_requires":
            "POTENTIALLY_NON_OBVIOUS",

        "all_core_atomic_claims_must_pass":
            True,

        "insufficient_is_not_routine":
            True,

        "action_policy_applied":
            True,
    }
