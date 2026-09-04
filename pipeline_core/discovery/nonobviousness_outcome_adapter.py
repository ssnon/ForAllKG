from __future__ import annotations

from typing import Any


_INTAKE_SCHEMA = "nonobviousness-shadow-v1"
_FULL_SCHEMA = "nonobviousness-full-shadow-v1"


def _resolve_atomic_outcome_from_n9(
    *,
    shadow_state: str,
    claim_id: str,
    full_by_claim: dict[str, dict[str, Any]],
    specification: dict[str, Any] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Translate frozen N9 semantics into explicit atomic outcomes.

    This intentionally mirrors the frozen production-v1
    nonobviousness _claim_outcome contract.

    It does NOT:
    - inspect novelty_selection_role;
    - inspect topology;
    - infer novelty from search absence;
    - copy historical judgments across claim IDs;
    - alter N9 adjudication.
    """

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
        detail_codes: list[str] = [
            "atomic_specification_incomplete",
        ]

        specification = (
            specification
            if isinstance(specification, dict)
            else {}
        )

        raw_reason_codes = specification.get(
            "reason_codes",
            [],
        )

        if isinstance(
            raw_reason_codes,
            (list, tuple),
        ):
            detail_codes.extend(
                str(value)
                for value in raw_reason_codes
                if str(value).strip()
            )

        raw_missing_fields = specification.get(
            "missing_fields",
            [],
        )

        if isinstance(
            raw_missing_fields,
            (list, tuple),
        ):
            detail_codes.extend(
                "missing_specification_field:"
                + str(value)
                for value in raw_missing_fields
                if str(value).strip()
            )

        return (
            "NEEDS_REFINEMENT",
            tuple(
                dict.fromkeys(
                    detail_codes
                )
            ),
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

    return (
        "INSUFFICIENT_FOR_JUDGMENT",
        (
            "missing_or_unsupported_final_"
            "nonobviousness_verdict",
        ),
    )


def build_atomic_outcomes_from_n9(
    *,
    intake_shadow: dict[str, Any],
    full_shadow: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compile N9 intake/full artifacts into explicit claim outcomes.

    Output is claim-ID keyed and selection-role agnostic.

    This is an adapter only. It carries no hypothesis-level selection
    authority and cannot be used to map an old claim judgment onto a
    new claim ID.
    """

    if (
        intake_shadow.get("schema_version")
        != _INTAKE_SCHEMA
    ):
        raise ValueError(
            "unexpected N9 intake shadow schema"
        )

    if (
        full_shadow.get("schema_version")
        != _FULL_SCHEMA
    ):
        raise ValueError(
            "unexpected N9 full shadow schema"
        )

    if (
        intake_shadow.get("shadow_only")
        is not True
        or full_shadow.get("shadow_only")
        is not True
    ):
        raise ValueError(
            "N9 outcome adapter requires shadow artifacts"
        )

    if (
        intake_shadow.get("source_portfolio_id")
        != full_shadow.get("source_portfolio_id")
    ):
        raise ValueError(
            "N9 intake/full source portfolio mismatch"
        )

    full_rows = full_shadow.get(
        "claims",
        [],
    )

    if not isinstance(
        full_rows,
        list,
    ):
        raise ValueError(
            "N9 full shadow claims must be a list"
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
                "N9 full shadow claim row must be an object"
            )

        claim_id = str(
            row.get("claim_id")
            or ""
        ).strip()

        if not claim_id:
            raise ValueError(
                "N9 full shadow claim missing claim_id"
            )

        if claim_id in full_by_claim:
            raise ValueError(
                "duplicate N9 full-shadow claim_id: "
                + claim_id
            )

        full_by_claim[
            claim_id
        ] = row

    hypotheses = intake_shadow.get(
        "hypotheses",
        [],
    )

    if not isinstance(
        hypotheses,
        list,
    ):
        raise ValueError(
            "N9 intake hypotheses must be a list"
        )

    outcomes: dict[
        str,
        dict[str, Any],
    ] = {}

    seen_hypotheses: set[str] = set()

    for hypothesis in hypotheses:
        if not isinstance(
            hypothesis,
            dict,
        ):
            raise ValueError(
                "N9 intake hypothesis row must be an object"
            )

        hypothesis_id = str(
            hypothesis.get("hypothesis_id")
            or ""
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "N9 intake hypothesis missing hypothesis_id"
            )

        if hypothesis_id in seen_hypotheses:
            raise ValueError(
                "duplicate N9 intake hypothesis_id: "
                + hypothesis_id
            )

        decisions = hypothesis.get(
            "claims",
            [],
        )

        if not isinstance(
            decisions,
            list,
        ):
            raise ValueError(
                "N9 intake claims must be a list"
            )

        for decision in decisions:
            if not isinstance(
                decision,
                dict,
            ):
                raise ValueError(
                    "N9 intake claim decision must be an object"
                )

            claim = decision.get(
                "claim"
            )

            if not isinstance(
                claim,
                dict,
            ):
                raise ValueError(
                    "N9 intake decision missing atomic claim"
                )

            claim_id = str(
                claim.get("claim_id")
                or ""
            ).strip()

            if not claim_id:
                raise ValueError(
                    "N9 atomic claim missing claim_id"
                )

            if claim_id in outcomes:
                raise ValueError(
                    "duplicate N9 intake claim_id: "
                    + claim_id
                )

            specification = decision.get(
                "specification"
            )

            if not isinstance(
                specification,
                dict,
            ):
                specification = {}

            outcome, reasons = (
                _resolve_atomic_outcome_from_n9(
                    shadow_state=str(
                        decision.get(
                            "shadow_state"
                        )
                        or ""
                    ),
                    claim_id=claim_id,
                    full_by_claim=full_by_claim,
                    specification=specification,
                )
            )

            outcomes[claim_id] = {
                "nonobviousness_outcome":
                    outcome,
                "reason_codes":
                    list(reasons),
                "source_hypothesis_id":
                    hypothesis_id,
                "source_shadow_state":
                    str(
                        decision.get(
                            "shadow_state"
                        )
                        or ""
                    ),
                "adapter_only":
                    True,
                "production_authority":
                    False,
            }

        seen_hypotheses.add(
            hypothesis_id
        )

    return outcomes
