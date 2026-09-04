from __future__ import annotations

import hashlib
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


_GENERATED_SURVIVOR_DECISIONS = {
    "accepted_refinement",
    "accepted_reaxis",
}


def _candidate_final_authority_projection(
    card: HypothesisCard,
) -> dict[str, Any]:
    """Return all authority-relevant card content minus compile identities.

    Candidate-stage and final-portfolio compilation intentionally occupy
    distinct identity namespaces.  Authority may cross that boundary only
    when every non-identity field is preserved.

    The only fields ignored here are identities deterministically regenerated
    by HypothesisCompiler:
      - hypothesis_id
      - predicted_observations[].observation_id
      - falsification_criteria[].criterion_id

    Any current or future non-identity HypothesisCard field remains part of
    the comparison by default, so schema evolution fails closed.
    """

    payload = card.model_dump(
        mode="json"
    )

    payload.pop(
        "hypothesis_id",
        None,
    )

    for row in payload.get(
        "predicted_observations",
        [],
    ):
        if isinstance(row, dict):
            row.pop(
                "observation_id",
                None,
            )

    for row in payload.get(
        "falsification_criteria",
        [],
    ):
        if isinstance(row, dict):
            row.pop(
                "criterion_id",
                None,
            )

    return payload


def assert_candidate_final_authority_equivalent(
    *,
    candidate: HypothesisCard,
    final: HypothesisCard,
) -> None:
    """Fail closed unless candidate and final differ only by compile IDs."""

    candidate_payload = (
        _candidate_final_authority_projection(
            candidate
        )
    )

    final_payload = (
        _candidate_final_authority_projection(
            final
        )
    )

    if candidate_payload == final_payload:
        return

    fields = sorted(
        {
            *candidate_payload,
            *final_payload,
        }
    )

    changed = [
        field
        for field in fields
        if (
            candidate_payload.get(field)
            != final_payload.get(field)
        )
    ]

    raise ValueError(
        "Alpha6 candidate/final authority transfer "
        "requires semantic/provenance equivalence; "
        "candidate="
        + candidate.hypothesis_id
        + "; final="
        + final.hypothesis_id
        + "; changed_fields="
        + repr(changed)
    )


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _validate_n10_gate(
    *,
    candidate_id: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    schema = gate.get(
        "schema_version"
    )

    if schema not in {
        "scientific-novelty-fallback-gate-v1",
        "scientific-novelty-fallback-gate-v2",
    }:
        raise ValueError(
            "unexpected N10 production gate schema"
        )

    if gate.get(
        "production_authority"
    ) is not True:
        raise ValueError(
            "N10 candidate gate lacks production authority"
        )

    if (
        schema
        == "scientific-novelty-fallback-gate-v1"
    ):
        if (
            gate.get("authority_source")
            != "n10_nonobviousness"
        ):
            raise ValueError(
                "post-generation candidate gate must come "
                "from N10 non-obviousness"
            )

    else:
        if (
            gate.get("authority_scope")
            != "alpha6_post_generation_candidate"
        ):
            raise ValueError(
                "role-aware post-generation N10 gate has "
                "wrong authority scope"
            )

        if (
            gate.get("authority_source")
            != "n10_role_aware_nonobviousness_v2"
        ):
            raise ValueError(
                "unexpected role-aware N10 authority source"
            )

        if (
            gate.get(
                "positive_authority_requires"
            )
            != (
                "ELIGIBLE_AND_ROLE_AWARE_"
                "POSITIVE_NONOBVIOUSNESS"
            )
        ):
            raise ValueError(
                "unexpected role-aware positive-authority "
                "contract"
            )

        if (
            gate.get("conditional_is_positive")
            is not False
        ):
            raise ValueError(
                "CONDITIONAL must not be positive authority"
            )

        if (
            gate.get("absence_is_novelty")
            is not False
        ):
            raise ValueError(
                "search-bounded absence must not become novelty"
            )

        if (
            gate.get(
                "candidate_semantics_preserved"
            )
            is not True
        ):
            raise ValueError(
                "candidate semantics must be preserved"
            )

    rows = gate.get(
        "gates"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise ValueError(
            "N10 gate rows must be a list"
        )

    matches = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and str(
                row.get(
                    "hypothesis_id"
                )
                or ""
            )
            == candidate_id
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "N10 post-generation gate must contain "
            "exactly one row for candidate "
            + candidate_id
        )

    row = matches[0]

    if (
        schema
        == "scientific-novelty-fallback-gate-v2"
    ):
        selection = row.get(
            "selection_class"
        )

        if selection not in {
            "ELIGIBLE",
            "CONDITIONAL",
            "INELIGIBLE",
        }:
            raise ValueError(
                "invalid role-aware N10 selection class"
            )

        positive = row.get(
            "positive_nonobviousness_authority"
        )

        fallback = row.get(
            "fallback_allowed"
        )

        if not isinstance(
            positive,
            bool,
        ):
            raise ValueError(
                "role-aware positive authority must be boolean"
            )

        if not isinstance(
            fallback,
            bool,
        ):
            raise ValueError(
                "role-aware fallback permission must be boolean"
            )

        if selection == "ELIGIBLE":
            if positive is not True:
                raise ValueError(
                    "ELIGIBLE role-aware candidate lacks "
                    "positive authority"
                )

            if fallback is not True:
                raise ValueError(
                    "ELIGIBLE role-aware candidate lacks "
                    "production permission"
                )

        else:
            if positive is not False:
                raise ValueError(
                    "non-ELIGIBLE role-aware candidate "
                    "cannot carry positive authority"
                )

            if fallback is not False:
                raise ValueError(
                    "non-ELIGIBLE role-aware candidate "
                    "cannot carry production permission"
                )

    return row


def filter_alpha6_portfolio_by_nonobviousness(
    *,
    portfolio: HypothesisPortfolio,
    refinement_report: NoveltyRefinementReport,
    gates_by_candidate_id: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    HypothesisPortfolio,
    dict[str, Any],
]:
    """Apply fresh N10 adjudication to Alpha6-generated survivors.

    kept_original:
        Already passed the pre-generation/original N10 fallback gate.

    accepted_refinement / accepted_reaxis:
        Must possess a fresh candidate-specific N10 production gate.
        Only POTENTIALLY_NON_OBVIOUS-derived ELIGIBLE authority survives.

    Other Alpha6 attempts:
        Must not claim final portfolio membership.
    """

    if (
        refinement_report.final_portfolio_id
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "Alpha6 report / portfolio mismatch before "
            "post-generation N10 enforcement"
        )

    cards_by_id = {
        card.hypothesis_id:
            card
        for card
        in portfolio.hypotheses
    }

    if (
        len(cards_by_id)
        != len(
            portfolio.hypotheses
        )
    ):
        raise ValueError(
            "duplicate hypothesis IDs in Alpha6 portfolio"
        )

    surviving_attempts = [
        attempt
        for attempt
        in refinement_report.attempts
        if attempt.final_hypothesis_id
        is not None
    ]

    final_attempt_ids = [
        str(
            attempt.final_hypothesis_id
        )
        for attempt
        in surviving_attempts
    ]

    if (
        len(final_attempt_ids)
        != len(
            set(
                final_attempt_ids
            )
        )
    ):
        raise ValueError(
            "duplicate final_hypothesis_id in Alpha6 report"
        )

    if (
        set(final_attempt_ids)
        != set(cards_by_id)
    ):
        raise ValueError(
            "Alpha6 survivor/report membership mismatch"
        )

    keep_ids: set[str] = set()
    decisions: list[
        dict[str, Any]
    ] = []

    consumed_candidates: set[
        str
    ] = set()

    consumed_gate_schemas: set[
        str
    ] = set()

    for attempt in surviving_attempts:
        final_id = str(
            attempt.final_hypothesis_id
        )

        candidate_id = str(
            attempt.candidate_hypothesis_id
            or ""
        )

        if (
            attempt.decision
            == "kept_original"
        ):
            keep_ids.add(
                final_id
            )

            decisions.append(
                {
                    "original_hypothesis_id":
                        attempt.original_hypothesis_id,
                    "candidate_hypothesis_id":
                        candidate_id,
                    "final_hypothesis_id":
                        final_id,
                    "alpha6_decision":
                        attempt.decision,
                    "n10_required":
                        False,
                    "n10_selection_class":
                        "PRE_GENERATION_GATE_ALREADY_PASSED",
                    "kept":
                        True,
                    "reason_codes": [
                        "original_survivor_already_passed_n10_fallback_gate",
                    ],
                }
            )

            continue

        if (
            attempt.decision
            not in _GENERATED_SURVIVOR_DECISIONS
        ):
            raise ValueError(
                "unsupported surviving Alpha6 decision: "
                + str(
                    attempt.decision
                )
            )

        if not candidate_id:
            raise ValueError(
                "generated Alpha6 survivor missing candidate ID"
            )

        gate = gates_by_candidate_id.get(
            candidate_id
        )

        if gate is None:
            raise ValueError(
                "missing fresh N10 gate for generated candidate "
                + candidate_id
            )

        row = _validate_n10_gate(
            candidate_id=candidate_id,
            gate=gate,
        )

        gate_schema = str(
            gate.get(
                "schema_version"
            )
            or ""
        )

        consumed_gate_schemas.add(
            gate_schema
        )

        consumed_candidates.add(
            candidate_id
        )

        selection_class = str(
            row.get(
                "selection_class"
            )
            or ""
        )

        fallback_allowed = (
            row.get(
                "fallback_allowed"
            )
        )

        if (
            gate_schema
            == "scientific-novelty-fallback-gate-v2"
        ):
            eligible = (
                selection_class
                == "ELIGIBLE"
                and row.get(
                    "positive_nonobviousness_authority"
                )
                is True
                and fallback_allowed
                is True
            )
        else:
            eligible = (
                selection_class
                == "ELIGIBLE"
                and fallback_allowed
                is True
            )

        if eligible:
            keep_ids.add(
                final_id
            )

        decisions.append(
            {
                "original_hypothesis_id":
                    attempt.original_hypothesis_id,
                "candidate_hypothesis_id":
                    candidate_id,
                "final_hypothesis_id":
                    final_id,
                "alpha6_decision":
                    attempt.decision,
                "n10_required":
                    True,
                "n10_selection_class":
                    selection_class,
                "n10_action":
                    row.get(
                        "action"
                    ),
                "kept":
                    eligible,
                "reason_codes":
                    list(
                        row.get(
                            "reason_codes"
                        )
                        or []
                    ),
            }
        )

    extra_gate_ids = (
        set(
            gates_by_candidate_id
        )
        - consumed_candidates
    )

    if extra_gate_ids:
        raise ValueError(
            "unused post-generation N10 gates: "
            + ",".join(
                sorted(
                    extra_gate_ids
                )
            )
        )

    selected_cards = [
        card
        for card
        in portfolio.hypotheses
        if card.hypothesis_id
        in keep_ids
    ]

    if selected_cards:
        abstention_reason = None
    elif not portfolio.hypotheses:
        # Alpha6 had already abstained before post-generation N10.
        # Preserve that upstream scientific/runtime explanation rather
        # than falsely attributing the empty portfolio to N10 removal.
        abstention_reason = portfolio.abstention_reason
    else:
        abstention_reason = (
            "All Alpha6 survivors were removed by fresh "
            "N10 post-generation non-obviousness enforcement."
        )

    portfolio_id = _stable_id(
        "hypothesis_portfolio",
        portfolio.domain_profile_id,
        portfolio.source_context_sha256,
        *(
            card.hypothesis_id
            for card
            in selected_cards
        ),
        abstention_reason or "",
    )

    result = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=(
            portfolio.domain_profile_id
        ),
        source_context_id=(
            portfolio.source_context_id
        ),
        source_context_sha256=(
            portfolio.source_context_sha256
        ),
        source_report_id=(
            portfolio.source_report_id
        ),
        source_report_sha256=(
            portfolio.source_report_sha256
        ),
        hypotheses=selected_cards,
        abstention_reason=(
            abstention_reason
        ),
    )

    if (
        not consumed_gate_schemas
        or consumed_gate_schemas
        == {
            "scientific-novelty-fallback-gate-v1"
        }
    ):
        report_schema_version = (
            "alpha6-post-generation-"
            "nonobviousness-enforcement-v1"
        )

        report_authority_source = (
            "n10_nonobviousness"
        )

        report_positive_requirement = (
            "POTENTIALLY_NON_OBVIOUS"
        )

        report_is_role_aware_v2 = False

    elif (
        consumed_gate_schemas
        == {
            "scientific-novelty-fallback-gate-v2"
        }
    ):
        report_schema_version = (
            "alpha6-post-generation-"
            "nonobviousness-enforcement-v2"
        )

        report_authority_source = (
            "n10_role_aware_nonobviousness_v2"
        )

        report_positive_requirement = (
            "ELIGIBLE_AND_ROLE_AWARE_"
            "POSITIVE_NONOBVIOUSNESS"
        )

        report_is_role_aware_v2 = True

    else:
        raise ValueError(
            "mixed post-generation N10 gate schemas "
            "are not permitted"
        )

    report = {
        "schema_version":
            report_schema_version,

        "source_alpha6_portfolio_id":
            portfolio.portfolio_id,

        "source_alpha6_refinement_report_id":
            refinement_report.report_id,

        "final_portfolio_id":
            result.portfolio_id,

        "generated_candidate_gate_count":
            len(
                gates_by_candidate_id
            ),

        "alpha6_survivor_count":
            len(
                portfolio.hypotheses
            ),

        "final_survivor_count":
            len(
                result.hypotheses
            ),

        "removed_by_post_generation_n10_count":
            len(
                portfolio.hypotheses
            )
            - len(
                result.hypotheses
            ),

        "decisions":
            decisions,

        "production_authority":
            True,

        "authority_source":
            report_authority_source,

        "generated_candidate_requires_fresh_n10":
            True,

        "positive_authority_requires":
            report_positive_requirement,
    }

    if report_is_role_aware_v2:
        report[
            "authority_scope"
        ] = (
            "alpha6_post_generation_candidate"
        )

        report[
            "conditional_is_positive"
        ] = False

        report[
            "absence_is_novelty"
        ] = False

        report[
            "candidate_semantics_preserved"
        ] = True

    return (
        result,
        report,
    )
