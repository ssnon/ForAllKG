from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    assess_residual_specification,
    extract_novelty_residue,
)


def compile_shadow_claim(
    claim: NoveltyResidueClaim,
) -> dict[str, object]:
    """Compile one claim into the N9 shadow-intake disposition.

    This stage deliberately stops before evidence closure.
    It must not infer scientific non-obviousness from residual status.
    """

    specification = assess_residual_specification(
        claim
    )

    if claim.disposition == "SATURATED":
        shadow_state = "SATURATED_PRIOR_ART"
        next_action = "NONE"

    elif claim.disposition == "UNRESOLVED_PARTIAL":
        shadow_state = "UNRESOLVED_PARTIAL"
        next_action = "RESOLVE_PARTIAL_PRIOR_ART"

    elif claim.disposition == "RESIDUAL":
        if (
            specification.status
            == "READY_FOR_CLOSURE"
        ):
            shadow_state = "READY_FOR_CLOSURE"
            next_action = "TARGETED_CLOSURE_REQUIRED"
        else:
            shadow_state = "NEEDS_REFINEMENT"
            next_action = (
                "REFINE_HYPOTHESIS_SPECIFICATION"
            )

    else:
        shadow_state = "UNRESOLVED"
        next_action = "REVIEW_EVIDENCE_STATE"

    return {
        "claim": asdict(claim),
        "specification": asdict(specification),
        "shadow_state": shadow_state,
        "next_action": next_action,

        # Explicitly prevent later consumers from mistaking
        # intake for completed N9 adjudication.
        "closure_status": (
            "PENDING_TARGETED_CLOSURE"
            if shadow_state == "READY_FOR_CLOSURE"
            else "NOT_RUN"
        ),
        "structural_status": "NOT_RUN",
        "adjudication_status": "NOT_RUN",
    }


def build_nonobviousness_shadow(
    *,
    plan: LiteratureQueryPlan,
    report: ExternalNoveltyReport,
) -> dict[str, object]:
    """Build shadow-only N9 residue/specification artifact."""

    if (
        plan.source_portfolio_id
        != report.source_portfolio_id
    ):
        raise ValueError(
            "N9 shadow provenance mismatch: "
            "query plan and external report refer "
            "to different source portfolios."
        )

    residues = extract_novelty_residue(
        plan,
        report,
    )

    cards = {
        card.hypothesis_id: card
        for card in report.cards
    }

    hypothesis_rows: list[dict[str, object]] = []
    states: Counter[str] = Counter()

    for residue in residues:
        card = cards.get(residue.hypothesis_id)

        decisions = [
            compile_shadow_claim(claim)
            for claim in residue.claims
        ]

        for decision in decisions:
            states[str(decision["shadow_state"])] += 1

        hypothesis_rows.append(
            {
                "hypothesis_id": residue.hypothesis_id,
                "title": (
                    card.title
                    if card is not None
                    else ""
                ),
                "external_status": (
                    residue.external_status
                ),
                "claims": decisions,
                "ready_for_closure_claim_ids": [
                    str(
                        row["claim"]["claim_id"]
                    )
                    for row in decisions
                    if (
                        row["shadow_state"]
                        == "READY_FOR_CLOSURE"
                    )
                ],
                "needs_refinement_claim_ids": [
                    str(
                        row["claim"]["claim_id"]
                    )
                    for row in decisions
                    if (
                        row["shadow_state"]
                        == "NEEDS_REFINEMENT"
                    )
                ],
            }
        )

    return {
        "schema_version": (
            "nonobviousness-shadow-v1"
        ),
        "shadow_only": True,
        "scientific_selection_changed": False,

        "source_portfolio_id": (
            report.source_portfolio_id
        ),
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": (
            plan.plan_sha256
        ),
        "source_external_report_id": (
            report.report_id
        ),
        "source_external_report_sha256": (
            report.report_sha256
        ),
        "source_prior_art_packet_id": (
            report.source_prior_art_packet_id
        ),

        "hypothesis_count": len(hypothesis_rows),
        "claim_count": sum(states.values()),
        "shadow_state_counts": dict(
            sorted(states.items())
        ),
        "hypotheses": hypothesis_rows,

        "epistemic_policy": {
            "residual_is_not_novelty": True,
            "missing_prior_art_is_not_positive_evidence": True,
            "under_specified_residue_requires_refinement": True,
            "ready_for_closure_is_not_nonobviousness": True,
        },
    }
