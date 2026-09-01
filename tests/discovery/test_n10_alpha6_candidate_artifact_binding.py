from pathlib import Path

from scripts.discovery.enforce_alpha6_nonobviousness import (
    find_final_external_triplet,
    generated_candidate_ids,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
    RefinementAttempt,
)


def _attempt(
    decision,
    candidate,
):
    return RefinementAttempt(
        original_hypothesis_id="hypothesis:o",
        candidate_hypothesis_id=candidate,
        final_hypothesis_id=(
            (
                "hypothesis:final-"
                + candidate.split(":")[-1]
            )
            if decision.startswith("accepted_")
            else None
        ),
        gap_id="gap:g",
        action="targeted_search_then_refine",
        decision=decision,
        original_external_status="PLAUSIBLY_NOVEL",
        targeted_external_status="PLAUSIBLY_NOVEL",
        final_external_status="PLAUSIBLY_NOVEL",
        grounding_preserved=True,
        refinement_generated=True,
        interpretation="control",
    )


def _report(
    attempts,
):
    return NoveltyRefinementReport(
        report_id="report:r",
        report_sha256="sha",
        source_portfolio_id="portfolio:p",
        source_external_report_id="external:r",
        source_gap_plan_id="gap:g",
        final_portfolio_id="portfolio:f",
        attempts=attempts,
        targeted_searches=[],
        accepted_refinement_count=sum(
            row.decision
            == "accepted_refinement"
            for row in attempts
        ),
        accepted_reaxis_count=sum(
            row.decision
            == "accepted_reaxis"
            for row in attempts
        ),
        kept_original_count=0,
        rejected_count=sum(
            row.decision
            not in {
                "accepted_refinement",
                "accepted_reaxis",
            }
            for row in attempts
        ),
        max_refinements_per_hypothesis=1,
        max_reaxes_per_hypothesis=1,
        external_prior_art_can_be_positive_premise=False,
        policy_version="novelty-refinement-policy-v2",
    )


def test_only_generated_alpha6_survivors_require_fresh_n10():
    report = _report(
        [
            _attempt(
                "accepted_refinement",
                "hypothesis:c1",
            ),
            _attempt(
                "accepted_reaxis",
                "hypothesis:c2",
            ),
            _attempt(
                "external_novelty_rejected",
                "hypothesis:c3",
            ),
        ]
    )

    assert generated_candidate_ids(
        report
    ) == (
        "hypothesis:c1",
        "hypothesis:c2",
    )
