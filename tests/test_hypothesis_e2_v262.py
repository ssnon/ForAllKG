from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_semantic_contracts import SEMANTIC_DIMENSIONS
from scripts.build_hypothesis_v262_e2_contexts import build_e2_contexts


def _gold_expectations():
    return [
        {
            "dimension": dimension,
            "allowed_verdicts": [
                "not_applicable"
                if dimension in {"candidate_calibration", "hypothesis_distinctness"}
                else "pass"
            ],
            "critical": True,
            "note": "fixture",
        }
        for dimension in SEMANTIC_DIMENSIONS
    ]


def _base_spec() -> HypothesisRealGoldSpec:
    return HypothesisRealGoldSpec.model_validate(
        {
            "suite_id": "base",
            "cases": [
                {
                    "case_id": "k9",
                    "description": "base K9",
                    "context_path": "k9.context.json",
                    "portfolio_path": "k9.portfolio.json",
                    "expectations": _gold_expectations(),
                }
            ],
        }
    )


def _worksheet(*, approved: bool) -> HypothesisE2HumanReviewWorksheet:
    cases = []
    for scenario in ("candidate", "alignment", "partial", "abstention"):
        dimensions = [
            HypothesisE2HumanDimension(
                dimension=dimension,
                critic_verdict=(
                    "not_applicable"
                    if dimension == "candidate_calibration" and scenario != "candidate"
                    else "pass"
                ),
                critic_rationale="fixture rationale",
                human_allowed_verdicts=["pass"],
                critical=True,
                human_note="human reviewed",
            )
            for dimension in SEMANTIC_DIMENSIONS
        ]
        cases.append(
            HypothesisE2HumanReviewCase(
                case_id=f"e2_{scenario}",
                scenario=scenario,
                description=f"{scenario} fixture",
                review_hint="fixture hint",
                context_path=f"data/{scenario}.context.json",
                portfolio_path=f"data/{scenario}.portfolio.json",
                review_path=f"reviews/{scenario}.json",
                generator_version="hypothesis-maker-v2.6.1",
                approval_status="approved" if approved else "pending",
                dimensions=dimensions,
            )
        )
    return HypothesisE2HumanReviewWorksheet(
        suite_id="worksheet",
        cases=cases,
    )


def test_e2_controlled_contexts_cover_required_scenarios():
    rows = build_e2_contexts()
    by_scenario = {case.scenario: (case, context) for case, context in rows}
    assert set(by_scenario) == {
        "candidate",
        "alignment",
        "partial",
        "abstention",
    }

    _, candidate = by_scenario["candidate"]
    assert any(
        row.requires_verification and row.eligible_as_premise
        for row in candidate.evidence_statements
    )

    _, alignment = by_scenario["alignment"]
    assert any(route.uses_alignment for route in alignment.mechanism_routes)
    assert len(
        {
            paper
            for row in alignment.evidence_statements
            if row.eligible_as_premise
            for paper in row.paper_ids
        }
    ) >= 2

    _, partial = by_scenario["partial"]
    assert "Kiwook_10" in partial.partial_absence_blocked_paper_ids
    assert any(
        row.eligible_as_premise and "Kiwook_10" in row.paper_ids
        for row in partial.evidence_statements
    )

    _, abstention = by_scenario["abstention"]
    assert not any(row.eligible_as_premise for row in abstention.evidence_statements)






