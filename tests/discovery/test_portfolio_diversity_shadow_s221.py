from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityReport,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGap,
    NoveltyGapPlan,
)
from pipeline_core.discovery.portfolio_diversity_shadow import (
    build_portfolio_diversity_shadow_report,
)


def _evidence(
    *,
    duplicates=1,
    max_jaccard=1.0,
    unused=True,
):
    return HypothesisEvidenceDiversityReport.model_construct(
        report_id="evidence:1",
        report_sha256="a" * 64,
        source_context_id="context:1",
        source_context_sha256="b" * 64,
        source_portfolio_id="portfolio:1",
        source_portfolio_sha256="c" * 64,
        hypothesis_count=3,
        eligible_statement_count=6,
        used_statement_count=(5 if unused else 6),
        eligible_statement_coverage=(5 / 6 if unused else 1.0),
        used_statement_ids=["A", "B", "C", "D", "E"],
        unused_eligible_statement_ids=(
            ["F"]
            if unused
            else []
        ),
        shared_core_statement_ids=["A", "B"],
        shared_core_statement_count=2,
        statement_usage_counts={},
        distinct_premise_set_count=2,
        exact_premise_set_duplicate_group_count=duplicates,
        mean_pairwise_statement_jaccard=0.7,
        max_pairwise_statement_jaccard=max_jaccard,
        multi_paper_used_statement_count=0,
        mean_papers_per_used_statement=1.0,
        pairwise_overlaps=[],
        exact_premise_set_groups=[],
        statement_usage=[],
        cards=[],
        diagnostic_only=True,
        scientific_selection_changed=False,
    )


def _gap_plan():
    gap = NoveltyGap(
        gap_id="gap:1",
        hypothesis_id="h1",
        source_external_status="LITERATURE_SUPPORTED_EXTENSION",
        action="gap_sharpen",
        target_claim_ids=["claim:1"],
        differentiator="test",
        sharpening_operators=[
            "MODERATOR",
            "BOUNDARY",
        ],
    )

    return NoveltyGapPlan.model_construct(
        plan_id="plan:1",
        plan_sha256="d" * 64,
        source_portfolio_id="portfolio:1",
        source_external_report_id="external:1",
        gaps=[gap],
        policy_version="novelty-gap-policy-v2",
    )


def test_redundancy_plus_unused_evidence_is_capacity_not_selection_authority():
    report = build_portfolio_diversity_shadow_report(
        evidence=_evidence(),
        novelty_gap_plan=_gap_plan(),
    )

    assert report.evidence_recommendation == "ALTERNATIVE_CAPACITY_PRESENT"
    assert report.operator_recommendation == "MULTI_OPERATOR_OPPORTUNITY"

    assert report.evidence_alternative_relevance_established is False
    assert report.operator_usage_inferred_from_hypothesis_text is False
    assert report.scientific_selection_changed is False
    assert report.production_selection_authority is False
    assert report.alpha6_trigger_authority is False


def test_redundancy_without_unused_evidence_only_requests_review():
    report = build_portfolio_diversity_shadow_report(
        evidence=_evidence(
            unused=False,
        ),
    )

    assert report.evidence_recommendation == "REDUNDANCY_REVIEW"
    assert report.operator_recommendation == "NO_OPERATOR_SIGNAL"


def test_no_redundancy_does_not_reward_diversity_for_its_own_sake():
    report = build_portfolio_diversity_shadow_report(
        evidence=_evidence(
            duplicates=0,
            max_jaccard=0.4,
            unused=True,
        ),
    )

    assert report.evidence_recommendation == "NO_ACTION"
    assert "UNUSED_ELIGIBLE_EVIDENCE_PRESENT" in report.reason_codes
