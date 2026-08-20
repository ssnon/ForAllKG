from types import SimpleNamespace

from domains.registry import get_domain_profile
from dac_her.novelty_gap_analysis import NoveltyGapAnalyzer


def _review(status, claim_id, text, importance="core", abstracts=3):
    return SimpleNamespace(
        status=status,
        claim_id=claim_id,
        claim_text=text,
        importance=importance,
        coverage=SimpleNamespace(abstract_work_count=abstracts),
        matches=[],
    )


def test_insufficient_prioritizes_unresolved_claim():
    card = SimpleNamespace(
        hypothesis_id="h1",
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        claim_reviews=[
            _review("PARTIAL_PRIOR_ART", "c1", "known relation"),
            _review("COMPONENTS_ONLY", "c2", "interaction not directly matched"),
        ],
        contextual_conflict_work_ids=[],
        coverage=SimpleNamespace(sufficient_for_absence_based_novelty=False),
    )
    portfolio = SimpleNamespace(
        portfolio_id="p1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                hypothesis_statement="hypothesis"
            )
        ],
    )
    external = SimpleNamespace(
        report_id="e1",
        source_portfolio_id="p1",
        cards=[card],
    )
    query_plan = SimpleNamespace(
        source_portfolio_id="p1",
        queries=[],
    )
    plan = NoveltyGapAnalyzer(
        max_target_claims=1, queries_per_gap=2,
        domain_profile=get_domain_profile("dac_her"),
    ).build(portfolio, external, query_plan)
    assert plan.gaps[0].action == "targeted_search_then_refine"
    assert plan.gaps[0].target_claim_ids == ["c2"]
    assert plan.gaps[0].targeted_queries
