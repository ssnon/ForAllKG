from pipeline_core.discovery.scientific_portfolio_audit import (
    ScientificPortfolioAudit,
    build_scientific_portfolio_cohort_audit,
)


def _audit(i, materialized):
    return ScientificPortfolioAudit(
        audit_id=f"a{i}", audit_sha256=f"s{i}", source_pool_id=f"p{i}",
        source_evaluation_report_id=f"e{i}", source_selection_id=f"sel{i}",
        source_materialization_report_id=f"m{i}", raw_frontier_idea_count=10,
        raw_evolution_idea_count=5, projected_candidate_count=8,
        evaluated_candidate_count=8, retained_candidate_count=4,
        retained_unique_family_count=4, materialized_hypothesis_count=materialized,
        materialization_success_fraction=materialized/4,
        exploration_retained_candidate_count=4,
        verification_ready_hypothesis_count=materialized,
        verification_ready_fraction_of_exploration=materialized/4,
        normalized_scientific_sketch_count=8,
        retained_count_by_profile={"EXPLORATORY_BRIDGE":1},
        retained_count_by_origin={"FRONTIER":2,"EVOLUTION":2},
        verification_burden_counts={"HIGH":2}, retained_verification_burden_counts={"HIGH":1},
        frontier_candidate_retained=True, evolution_candidate_retained=True,
        exploratory_bridge_retained=True, reframe_retained=False,
        downstream_verification_ready=materialized>0,
    )


def test_cohort_audit_does_not_rank_cases():
    cohort = build_scientific_portfolio_cohort_audit([("c1", _audit(1,3)), ("c2", _audit(2,2))])
    assert cohort.case_count == 2
    assert cohort.total_materialized_hypothesis_count == 5
    assert cohort.total_exploration_retained_candidate_count == 8
    assert cohort.total_verification_ready_hypothesis_count == 5
    assert cohort.cross_case_winner_selected is False
    assert cohort.production_selection_authority is False
