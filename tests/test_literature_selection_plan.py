from pathlib import Path

from dac_her.literature_discovery import load_query_plan, load_selection_plan, scaled_bucket_quotas


ROOT = Path(__file__).resolve().parents[1]


def test_broad_selection_plan_matches_query_plan_and_expected_100_paper_quotas():
    query = load_query_plan(ROOT / "configs/literature/broad_catalysis_v1.yaml")
    selection = load_selection_plan(
        ROOT / "configs/literature/broad_catalysis_selection_v1.yaml"
    )
    assert selection.query_plan_id == query.plan_id
    assert selection.target_count == 100
    assert selection.max_abstract_chars == 9000
    assert scaled_bucket_quotas(query, target_count=100) == {
        "working_state_reconstruction": 18,
        "elementary_step_kinetics": 18,
        "interfacial_environment": 15,
        "structural_landscape": 15,
        "active_site_attribution": 10,
        "cooperative_bifunctional_mechanisms": 10,
        "descriptor_failure_counterexamples": 7,
        "cross_reaction_atomic_site_mechanisms": 7,
    }
