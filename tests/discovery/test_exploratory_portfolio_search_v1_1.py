from pipeline_core.discovery.exploratory_portfolio_search_v1 import (
    ExploratoryCandidateRecord,
)
from pipeline_core.discovery.exploratory_portfolio_search_v1_1 import (
    reserve_diverse_search_portfolio,
)


def row(
    hid,
    route,
    *,
    prospective="PROSPECTIVELY_IDENTIFIABLE",
    family=None,
    q=.6,
    u=.6,
    n=.5,
    reproductive=True,
):
    return ExploratoryCandidateRecord(
        hypothesis_id=hid,
        source_hypothesis_id="root",
        route=route,
        prospective_identifiability=prospective,
        current_epistemic_state="UNRESOLVED_EVIDENCE_GAP",
        external_status="INSUFFICIENT_SEARCH_EVIDENCE",
        premise_statement_ids=["s", hid],
        hypothesis_type="mechanistic_extension",
        title=hid,
        hypothesis_statement="statement " + hid,
        family_signature=family or ("f_" + hid),
        reproductive_eligible=reproductive,
        quality_prior=q,
        uncertainty=u,
        novelty_of_search_path=n,
        verification_cost_prior=.2,
        ucb_exploration_bonus=1.0,
        base_search_value=q + u + n,
    )


def test_small_population_reserves_exploration():
    rows = [
        row("elite", "KEEP_ELITE", q=.95, u=.1, n=0),
        row("reaxis", "EVIDENCE_REAXIS", q=.7, u=.7, n=.45),
        row("axis", "AXIS_MUTATION", q=.55, u=.85, n=.8),
    ]
    result = reserve_diverse_search_portfolio(
        rows,
        max_retained_candidates=3,
    )
    slots = set(result["retained_count_by_slot"])
    assert "EXPLOIT" in slots
    assert "WILDCARD" in slots
    assert "UNCERTAINTY" in slots
    assert len(result["retained_candidate_ids"]) == 3


def test_sterile_not_operationalizable_incumbent_is_not_selected():
    rows = [
        row(
            "sterile",
            "KEEP_ELITE",
            prospective="NOT_OPERATIONALIZABLE",
            q=1,
            u=1,
            n=1,
            reproductive=False,
        ),
        row("axis", "AXIS_MUTATION", q=.6, u=.8, n=.8),
    ]
    result = reserve_diverse_search_portfolio(rows, max_retained_candidates=2)
    assert "sterile" not in result["retained_candidate_ids"]
    assert "axis" in result["retained_candidate_ids"]


def test_family_capture_is_blocked():
    rows = [
        row("a", "KEEP_ELITE", family="same", q=.9),
        row("b", "AXIS_MUTATION", family="same", q=.8, n=.9),
        row("c", "EVIDENCE_REAXIS", family="other", q=.7, n=.5),
    ]
    result = reserve_diverse_search_portfolio(rows, max_retained_candidates=3)
    assert not (
        "a" in result["retained_candidate_ids"]
        and "b" in result["retained_candidate_ids"]
    )


def test_production_authority_remains_false():
    result = reserve_diverse_search_portfolio(
        [row("a", "KEEP_ELITE")],
        max_retained_candidates=1,
    )
    assert result["search_parent_selection_authority"] is True
    assert result["compute_allocation_authority"] is True
    assert result["scientific_truth_authority"] is False
    assert result["production_selection_authority"] is False
    assert result["stage8_input_changed"] is False
    assert result["canonical_graph_mutated"] is False
