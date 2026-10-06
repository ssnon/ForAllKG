from pipeline_core.discovery.exploratory_portfolio_search_v1 import (
    ExploratoryCandidateRecord,
    ExploratoryParentState,
    ExploratoryPortfolioPlan,
    allocated_graph_parent_ids,
    select_exploratory_portfolio,
)


def parent(hid, prospective, quality, uncertainty, graph):
    return ExploratoryParentState(
        hypothesis_id=hid,
        prospective_identifiability=prospective,
        current_epistemic_state="UNRESOLVED_EVIDENCE_GAP",
        external_status="INSUFFICIENT_SEARCH_EVIDENCE",
        safe_unused_premise_statement_ids=["s3"],
        reproductive_eligible=prospective != "NOT_OPERATIONALIZABLE",
        reproductive_block_reason=None if prospective != "NOT_OPERATIONALIZABLE" else "NOT_OPERATIONALIZABLE_PARENT_CANNOT_REPRODUCE_DIRECTLY",
        quality_prior=quality,
        uncertainty=uncertainty,
        underexplored_bonus=1.0,
        verification_cost_prior=.2,
        search_priority=quality + uncertainty,
        graph_retraversal_priority=graph,
        branch_actions=["EVIDENCE_REAXIS", "AXIS_MUTATION"],
    )


def plan(parents):
    return ExploratoryPortfolioPlan(
        plan_id="eps:test",
        source_portfolio_id="portfolio:test",
        source_context_id="context:test",
        parent_states=parents,
        max_local_branches_per_parent=3,
        max_graph_retraversal_allocations=2,
        exploration_temperature=.9,
    )


def record(hid, route, prospective, family, q=.6, u=.6, n=.5):
    return ExploratoryCandidateRecord(
        hypothesis_id=hid,
        source_hypothesis_id="root:"+hid,
        route=route,
        prospective_identifiability=prospective,
        current_epistemic_state="UNRESOLVED_EVIDENCE_GAP",
        external_status="INSUFFICIENT_SEARCH_EVIDENCE",
        premise_statement_ids=["s1", hid],
        hypothesis_type="mechanistic",
        title=hid,
        hypothesis_statement="statement "+hid,
        family_signature=family,
        reproductive_eligible=prospective != "NOT_OPERATIONALIZABLE",
        quality_prior=q,
        uncertainty=u,
        novelty_of_search_path=n,
        verification_cost_prior=.2,
        ucb_exploration_bonus=1.0,
        base_search_value=q+u+n,
    )


def test_not_operationalizable_is_sterile_and_graph_prioritized():
    p = plan([
        parent("h1", "NOT_OPERATIONALIZABLE", .1, 1.0, 1.4),
        parent("h2", "PROSPECTIVELY_IDENTIFIABLE", .8, .6, .7),
        parent("h3", "CURRENTLY_IDENTIFIED", .9, .2, .3),
    ])
    assert allocated_graph_parent_ids(p)[0] == "h1"
    assert p.parent_states[0].reproductive_eligible is False


def test_selection_excludes_not_operationalizable_and_retains_wildcard():
    p = plan([parent("root", "PROSPECTIVELY_IDENTIFIABLE", .8, .6, .5)])
    rows = [
        record("bad", "KEEP_ELITE", "NOT_OPERATIONALIZABLE", "f0", 1, 1, 1),
        record("elite", "KEEP_ELITE", "CURRENTLY_IDENTIFIED", "f1", .95, .1, 0),
        record("pro", "EVIDENCE_REAXIS", "PROSPECTIVELY_IDENTIFIABLE", "f2", .7, .7, .5),
        record("wild", "AXIS_MUTATION", "PROSPECTIVELY_IDENTIFIABLE", "f3", .55, .8, .9),
    ]
    s = select_exploratory_portfolio(
        plan=p,
        records=rows,
        max_retained_candidates=3,
        slot_quotas={"EXPLOIT":1,"PROSPECTIVE":1,"UNCERTAINTY":0,"WILDCARD":1},
    )
    assert "bad" not in s.retained_candidate_ids
    assert "elite" in s.retained_candidate_ids
    assert "wild" in s.retained_candidate_ids
    assert s.search_parent_selection_authority is True
    assert s.production_selection_authority is False


def test_family_diversity_blocks_duplicate_capture():
    p = plan([parent("root", "PROSPECTIVELY_IDENTIFIABLE", .8, .6, .5)])
    rows = [
        record("h1", "KEEP_ELITE", "CURRENTLY_IDENTIFIED", "same", .9),
        record("h2", "SAME_PREMISE_SHARPEN", "PROSPECTIVELY_IDENTIFIABLE", "same", .85),
        record("h3", "AXIS_MUTATION", "PROSPECTIVELY_IDENTIFIABLE", "different", .6, .7, .9),
    ]
    s = select_exploratory_portfolio(
        plan=p,
        records=rows,
        max_retained_candidates=2,
        slot_quotas={"EXPLOIT":1,"PROSPECTIVE":0,"UNCERTAINTY":0,"WILDCARD":1},
    )
    assert not ("h1" in s.retained_candidate_ids and "h2" in s.retained_candidate_ids)


def test_search_authority_does_not_grant_production_authority():
    p = plan([parent("h1", "PROSPECTIVELY_IDENTIFIABLE", .7, .7, .6)])
    assert p.parallel_branching_authority is True
    assert p.search_parent_selection_authority is True
    assert p.compute_allocation_authority is True
    assert p.graph_retraversal_allocation_authority is True
    assert p.scientific_truth_authority is False
    assert p.literature_wide_novelty_authority is False
    assert p.production_selection_authority is False
    assert p.stage8_input_changed is False
    assert p.canonical_graph_mutated is False
