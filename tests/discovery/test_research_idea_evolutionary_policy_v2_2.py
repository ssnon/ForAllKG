from types import SimpleNamespace as NS

from pipeline_core.discovery.research_idea_evolutionary_policy import (
    EvolutionSlotBudget,
    build_evolutionary_idea_search_shadow,
    compile_evolution_policy_states,
    derive_slot_budget,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    IdeaOutcomeObservation,
)


def _obs(
    idea_id: str,
    *,
    status: str = "MATERIALIZED",
    residual: str | None = None,
    prospective: str | None = None,
    integrity: bool | None = True,
):
    return IdeaOutcomeObservation(
        observation_id=f"obs:{idea_id}",
        idea_id=idea_id,
        target_scope="REALIZATION",
        source_systems=["TEST_VERIFIER"],
        source_versions=["v1"],
        candidate_id=f"c:{idea_id}",
        hypothesis_id=f"h:{idea_id}" if status == "MATERIALIZED" else None,
        materialization_status=status,
        residual_epistemic_state=residual,
        prospective_identifiability=prospective,
        prospective_contract_integrity_passed=integrity,
    )


def _candidate(idea_id: str, *, priority="MEDIUM", bonus=1.0, profiles=None):
    return NS(
        idea_id=idea_id,
        eligible_profiles=list(profiles or ["TASK_NEAR_VALIDATION"]),
        pareto_layer_by_profile={"TASK_NEAR_VALIDATION": 1},
        dimension_levels={
            "task_relevance": 3,
            "mechanistic_coherence": 3,
            "falsifiability": 3,
            "discriminating_power": 3,
            "operationalizability": 3,
            "information_gain": 3,
        },
        verification_burden="MODERATE",
        priority_band=priority,
        underexplored_bonus=bonus,
    )


def _report(observations, idea_ids, *, old_parents=None, candidates=None):
    return NS(
        report_id="sis:v211",
        report_sha256="a" * 64,
        observations=list(observations),
        research_ideas=[
            NS(idea_id=idea_id, kernel_sha256=(f"{idx:064x}"))
            for idx, idea_id in enumerate(idea_ids, start=1)
        ],
        candidate_states=list(candidates or [_candidate(x) for x in idea_ids]),
        g2_parent_idea_ids=list(old_parents or []),
        g2_parent_count=len(old_parents or []),
    )


def test_observation_scope_is_typed_and_realization_scoped_by_default():
    row = _obs("i1")
    assert row.target_scope == "REALIZATION"
    assert row.search_policy_authority is False


def test_prior_art_reduces_exploit_but_increases_transformation_not_termination():
    report = _report(
        [_obs("i1", residual="PRIOR_ART_BACKED_OR_NO_RESIDUAL")],
        ["i1"],
    )
    policy = compile_evolution_policy_states(report)[0]
    assert policy.realization_viability == "LOW"
    assert policy.exploit_value == "LOW"
    assert policy.transformation_pressure == "HIGH"
    assert policy.exploration_value == "HIGH"
    assert policy.idea_reproductive_value == "ENCOURAGED"
    assert "AXIS_MUTATION" in policy.operator_hints
    assert "PRIOR_ART_BOUNDARY_INCREASES_MUTATIONAL_PRESSURE" in policy.reason_codes


def test_not_operationalizable_is_transform_signal_not_dead_idea():
    report = _report(
        [_obs("i1", prospective="NOT_OPERATIONALIZABLE")],
        ["i1"],
    )
    policy = compile_evolution_policy_states(report)[0]
    assert policy.realization_viability == "LOW"
    assert policy.idea_reproductive_value == "ENCOURAGED"
    assert policy.transformation_pressure == "HIGH"
    assert "LATENT_VARIABLE" in policy.operator_hints
    assert "REGIME_BOUNDARY" in policy.operator_hints


def test_materialization_failure_does_not_become_idea_level_hard_constraint():
    report = _report(
        [_obs("i1", status="HARD_GATE_REJECTED", integrity=None)],
        ["i1"],
    )
    policy = compile_evolution_policy_states(report)[0]
    assert policy.idea_reproductive_value != "TERMINATED"
    assert policy.hard_constraint_codes == []
    assert policy.soft_feedback_only is True
    assert policy.exploration_value == "HIGH"


def test_only_explicit_hard_idea_constraint_can_terminate():
    report = _report([_obs("i1")], ["i1"])
    policy = compile_evolution_policy_states(
        report,
        hard_constraint_codes_by_idea={"i1": ["FABRICATED_PROVENANCE"]},
    )[0]
    assert policy.idea_reproductive_value == "TERMINATED"
    assert policy.hard_constraint_codes == ["FABRICATED_PROVENANCE"]
    assert policy.soft_feedback_only is False


def test_default_budget_for_eight_is_two_three_two_one():
    budget = derive_slot_budget(8)
    assert budget.exploit_slots == 2
    assert budget.transform_slots == 3
    assert budget.explore_slots == 2
    assert budget.wildcard_slots == 1


def test_transform_lane_can_rescue_v211_low_priority_failed_realization():
    observations = [
        _obs("exploit", prospective="CURRENTLY_IDENTIFIED"),
        _obs("transform_prior", residual="PRIOR_ART_BACKED_OR_NO_RESIDUAL"),
        _obs("transform_nonop", prospective="NOT_OPERATIONALIZABLE"),
        _obs("explore", residual="UNRESOLVED_EVIDENCE_GAP"),
        _obs("old_only", prospective="CURRENTLY_IDENTIFIED"),
    ]
    idea_ids = ["exploit", "transform_prior", "transform_nonop", "explore", "old_only"]
    candidates = [
        _candidate("exploit", priority="HIGH"),
        _candidate("transform_prior", priority="LOW"),
        _candidate("transform_nonop", priority="MEDIUM"),
        _candidate("explore", priority="MEDIUM", bonus=2.0),
        _candidate("old_only", priority="HIGH"),
    ]
    source = _report(
        observations,
        idea_ids,
        old_parents=["exploit", "old_only", "explore", "transform_nonop"],
        candidates=candidates,
    )
    budget = EvolutionSlotBudget(
        max_parent_budget=4,
        exploit_slots=1,
        transform_slots=2,
        explore_slots=1,
        wildcard_slots=0,
    )
    result = build_evolutionary_idea_search_shadow(
        source,
        max_parent_budget=4,
        slot_budget=budget,
    )
    transform_rows = [row for row in result.allocations if row.channel == "TRANSFORM"]
    assert len(transform_rows) == 2
    assert any(row.idea_id == "transform_prior" for row in transform_rows)
    assert result.selected_low_viability_transform_count >= 1
    assert result.selected_v2_1_1_low_priority_count >= 1
    assert result.parent_set_changed is True
    assert "transform_prior" in result.parent_added_idea_ids
    assert result.soft_feedback_cannot_terminate_idea is True
    assert result.verification_is_not_fertility_authority is True
    assert result.single_scalar_fitness_used is False


def test_transform_explore_and_wildcard_do_not_require_legacy_profile_eligibility():
    observations = [
        _obs("transform", prospective="NOT_OPERATIONALIZABLE"),
        _obs("explore", status="ABSTAINED", integrity=None),
        _obs("wild", status="ABSTAINED", integrity=None),
    ]
    candidates = [
        _candidate("transform", priority="LOW", profiles=[]),
        _candidate("explore", priority="MEDIUM", profiles=[]),
        _candidate("wild", priority="MEDIUM", profiles=[]),
    ]
    for row in candidates:
        row.eligible_profiles = []
        row.pareto_layer_by_profile = {}
    source = _report(
        observations,
        ["transform", "explore", "wild"],
        old_parents=[],
        candidates=candidates,
    )
    budget = EvolutionSlotBudget(
        max_parent_budget=3,
        exploit_slots=0,
        transform_slots=1,
        explore_slots=1,
        wildcard_slots=1,
    )
    result = build_evolutionary_idea_search_shadow(
        source,
        max_parent_budget=3,
        slot_budget=budget,
    )
    assert result.selected_parent_count == 3
    assert set(result.selected_parent_channel_counts) == {
        "EXPLORE",
        "TRANSFORM",
        "WILDCARD",
    }
    assert result.transform_lane_not_profile_gated is True
    assert result.explore_lane_not_profile_gated is True
    assert result.wildcard_lane_not_profile_gated is True
