from __future__ import annotations

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaKernel, ResearchIdeaNode
from pipeline_core.discovery.research_idea_program_family_v3_3_1 import (
    ScientificProgramPairAssessment,
    ScientificProgramPairBatch,
    build_scientific_program_family_report,
)
from pipeline_core.discovery.research_idea_search_assessment import (
    ResearchIdeaNoveltySignal,
    ResearchIdeaSearchAssessmentItem,
    ResearchIdeaSearchAssessmentReport,
    ResearchIdeaSearchDecision,
)
from pipeline_core.discovery.research_idea_search_pressure_v3_3_2 import (
    ResearchIdeaNoveltyDepthSignal,
    build_bounded_parent_schedule,
    build_research_idea_search_pressure_report,
)


def _node(idea_id: str) -> ResearchIdeaNode:
    kernel = ResearchIdeaKernel(
        canonical_intent=f"intent {idea_id}",
        core_scientific_commitments=[f"A --REL--> {idea_id}"],
    )
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=2,
        source_context_id="ctx",
        source_context_sha256="sha",
        origin_kind="GENERATIONAL_OFFSPRING",
        source_object_id=idea_id,
        kernel=kernel,
        kernel_sha256="0" * 64,
        task_relation_mode="DIRECT",
        differential_prediction="prediction",
        falsification_condition="falsifier",
        discriminating_observation="discriminator",
    )


def _assessment_item(idea_id: str, *, grounded: int = 1) -> ResearchIdeaSearchAssessmentItem:
    return ResearchIdeaSearchAssessmentItem(
        idea_id=idea_id,
        idea_birth_generation_index=2,
        cycle_generation_index=2,
        remains_active=True,
        realization_count=1,
        usable_grounded_realization_count=grounded,
        failed_or_abstained_count=0,
        not_operationalizable_count=0,
        local_search_budget=2,
        local_search_budget_used=1,
        local_search_exhausted=False,
        active_maturity_counts={"STRICT_GROUNDED": grounded} if grounded else {"IDEA_ONLY": 1},
        active_grounded_member_count=grounded,
        active_non_grounded_member_count=0 if grounded else 1,
        speculative_member_count=0,
        epistemic_debt_ids=[],
        epistemic_debt_kinds=[],
        acquisition_escalation_eligible_debt_count=0,
        program_family_key=f"old:{idea_id}",
        program_family_size=1,
        tight_neighborhood_key=f"old-tight:{idea_id}",
        tight_neighborhood_size=1,
        family_pressure="ISOLATED",
        differential_prediction_present=True,
        falsification_condition_present=True,
        discriminating_observation_present=True,
        discrimination_sketch_complete=True,
        novelty=ResearchIdeaNoveltySignal(
            idea_id=idea_id,
            novelty_state="RESIDUAL",
            direct_prior_art_claim_count=0,
            partial_prior_art_claim_count=0,
            residual_claim_count=1,
            unresolved_claim_count=0,
        ),
    )


def _assessment_report(ids: list[str]) -> ResearchIdeaSearchAssessmentReport:
    assessments = [_assessment_item(i) for i in ids]
    decisions = [
        ResearchIdeaSearchDecision(
            decision_id=f"d:{i}",
            idea_id=i,
            idea_birth_generation_index=2,
            cycle_generation_index=2,
            persistence="RETAIN",
            search_action="EVOLVE",
            search_complete_candidate=False,
        )
        for i in ids
    ]
    return ResearchIdeaSearchAssessmentReport(
        report_id="r",
        report_sha256="s",
        generation_index=2,
        source_parallel_report_id="p",
        source_lifecycle_report_id="l",
        source_family_calibration_report_id="f",
        novelty_source_mode="RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        assessments=assessments,
        decisions=decisions,
        assessment_count=len(assessments),
        decision_count=len(decisions),
        persistence_counts={"RETAIN": len(ids)},
        search_action_counts={"EVOLVE": len(ids)},
        novelty_state_counts={"RESIDUAL": len(ids)},
        family_pressure_counts={"ISOLATED": len(ids)},
    )


def _novelty(idea_id: str, shape: str) -> ResearchIdeaNoveltyDepthSignal:
    return ResearchIdeaNoveltyDepthSignal(
        idea_id=idea_id,
        novelty_shape=shape,
        search_coverage_sufficient=shape not in {"NOT_ASSESSED", "UNRESOLVED"},
        core_claim_count=1,
        relation_backed_core_claim_count=0,
        known_core_relation_fraction_max=0.0,
        novelty_bearing_claim_count=1,
        novelty_bearing_gap_like_claim_count=1,
        novelty_bearing_relation_backed_claim_count=0,
        novelty_bearing_conflicting_claim_count=0,
        novelty_bearing_unresolved_claim_count=0,
    )


def test_semantic_program_family_groups_same_program_without_touching_idea_identity():
    nodes = [_node("a"), _node("b"), _node("c")]
    batch = ScientificProgramPairBatch(
        pairs=[
            ScientificProgramPairAssessment(
                idea_id_a="a", idea_id_b="b", relation="SAME_PROGRAM", rationale="same causal program"
            ),
            ScientificProgramPairAssessment(
                idea_id_a="a", idea_id_b="c", relation="ADJACENT_PROGRAM", rationale="same question, different perturbation"
            ),
            ScientificProgramPairAssessment(
                idea_id_a="b", idea_id_b="c", relation="ADJACENT_PROGRAM", rationale="same question, different perturbation"
            ),
        ]
    )
    report = build_scientific_program_family_report(nodes=nodes, pair_batch=batch)
    by_id = {row.idea_id: row for row in report.assignments}
    assert by_id["a"].scientific_program_key == by_id["b"].scientific_program_key
    assert by_id["a"].same_program_size == 2
    assert by_id["c"].same_program_size == 1
    assert by_id["c"].adjacent_program_neighbor_count == 2


def test_strong_residue_can_be_terminal_without_forcing_evolution():
    ids = ["a"]
    family = build_scientific_program_family_report(
        nodes=[_node("a")],
        pair_batch=ScientificProgramPairBatch(pairs=[]),
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(ids),
        program_family=family,
        novelty_by_idea={"a": _novelty("a", "STRONG_RESIDUE")},
    )
    row = pressure.pressures[0]
    assert row.terminal_worthy is True
    assert row.preferred_action == "NONE"
    assert row.search_worthy is False


def test_unresolved_prior_art_prefers_probe_not_evolve():
    family = build_scientific_program_family_report(
        nodes=[_node("a")],
        pair_batch=ScientificProgramPairBatch(pairs=[]),
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(["a"]),
        program_family=family,
        novelty_by_idea={"a": _novelty("a", "UNRESOLVED")},
    )
    assert pressure.pressures[0].preferred_action == "PROBE"


def test_bounded_scheduler_does_not_select_every_residual_candidate():
    ids = ["a", "b", "c", "d"]
    nodes = [_node(i) for i in ids]
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            relation = "SAME_PROGRAM" if {a, b} == {"a", "b"} else "DISTINCT_PROGRAM"
            pairs.append(
                ScientificProgramPairAssessment(
                    idea_id_a=a,
                    idea_id_b=b,
                    relation=relation,
                    rationale="test relation",
                )
            )
    family = build_scientific_program_family_report(
        nodes=nodes,
        pair_batch=ScientificProgramPairBatch(pairs=pairs),
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(ids),
        program_family=family,
        novelty_by_idea={i: _novelty(i, "SHALLOW_RESIDUE") for i in ids},
    )
    schedule = build_bounded_parent_schedule(
        pressure_report=pressure,
        program_family=family,
        max_selected_parents=2,
        max_selected_per_same_program=1,
    )
    assert schedule.selected_parent_count == 2
    selected = {row.idea_id for row in schedule.selected_parents}
    assert not ({"a", "b"} <= selected)
    assert len(schedule.deferred_search_worthy_idea_ids) == 2


def test_terminal_worthy_and_search_worthy_are_independent():
    nodes = [_node("a"), _node("b")]
    family = build_scientific_program_family_report(
        nodes=nodes,
        pair_batch=ScientificProgramPairBatch(
            pairs=[
                ScientificProgramPairAssessment(
                    idea_id_a="a",
                    idea_id_b="b",
                    relation="SAME_PROGRAM",
                    rationale="same program",
                )
            ]
        ),
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(["a", "b"]),
        program_family=family,
        novelty_by_idea={
            "a": _novelty("a", "CENTRAL_RESIDUE"),
            "b": _novelty("b", "CENTRAL_RESIDUE"),
        },
    )
    assert all(row.terminal_worthy for row in pressure.pressures)
    assert all(row.search_worthy for row in pressure.pressures)


def test_adjacent_program_neighbors_do_not_create_family_saturation_pressure():
    ids = ["a", "b", "c", "d"]
    nodes = [_node(i) for i in ids]
    pairs = [
        ScientificProgramPairAssessment(
            idea_id_a=a, idea_id_b=b, relation="ADJACENT_PROGRAM", rationale="adjacent branch"
        )
        for i, a in enumerate(ids) for b in ids[i + 1 :]
    ]
    family = build_scientific_program_family_report(
        nodes=nodes, pair_batch=ScientificProgramPairBatch(pairs=pairs)
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(ids),
        program_family=family,
        novelty_by_idea={i: _novelty(i, "CENTRAL_RESIDUE") for i in ids},
    )
    assert all(row.family_diversification_pressure == "NONE" for row in pressure.pressures)
    assert all(row.preferred_action == "EVOLVE" for row in pressure.pressures)
    assert all(row.search_priority == "MODERATE" for row in pressure.pressures)


def test_central_residue_is_moderate_without_other_high_pressure():
    family = build_scientific_program_family_report(
        nodes=[_node("a")], pair_batch=ScientificProgramPairBatch(pairs=[])
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(["a"]),
        program_family=family,
        novelty_by_idea={"a": _novelty("a", "CENTRAL_RESIDUE")},
    )
    row = pressure.pressures[0]
    assert row.residual_refinement_pressure == "MODERATE"
    assert row.search_priority == "MODERATE"
    assert row.preferred_action == "EVOLVE"


def test_scheduler_prefers_distinct_program_over_adjacent_at_equal_pressure():
    ids = ["a", "b", "c"]
    nodes = [_node(i) for i in ids]
    family = build_scientific_program_family_report(
        nodes=nodes,
        pair_batch=ScientificProgramPairBatch(
            pairs=[
                ScientificProgramPairAssessment(idea_id_a="a", idea_id_b="b", relation="ADJACENT_PROGRAM", rationale="adjacent"),
                ScientificProgramPairAssessment(idea_id_a="a", idea_id_b="c", relation="DISTINCT_PROGRAM", rationale="distinct"),
                ScientificProgramPairAssessment(idea_id_a="b", idea_id_b="c", relation="DISTINCT_PROGRAM", rationale="distinct"),
            ]
        ),
    )
    pressure = build_research_idea_search_pressure_report(
        search_assessment=_assessment_report(ids),
        program_family=family,
        novelty_by_idea={i: _novelty(i, "CENTRAL_RESIDUE") for i in ids},
    )
    # deterministic first choice is c by the final idea-id tie-break; its second
    # parent should then be program-distinct from c rather than same/adjacent.
    schedule = build_bounded_parent_schedule(
        pressure_report=pressure,
        program_family=family,
        max_selected_parents=2,
        max_selected_per_same_program=1,
    )
    assert schedule.selected_parent_count == 2
    assert schedule.selected_pair_relation_counts == {"DISTINCT_PROGRAM": 1}
