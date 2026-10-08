from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.research_idea_adaptive_fertility import (
    AdaptiveFertilityReport,
    ResearchIdeaFertilityDecision,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4Prompt,
)
from pipeline_core.discovery.research_idea_novelty_aware_reproduction_v3_4 import (
    NoveltyAwareOffspringBackendAdapter,
    PriorArtSearchClaimContext,
    ResearchIdeaMutationSearchContext,
    adapt_fertility_report_v3_4,
)
from pipeline_core.discovery.research_idea_search_pressure_v3_3_2 import (
    BoundedResearchIdeaParentSchedule,
    ResearchIdeaSearchPressure,
    ResearchIdeaSearchPressureReport,
    ScheduledResearchIdeaParent,
)


class _CaptureBackend:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(draft={})

    def repair(self, prompt, previous_draft, feedback):
        self.prompt = prompt
        return SimpleNamespace(draft={})


def test_novelty_context_is_injected_without_positive_premise_authority():
    context = ResearchIdeaMutationSearchContext(
        idea_id="idea:a",
        novelty_shape="CENTRAL_RESIDUE",
        external_status_counts={"KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 1},
        search_coverage_sufficient=True,
        saturated_claims=[
            PriorArtSearchClaimContext(
                claim_id="c-known",
                claim_text="known relation",
                importance="core",
                disposition="SATURATED",
                prior_art_status="DIRECT_PRIOR_ART",
            )
        ],
        residual_claims=[
            PriorArtSearchClaimContext(
                claim_id="c-residual",
                claim_text="unresolved relation",
                importance="core",
                disposition="RESIDUAL",
                prior_art_status="NO_DIRECT_MATCH_FOUND",
                required_bridge="bridge",
                predicted_observation="prediction",
                falsification_condition="falsifier",
            )
        ],
        preferred_action="EVOLVE",
        search_priority="HIGH",
        terminal_worthy=True,
    )
    base = EpistemicG4Prompt(
        task_id="task:1",
        system_prompt="base system",
        user_prompt='{"task_id":"task:1"}',
        prompt_sha256="base",
    )
    backend = _CaptureBackend()
    adapter = NoveltyAwareOffspringBackendAdapter(
        backend=backend,
        context_by_idea={"idea:a": context},
        task_to_idea={"task:1": "idea:a"},
    )
    adapter.generate(base)
    assert backend.prompt is not None
    assert "SEARCH FEEDBACK ONLY" in backend.prompt.system_prompt
    assert "unresolved relation" in backend.prompt.user_prompt
    assert "known relation" in backend.prompt.user_prompt
    assert '"external_prior_art_promoted_to_positive_premise": false' in backend.prompt.user_prompt


def _decision(idea_id: str, *, handoff: str | None = None):
    return ResearchIdeaFertilityDecision.model_construct(
        decision_id=f"d:{idea_id}",
        idea_id=idea_id,
        cycle_generation_index=2,
        idea_birth_generation_index=0,
        disposition="HOLD_STABLE",
        remains_active=True,
        fertile_for_child_generation=False,
        carry_forward_if_not_replaced=True,
        source_handoff_id=handoff,
        source_member_ids=[],
        source_epistemic_realization_ids=[],
        maturity_counts={},
        active_non_grounded_member_count=0,
        active_grounded_member_count=1,
        speculative_member_count=0,
        realization_count=1,
        usable_grounded_realization_count=1,
        failed_or_abstained_count=0,
        not_operationalizable_count=0,
        local_search_budget=2,
        local_search_budget_used=1,
        local_search_exhausted=False,
        rescued_within_same_idea=False,
        epistemic_debt_ids=[],
        epistemic_debt_kinds=[],
        acquisition_escalation_eligible_debt_count=0,
        recommended_channels=[],
        nonbinding_operator_hints=[],
        reason_codes=[],
    )


def _pressure(idea_id: str, *, action: str, worthy: bool):
    return ResearchIdeaSearchPressure.model_construct(
        idea_id=idea_id,
        generation_index=2,
        remains_active=True,
        persistence="RETAIN",
        novelty_shape="CENTRAL_RESIDUE",
        search_coverage_sufficient=True,
        same_program_key=f"p:{idea_id}",
        same_program_size=1,
        adjacent_program_neighbor_count=0,
        terminal_worthy=True,
        terminal_caveat_codes=[],
        search_worthy=worthy,
        preferred_action=action,
        search_priority="HIGH" if worthy else "DEFERRED",
        prior_art_escape_pressure="LOW",
        residual_refinement_pressure="MODERATE",
        discrimination_upside="HIGH",
        evidence_pressure="NONE",
        family_diversification_pressure="NONE",
        realization_pressure="NONE",
        recommended_channels=["TRANSFORM", "EXPLORE"] if worthy else [],
        nonbinding_operator_hints=["REGIME_BOUNDARY"] if worthy else [],
        reason_codes=[],
    )


def test_v3_4_reproduction_selection_is_not_survival_selection():
    base = AdaptiveFertilityReport.model_construct(
        report_id="base",
        report_sha256="sha",
        source_parallel_report_id="parallel",
        source_lifecycle_report_id="life",
        cycle_generation_index=2,
        decisions=[
            _decision("idea:selected", handoff="h:selected"),
            _decision("idea:deferred", handoff="h:deferred"),
            _decision("idea:stable", handoff="h:stable"),
        ],
        decision_count=3,
        active_idea_count=3,
        fertile_idea_count=0,
        persistent_nonfertile_idea_count=3,
        dropped_inactive_idea_count=0,
        source_evolution_handoff_count=3,
        fertile_handoff_count=0,
        disposition_counts={"HOLD_STABLE": 3},
        active_idea_ids=["idea:selected", "idea:deferred", "idea:stable"],
        fertile_idea_ids=[],
        persistent_nonfertile_idea_ids=["idea:selected", "idea:deferred", "idea:stable"],
    )
    pressure = ResearchIdeaSearchPressureReport.model_construct(
        report_id="pressure",
        report_sha256="sha",
        generation_index=2,
        source_search_assessment_report_id="assessment",
        source_program_family_report_id="family",
        pressures=[
            _pressure("idea:selected", action="EVOLVE", worthy=True),
            _pressure("idea:deferred", action="EVOLVE", worthy=True),
            _pressure("idea:stable", action="NONE", worthy=False),
        ],
        pressure_count=3,
        preferred_action_counts={"EVOLVE": 2, "NONE": 1},
        search_priority_counts={"HIGH": 2, "DEFERRED": 1},
        novelty_shape_counts={"CENTRAL_RESIDUE": 3},
        terminal_worthy_count=3,
        search_worthy_count=2,
    )
    schedule = BoundedResearchIdeaParentSchedule.model_construct(
        schedule_id="schedule",
        schedule_sha256="sha",
        generation_index=2,
        source_pressure_report_id="pressure",
        source_program_family_report_id="family",
        max_selected_parents=1,
        max_selected_per_same_program=1,
        candidate_parent_count=2,
        selected_parents=[
            ScheduledResearchIdeaParent.model_construct(
                idea_id="idea:selected",
                scientific_program_key="p:selected",
                selected_rank=1,
                preferred_action="EVOLVE",
                search_priority="HIGH",
                terminal_worthy=True,
                recommended_channels=["TRANSFORM", "EXPLORE"],
                nonbinding_operator_hints=["REGIME_BOUNDARY"],
                selection_reason_codes=["BOUNDED_PARENT_BUDGET"],
            )
        ],
        selected_parent_count=1,
        deferred_search_worthy_idea_ids=["idea:deferred"],
        selected_program_count=1,
        selected_action_counts={"EVOLVE": 1},
        selected_priority_counts={"HIGH": 1},
        selected_pair_relation_counts={},
    )

    compat, control = adapt_fertility_report_v3_4(
        base_fertility=base,
        pressure_report=pressure,
        schedule=schedule,
    )
    by_id = {row.idea_id: row for row in compat.decisions}
    assert by_id["idea:selected"].fertile_for_child_generation is True
    assert by_id["idea:selected"].disposition == "EVOLVE_CHILD"
    assert by_id["idea:deferred"].fertile_for_child_generation is False
    assert by_id["idea:deferred"].remains_active is True
    assert by_id["idea:deferred"].disposition == "HOLD_STABLE"
    assert by_id["idea:stable"].remains_active is True
    assert control.selected_parent_idea_ids == ["idea:selected"]
    assert control.deferred_search_worthy_idea_ids == ["idea:deferred"]
