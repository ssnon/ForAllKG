from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    ProspectiveIdentificationShadowArtifact,
)
from pipeline_core.discovery.research_idea_contracts import (
    ConceptualFamilyAssessment,
    IdeaFacetAssessment,
    IdeaParentComparison,
    IdeaTransitionAssessment,
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    OffspringExecutionReport,
    OffspringRealizationRecord,
    OffspringRealizationReport,
    OffspringSemanticRecord,
)
from pipeline_core.discovery.research_idea_population_semantics import (
    FamilyPopulationReport,
    FamilySearchState,
    IdeaLocalSearchState,
    IdeaRealizationLink,
    ScientificFeedbackFacetObservation,
    adapt_adaptive_controller_plans,
    assign_conceptual_families,
    build_credit_continuity_traces,
    build_idea_learning_decisions,
    enrich_observations_from_residual_reports,
)
from pipeline_core.discovery.research_idea_closed_loop import run_realization_lifecycle


CTX_ID = "ctx:test"
CTX_SHA = "c" * 64


def _kernel(core: str, *, scope: list[str] | None = None) -> ResearchIdeaKernel:
    return ResearchIdeaKernel(
        canonical_intent=core,
        core_scientific_commitments=[core],
        scope_commitments=scope or [],
    )


def _node(
    idea_id: str,
    core: str,
    *,
    generation: int,
    parents: list[str] | None = None,
) -> ResearchIdeaNode:
    kernel = _kernel(core)
    raw = json.dumps(kernel.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=parents or [],
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        origin_kind="FRONTIER" if generation == 0 else "GENERATIONAL_OFFSPRING",
        source_object_id=f"source:{idea_id}",
        source_parent_object_ids=[f"source:{p}" for p in (parents or [])],
        source_kind="test",
        operator_id=None if generation == 0 else "AXIS_MUTATION",
        source_artifact_refs=[],
        kernel=kernel,
        kernel_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        task_relation_mode="DIRECT",
    )


def _transition(child_id: str, parent_id: str, identity: str = "DIFFERENT_IDEA"):
    changed = identity == "DIFFERENT_IDEA"
    facets = [
        IdeaFacetAssessment(
            facet="CORE_COMMITMENTS",
            relation="REPLACED" if changed else "PRESERVED",
            similarity=0.0 if changed else 1.0,
            rationale="test fixture core comparison",
        ),
        IdeaFacetAssessment(
            facet="SCOPE",
            relation="PRESERVED",
            similarity=1.0,
            rationale="test fixture scope comparison",
        ),
        IdeaFacetAssessment(
            facet="CONTRAST",
            relation="PRESERVED",
            similarity=1.0,
            rationale="test fixture contrast comparison",
        ),
        IdeaFacetAssessment(
            facet="QUESTION",
            relation="PRESERVED",
            similarity=1.0,
            rationale="test fixture question comparison",
        ),
    ]
    family = ConceptualFamilyAssessment(
        idea_id_a=parent_id,
        idea_id_b=child_id,
        relation="ADJACENT_FAMILY" if changed else "SAME_FAMILY",
        similarity=0.5 if changed else 1.0,
        confidence=0.8,
    )
    comparison = IdeaParentComparison(
        parent_idea_id=parent_id,
        identity_relation=identity,
        facet_assessments=facets,
        family_assessment=family,
    )
    return IdeaTransitionAssessment(
        proposed_idea_id=child_id,
        parent_idea_ids=[parent_id],
        parent_comparisons=[comparison],
        identity_relation=identity,
        genealogy_relation="CHILD_OF" if changed else "REFINEMENT_OF",
        operator_id="AXIS_MUTATION",
        operator_expectation_consistent=None,
        diagnostic_codes=[],
    )


def _execution(node: ResearchIdeaNode, parent_id: str = "parent") -> OffspringExecutionReport:
    semantic = OffspringSemanticRecord(
        idea_id=node.idea_id,
        task_id="task:1",
        channel="TRANSFORM",
        chosen_operator_id="AXIS_MUTATION",
        parent_idea_ids=[parent_id],
        transition=_transition(node.idea_id, parent_id),
        disposition="ACCEPTED_CHILD",
        accepted_for_realization=True,
        conceptual_change_summary="test child",
    )
    return OffspringExecutionReport.model_construct(
        report_id=f"execution:{node.idea_id}",
        report_sha256="e" * 64,
        source_plan_id="plan:1",
        source_v2_2_report_id="v22:1",
        source_generational_report_id="gen:1",
        generation_index=node.generation_index,
        generation_tasks=[],
        run_records=[],
        offspring_nodes=[node],
        semantic_records=[semantic],
        raw_offspring_count=1,
        accepted_for_realization_count=1,
        identity_relation_counts={"DIFFERENT_IDEA": 1},
        disposition_counts={"ACCEPTED_CHILD": 1},
        generated_count_by_channel={"TRANSFORM": 1},
        generated_count_by_operator={"AXIS_MUTATION": 1},
        semantic_noop_count=0,
        mutation_attempt_count=1,
        distinct_child_count=1,
        mutation_semantic_yield_fraction=1.0,
        indeterminate_probe_count=0,
        channel_drift_child_count=0,
        exact_kernel_duplicate_suppressed_count=0,
        llm_call_count=1,
        input_tokens=0,
        output_tokens=0,
        semantic_retry_count=0,
    )


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id=CTX_ID,
        context_sha256=CTX_SHA,
        source_packet_id="packet:1",
        source_packet_sha256="p" * 64,
        source_report_id="report:1",
        source_report_sha256="r" * 64,
        task_id="task:1",
        question="How does A affect B?",
        corpus_id="corpus:1",
        domain_profile_id="domain:test",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s1",
                text="A is associated with increased B under condition C.",
                epistemic_role="reported",
                claim_kind="relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            )
        ],
    )


def _valid_draft(local_id: str) -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id=local_id,
                title="Grounded realization",
                hypothesis_statement="Under condition C, A increases B through a testable dependency.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["s1"],
                inferential_bridge="The grounded A-B relation motivates testing a bounded intermediate dependency.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="B response",
                        expected_direction="increase",
                        rationale="The proposed bridge predicts increased B.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="B response",
                        falsifying_outcome="B does not increase under the matched condition.",
                    )
                ],
            )
        ],
        abstention_reason=None,
    )


class AbstainThenMaterializeBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return HypothesisDraftGeneration(
                draft=HypothesisPortfolioDraft(hypotheses=[], abstention_reason="first formulation not grounded")
            )
        return HypothesisDraftGeneration(draft=_valid_draft(f"h{self.calls}"))

    def repair(self, prompt, previous_draft, feedback):
        return HypothesisDraftGeneration(draft=_valid_draft("repair"))


class MaterializeBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return HypothesisDraftGeneration(draft=_valid_draft(f"h{self.calls}"))

    def repair(self, prompt, previous_draft, feedback):
        return HypothesisDraftGeneration(draft=_valid_draft("repair"))


class SequenceProspective:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def __call__(self, *, context, candidate, source_stage, output_prefix):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return ProspectiveIdentificationShadowArtifact(
            status="COMPLETE",
            source_stage=source_stage,
            source_context_id=context.context_id,
            source_hypothesis_id=candidate.hypothesis_id,
            contract=None,
            contract_integrity_passed=True,
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability=status,
            directionality_mode=("NOT_IDENTIFIABLE" if status == "NOT_OPERATIONALIZABLE" else "GROUNDED_PREDICTION"),
            measurement_compatibility_mode=("NOT_COMPARABLE" if status == "NOT_OPERATIONALIZABLE" else "PROSPECTIVE_MATCH_REQUIRED"),
            would_abstain_if_authoritative=(status == "NOT_OPERATIONALIZABLE"),
            artifact_path=str(output_prefix) + ".result.json",
        )


def test_family_assignment_is_soft_recomputable_and_not_node_identity():
    a = _node("a", "gap confinement --MODULATES--> hotspot distribution", generation=0)
    b = _node("b", "gap confinement --MODULATES--> hotspot distribution", generation=1, parents=["a"])
    c = _node("c", "surface charge --CONTROLS--> catalytic selectivity", generation=1, parents=["a"])
    assignments = assign_conceptual_families([a, b, c])
    by_id = {row.idea_id: row for row in assignments}
    assert by_id["a"].family_key == by_id["b"].family_key
    assert by_id["c"].family_key != by_id["a"].family_key
    assert by_id["a"].recomputable is True
    assert by_id["a"].immutable_idea_identity_field is False
    assert not hasattr(a, "family_key")


def test_adaptive_axis_mutation_is_boundary_sensitive_not_child_authority():
    plan = {
        "schema_version": "adaptive-discovery-controller-plan-v1",
        "plan_id": "adaptive:1",
        "decisions": [
            {
                "root_hypothesis_id": "h1",
                "current_hypothesis_id": "h1",
                "action": "AXIS_MUTATION",
                "action_scope_rank": 3,
            },
            {
                "root_hypothesis_id": "h1",
                "current_hypothesis_id": "h1",
                "action": "EVIDENCE_REAXIS",
                "action_scope_rank": 2,
            },
        ],
    }
    events = adapt_adaptive_controller_plans([plan], hypothesis_to_idea={"h1": "idea:1"})
    assert [row.interpreted_scope for row in events] == ["BOUNDARY_SENSITIVE", "LOCAL_REAXIS"]
    assert events[0].requires_research_idea_semantic_comparison is True
    assert events[0].idea_identity_authority is False


def test_failed_initial_realization_is_rescued_by_same_idea_alternate(tmp_path: Path):
    node = _node("g2", "A --CAUSES--> B", generation=2, parents=["parent"])
    lifecycle, portfolio = run_realization_lifecycle(
        execution=_execution(node),
        context=_context(),
        backend=AbstainThenMaterializeBackend(),
        output_dir=tmp_path,
        max_ideas=1,
        max_realizations_per_idea=2,
        max_per_parent=1,
        max_repair_attempts=0,
        prospective_runner=SequenceProspective(["PROSPECTIVELY_IDENTIFIABLE"]),
    )
    assert len(lifecycle.links) == 2
    assert lifecycle.links[0].materialization_status == "ABSTAINED"
    assert lifecycle.links[1].materialization_status == "MATERIALIZED"
    assert lifecycle.links[0].idea_id == lifecycle.links[1].idea_id == "g2"
    assert lifecycle.local_states[0].rescued_within_same_idea is True
    assert lifecycle.usable_grounded_realization_count == 1
    assert len(portfolio.hypotheses) == 1


def test_not_operationalizable_realization_triggers_same_idea_alternate_before_escalation(tmp_path: Path):
    node = _node("g2", "A --CAUSES--> B", generation=2, parents=["parent"])
    execution = _execution(node)
    context = _context()
    # Build a seed materialized card through the standard compiler path once.
    from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler

    seed_portfolio = HypothesisCompiler().compile(context, _valid_draft("seed"))
    seed_card = seed_portfolio.hypotheses[0]
    seed_report = OffspringRealizationReport.model_construct(
        report_id="seed-report",
        report_sha256="s" * 64,
        source_offspring_execution_report_id=execution.report_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        selected_idea_ids=[node.idea_id],
        selected_idea_count=1,
        records=[
            OffspringRealizationRecord(
                idea_id=node.idea_id,
                channel="TRANSFORM",
                semantic_disposition="ACCEPTED_CHILD",
                status="MATERIALIZED",
                hypothesis_id=seed_card.hypothesis_id,
                generation_attempt_count=1,
            )
        ],
        status_counts={"MATERIALIZED": 1},
        materialized_hypothesis_count=1,
        materialization_success_fraction=1.0,
        output_portfolio_id=seed_portfolio.portfolio_id,
        llm_call_count=1,
        input_tokens=0,
        output_tokens=0,
    )
    prospective = SequenceProspective([
        "NOT_OPERATIONALIZABLE",
        "PROSPECTIVELY_IDENTIFIABLE",
    ])
    lifecycle, _ = run_realization_lifecycle(
        execution=execution,
        context=context,
        backend=MaterializeBackend(),
        output_dir=tmp_path,
        max_ideas=1,
        max_realizations_per_idea=2,
        max_per_parent=1,
        max_repair_attempts=0,
        prospective_runner=prospective,
        seed_realization_report=seed_report,
        seed_portfolio=seed_portfolio,
    )
    assert len(lifecycle.links) == 2
    assert lifecycle.observations[0].prospective_identifiability == "NOT_OPERATIONALIZABLE"
    assert lifecycle.observations[1].prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE"
    assert lifecycle.local_states[0].rescued_within_same_idea is True
    assert lifecycle.not_operationalizable_realization_is_not_idea_failure is True


def _family_report_for_decision(idea_id: str, compute_share: float) -> FamilyPopulationReport:
    assignment = NS(idea_id=idea_id, family_key="f1")
    family = FamilySearchState(
        family_key="f1",
        representative_idea_id=idea_id,
        member_idea_ids=[idea_id],
        idea_count=1,
        generation_counts={"G2": 1},
        exact_kernel_unique_count=1,
        realization_count=1,
        materialized_realization_count=0,
        usable_grounded_realization_count=0,
        not_operationalizable_realization_count=0,
        adaptive_local_event_count=0,
        offspring_generation_llm_calls=0,
        realization_llm_calls=1,
        prospective_audit_llm_calls=0,
        total_tracked_llm_calls=1,
        compute_share=compute_share,
        productive_offspring_count=0,
        family_expansion_event_count=0,
        productive_operator_counts={},
    )
    return FamilyPopulationReport.model_construct(
        assignments=[assignment], family_states=[family]
    )


def test_local_exhaustion_escalates_transform_without_terminating_idea():
    state = IdeaLocalSearchState(
        idea_id="idea:1",
        realization_ids=["r1", "r2"],
        realization_count=2,
        materialized_count=0,
        usable_grounded_realization_count=0,
        not_operationalizable_count=0,
        failed_or_abstained_count=2,
        adaptive_event_ids=[],
        local_search_budget=2,
        local_search_budget_used=2,
        local_search_exhausted=True,
        rescued_within_same_idea=False,
    )
    decisions = build_idea_learning_decisions(
        local_states=[state],
        observations=[],
        adaptive_events=[],
        family_report=_family_report_for_decision("idea:1", 0.1),
    )
    row = decisions[0]
    assert row.disposition == "ESCALATE_TRANSFORM"
    assert "TRANSFORM" in row.recommended_channels
    assert row.idea_termination_authority is False
    assert row.one_realization_failure_is_not_idea_failure is True


def test_family_overconcentration_adds_exploration_pressure_not_gate():
    state = IdeaLocalSearchState(
        idea_id="idea:1",
        realization_ids=[],
        realization_count=0,
        materialized_count=0,
        usable_grounded_realization_count=0,
        not_operationalizable_count=0,
        failed_or_abstained_count=0,
        adaptive_event_ids=[],
        local_search_budget=2,
        local_search_budget_used=0,
        local_search_exhausted=False,
        rescued_within_same_idea=False,
    )
    row = build_idea_learning_decisions(
        local_states=[state],
        observations=[],
        adaptive_events=[],
        family_report=_family_report_for_decision("idea:1", 0.7),
        overconcentrated_family_share=0.35,
    )[0]
    assert "OVERCONCENTRATED_FAMILY_ADDS_EXPLORATION_PRESSURE_NOT_HARD_GATE" in row.reason_codes
    assert "WILDCARD" in row.recommended_channels
    assert row.family_is_soft_pressure_not_gate is True


def test_failed_g2_realization_can_receive_credit_from_materialized_g3_child():
    g2 = _node("g2", "A --CAUSES--> B", generation=2, parents=["p"])
    g3 = _node("g3", "A --CAUSES--> mediator M", generation=3, parents=["g2"])
    g2_state = IdeaLocalSearchState(
        idea_id="g2",
        realization_ids=["r2"],
        realization_count=1,
        materialized_count=0,
        usable_grounded_realization_count=0,
        not_operationalizable_count=0,
        failed_or_abstained_count=1,
        adaptive_event_ids=[],
        local_search_budget=1,
        local_search_budget_used=1,
        local_search_exhausted=True,
        rescued_within_same_idea=False,
    )
    g3_state = IdeaLocalSearchState(
        idea_id="g3",
        realization_ids=["r3"],
        realization_count=1,
        materialized_count=1,
        usable_grounded_realization_count=1,
        not_operationalizable_count=0,
        failed_or_abstained_count=0,
        adaptive_event_ids=[],
        local_search_budget=1,
        local_search_budget_used=1,
        local_search_exhausted=False,
        rescued_within_same_idea=False,
    )
    trace = build_credit_continuity_traces(
        generation2_nodes=[g2],
        generation3_nodes=[g3],
        g2_states=[g2_state],
        g3_states=[g3_state],
    )[0]
    assert trace.outcome == "RESCUED_BY_CHILD"
    assert trace.generation3_usable_child_idea_ids == ["g3"]


def test_observation_contract_cannot_decide_search_policy():
    obs = ScientificFeedbackFacetObservation(
        observation_id="obs:1",
        idea_id="idea:1",
        realization_id="r1",
        materialization_status="MATERIALIZED",
        grounding_integrity="PASSED",
        prospective_identifiability="NOT_OPERATIONALIZABLE",
    )
    assert obs.observation_only is True
    assert obs.search_policy_authority is False
    assert obs.idea_termination_authority is False
    assert not hasattr(obs, "next_operator")


def test_residual_feedback_adapter_requires_exact_portfolio_lineage_and_preserves_observation_policy_separation():
    obs = ScientificFeedbackFacetObservation(
        observation_id="obs:residual",
        idea_id="idea:1",
        realization_id="r1",
        hypothesis_id="h1",
        materialization_status="MATERIALIZED",
        grounding_integrity="PASSED",
    )
    wrong = {
        "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
        "source_portfolio_id": "portfolio:wrong",
        "hypotheses": [{
            "hypothesis_id": "h1",
            "final_epistemic_state": "UNRESOLVED_TOPOLOGY_GAP",
            "state_reason": "wrong lineage",
        }],
    }
    right = {
        "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
        "source_portfolio_id": "portfolio:right",
        "hypotheses": [{
            "hypothesis_id": "h1",
            "final_epistemic_state": "UNRESOLVED_EVIDENCE_GAP",
            "state_reason": "matched",
            "fresh_external_status": "INSUFFICIENT_SEARCH_EVIDENCE",
        }],
    }
    enriched, report = enrich_observations_from_residual_reports(
        [obs],
        residual_reports=[wrong, right],
        allowed_source_portfolio_ids=["portfolio:right"],
    )
    assert enriched[0].residual_epistemic_state == "UNRESOLVED_EVIDENCE_GAP"
    assert enriched[0].residual_state_reason == "matched"
    assert enriched[0].external_novelty_status == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert enriched[0].search_policy_authority is False
    assert report.residual_report_count == 1
    assert report.residual_matched_hypothesis_count == 1
