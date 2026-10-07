from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    ProspectiveIdentificationShadowArtifact,
)
from pipeline_core.discovery.research_idea_active_realization import (
    ActiveRealizationActionRecord,
    ActiveRealizationSearchReport,
    ExactFeedbackAttachmentReport,
    FeedbackAttachmentRecord,
    attach_exact_feedback,
    build_boundary_escalation_traces,
    choose_active_action,
    run_active_realization_search,
)
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_closed_loop_v2_6 import build_credit_continuity_v2
from pipeline_core.discovery.research_idea_contracts import (
    ConceptualFamilyAssessment,
    IdeaFacetAssessment,
    IdeaParentComparison,
    IdeaTransitionAssessment,
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_family_calibration import build_family_calibration
from pipeline_core.discovery.research_idea_offspring_execution import (
    OffspringExecutionReport,
    OffspringSemanticRecord,
)
from pipeline_core.discovery.research_idea_population_semantics import (
    IdeaLocalSearchState,
    IdeaRealizationLink,
    ScientificFeedbackFacetObservation,
)


CTX_ID = "ctx:test"
CTX_SHA = "c" * 64


def _kernel(core: str) -> ResearchIdeaKernel:
    return ResearchIdeaKernel(
        canonical_intent=core,
        core_scientific_commitments=[core],
    )


def _node(idea_id: str, core: str, generation: int, parents=None) -> ResearchIdeaNode:
    kernel = _kernel(core)
    raw = json.dumps(kernel.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=list(parents or []),
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        origin_kind="FRONTIER" if generation == 0 else "GENERATIONAL_OFFSPRING",
        source_object_id=f"source:{idea_id}",
        source_parent_object_ids=[f"source:{x}" for x in (parents or [])],
        source_kind="test",
        operator_id=None if generation == 0 else "AXIS_MUTATION",
        source_artifact_refs=[],
        kernel=kernel,
        kernel_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        task_relation_mode="DIRECT",
    )


def _transition(child_id: str, parent_id: str, identity="DIFFERENT_IDEA"):
    changed = identity == "DIFFERENT_IDEA"
    facets = [
        IdeaFacetAssessment(
            facet="CORE_COMMITMENTS",
            relation="REPLACED" if changed else "PRESERVED",
            similarity=0.0 if changed else 1.0,
            rationale="fixture",
        ),
        IdeaFacetAssessment(facet="SCOPE", relation="PRESERVED", similarity=1.0, rationale="fixture"),
        IdeaFacetAssessment(facet="CONTRAST", relation="PRESERVED", similarity=1.0, rationale="fixture"),
        IdeaFacetAssessment(facet="QUESTION", relation="PRESERVED", similarity=1.0, rationale="fixture"),
    ]
    comparison = IdeaParentComparison(
        parent_idea_id=parent_id,
        identity_relation=identity,
        facet_assessments=facets,
        family_assessment=ConceptualFamilyAssessment(
            idea_id_a=parent_id,
            idea_id_b=child_id,
            relation="ADJACENT_FAMILY" if changed else "SAME_FAMILY",
            similarity=0.6 if changed else 1.0,
            confidence=0.8,
        ),
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


def _execution(node: ResearchIdeaNode, *, parent_id="parent") -> OffspringExecutionReport:
    semantic = OffspringSemanticRecord(
        idea_id=node.idea_id,
        task_id=f"task:{node.idea_id}",
        channel="TRANSFORM",
        chosen_operator_id="AXIS_MUTATION",
        parent_idea_ids=[parent_id],
        transition=_transition(node.idea_id, parent_id),
        disposition="ACCEPTED_CHILD",
        accepted_for_realization=True,
        conceptual_change_summary="fixture child",
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
                text="A is associated with B under condition C.",
                epistemic_role="reported",
                claim_kind="relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            ),
            HypothesisEvidenceStatement(
                statement_id="s2",
                text="A changes intermediate M under condition C.",
                epistemic_role="reported",
                claim_kind="relation",
                paper_ids=["paper:2"],
                eligible_as_premise=True,
            ),
        ],
    )


def _draft(local_id: str, premise="s1") -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id=local_id,
                title="Grounded realization",
                hypothesis_statement="A affects B through a bounded testable dependency.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=[premise],
                inferential_bridge="The grounded relation motivates a bounded test.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="B response",
                        expected_direction="increase",
                        rationale="The bridge predicts increased B.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="B response",
                        falsifying_outcome="B does not increase.",
                    )
                ],
            )
        ],
        abstention_reason=None,
    )


class ReaxisBackend:
    backend_name = "fake"
    model_name = "fake"

    def generate(self, prompt):
        return HypothesisDraftGeneration(draft=_draft("reaxis", premise="s2"))

    def repair(self, prompt, previous_draft, feedback):
        return HypothesisDraftGeneration(draft=_draft("repair", premise="s2"))


class ProspectiveIdentifiable:
    def __call__(self, *, context, candidate, source_stage, output_prefix):
        return ProspectiveIdentificationShadowArtifact(
            status="COMPLETE",
            source_stage=source_stage,
            source_context_id=context.context_id,
            source_hypothesis_id=candidate.hypothesis_id,
            contract=None,
            contract_integrity_passed=True,
            current_evidence_status="PARTIAL_GROUNDING",
            prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
            directionality_mode="GROUNDED_PREDICTION",
            measurement_compatibility_mode="PROSPECTIVE_MATCH_REQUIRED",
            artifact_path=str(output_prefix) + ".result.json",
        )


def _not_op_observation(idea_id: str, link: IdeaRealizationLink):
    return ScientificFeedbackFacetObservation(
        observation_id=f"obs:{idea_id}",
        idea_id=idea_id,
        realization_id=link.realization_id,
        hypothesis_id=link.hypothesis_id,
        materialization_status="MATERIALIZED",
        grounding_integrity="PASSED",
        prospective_status="COMPLETE",
        prospective_identifiability="NOT_OPERATIONALIZABLE",
        prospective_contract_integrity_passed=True,
        source_systems=["fixture"],
        source_versions=["v1"],
    )


def _lifecycle(idea_id: str, hypothesis_id: str, observation, generation=2):
    link = IdeaRealizationLink(
        realization_id=f"realization:{idea_id}:1",
        idea_id=idea_id,
        generation_index=generation,
        hypothesis_id=hypothesis_id,
        source_context_id=CTX_ID,
        realization_kind="INITIAL",
        attempt_index=1,
        materialization_status="MATERIALIZED",
    )
    if observation is None:
        observation = ScientificFeedbackFacetObservation(
            observation_id=f"obs:{idea_id}",
            idea_id=idea_id,
            realization_id=link.realization_id,
            hypothesis_id=hypothesis_id,
            materialization_status="MATERIALIZED",
            grounding_integrity="PASSED",
            source_systems=["fixture"],
            source_versions=["v1"],
        )
    state = IdeaLocalSearchState(
        idea_id=idea_id,
        realization_ids=[link.realization_id],
        realization_count=1,
        materialized_count=1,
        usable_grounded_realization_count=0,
        not_operationalizable_count=int(observation.prospective_identifiability == "NOT_OPERATIONALIZABLE"),
        failed_or_abstained_count=0,
        local_search_budget=2,
        local_search_budget_used=1,
        local_search_exhausted=False,
        rescued_within_same_idea=False,
    )
    return RealizationLifecycleReport.model_construct(
        report_id=f"lifecycle:{idea_id}",
        report_sha256="l" * 64,
        generation_index=generation,
        source_offspring_execution_report_id=f"execution:{idea_id}",
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        target_idea_ids=[idea_id],
        links=[link],
        observations=[observation],
        local_states=[state],
        output_portfolio_id=f"portfolio:{idea_id}",
        materialized_hypothesis_count=1,
        usable_grounded_realization_count=0,
        rescued_within_same_idea_count=0,
        local_search_exhausted_count=0,
        prospective_audit_count=1,
        prospective_not_operationalizable_count=state.not_operationalizable_count,
        realization_llm_call_count=1,
        prospective_audit_llm_call_count=1,
        max_realizations_per_idea=2,
        max_repair_attempts=1,
    )


def _active_empty(generation: int, execution_id: str, lifecycle_id: str, *, rescued=()):
    return ActiveRealizationSearchReport.model_construct(
        report_id=f"active:g{generation}",
        report_sha256="a" * 64,
        generation_index=generation,
        source_lifecycle_report_id=lifecycle_id,
        source_offspring_execution_report_id=execution_id,
        source_feedback_attachment_report_id="feedback:1",
        target_idea_ids=[],
        action_records=[],
        new_links=[],
        new_observations=[],
        resolved_usable_idea_ids=list(rescued),
        rescued_within_same_idea_ids=list(rescued),
        unresolved_idea_ids=[],
        boundary_escalation_requested_idea_ids=[],
        deferred_retrieval_idea_ids=[],
        deferred_graph_retraversal_idea_ids=[],
        action_counts={},
        execution_status_counts={},
        local_llm_call_count=0,
        prospective_audit_llm_call_count=0,
        max_actions_per_idea=2,
    )


def test_graph_family_projection_merges_high_similarity_adjacent_program_without_collapsing_tight_identity():
    a = _node("a", "gap confinement --MODULATES--> hotspot distribution", 0)
    b = _node("b", "gap confinement --MODULATES--> hotspot distribution", 1, ["a"])
    c = _node("c", "gap confinement --CONTROLS--> hotspot distribution", 1, ["a"])
    d = _node("d", "surface charge --CONTROLS--> catalytic selectivity", 1, ["a"])
    report = build_family_calibration([a, b, c, d], program_similarity_floor=0.58)
    by_id = {row.idea_id: row for row in report.assignments}
    assert by_id["a"].tight_neighborhood_key == by_id["b"].tight_neighborhood_key
    assert by_id["c"].tight_neighborhood_key != by_id["a"].tight_neighborhood_key
    assert by_id["c"].program_family_key == by_id["a"].program_family_key
    assert by_id["d"].program_family_key != by_id["a"].program_family_key
    assert report.program_family_count < report.tight_neighborhood_count
    assert report.family_is_not_hard_selection_gate is True


def test_family_threshold_curve_component_count_is_monotone_with_stricter_threshold():
    nodes = [
        _node("a", "A --CAUSES--> B", 0),
        _node("b", "A --MODULATES--> B", 1, ["a"]),
        _node("c", "A --CONTROLS--> B", 1, ["a"]),
    ]
    report = build_family_calibration(nodes, curve_thresholds=[0.4, 0.6, 0.8])
    counts = [row.connected_component_count for row in report.threshold_curve]
    assert counts == sorted(counts)


def test_not_evaluated_materialized_realization_is_preserved_not_treated_as_failure():
    context = _context()
    portfolio = HypothesisCompiler().compile(context, _draft("seed", premise="s1"))
    card = portfolio.hypotheses[0]
    link = IdeaRealizationLink(
        realization_id="r1", idea_id="idea", generation_index=2,
        hypothesis_id=card.hypothesis_id, source_context_id=CTX_ID,
        realization_kind="INITIAL", attempt_index=1, materialization_status="MATERIALIZED",
    )
    obs = ScientificFeedbackFacetObservation(
        observation_id="o1", idea_id="idea", realization_id="r1",
        hypothesis_id=card.hypothesis_id, materialization_status="MATERIALIZED",
        grounding_integrity="PASSED", source_systems=["fixture"], source_versions=["v1"],
    )
    feedback = FeedbackAttachmentRecord(
        hypothesis_id=card.hypothesis_id,
        idea_id="idea",
        prospective_evaluation_state="NOT_EVALUATED",
        residual_evaluation_state="NOT_EVALUATED",
    )
    action, reasons = choose_active_action(
        context=context,
        current_card=card,
        current_observation=obs,
        feedback=feedback,
        prior_actions=[],
    )
    assert action == "KEEP_UNTIL_EVALUATED"
    assert "PROSPECTIVE_NOT_EVALUATED_IS_NOT_FAILURE" in reasons


def test_not_operationalizable_uses_active_evidence_reaxis_and_can_rescue_same_idea(tmp_path: Path):
    context = _context()
    seed_portfolio = HypothesisCompiler().compile(context, _draft("seed", premise="s1"))
    card = seed_portfolio.hypotheses[0]
    node = _node("g2", "A --CAUSES--> B", 2, ["parent"])
    execution = _execution(node)
    link = IdeaRealizationLink(
        realization_id="r1", idea_id="g2", generation_index=2,
        hypothesis_id=card.hypothesis_id, source_context_id=CTX_ID,
        realization_kind="INITIAL", attempt_index=1, materialization_status="MATERIALIZED",
    )
    obs = _not_op_observation("g2", link)
    lifecycle = _lifecycle("g2", card.hypothesis_id, obs, generation=2)
    # Replace fixture-generated link so observation/link IDs align exactly.
    lifecycle = lifecycle.model_copy(update={"links": [link], "observations": [obs]})
    feedback = ExactFeedbackAttachmentReport.model_construct(
        report_id="feedback:1", report_sha256="f" * 64,
        source_lifecycle_report_id=lifecycle.report_id, search_roots=[],
        records=[FeedbackAttachmentRecord(
            hypothesis_id=card.hypothesis_id, idea_id="g2",
            prospective_evaluation_state="EVALUATED",
            residual_evaluation_state="NOT_EVALUATED",
            prospective_identifiability="NOT_OPERATIONALIZABLE",
        )],
        prospective_evaluated_count=1, prospective_not_evaluated_count=0,
        prospective_ambiguous_count=0, residual_evaluated_count=0,
        residual_not_evaluated_count=1, residual_ambiguous_count=0,
    )
    result = run_active_realization_search(
        execution=execution,
        lifecycle=lifecycle,
        portfolio=seed_portfolio,
        context=context,
        feedback=feedback,
        backend=ReaxisBackend(),
        output_dir=tmp_path,
        prospective_runner=ProspectiveIdentifiable(),
        max_active_ideas=1,
        max_actions_per_idea=2,
        max_repair_attempts=0,
    )
    assert result.report.action_counts == {"EVIDENCE_REAXIS": 1}
    assert result.report.rescued_within_same_idea_ids == ["g2"]
    assert result.report.new_links[0].realization_kind == "EVIDENCE_REAXIS"
    assert result.report.new_links[0].idea_id == "g2"
    assert result.report.production_generation_authority is False


def test_exact_feedback_with_no_matching_artifacts_is_explicitly_not_evaluated(tmp_path: Path):
    context = _context()
    portfolio = HypothesisCompiler().compile(context, _draft("seed", premise="s1"))
    card = portfolio.hypotheses[0]
    lifecycle = _lifecycle("g2", card.hypothesis_id, None, generation=2)
    report = attach_exact_feedback(lifecycle=lifecycle, search_roots=[tmp_path])
    assert len(report.records) == 1
    assert report.records[0].prospective_evaluation_state == "NOT_EVALUATED"
    assert report.records[0].residual_evaluation_state == "NOT_EVALUATED"
    assert report.not_evaluated_is_not_failure is True


def test_axis_mutation_boundary_uses_existing_child_semantics_not_operator_name_alone():
    g2 = _node("g2", "A --CAUSES--> B", 2, ["p"])
    g3 = _node("g3", "A --CAUSES--> M", 3, ["g2"])
    g3_execution = _execution(g3, parent_id="g2")
    action = ActiveRealizationActionRecord(
        action_id="a1", idea_id="g2", generation_index=2,
        action="AXIS_MUTATION", action_index=1,
        execution_status="BOUNDARY_ESCALATION_REQUESTED",
        requires_research_idea_semantic_comparison=True,
        local_action_cannot_change_idea_identity=False,
    )
    g2_active = _active_empty(2, "execution:g2", "lifecycle:g2").model_copy(
        update={
            "action_records": [action],
            "boundary_escalation_requested_idea_ids": ["g2"],
            "action_counts": {"AXIS_MUTATION": 1},
        }
    )
    g3_active = _active_empty(3, "execution:g3", "lifecycle:g3")
    traces = build_boundary_escalation_traces(
        generation2_active=g2_active,
        generation3_active=g3_active,
        generation3_execution=g3_execution,
    )
    assert traces[0].resolution == "CHILD_IDEA_OBSERVED"
    assert traces[0].idea_identity_decided_by_operator_name is False
    assert traces[0].existing_next_generation_is_not_causally_attributed_to_v2_6_action is True


def test_credit_v2_counts_generation3_same_idea_rescue_and_child_rescue_separately():
    g2 = _node("g2", "A --CAUSES--> B", 2, ["p"])
    g3 = _node("g3", "A --CAUSES--> M", 3, ["g2"])
    g2_execution = _execution(g2)
    g3_execution = _execution(g3, parent_id="g2")
    context = _context()
    p2 = HypothesisCompiler().compile(context, _draft("g2h", premise="s1"))
    p3 = HypothesisCompiler().compile(context, _draft("g3h", premise="s1"))
    g2_lifecycle = _lifecycle("g2", p2.hypotheses[0].hypothesis_id, None, generation=2)
    g3_lifecycle = _lifecycle("g3", p3.hypotheses[0].hypothesis_id, None, generation=3)
    g2_active = _active_empty(2, g2_execution.report_id, g2_lifecycle.report_id)
    # G3 active rescue must make the child usable. Add a new usable observation.
    rescue_obs = ScientificFeedbackFacetObservation(
        observation_id="g3-rescue", idea_id="g3", realization_id="new-r",
        hypothesis_id="new-h", materialization_status="MATERIALIZED",
        grounding_integrity="PASSED", prospective_status="COMPLETE",
        prospective_identifiability="PROSPECTIVELY_IDENTIFIABLE",
        prospective_contract_integrity_passed=True,
        source_systems=["fixture"], source_versions=["v1"],
    )
    g3_active = _active_empty(3, g3_execution.report_id, g3_lifecycle.report_id, rescued=["g3"]).model_copy(
        update={"new_observations": [rescue_obs]}
    )
    report = build_credit_continuity_v2(
        generation2_execution=g2_execution,
        generation2_lifecycle=g2_lifecycle,
        generation2_active=g2_active,
        generation3_execution=g3_execution,
        generation3_lifecycle=g3_lifecycle,
        generation3_active=g3_active,
    )
    assert report.generation3_same_idea_rescue_count == 1
    assert report.total_same_idea_rescue_count == 1
    assert report.child_rescue_count == 1
    assert report.outcome_counts["RESCUED_BY_CHILD"] == 1


def test_family_calibration_recovers_four_backbone_components_in_512_candidate_control():
    nodes = []
    for family_index in range(4):
        core = f"backbone_{family_index} --CONTROLS--> phenotype_{family_index}"
        for member_index in range(128):
            nodes.append(_node(f"f{family_index}:{member_index}", core, 0))
    report = build_family_calibration(nodes, program_similarity_floor=0.58)
    assert report.idea_count == 512
    assert report.tight_neighborhood_count == 4
    assert report.program_family_count == 4
    assert report.program_singleton_count == 0


def test_exact_feedback_resolver_recovers_matching_residual_and_prospective(tmp_path: Path):
    context = _context()
    portfolio = HypothesisCompiler().compile(context, _draft("seed", premise="s1"))
    card = portfolio.hypotheses[0]
    lifecycle = _lifecycle("g2", card.hypothesis_id, None, generation=2).model_copy(
        update={"output_portfolio_id": "portfolio:matched"}
    )
    residual = {
        "schema_version": "scientific-portfolio-residual-epistemic-state-v1",
        "source_portfolio_id": "portfolio:matched",
        "report_id": "residual:1",
        "hypotheses": [
            {
                "hypothesis_id": card.hypothesis_id,
                "final_epistemic_state": "UNRESOLVED_EVIDENCE_GAP",
                "state_reason": "fixture_gap",
            }
        ],
    }
    prospective = {
        "schema_version": "prospective-identification-materialization-shadow-v1",
        "status": "COMPLETE",
        "source_context_id": CTX_ID,
        "source_hypothesis_id": card.hypothesis_id,
        "contract_integrity_passed": True,
        "current_evidence_status": "PARTIAL_GROUNDING",
        "prospective_identifiability": "PROSPECTIVELY_IDENTIFIABLE",
        "directionality_mode": "GROUNDED_PREDICTION",
        "measurement_compatibility_mode": "PROSPECTIVE_MATCH_REQUIRED",
    }
    (tmp_path / "residual_epistemic_state.json").write_text(json.dumps(residual))
    (tmp_path / "prospective.result.json").write_text(json.dumps(prospective))
    report = attach_exact_feedback(lifecycle=lifecycle, search_roots=[tmp_path])
    assert report.records[0].prospective_evaluation_state == "EVALUATED"
    assert report.records[0].residual_evaluation_state == "EVALUATED"
    assert report.records[0].residual_epistemic_state == "UNRESOLVED_EVIDENCE_GAP"
