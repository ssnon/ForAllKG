from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace as NS

from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_multigeneration import (
    ConceptualDeltaAuditBatchDraft,
    ConceptualDeltaAuditDraft,
    ConceptualDeltaAuditGeneration,
    build_calibrated_g3_search,
    build_multigeneration_case_summary,
    run_conceptual_delta_audit,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    GenerationalOffspringBatchDraft,
    GenerationalOffspringCandidateDraft,
    OffspringGeneration,
    OffspringGenerationPlan,
    OffspringGenerationTask,
    OffspringRealizationRecord,
    OffspringRealizationReport,
    build_offspring_generation_plan,
    execute_offspring_plan,
)


CTX_ID = "ctx:v24"
CTX_SHA = "c" * 64


def _kernel(core: str):
    return ResearchIdeaKernel(
        canonical_intent=core,
        core_scientific_commitments=[core],
    )


def _node(idea_id: str, core: str, generation: int, parent_ids=None):
    kernel = _kernel(core)
    digest = hashlib.sha256(
        json.dumps(
            kernel.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=list(parent_ids or []),
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        origin_kind=("FRONTIER" if generation == 0 else "GENERATIONAL_OFFSPRING"),
        source_object_id=f"source:{idea_id}",
        source_parent_object_ids=[f"source:{x}" for x in (parent_ids or [])],
        source_kind="TEST",
        operator_id=None,
        source_artifact_refs=["test.json"],
        kernel=kernel,
        kernel_sha256=digest,
        task_relation_mode="DIRECT",
    )


def _task(channel="TRANSFORM", generation=2):
    allowed = {
        "EXPLOIT": ["SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"],
        "TRANSFORM": ["AXIS_MUTATION", "REGIME_BOUNDARY"],
        "EXPLORE": ["AXIS_MUTATION", "CROSS_SOURCE_BRIDGE"],
        "WILDCARD": ["AXIS_MUTATION", "CROSS_SOURCE_BRIDGE"],
    }[channel]
    identity = {
        "EXPLOIT": "PRESERVE_IDEA",
        "TRANSFORM": "CREATE_CHILD",
        "EXPLORE": "PREFER_CHILD",
        "WILDCARD": "OPEN_BOUNDED",
    }[channel]
    return OffspringGenerationTask(
        task_id=f"task:g{generation}:{channel}",
        primary_parent_idea_id="p1",
        eligible_secondary_parent_idea_ids=["p2"],
        channel=channel,
        identity_goal=identity,
        operator_hints=allowed[:1],
        allowed_operator_ids=allowed,
        max_output_count=1,
    )


class FakeOffspringBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self, batch):
        self.batch = batch

    def generate(self, prompt):
        return OffspringGeneration(draft=self.batch, input_tokens=4, output_tokens=3)

    def repair(self, prompt, previous_draft, feedback):
        return OffspringGeneration(draft=previous_draft, input_tokens=2, output_tokens=1)


class FakeAuditBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self, rows):
        self.rows = rows
        self.user_prompt = None

    def audit(self, *, system_prompt, user_prompt, generation_label):
        self.user_prompt = user_prompt
        return ConceptualDeltaAuditGeneration(
            draft=ConceptualDeltaAuditBatchDraft(assessments=self.rows),
            input_tokens=7,
            output_tokens=5,
        )


def _execution(channel="TRANSFORM", child_core="A --CAUSES--> M"):
    p1 = _node("p1", "A --CAUSES--> B", 1, ["root1"])
    p2 = _node("p2", "C --CAUSES--> D", 1, ["root2"])
    task = _task(channel)
    operator = "SAME_PREMISE_SHARPEN" if channel == "EXPLOIT" else "AXIS_MUTATION"
    batch = GenerationalOffspringBatchDraft(
        task_id=task.task_id,
        primary_parent_idea_id="p1",
        channel=channel,
        candidates=[
            GenerationalOffspringCandidateDraft(
                local_id="c1",
                chosen_operator_id=operator,
                conceptual_change_summary="test child",
                kernel=_kernel(child_core),
                differential_prediction="prediction",
                falsification_condition="falsifier",
                discriminating_observation="observation",
                task_relation_mode="DIRECT",
            )
        ],
        abstention_reason=None,
    )
    plan = OffspringGenerationPlan(
        plan_id="plan:g2",
        source_v2_2_report_id="search:g2",
        source_generational_report_id="population:g1",
        generation_index=2,
        tasks=[task],
        task_count=1,
        max_raw_offspring=1,
        planned_max_offspring=1,
        task_count_by_channel={channel: 1},
    )
    search = NS(report_id="search:g2", source_generational_report_id="population:g1")
    population = NS(report_id="population:g1", research_ideas=[p1, p2])
    execution, _ = execute_offspring_plan(
        plan=plan,
        evolutionary_report=search,
        generational_report=population,
        research_question="How does A affect B?",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    return p1, p2, execution


def _realization(execution, status="MATERIALIZED"):
    row = execution.semantic_records[0]
    materialized = status == "MATERIALIZED"
    return OffspringRealizationReport(
        report_id="real:g2",
        report_sha256="r" * 64,
        source_offspring_execution_report_id=execution.report_id,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        selected_idea_ids=[row.idea_id],
        selected_idea_count=1,
        records=[
            OffspringRealizationRecord(
                idea_id=row.idea_id,
                channel=row.channel,
                semantic_disposition=row.disposition,
                status=status,
                hypothesis_id=("hyp:1" if materialized else None),
                generation_attempt_count=1,
            )
        ],
        status_counts={status: 1},
        materialized_hypothesis_count=int(materialized),
        materialization_success_fraction=float(materialized),
        output_portfolio_id="portfolio:g2",
        llm_call_count=1,
        input_tokens=3,
        output_tokens=2,
    )


def test_independent_audit_does_not_receive_deterministic_comparator_result():
    p1, p2, execution = _execution()
    child_id = execution.semantic_records[0].idea_id
    backend = FakeAuditBackend([
        ConceptualDeltaAuditDraft(
            child_idea_id=child_id,
            category="REFINEMENT",
            confidence=0.9,
            rationale="The core relation remains substantively the same.",
        )
    ])
    report = run_conceptual_delta_audit(
        execution=execution,
        parent_nodes=[p1, p2],
        research_question="q",
        backend=backend,
    )
    assert "DIFFERENT_IDEA" not in backend.user_prompt
    assert report.disagreement_count == 1
    assert report.potential_semantic_inflation_count == 1
    assert report.calibration_is_not_hard_gate is True
    assert report.records[0].candidate_deletion_authority is False


def test_strong_distinct_requires_independent_mutation_confirmation():
    p1, p2, execution = _execution()
    child_id = execution.semantic_records[0].idea_id
    backend = FakeAuditBackend([
        ConceptualDeltaAuditDraft(
            child_idea_id=child_id,
            category="GENUINE_MUTATION",
            confidence=0.85,
            rationale="The mediator commitment changes the scientific mechanism.",
            identity_bearing_changes=["B is replaced by mediator M"],
        )
    ])
    report = run_conceptual_delta_audit(
        execution=execution,
        parent_nodes=[p1, p2],
        research_question="q",
        backend=backend,
    )
    assert report.strong_distinct_child_count == 1
    assert report.calibrated_mutation_yield_fraction == 1.0


def test_failed_g2_realization_increases_transform_pressure_without_termination():
    p1, p2, execution = _execution()
    child_id = execution.semantic_records[0].idea_id
    audit = run_conceptual_delta_audit(
        execution=execution,
        parent_nodes=[p1, p2],
        research_question="q",
        backend=FakeAuditBackend([
            ConceptualDeltaAuditDraft(
                child_idea_id=child_id,
                category="GENUINE_MUTATION",
                confidence=0.9,
                rationale="Distinct mechanism.",
            )
        ]),
    )
    population, search = build_calibrated_g3_search(
        execution=execution,
        realization=_realization(execution, "ABSTAINED"),
        conceptual_audit=audit,
        max_parent_budget=1,
    )
    credit = search.credit_states[0]
    assert population.generation_index == 2
    assert credit.realization_fertility == "LOW"
    assert credit.transformation_pressure == "HIGH"
    assert credit.idea_reproductive_value != "TERMINATED"
    assert search.low_realization_transform_parent_count == 1


def test_materialized_g2_child_can_receive_exploit_credit():
    p1, p2, execution = _execution()
    child_id = execution.semantic_records[0].idea_id
    audit = run_conceptual_delta_audit(
        execution=execution,
        parent_nodes=[p1, p2],
        research_question="q",
        backend=FakeAuditBackend([
            ConceptualDeltaAuditDraft(
                child_idea_id=child_id,
                category="GENUINE_MUTATION",
                confidence=0.9,
                rationale="Distinct mechanism.",
            )
        ]),
    )
    _, search = build_calibrated_g3_search(
        execution=execution,
        realization=_realization(execution, "MATERIALIZED"),
        conceptual_audit=audit,
        max_parent_budget=1,
    )
    credit = search.credit_states[0]
    assert credit.realization_fertility == "HIGH"
    assert credit.exploit_value == "HIGH"
    assert search.selected_parent_count == 1


def test_generic_offspring_executor_can_create_generation3_nodes():
    p1 = _node("p1", "A --CAUSES--> B", 2, ["g1a"])
    p2 = _node("p2", "C --CAUSES--> D", 2, ["g1b"])
    credit = NS(
        idea_id="p1",
        reason_codes=["TEST"],
    )
    allocation = NS(
        idea_id="p1",
        channel="TRANSFORM",
        operator_hints=["AXIS_MUTATION"],
    )
    search = NS(
        report_id="search:g3",
        source_generational_report_id="population:g2",
        selected_parent_idea_ids=["p1"],
        allocations=[allocation],
        policy_states=[credit],
    )
    population = NS(report_id="population:g2", research_ideas=[p1, p2])
    plan = build_offspring_generation_plan(
        evolutionary_report=search,
        generational_report=population,
        max_raw_offspring=1,
        generation_index=3,
    )
    task = plan.tasks[0]
    batch = GenerationalOffspringBatchDraft(
        task_id=task.task_id,
        primary_parent_idea_id="p1",
        channel="TRANSFORM",
        candidates=[
            GenerationalOffspringCandidateDraft(
                local_id="g3c",
                chosen_operator_id="AXIS_MUTATION",
                conceptual_change_summary="new G3 relation",
                kernel=_kernel("A --CAUSES--> Z"),
                differential_prediction="p",
                falsification_condition="f",
                discriminating_observation="o",
                task_relation_mode="DIRECT",
            )
        ],
        abstention_reason=None,
    )
    execution, _ = execute_offspring_plan(
        plan=plan,
        evolutionary_report=search,
        generational_report=population,
        research_question="q",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    assert plan.generation_index == 3
    assert execution.generation_index == 3
    assert execution.offspring_nodes[0].generation_index == 3
    assert execution.report_id.startswith("g3_offspring_execution:")


def test_multigeneration_summary_keeps_conceptual_and_realization_fertility_separate():
    p1, p2, g2_execution = _execution()
    child_id = g2_execution.semantic_records[0].idea_id
    g2_audit = run_conceptual_delta_audit(
        execution=g2_execution,
        parent_nodes=[p1, p2],
        research_question="q",
        backend=FakeAuditBackend([
            ConceptualDeltaAuditDraft(
                child_idea_id=child_id,
                category="GENUINE_MUTATION",
                confidence=0.9,
                rationale="Distinct.",
            )
        ]),
    )
    g2_real = _realization(g2_execution, "MATERIALIZED")
    population, g3_search = build_calibrated_g3_search(
        execution=g2_execution,
        realization=g2_real,
        conceptual_audit=g2_audit,
        max_parent_budget=1,
    )
    # Reuse a generated G2 execution as a minimal stand-in for G3, but correct
    # the generation metadata to exercise summary accounting only.
    g3_execution = g2_execution.model_copy(update={"generation_index": 3})
    g3_audit = g2_audit.model_copy(update={"generation_index": 3})
    summary = build_multigeneration_case_summary(
        source_generational_report=NS(
            generation0_idea_count=3,
            generation1_idea_count=2,
            research_ideas=[p1, p2],
        ),
        g2_execution=g2_execution,
        g2_audit=g2_audit,
        g2_realization=g2_real,
        g3_search=g3_search,
        g3_execution=g3_execution,
        g3_audit=g3_audit,
    )
    assert summary.generation0_idea_count == 3
    assert summary.generation2_materialized_hypothesis_count == 1
    assert summary.generation2_parent_fertility[0].materialized_child_count == 1
    assert summary.generation2_parent_fertility[0].conceptual_fertility_and_realization_fertility_kept_separate is True
