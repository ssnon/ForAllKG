from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace as NS

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    GenerationalOffspringBatchDraft,
    GenerationalOffspringCandidateDraft,
    OffspringGeneration,
    OffspringGenerationPlan,
    OffspringGenerationTask,
    build_g3_feedback_seed,
    build_offspring_generation_plan,
    execute_offspring_plan,
    realize_offspring_bounded,
    select_offspring_for_realization,
)


CTX_ID = "ctx:v23"
CTX_SHA = "c" * 64


def _kernel(
    core: str,
    *,
    intent: str | None = None,
    scope: list[str] | None = None,
    contrast: list[str] | None = None,
    question: str | None = None,
):
    return ResearchIdeaKernel(
        canonical_intent=intent or core,
        core_scientific_commitments=[core],
        scope_commitments=list(scope or []),
        contrastive_commitments=list(contrast or []),
        question_commitment=question,
    )


def _node(idea_id: str, core: str, *, generation=1):
    kernel = _kernel(core)
    import hashlib, json
    digest = hashlib.sha256(
        json.dumps(kernel.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=[] if generation == 0 else ["root"] if idea_id != "root" else [],
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        origin_kind="EVOLUTION" if generation else "FRONTIER",
        source_object_id=f"source:{idea_id}",
        source_parent_object_ids=[],
        source_kind="TEST",
        operator_id=None,
        source_artifact_refs=["test.json"],
        kernel=kernel,
        kernel_sha256=digest,
        task_relation_mode="DIRECT",
    )


def _task(channel: str, primary="p1", max_outputs=1):
    allowed = {
        "EXPLOIT": ["SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"],
        "TRANSFORM": ["AXIS_MUTATION", "REGIME_BOUNDARY", "CROSS_SOURCE_BRIDGE"],
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
        task_id=f"task:{channel}:{primary}",
        primary_parent_idea_id=primary,
        eligible_secondary_parent_idea_ids=["p2"] if primary != "p2" else ["p1"],
        channel=channel,
        identity_goal=identity,
        operator_hints=allowed[:1],
        allowed_operator_ids=allowed,
        max_output_count=max_outputs,
    )


def _plan(task):
    return OffspringGenerationPlan(
        plan_id="plan:v23",
        source_v2_2_report_id="v22:1",
        source_generational_report_id="v211:1",
        tasks=[task],
        task_count=1,
        max_raw_offspring=task.max_output_count,
        planned_max_offspring=task.max_output_count,
        task_count_by_channel={task.channel: 1},
    )


class FakeOffspringBackend:
    backend_name = "fake"
    model_name = "fake"

    def __init__(self, first, repair=None):
        self.first = first
        self.repaired = repair
        self.repairs = 0

    def generate(self, prompt):
        return OffspringGeneration(draft=self.first, input_tokens=10, output_tokens=5)

    def repair(self, prompt, previous_draft, feedback):
        self.repairs += 1
        return OffspringGeneration(
            draft=self.repaired or previous_draft,
            input_tokens=7,
            output_tokens=4,
        )


def _generational(nodes):
    return NS(report_id="v211:1", report_sha256="a" * 64, research_ideas=nodes)


def _evolutionary():
    return NS(report_id="v22:1", source_generational_report_id="v211:1")


def _batch(task, candidates):
    return GenerationalOffspringBatchDraft(
        task_id=task.task_id,
        primary_parent_idea_id=task.primary_parent_idea_id,
        channel=task.channel,
        candidates=candidates,
        abstention_reason=None,
    )


def _candidate(local_id, operator, kernel, secondary=None):
    return GenerationalOffspringCandidateDraft(
        local_id=local_id,
        chosen_operator_id=operator,
        secondary_parent_idea_id=secondary,
        conceptual_change_summary=f"change {local_id}",
        kernel=kernel,
        differential_prediction="response changes",
        falsification_condition="response does not change",
        discriminating_observation="measure response",
        task_relation_mode="DIRECT",
    )


def test_generational_offspring_origin_kind_is_available():
    parent = _node("p1", "A --CAUSES--> B")
    task = _task("EXPLOIT")
    batch = _batch(
        task,
        [_candidate("x", "SAME_PREMISE_SHARPEN", parent.kernel)],
    )
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([parent, _node("p2", "C --CAUSES--> D")]),
        research_question="How does A affect B?",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    assert result.offspring_nodes[0].origin_kind == "GENERATIONAL_OFFSPRING"
    assert result.offspring_nodes[0].generation_index == 2


def test_exploit_same_kernel_is_accepted_refinement():
    parent = _node("p1", "A --CAUSES--> B")
    task = _task("EXPLOIT")
    batch = _batch(task, [_candidate("x", "EVIDENCE_REAXIS", parent.kernel)])
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([parent, _node("p2", "C --CAUSES--> D")]),
        research_question="q",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    row = result.semantic_records[0]
    assert row.transition.identity_relation == "SAME_IDEA"
    assert row.disposition == "ACCEPTED_REFINEMENT"
    assert row.accepted_for_realization is True


def test_transform_semantic_noop_retries_and_keeps_real_child():
    parent = _node("p1", "A --CAUSES--> B")
    other = _node("p2", "C --CAUSES--> D")
    task = _task("TRANSFORM")
    first = _batch(task, [_candidate("noop", "AXIS_MUTATION", parent.kernel)])
    child_kernel = _kernel("A --CAUSES--> M")
    repaired = _batch(task, [_candidate("child", "AXIS_MUTATION", child_kernel)])
    backend = FakeOffspringBackend(first, repaired)
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([parent, other]),
        research_question="q",
        backend=backend,
        semantic_retry_limit=1,
    )
    assert result.semantic_retry_count == 1
    assert backend.repairs == 1
    assert result.disposition_counts["SEMANTIC_NOOP"] == 1
    assert result.disposition_counts["ACCEPTED_CHILD"] == 1
    assert result.accepted_for_realization_count == 1
    assert result.mutation_attempt_count == 2
    assert result.distinct_child_count == 1
    assert result.mutation_semantic_yield_fraction == 0.5


def test_indeterminate_identity_is_retained_as_probe_not_hard_rejected():
    parent = _node("p1", "A --CAUSES--> B")
    other = _node("p2", "C --CAUSES--> D")
    task = _task("TRANSFORM")
    child = parent.kernel.model_copy(update={"scope_commitments": ["regime: low temperature"]})
    batch = _batch(task, [_candidate("scope", "REGIME_BOUNDARY", child)])
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([parent, other]),
        research_question="q",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    row = result.semantic_records[0]
    assert row.transition.identity_relation == "INDETERMINATE"
    assert row.disposition == "ACCEPTED_INDETERMINATE_PROBE"
    assert row.accepted_for_realization is True
    assert result.indeterminate_identity_is_not_automatic_rejection is True


def test_cross_source_bridge_can_create_multi_parent_composed_child():
    p1 = _node("p1", "A --CAUSES--> B")
    p2 = _node("p2", "C --CAUSES--> D")
    task = _task("TRANSFORM")
    bridge = _kernel("A --MODULATES--> D")
    batch = _batch(
        task,
        [_candidate("bridge", "CROSS_SOURCE_BRIDGE", bridge, secondary="p2")],
    )
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([p1, p2]),
        research_question="q",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    row = result.semantic_records[0]
    assert row.parent_idea_ids == ["p1", "p2"]
    assert row.transition.genealogy_relation == "COMPOSED_FROM"


def test_plan_preserves_transform_capacity_without_exceeding_global_bound():
    nodes = [_node(f"i{i}", f"A{i} --CAUSES--> B{i}") for i in range(4)]
    generational = NS(report_id="g:1", research_ideas=nodes)
    allocations = [
        NS(idea_id="i0", channel="EXPLOIT", operator_hints=[]),
        NS(idea_id="i1", channel="TRANSFORM", operator_hints=["AXIS_MUTATION"]),
        NS(idea_id="i2", channel="TRANSFORM", operator_hints=["REGIME_BOUNDARY"]),
        NS(idea_id="i3", channel="EXPLORE", operator_hints=[]),
    ]
    evolutionary = NS(
        report_id="e:1",
        source_generational_report_id="g:1",
        selected_parent_idea_ids=[x.idea_id for x in allocations],
        allocations=allocations,
        policy_states=[NS(idea_id=x.idea_id, reason_codes=[]) for x in allocations],
    )
    plan = build_offspring_generation_plan(
        evolutionary_report=evolutionary,
        generational_report=generational,
        max_raw_offspring=5,
    )
    assert plan.planned_max_offspring == 5
    assert plan.planned_max_offspring <= plan.max_raw_offspring
    assert any(row.channel == "TRANSFORM" and row.max_output_count == 2 for row in plan.tasks)


def _context():
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
                text="A is associated with an increase in B under condition C.",
                epistemic_role="reported",
                claim_kind="relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            )
        ],
    )


def _valid_draft(local_id="h1"):
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id=local_id,
                title="Grounded test of the generated idea",
                hypothesis_statement="Under condition C, A will increase B through a testable intermediate dependency.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["s1"],
                gap_statement_ids=[],
                inferential_bridge="The reported A-B association motivates testing whether an intermediate dependency mediates the response.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="B response",
                        expected_direction="increase",
                        rationale="The proposed bridge predicts a larger B response under the matched condition.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="B response",
                        falsifying_outcome="B does not increase under the matched comparison.",
                    )
                ],
                assumptions=["Condition C is comparable across the planned observation."],
            )
        ],
        abstention_reason=None,
    )


class FakeHypothesisBackend:
    backend_name = "fake"
    model_name = "fake"

    def generate(self, prompt):
        return HypothesisDraftGeneration(draft=_valid_draft(), input_tokens=9, output_tokens=5)

    def repair(self, prompt, previous_draft, feedback):
        return HypothesisDraftGeneration(draft=_valid_draft("h2"), input_tokens=6, output_tokens=4)


def _execution_for_realization():
    p1 = _node("p1", "A --CAUSES--> B")
    p2 = _node("p2", "C --CAUSES--> D")
    task = _task("TRANSFORM")
    batch = _batch(task, [_candidate("child", "AXIS_MUTATION", _kernel("A --CAUSES--> M"))])
    result, _ = execute_offspring_plan(
        plan=_plan(task),
        evolutionary_report=_evolutionary(),
        generational_report=_generational([p1, p2]),
        research_question="q",
        backend=FakeOffspringBackend(batch),
        semantic_retry_limit=0,
    )
    return result


def test_bounded_realization_uses_standard_grounded_compiler_and_builds_g3_seed():
    execution = _execution_for_realization()
    report, portfolio, _ = realize_offspring_bounded(
        execution=execution,
        context=_context(),
        backend=FakeHypothesisBackend(),
        max_realizations=1,
        max_repair_attempts=0,
    )
    assert report.selected_idea_count == 1
    assert report.materialized_hypothesis_count == 1
    assert report.status_counts == {"MATERIALIZED": 1}
    assert report.materialization_success_fraction == 1.0
    assert len(portfolio.hypotheses) == 1
    assert portfolio.hypotheses[0].premise_statement_ids == ["s1"]
    seed = build_g3_feedback_seed(execution=execution, realization=report)
    assert seed.observation_count == 1
    assert seed.realization_observations[0].target_scope == "REALIZATION"
    assert seed.realization_observations[0].materialization_status == "MATERIALIZED"
    assert seed.verification_is_not_fertility_authority is True


def test_realization_selector_prefers_distinct_child_over_refinement_within_channel_cycle():
    # The runtime helper already assigns transform before exploit in its lane cycle.
    execution = _execution_for_realization()
    ids = select_offspring_for_realization(execution, max_realizations=1)
    assert ids == [execution.semantic_records[0].idea_id]
