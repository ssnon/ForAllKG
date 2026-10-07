import hashlib
import json
from types import SimpleNamespace

from pydantic import BaseModel, Field

from pipeline_core.discovery.research_idea_closed_generation_cycle import (
    build_cycle_report,
    build_legacy_offspring_execution_payload,
    population_nodes,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicOffspringBatchDraft,
    EpistemicOffspringCandidateDraft,
    build_epistemic_generation_plan,
    execute_epistemic_generation,
)


class FakeKernel(BaseModel):
    canonical_intent: str
    core_scientific_commitments: list[str]
    scope_commitments: list[str] = Field(default_factory=list)
    contrastive_commitments: list[str] = Field(default_factory=list)
    question_commitment: str | None = None


class FakeNode(BaseModel):
    idea_id: str
    generation_index: int
    parent_idea_ids: list[str] = Field(default_factory=list)
    source_context_id: str = "ctx"
    source_context_sha256: str = "ctxsha"
    origin_kind: str = "GENERATIONAL_OFFSPRING"
    source_object_id: str = "src"
    kernel: FakeKernel
    kernel_sha256: str
    task_relation_mode: str = "DIRECT"
    differential_prediction: str = ""
    falsification_condition: str = ""
    discriminating_observation: str = ""


class FakeTransition(BaseModel):
    schema_version: str = "idea-transition-assessment-v1"
    proposed_idea_id: str = "child"
    parent_idea_ids: list[str] = Field(default_factory=list)
    parent_comparisons: list[dict] = Field(default_factory=list)
    identity_relation: str
    genealogy_relation: str = "CHILD_OF"
    operator_id: str | None = None
    operator_expectation_consistent: bool | None = True
    diagnostic_codes: list[str] = Field(default_factory=list)


def h(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parent(idea_id="i4"):
    kernel = FakeKernel(
        canonical_intent="Study hotspot organization",
        core_scientific_commitments=["geometry --CONTROLS--> hotspot organization"],
        scope_commitments=["SERS"],
        question_commitment="Does geometry control hotspot organization?",
    )
    return FakeNode(
        idea_id=idea_id,
        generation_index=4,
        source_object_id=f"src:{idea_id}",
        kernel=kernel,
        kernel_sha256=h(kernel.model_dump(mode="json")),
    )


def parallel_fixture():
    member = SimpleNamespace(
        member_id="m4",
        idea_id="i4",
        epistemic_realization_id="e4",
        epistemic_maturity="EVIDENCE_SEEKING",
        grounding_coverage="NONE",
        materialization_status="ABSTAINED",
        prediction_present=True,
        falsifier_present=True,
        missing_evidence_requirement_count=1,
        residual_epistemic_state=None,
        prospective_identifiability=None,
    )
    debt = SimpleNamespace(
        debt_id="d4",
        idea_id="i4",
        requirement_kind="RELATION_OR_MECHANISM_EVIDENCE",
        normalized_description="Need mechanism evidence",
        occurrence_count=2,
        disposition="ACQUISITION_ESCALATION_ELIGIBLE",
    )
    handoff = SimpleNamespace(
        handoff_id="h4",
        idea_id="i4",
        generation_index=4,
        source_member_ids=["m4"],
        source_epistemic_realization_ids=["e4"],
        recommended_channels=["TRANSFORM", "EXPLORE"],
        nonbinding_operator_hints=["LATENT_VARIABLE"],
        reason_codes=["TEST"],
    )
    return SimpleNamespace(
        report_id="parallel:g4",
        members=[member],
        epistemic_debts=[debt],
        evolution_handoffs=[handoff],
    )


class StaticBackend:
    def __init__(self, draft):
        self.draft = draft

    def generate(self, prompt):
        return SimpleNamespace(draft=self.draft, input_tokens=10, output_tokens=5)

    def repair(self, prompt, previous_draft, feedback):
        return SimpleNamespace(draft=self.draft, input_tokens=10, output_tokens=5)


def fake_node_builder(*, candidate, task, parent_by_id, source_parallel_report_id):
    primary = parent_by_id[task.primary_parent_idea_id]
    kernel = FakeKernel.model_validate(candidate.kernel)
    node = FakeNode(
        idea_id=f"g{task.generation_index}:{candidate.local_id}",
        generation_index=task.generation_index,
        parent_idea_ids=[primary.idea_id],
        source_context_id=primary.source_context_id,
        source_context_sha256=primary.source_context_sha256,
        source_object_id=f"src:g{task.generation_index}:{candidate.local_id}",
        kernel=kernel,
        kernel_sha256=h(kernel.model_dump(mode="json")),
    )
    return node, [primary]


def fake_transition_assessor(*, parents, proposed, operator_id):
    return FakeTransition(
        proposed_idea_id=proposed.idea_id,
        parent_idea_ids=[parents[0].idea_id],
        identity_relation="DIFFERENT_IDEA",
        operator_id=operator_id,
    )


def test_v2_9_generation_contract_generalizes_to_g5():
    p = parent()
    parallel = parallel_fixture()
    plan = build_epistemic_generation_plan(
        parallel_report=parallel,
        parent_by_id={p.idea_id: p},
        generation_index=5,
        max_parents=1,
    )
    assert plan.generation_index == 5
    assert plan.tasks[0].generation_index == 5
    assert plan.plan_id.startswith("g5_epistemic_generation_plan:")

    child_kernel = p.kernel.model_copy(
        update={
            "core_scientific_commitments": [
                "latent mode participation --MEDIATES--> hotspot organization"
            ],
            "question_commitment": "Does latent mode participation mediate hotspot organization?",
        }
    )
    task = plan.tasks[0]
    draft = EpistemicOffspringBatchDraft(
        task_id=task.task_id,
        primary_parent_idea_id=task.primary_parent_idea_id,
        channel=task.channel,
        candidates=[
            EpistemicOffspringCandidateDraft(
                local_id="c1",
                chosen_operator_id="LATENT_VARIABLE",
                conceptual_change_summary="Introduce a latent mediator",
                kernel=child_kernel.model_dump(mode="json"),
                differential_prediction="Mediator stratification changes hotspots",
                falsification_condition="No change after mediator stratification",
                task_relation_mode="DIRECT",
            )
        ],
    )
    execution, _ = execute_epistemic_generation(
        plan=plan,
        parallel_report=parallel,
        parent_by_id={p.idea_id: p},
        research_question="How does geometry affect SERS hotspots?",
        backend=StaticBackend(draft),
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.generation_index == 5
    assert execution.genuine_child_count == 1
    assert execution.g4_population_nodes[0].generation_index == 5
    assert execution.semantic_records[0].generation_index == 5
    assert execution.run_records[0].generation_index == 5
    assert execution.report_id.startswith("g5_epistemic_evolution:")


def test_realization_adapter_preserves_generation_and_only_retained_population():
    p = parent()
    child = p.model_copy(
        update={
            "idea_id": "g5:c1",
            "generation_index": 5,
            "parent_idea_ids": [p.idea_id],
            "kernel": p.kernel.model_copy(
                update={"core_scientific_commitments": ["new mechanism"]}
            ),
        }
    )
    execution = SimpleNamespace(
        report_id="g5:report",
        source_plan_id="g5:plan",
        source_parallel_report_id="parallel:g4",
        generation_index=5,
        offspring_nodes=[child],
        g4_population_nodes=[child],
        semantic_records=[
            SimpleNamespace(
                idea_id=child.idea_id,
                task_id="t",
                channel="TRANSFORM",
                chosen_operator_id="LATENT_VARIABLE",
                parent_idea_ids=[p.idea_id],
                transition={
                    "schema_version": "idea-transition-assessment-v1",
                    "proposed_idea_id": child.idea_id,
                    "parent_idea_ids": [p.idea_id],
                    "parent_comparisons": [],
                    "identity_relation": "DIFFERENT_IDEA",
                    "genealogy_relation": "CHILD_OF",
                    "operator_id": "LATENT_VARIABLE",
                    "operator_expectation_consistent": True,
                    "diagnostic_codes": [],
                },
                disposition="GENUINE_CHILD",
                retained_in_g4_population=True,
                conceptual_change_summary="new mechanism",
                diagnostic_codes=[],
                identity_relation="DIFFERENT_IDEA",
            )
        ],
        llm_call_count=1,
        input_tokens=10,
        output_tokens=5,
        semantic_retry_count=0,
    )
    payload = build_legacy_offspring_execution_payload(execution)
    assert payload["generation_index"] == 5
    assert payload["raw_offspring_count"] == 1
    assert payload["accepted_for_realization_count"] == 1
    assert payload["semantic_records"][0]["disposition"] == "ACCEPTED_CHILD"
    assert payload["semantic_records"][0]["accepted_for_realization"] is True
    assert population_nodes(execution)[0].idea_id == child.idea_id


def test_cycle_report_marks_next_execution_as_recursively_consumable():
    execution = SimpleNamespace(
        generation_index=4,
        report_id="g4:execution",
        g4_population_nodes=[SimpleNamespace(idea_id="i1")],
    )
    lifecycle = SimpleNamespace(
        source_context_id="ctx",
        source_context_sha256="sha",
        target_idea_ids=["i1"],
        links=[SimpleNamespace(realization_id="r1")],
        materialized_hypothesis_count=0,
        usable_grounded_realization_count=0,
        rescued_within_same_idea_count=0,
        realization_llm_call_count=1,
        prospective_audit_llm_call_count=0,
    )
    decomposition = SimpleNamespace(record_count=1, maturity_counts={"EVIDENCE_SEEKING": 1})
    archive = SimpleNamespace(retained_entry_count=1)
    parallel = SimpleNamespace(
        active_member_count=1,
        active_idea_count=1,
        epistemic_debt_count=2,
        evolution_handoff_count=1,
    )
    next_plan = SimpleNamespace(plan_id="g5:plan", task_count=1)
    next_execution = SimpleNamespace(
        report_id="g5:execution",
        raw_offspring_count=1,
        g4_population_count=1,
        genuine_child_count=1,
    )
    report = build_cycle_report(
        execution=execution,
        lifecycle=lifecycle,
        decomposition=decomposition,
        archive=archive,
        parallel=parallel,
        next_plan=next_plan,
        next_execution=next_execution,
    )
    assert report.current_generation_index == 4
    assert report.next_generation_index == 5
    assert report.feedback_loop_closed is True
    assert report.same_cycle_runner_can_consume_next_execution is True
    assert report.next_generation_genuine_child_count == 1


def test_genericization_keeps_v2_9_legacy_transport_compatibility():
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
        _normalize_generation,
    )

    payload = {
        "schema_version": "generational-offspring-batch-draft-v1",
        "task_id": "t",
        "primary_parent_idea_id": "i",
        "channel": "TRANSFORM",
        "candidates": [],
        "abstention_reason": "test abstention",
    }
    batch, input_tokens, output_tokens = _normalize_generation(
        SimpleNamespace(draft=payload, input_tokens=3, output_tokens=2)
    )
    assert batch.task_id == "t"
    assert batch.abstention_reason == "test abstention"
    assert input_tokens == 3
    assert output_tokens == 2


def test_cycle_epistemic_artifacts_feed_lifecycle_observations_back_without_claim_authority():
    from pipeline_core.discovery.research_idea_closed_generation_cycle import (
        build_cycle_epistemic_artifacts,
    )

    execution = SimpleNamespace(
        g4_population_nodes=[SimpleNamespace(idea_id="i4", generation_index=4)]
    )
    lifecycle = SimpleNamespace(
        observations=[
            SimpleNamespace(
                hypothesis_id="h1",
                idea_id="i4",
                prospective_status=None,
                current_evidence_status=None,
                prospective_identifiability=None,
                directionality_mode=None,
                measurement_compatibility_mode=None,
                residual_epistemic_state=None,
                residual_state_reason=None,
            )
        ]
    )
    portfolio = SimpleNamespace(hypotheses=[])
    seen = {}

    def decompose(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(record_count=1)

    def archive(decomposition):
        assert decomposition.record_count == 1
        return SimpleNamespace(retained_entry_count=1)

    def parallel(**kwargs):
        assert kwargs["acquisition_persistence_threshold"] == 3
        return SimpleNamespace(evolution_handoff_count=1)

    decomposition, archive_report, parallel_report = build_cycle_epistemic_artifacts(
        execution=execution,
        lifecycle=lifecycle,
        portfolio=portfolio,
        acquisition_persistence_threshold=3,
        decomposition_builder=decompose,
        archive_builder=archive,
        parallel_builder=parallel,
    )
    assert seen["research_ideas"][0].idea_id == "i4"
    assert seen["feedback_reports"][0]["records"][0]["hypothesis_id"] == "h1"
    assert decomposition.record_count == 1
    assert archive_report.retained_entry_count == 1
    assert parallel_report.evolution_handoff_count == 1


def test_zero_next_population_is_not_claimed_as_recursively_consumable():
    execution = SimpleNamespace(
        generation_index=4,
        report_id="g4:execution",
        g4_population_nodes=[SimpleNamespace(idea_id="i1")],
    )
    lifecycle = SimpleNamespace(
        source_context_id="ctx",
        source_context_sha256="sha",
        target_idea_ids=["i1"],
        links=[],
        materialized_hypothesis_count=0,
        usable_grounded_realization_count=0,
        rescued_within_same_idea_count=0,
        realization_llm_call_count=0,
        prospective_audit_llm_call_count=0,
    )
    decomposition = SimpleNamespace(record_count=1, maturity_counts={"IDEA_ONLY": 1})
    archive = SimpleNamespace(retained_entry_count=1)
    parallel = SimpleNamespace(
        active_member_count=1,
        active_idea_count=1,
        epistemic_debt_count=1,
        evolution_handoff_count=1,
    )
    report = build_cycle_report(
        execution=execution,
        lifecycle=lifecycle,
        decomposition=decomposition,
        archive=archive,
        parallel=parallel,
        next_plan=SimpleNamespace(plan_id="g5:plan", task_count=1),
        next_execution=SimpleNamespace(
            report_id="g5:empty",
            raw_offspring_count=1,
            g4_population_count=0,
            genuine_child_count=0,
        ),
    )
    assert report.same_cycle_runner_can_consume_next_execution is False
