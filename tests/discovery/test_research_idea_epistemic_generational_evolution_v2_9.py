import hashlib
import json
from types import SimpleNamespace

from pydantic import BaseModel, Field

from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicOffspringBatchDraft,
    EpistemicOffspringCandidateDraft,
    build_epistemic_g4_generation_plan,
    build_epistemic_g4_prompt,
    execute_epistemic_g4_generation,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    GenerationalOffspringBatchDraft,
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
    identity_relation: str
    diagnostic_codes: list[str] = Field(default_factory=list)


def h(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parent(idea_id, core, *, intent=None):
    kernel = FakeKernel(
        canonical_intent=intent or f"Study {core}",
        core_scientific_commitments=[core],
        scope_commitments=["SERS"],
        contrastive_commitments=[],
        question_commitment=f"Does {core}?",
    )
    return FakeNode(
        idea_id=idea_id,
        generation_index=3,
        source_object_id=f"src:{idea_id}",
        kernel=kernel,
        kernel_sha256=h(kernel.model_dump(mode="json")),
    )


def member(mid, idea, maturity):
    return SimpleNamespace(
        member_id=mid,
        idea_id=idea,
        epistemic_realization_id=f"e:{mid}",
        epistemic_maturity=maturity,
        grounding_coverage=(
            "STRICT" if maturity in {"STRICT_GROUNDED", "OPERATIONAL_GROUNDED"} else "NONE"
        ),
        materialization_status=(
            "MATERIALIZED" if maturity in {"STRICT_GROUNDED", "OPERATIONAL_GROUNDED"} else "ABSTAINED"
        ),
        prediction_present=True,
        falsifier_present=True,
        missing_evidence_requirement_count=(0 if "GROUNDED" in maturity else 1),
        residual_epistemic_state=None,
        prospective_identifiability=None,
    )


def debt(did, idea, kind):
    return SimpleNamespace(
        debt_id=did,
        idea_id=idea,
        requirement_kind=kind,
        normalized_description=f"Need {kind}",
        occurrence_count=2,
        disposition="ACQUISITION_ESCALATION_ELIGIBLE",
    )


def handoff(hid, idea, mid, *, channels=("TRANSFORM", "EXPLORE"), hints=("REGIME_BOUNDARY",)):
    return SimpleNamespace(
        handoff_id=hid,
        idea_id=idea,
        generation_index=3,
        source_member_ids=[mid],
        source_epistemic_realization_ids=[f"e:{mid}"],
        recommended_channels=list(channels),
        nonbinding_operator_hints=list(hints),
        reason_codes=["TEST_FEEDBACK"],
    )


def parallel_report(rows, debts, handoffs):
    return SimpleNamespace(
        report_id="parallel:test",
        members=rows,
        epistemic_debts=debts,
        evolution_handoffs=handoffs,
    )


def fake_node_builder(*, candidate, task, parent_by_id, source_parallel_report_id):
    primary = parent_by_id[task.primary_parent_idea_id]
    kernel = FakeKernel.model_validate(candidate.kernel)
    node = FakeNode(
        idea_id=f"g4:{task.task_id}:{candidate.local_id}",
        generation_index=4,
        parent_idea_ids=[task.primary_parent_idea_id],
        source_context_id=primary.source_context_id,
        source_context_sha256=primary.source_context_sha256,
        source_object_id=f"src:{task.task_id}:{candidate.local_id}",
        kernel=kernel,
        kernel_sha256=h(kernel.model_dump(mode="json")),
        task_relation_mode=candidate.task_relation_mode,
        differential_prediction=candidate.differential_prediction,
        falsification_condition=candidate.falsification_condition,
        discriminating_observation=candidate.discriminating_observation,
    )
    return node, [primary]


def fake_transition_assessor(*, parents, proposed, operator_id):
    parent_kernel = parents[0].kernel
    child_kernel = proposed.kernel
    if parent_kernel.model_dump(mode="json") == child_kernel.model_dump(mode="json"):
        relation = "SAME_IDEA"
    elif (
        parent_kernel.core_scientific_commitments
        == child_kernel.core_scientific_commitments
        and parent_kernel.question_commitment == child_kernel.question_commitment
    ):
        relation = "INDETERMINATE"
    else:
        relation = "DIFFERENT_IDEA"
    return FakeTransition(identity_relation=relation)


class StaticBackend:
    def __init__(self, first, repair=None):
        self.first = first
        self.repair_value = repair
        self.generate_calls = 0
        self.repair_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        return SimpleNamespace(draft=self.first, input_tokens=10, output_tokens=5)

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        if self.repair_value is None:
            raise RuntimeError("no repair configured")
        return SimpleNamespace(draft=self.repair_value, input_tokens=11, output_tokens=6)


def batch(task, kernel, *, local_id="c1"):
    return EpistemicOffspringBatchDraft(
        task_id=task.task_id,
        primary_parent_idea_id=task.primary_parent_idea_id,
        channel=task.channel,
        candidates=[
            EpistemicOffspringCandidateDraft(
                local_id=local_id,
                chosen_operator_id=task.operator_hints[0],
                conceptual_change_summary="Substantive conceptual change",
                kernel=kernel.model_dump(mode="json"),
                differential_prediction="Different regime changes the response",
                falsification_condition="The response is invariant across regimes",
                task_relation_mode="DIRECT",
            )
        ],
    )


def one_parent_fixture(maturity="SPECULATIVE_BUT_FALSIFIABLE"):
    p = parent("i1", "geometry --CONTROLS--> hotspot distribution")
    m = member("m1", "i1", maturity)
    d = debt("d1", "i1", "RELATION_OR_MECHANISM_EVIDENCE")
    pr = parallel_report([m], [d], [handoff("h1", "i1", "m1")])
    plan = build_epistemic_g4_generation_plan(
        parallel_report=pr,
        parent_by_id={"i1": p},
        max_parents=1,
    )
    return p, pr, plan


def test_bounded_parent_selection_prefers_epistemically_diverse_sources():
    parents = {
        "i1": parent("i1", "geometry --CONTROLS--> hotspots"),
        "i2": parent("i2", "modal length --MODULATES--> hotspots"),
        "i3": parent("i3", "orientation --MODULATES--> Raman intensity"),
    }
    rows = [
        member("m1", "i1", "SPECULATIVE_BUT_FALSIFIABLE"),
        member("m2", "i2", "EVIDENCE_SEEKING"),
        member("m3", "i3", "STRICT_GROUNDED"),
    ]
    debts = [
        debt("d1", "i1", "RELATION_OR_MECHANISM_EVIDENCE"),
        debt("d2", "i2", "MEASUREMENT_OR_OBSERVABLE_SUPPORT"),
    ]
    handoffs = [
        handoff("h1", "i1", "m1"),
        handoff("h2", "i2", "m2"),
        handoff("h3", "i3", "m3"),
    ]
    report = parallel_report(rows, debts, handoffs)
    plan = build_epistemic_g4_generation_plan(
        parallel_report=report,
        parent_by_id=parents,
        max_parents=2,
    )
    assert plan.task_count == 2
    assert plan.selected_parent_source_mode_counts["SPECULATIVE"] == 1
    assert plan.selected_parent_source_mode_counts["EVIDENCE_SEEKING"] == 1
    assert len(set(task.channel for task in plan.tasks)) == 2
    assert plan.evidence_acquisition_required_before_generation is False


def test_speculative_feedback_can_generate_genuine_g4_child_without_grounding():
    p, pr, plan = one_parent_fixture()
    task = plan.tasks[0]
    child = p.kernel.model_copy(
        update={
            "core_scientific_commitments": [
                "geometry --CONTROLS--> modal regime --CONTROLS--> hotspot distribution"
            ],
            "question_commitment": "Does a modal regime mediate geometry-dependent hotspots?",
        }
    )
    backend = StaticBackend(batch(task, child))
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.genuine_child_count == 1
    assert execution.g4_population_count == 1
    assert execution.genuine_child_from_non_grounded_feedback_count == 1
    assert execution.genuine_child_from_speculative_feedback_count == 1
    assert execution.grounding_required_for_child_generation is False
    assert execution.grounding_required_before_scientific_claim is True
    assert execution.evidence_acquisition_executed is False
    assert execution.canonical_graph_mutated is False


def test_same_idea_generation_returns_to_realization_lane_not_g4_population():
    p, pr, plan = one_parent_fixture("EVIDENCE_SEEKING")
    task = plan.tasks[0]
    backend = StaticBackend(batch(task, p.kernel))
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.same_idea_refinement_count == 1
    assert execution.genuine_child_count == 0
    assert execution.g4_population_count == 0
    assert execution.same_idea_is_routed_back_to_realization_lane


def test_indeterminate_identity_is_retained_as_bounded_g4_probe():
    p, pr, plan = one_parent_fixture("EVIDENCE_SEEKING")
    task = plan.tasks[0]
    changed_scope = p.kernel.model_copy(update={"scope_commitments": ["SERS", "narrow regime"]})
    backend = StaticBackend(batch(task, changed_scope))
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.indeterminate_probe_count == 1
    assert execution.genuine_child_count == 0
    assert execution.g4_population_count == 1
    assert execution.indeterminate_identity_is_retained_as_bounded_probe


def test_semantic_retry_can_convert_same_idea_noop_into_genuine_child():
    p, pr, plan = one_parent_fixture("EVIDENCE_SEEKING")
    task = plan.tasks[0]
    child = p.kernel.model_copy(
        update={
            "core_scientific_commitments": [
                "latent coupling --MEDIATES--> hotspot distribution"
            ],
            "question_commitment": "Does latent coupling mediate the hotspot pattern?",
        }
    )
    backend = StaticBackend(
        batch(task, p.kernel, local_id="same"),
        repair=batch(task, child, local_id="child"),
    )
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=1,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert backend.generate_calls == 1
    assert backend.repair_calls == 1
    assert execution.semantic_retry_count == 1
    assert execution.same_idea_refinement_count == 1
    assert execution.genuine_child_count == 1
    assert execution.llm_call_count == 2


def test_prompt_marks_epistemic_feedback_as_search_context_not_evidence():
    p, pr, plan = one_parent_fixture()
    prompt = build_epistemic_g4_prompt(
        task=plan.tasks[0],
        parent_by_id={"i1": p},
        parallel_report=pr,
        research_question="How does structure affect SERS response?",
    )
    assert "SEARCH FEEDBACK ONLY" in prompt.system_prompt
    assert "not positive premises" in prompt.system_prompt
    assert "epistemic_debt_search_context_only" in prompt.user_prompt
    assert "Grounding is deliberately NOT a precondition" in prompt.system_prompt


def test_exact_kernel_duplicate_of_other_existing_idea_is_suppressed_from_g4():
    p, pr, plan = one_parent_fixture("EVIDENCE_SEEKING")
    task = plan.tasks[0]
    other = parent("i2", "latent coupling --MEDIATES--> hotspot distribution")
    backend = StaticBackend(batch(task, other.kernel))
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p, "i2": other},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.exact_kernel_duplicate_suppressed_count == 1
    assert execution.genuine_child_count == 0
    assert execution.g4_population_count == 0
    assert any(
        code.startswith("DUPLICATE_EXISTING_RESEARCH_IDEA_KERNEL")
        for code in execution.semantic_records[0].diagnostic_codes
    )


def test_legacy_v23_backend_batch_schema_is_normalized_at_v29_adapter_boundary():
    p, pr, plan = one_parent_fixture("EVIDENCE_SEEKING")
    task = plan.tasks[0]
    child = p.kernel.model_copy(
        update={
            "core_scientific_commitments": [
                "geometry --CONTROLS--> modal regime --CONTROLS--> hotspot distribution"
            ],
            "question_commitment": "Does a modal regime mediate geometry-dependent hotspots?",
        }
    )
    legacy = GenerationalOffspringBatchDraft.model_validate(
        {
            "schema_version": "generational-offspring-batch-draft-v1",
            "task_id": task.task_id,
            "primary_parent_idea_id": task.primary_parent_idea_id,
            "channel": task.channel,
            "candidates": [
                {
                    "local_id": "legacy-c1",
                    "chosen_operator_id": task.operator_hints[0],
                    "secondary_parent_idea_id": None,
                    "conceptual_change_summary": "Insert a modal-regime mediator.",
                    "kernel": child.model_dump(mode="json"),
                    "differential_prediction": "Different modal regimes change hotspot distribution.",
                    "falsification_condition": "Hotspot distribution is invariant to modal regime.",
                    "discriminating_observation": "Resolve modal regime and hotspot topology jointly.",
                    "task_relation_mode": "DIRECT",
                }
            ],
            "abstention_reason": None,
        }
    )
    backend = StaticBackend(legacy)
    execution, _ = execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=pr,
        parent_by_id={"i1": p},
        research_question="How does structure affect SERS response?",
        backend=backend,
        semantic_retry_limit=0,
        node_builder=fake_node_builder,
        transition_assessor=fake_transition_assessor,
    )
    assert execution.raw_offspring_count == 1
    assert execution.genuine_child_count == 1
    assert execution.g4_population_count == 1
    assert execution.run_records[0].decision == "GENERATED"
    assert execution.run_records[0].generation_error is None
