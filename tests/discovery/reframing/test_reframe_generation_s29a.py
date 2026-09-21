from __future__ import annotations


from pipeline_core.corpus.semantic_ir.backfill import build_backfill_plan
from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRequirementCheck,
)
from pipeline_core.corpus.semantic_ir.task_capability import TaskCapabilitySnapshot
from pipeline_core.discovery.reframing.operator_readiness import (
    OperatorReadinessAssessment,
    ReframingOperatorReadinessReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ScientificModelDraft,
    ScientificReframeBatchDraft,
    ScientificReframeDraft,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ConditionEvidenceExample,
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.reframe_llm import (
    ReframeDraftGeneration,
)
from pipeline_core.discovery.reframing.reframe_prompt import (
    ScientificReframePromptAssembler,
)
from pipeline_core.discovery.reframing.reframe_runtime import (
    ScientificReframingShadowRuntime,
)


def _model(summary: str, explained: list[str]) -> ScientificModelDraft:
    return ScientificModelDraft(
        summary=summary,
        assumptions=["assumption"],
        explained_statement_ids=explained,
        expected_observations=[summary + " observation"],
    )


def _prediction() -> DifferentialPredictionDraft:
    return DifferentialPredictionDraft(
        local_id="dp1",
        observable="response curve",
        baseline_expectation="one continuous trend",
        alternative_expectation="a breakpoint separates two response regimes",
        discriminating_outcome="a reproducible breakpoint after controlling other conditions",
    )


def _test() -> DiscriminatingTestDraft:
    return DiscriminatingTestDraft(
        test_design="sweep the candidate boundary variable while measuring response",
        primary_observables=["response curve"],
        baseline_favoring_outcome="one smooth response law fits the full range",
        alternative_favoring_outcome="two regimes fit better with a reproducible transition",
    )


def _regime_draft(*, premise_ids: list[str] | None = None) -> ScientificReframeDraft:
    premise_ids = premise_ids or ["s1", "s2"]
    return ScientificReframeDraft(
        local_id="r1",
        operator_id="REGIME_BOUNDARY",
        title="Response-law transition",
        premise_statement_ids=premise_ids,
        gap_statement_ids=[],
        baseline_model=_model("single response model", premise_ids),
        alternative_model=_model("condition-dependent response regimes", premise_ids),
        challenged_assumption="one response model applies across all conditions",
        proposed_constructs=[],
        latent_constructs=[],
        boundary_variables=["dielectric environment"],
        regime_change_kind="slope_change",
        differential_predictions=[_prediction()],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="a single response law remains adequate across the sweep",
            )
        ],
        discriminating_test=_test(),
    )


def _latent_draft() -> ScientificReframeDraft:
    return ScientificReframeDraft(
        local_id="l1",
        operator_id="LATENT_VARIABLE",
        title="Shared electromagnetic accessibility",
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=[],
        baseline_model=_model("two unrelated associations", ["s1", "s2"]),
        alternative_model=_model("one latent construct links both observations", ["s1", "s2"]),
        challenged_assumption="the two observations require independent explanations",
        proposed_constructs=["effective electromagnetic accessibility"],
        latent_constructs=["effective electromagnetic accessibility"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[_prediction()],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="the two observations vary independently after controlling the proposed construct",
            )
        ],
        discriminating_test=_test(),
    )


def _evidence() -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question="How does dielectric environment affect SERS response?",
        source_context_id="context:1",
        source_context_sha256="sha-context",
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="s1",
                text="Observation one",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["P1"],
            ),
            ReframeEvidenceStatement(
                statement_id="s2",
                text="Observation two",
                epistemic_role="evidence_synthesis",
                claim_kind="association",
                paper_ids=["P2"],
            ),
        ],
        gap_statements=[
            ReframeEvidenceStatement(
                statement_id="g1",
                text="Boundary remains unresolved",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["P1"],
            )
        ],
        condition_examples=[
            ConditionEvidenceExample(
                object_kind="measurement",
                paper_id="P1",
                chunk_id="c1",
                object_id="m1",
                label="SERS intensity",
                conditions=[{"name": "wavelength", "value_numeric": 633}],
                value_summary="10 a.u.",
                direct_grounded_object=True,
            )
        ],
        grounded_source_chunk_count=3,
        unresolved_grounded_node_count=0,
    )


def _empty_plan(name: str):
    return build_backfill_plan(
        plan_id=name,
        requirements=[],
        targets=[],
    )


def _readiness() -> ReframingOperatorReadinessReport:
    manifest = CapabilityManifest(
        manifest_id="manifest:task",
        scope_kind="task",
        scope_id="task:1",
        capabilities={},
    )
    snapshot = TaskCapabilitySnapshot(
        scope_id="task:1",
        manifest=manifest,
        metrics={},
        source_chunk_count=3,
        semantic_record_count=10,
    )
    check = CapabilityRequirementCheck(
        manifest_id=manifest.manifest_id,
        requirement_count=0,
        satisfied_count=0,
        backfill_required_count=0,
        assessments=[],
    )
    return ReframingOperatorReadinessReport(
        scope_id="task:1",
        task_capabilities=snapshot,
        assessments=[
            OperatorReadinessAssessment(
                operator_id="LATENT_VARIABLE",
                status="ready_now",
                hard_requirement_check=check,
                mandatory_backfill_plan=_empty_plan("latent-m"),
                optional_backfill_plan=_empty_plan("latent-o"),
                reasons=["ready"],
                task_source_chunk_count=3,
                measurement_bearing_chunk_count=1,
            ),
            OperatorReadinessAssessment(
                operator_id="REGIME_BOUNDARY",
                status="ready_with_optional_enrichment",
                hard_requirement_check=check,
                mandatory_backfill_plan=_empty_plan("regime-m"),
                optional_backfill_plan=_empty_plan("regime-o"),
                reasons=["optional enrichment"],
                task_source_chunk_count=3,
                measurement_bearing_chunk_count=1,
            ),
        ],
    )


class FakeBackend:
    backend_name = "fake"
    model_name = "fake-model"

    def generate(self, prompt):
        draft = (
            _latent_draft()
            if prompt.operator_id == "LATENT_VARIABLE"
            else _regime_draft()
        )
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id=prompt.operator_id,
                candidates=[draft],
                abstention_reason=None,
            ),
            input_tokens=100,
            output_tokens=50,
            response_id="resp",
            elapsed_seconds=0.1,
        )


class InvalidPremiseBackend(FakeBackend):
    def generate(self, prompt):
        draft = _regime_draft(premise_ids=["s1", "not_grounded"])
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="REGIME_BOUNDARY",
                candidates=[draft],
            )
        )


class OmittedExplainedPremiseBackend(FakeBackend):
    def generate(self, prompt):
        draft = _latent_draft()
        payload = draft.model_dump(mode="json")
        payload["premise_statement_ids"] = ["s1"]
        payload["baseline_model"]["explained_statement_ids"] = ["s1", "s2"]
        payload["alternative_model"]["explained_statement_ids"] = ["s1", "s2"]
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="LATENT_VARIABLE",
                candidates=[draft],
            )
        )


class UnknownExplainedPremiseBackend(FakeBackend):
    def generate(self, prompt):
        draft = _latent_draft()
        payload = draft.model_dump(mode="json")
        payload["baseline_model"]["explained_statement_ids"] = ["s1", "outside"]
        payload["alternative_model"]["explained_statement_ids"] = ["s1", "s2"]
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="LATENT_VARIABLE",
                candidates=[draft],
            )
        )


class LatentWithRegimeFieldsBackend(FakeBackend):
    def generate(self, prompt):
        payload = _latent_draft().model_dump(mode="json")
        payload["boundary_variables"] = ["dielectric environment"]
        payload["regime_change_kind"] = "qualitative_response_change"
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="LATENT_VARIABLE",
                candidates=[draft],
            )
        )


class RegimeWithLatentFieldsBackend(FakeBackend):
    def generate(self, prompt):
        payload = _regime_draft().model_dump(mode="json")
        payload["latent_constructs"] = ["accidental latent label"]
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="REGIME_BOUNDARY",
                candidates=[draft],
            )
        )


class LatentTooNarrowBackend(FakeBackend):
    def generate(self, prompt):
        payload = _latent_draft().model_dump(mode="json")
        payload["alternative_model"]["explained_statement_ids"] = ["s1"]
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="LATENT_VARIABLE",
                candidates=[draft],
            )
        )


class RegimeMissingBoundaryBackend(FakeBackend):
    def generate(self, prompt):
        payload = _regime_draft().model_dump(mode="json")
        payload["boundary_variables"] = []
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="REGIME_BOUNDARY",
                candidates=[draft],
            )
        )


class NoDifferentialPredictionBackend(FakeBackend):
    def generate(self, prompt):
        payload = _regime_draft().model_dump(mode="json")
        payload["differential_predictions"] = []
        draft = ScientificReframeDraft.model_validate(payload)
        return ReframeDraftGeneration(
            draft=ScientificReframeBatchDraft(
                operator_id="REGIME_BOUNDARY",
                candidates=[draft],
            )
        )


class RaisingBackend(FakeBackend):
    def generate(self, prompt):
        raise RuntimeError("synthetic structured generation failure")


def test_semantic_operator_rules_are_not_instructor_parse_rules() -> None:
    latent_payload = _latent_draft().model_dump(mode="json")
    latent_payload["boundary_variables"] = ["dielectric environment"]
    latent_payload["regime_change_kind"] = "qualitative_response_change"
    latent_payload["alternative_model"]["explained_statement_ids"] = ["s1"]
    assert ScientificReframeDraft.model_validate(latent_payload)

    regime_payload = _regime_draft().model_dump(mode="json")
    regime_payload["boundary_variables"] = []
    regime_payload["differential_predictions"] = []
    assert ScientificReframeDraft.model_validate(regime_payload)


def test_latent_regime_fields_are_stripped_at_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(LatentWithRegimeFieldsBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["LATENT_VARIABLE"],
    )
    candidate = outcome.report.candidates[0]
    assert candidate.boundary_variables == []
    assert candidate.regime_change_kind is None
    assert any(
        "stripped regime-specific fields" in note
        for note in candidate.provenance_normalizations
    )


def test_regime_latent_fields_are_stripped_at_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(RegimeWithLatentFieldsBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["REGIME_BOUNDARY"],
    )
    candidate = outcome.report.candidates[0]
    assert candidate.latent_constructs == []
    assert any(
        "stripped latent_constructs" in note
        for note in candidate.provenance_normalizations
    )


def test_latent_requires_two_explained_statements_at_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(LatentTooNarrowBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["LATENT_VARIABLE"],
    )
    assert outcome.report.candidates == []
    assert outcome.report.runs[0].decision == "rejected_invalid_draft"
    assert "at least two grounded premises" in outcome.report.runs[0].compile_issues[0]


def test_regime_requires_boundary_variable_at_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(RegimeMissingBoundaryBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["REGIME_BOUNDARY"],
    )
    assert outcome.report.candidates == []
    assert outcome.report.runs[0].decision == "rejected_invalid_draft"
    assert "boundary_variable" in outcome.report.runs[0].compile_issues[0]


def test_differential_prediction_requirement_is_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(NoDifferentialPredictionBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["REGIME_BOUNDARY"],
    )
    assert outcome.report.candidates == []
    assert outcome.report.runs[0].decision == "rejected_invalid_draft"
    assert "differential prediction" in outcome.report.runs[0].compile_issues[0]


def test_generation_failure_is_recorded_without_crashing_shadow_lane() -> None:
    outcome = ScientificReframingShadowRuntime(RaisingBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["LATENT_VARIABLE"],
    )
    assert outcome.report.candidates == []
    assert outcome.report.llm_calls_performed == 1
    assert outcome.report.runs[0].decision == "generation_failed"
    assert outcome.report.runs[0].generation_error_type == "RuntimeError"


def test_prompt_exposes_conditions_only_to_regime_operator() -> None:
    evidence = _evidence()
    assembler = ScientificReframePromptAssembler()
    latent = assembler.build(operator_id="LATENT_VARIABLE", evidence=evidence)
    regime = assembler.build(operator_id="REGIME_BOUNDARY", evidence=evidence)
    assert '"structured_condition_examples": []' in latent.user_prompt
    assert '"object_id": "m1"' in regime.user_prompt
    assert "effect becomes stronger or weaker is NOT a regime reframe" in regime.system_prompt
    assert "Set boundary_variables=[] and regime_change_kind=null" in latent.system_prompt
    assert "Set latent_constructs=[]" in regime.system_prompt


def test_runtime_generates_shadow_candidates_without_authority() -> None:
    outcome = ScientificReframingShadowRuntime(FakeBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
    )
    assert outcome.report.llm_calls_performed == 2
    assert len(outcome.report.candidates) == 2
    assert {row.operator_id for row in outcome.report.candidates} == {
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
    }
    assert all(row.shadow_only for row in outcome.report.candidates)
    assert all(row.requires_verification for row in outcome.report.candidates)
    assert not outcome.report.production_selection_changed
    assert not outcome.report.external_novelty_evaluated


def test_runtime_skips_operator_that_is_not_ready() -> None:
    readiness = _readiness()
    regime = readiness.assessments[1].model_copy(
        update={"status": "targeted_enrichment_required"}
    )
    readiness = readiness.model_copy(
        update={"assessments": [readiness.assessments[0], regime]}
    )
    outcome = ScientificReframingShadowRuntime(FakeBackend()).run(
        evidence=_evidence(),
        readiness=readiness,
    )
    assert outcome.report.llm_calls_performed == 1
    assert outcome.report.runs[1].decision == "skipped_not_ready"
    assert all(
        candidate.operator_id != "REGIME_BOUNDARY"
        for candidate in outcome.report.candidates
    )


def test_runtime_rejects_non_grounded_premise_id_without_crashing_batch() -> None:
    outcome = ScientificReframingShadowRuntime(InvalidPremiseBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["REGIME_BOUNDARY"],
    )
    assert outcome.report.llm_calls_performed == 1
    assert outcome.report.candidates == []
    assert outcome.report.runs[0].decision == "rejected_invalid_draft"
    assert outcome.report.runs[0].rejected_candidate_count == 1
    assert "not_grounded" in outcome.report.runs[0].compile_issues[0]




def test_grounded_explained_statement_omitted_from_candidate_list_is_promoted() -> None:
    outcome = ScientificReframingShadowRuntime(OmittedExplainedPremiseBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["LATENT_VARIABLE"],
    )
    assert outcome.report.runs[0].decision == "generated"
    assert len(outcome.report.candidates) == 1
    candidate = outcome.report.candidates[0]
    assert candidate.premise_statement_ids == ["s1", "s2"]
    assert candidate.provenance_normalizations
    assert "s2" in candidate.provenance_normalizations[0]


def test_non_grounded_explained_statement_is_rejected_at_compile_time() -> None:
    outcome = ScientificReframingShadowRuntime(UnknownExplainedPremiseBackend()).run(
        evidence=_evidence(),
        readiness=_readiness(),
        operators=["LATENT_VARIABLE"],
    )
    assert outcome.report.candidates == []
    assert outcome.report.runs[0].decision == "rejected_invalid_draft"
    assert "outside" in outcome.report.runs[0].compile_issues[0]


def test_prompt_requires_explained_ids_to_be_declared_as_premises() -> None:
    prompt = ScientificReframePromptAssembler().build(
        operator_id="LATENT_VARIABLE",
        evidence=_evidence(),
    )
    assert "explained_statement_ids" in prompt.system_prompt
    assert "premise_statement_ids" in prompt.system_prompt

def test_batch_may_abstain_instead_of_filling_quota() -> None:
    batch = ScientificReframeBatchDraft(
        operator_id="REGIME_BOUNDARY",
        candidates=[],
        abstention_reason="No grounded evidence distinguishes a regime change from continuous modulation.",
    )
    assert batch.candidates == []
    assert batch.abstention_reason
