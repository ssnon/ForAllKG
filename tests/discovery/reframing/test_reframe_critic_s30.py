from __future__ import annotations

from dataclasses import dataclass

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ReframeCriticDimensionDraft,
    ReframeCriticDraft,
)
from pipeline_core.discovery.reframing.critic_llm import ReframeCriticGeneration
from pipeline_core.discovery.reframing.critic_prompt import (
    ScientificReframeCriticPromptAssembler,
)
from pipeline_core.discovery.reframing.critic_runtime import (
    ScientificReframeCriticRuntime,
    compile_critic_review,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ReframeOperatorRunRecord,
    ScientificModelDraft,
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)


def _evidence() -> ScientificReframeEvidencePacket:
    return ScientificReframeEvidencePacket(
        task_id="task:t",
        question="How does environment affect SERS response?",
        source_context_id="ctx",
        source_context_sha256="sha",
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="s1",
                text="LSPR shifts with dielectric environment.",
                epistemic_role="reported",
                claim_kind="observation",
            ),
            ReframeEvidenceStatement(
                statement_id="s2",
                text="SERS maximum can be offset from far-field LSPR.",
                epistemic_role="reported",
                claim_kind="observation",
            ),
        ],
        gap_statements=[],
        grounded_source_chunk_count=2,
        unresolved_grounded_node_count=0,
    )


def _candidate() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="scientific_reframe:r1",
        operator_id="LATENT_VARIABLE",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        title="latent local coupling",
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=[],
        baseline_model=ScientificModelDraft(
            summary="far-field match controls response",
            explained_statement_ids=["s1"],
            expected_observations=["tracks LSPR"],
        ),
        alternative_model=ScientificModelDraft(
            summary="latent local coupling controls response",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["offset can persist"],
        ),
        challenged_assumption="far-field peak is sufficient",
        proposed_constructs=["local coupling landscape"],
        latent_constructs=["local coupling landscape"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp1",
                observable="SERS maximum",
                baseline_expectation="tracks LSPR",
                alternative_expectation="can remain offset",
                discriminating_outcome="persistent offset",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="LSPR alone predicts all cases",
            )
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="measure both spectra",
            primary_observables=["LSPR", "SERS maximum"],
            baseline_favoring_outcome="tracking",
            alternative_favoring_outcome="offset",
        ),
        unresolved_questions=[],
    )


def _shadow(candidate: ScientificReframeCandidate | None = None) -> ScientificReframingShadowReport:
    candidate = candidate or _candidate()
    return ScientificReframingShadowReport(
        report_id="shadow:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake-gen",
        model_name="fake-model",
        runs=[
            ReframeOperatorRunRecord(
                operator_id="LATENT_VARIABLE",
                readiness_status="ready_now",
                decision="generated",
                candidate_ids=[candidate.reframe_id],
            )
        ],
        candidates=[candidate],
        llm_calls_performed=1,
    )


def _full_draft(candidate_id: str = "scientific_reframe:r1") -> ReframeCriticDraft:
    return ReframeCriticDraft(
        candidate_id=candidate_id,
        dimensions=[
            ReframeCriticDimensionDraft(
                dimension=dimension,
                rating=3,
                rationale=f"{dimension} rationale",
                supporting_statement_ids=["s1"],
            )
            for dimension in CRITIC_DIMENSIONS
        ],
    )


def test_prompt_requires_vector_and_forbids_overall_ranking():
    prompt = ScientificReframeCriticPromptAssembler().build(
        candidate=_candidate(),
        evidence=_evidence(),
    )
    assert "Do not compute an overall score" in prompt.system_prompt
    assert "external novelty" in prompt.system_prompt
    assert "reframe_depth" in prompt.user_prompt
    assert "s1" in prompt.user_prompt


def test_compiler_preserves_canonical_dimension_order():
    review = compile_critic_review(
        candidate=_candidate(),
        evidence=_evidence(),
        draft=_full_draft(),
    )
    assert tuple(row.dimension for row in review.dimensions) == CRITIC_DIMENSIONS
    assert all(row.rating == 3 for row in review.dimensions)
    assert review.llm_review_complete is True
    assert review.structural_audit.operator_shape_valid is True


def test_missing_dimension_is_null_not_batch_failure():
    draft = _full_draft()
    draft = draft.model_copy(update={"dimensions": draft.dimensions[:-1]})
    review = compile_critic_review(
        candidate=_candidate(),
        evidence=_evidence(),
        draft=draft,
    )
    assert review.dimensions[-1].dimension == "reframe_depth"
    assert review.dimensions[-1].rating is None
    assert review.dimensions[-1].review_status == "missing"
    assert review.llm_review_complete is False


def test_unknown_duplicate_and_out_of_range_values_are_normalized():
    draft = ReframeCriticDraft(
        candidate_id="scientific_reframe:r1",
        dimensions=[
            ReframeCriticDimensionDraft(
                dimension="operator_validity",
                rating=7,
                supporting_statement_ids=["s1", "made_up"],
            ),
            ReframeCriticDimensionDraft(
                dimension="operator_validity",
                rating=0,
            ),
            ReframeCriticDimensionDraft(
                dimension="invented_dimension",
                rating=2,
            ),
        ],
    )
    review = compile_critic_review(
        candidate=_candidate(),
        evidence=_evidence(),
        draft=draft,
    )
    operator = review.dimensions[0]
    assert operator.rating == 3
    assert operator.supporting_statement_ids == ["s1"]
    assert any("clamped" in note for note in review.normalization_notes)
    assert any("unknown critic dimension" in note for note in review.normalization_notes)
    assert any("duplicate critic dimension" in note for note in review.normalization_notes)
    assert any("made_up" in note for note in review.normalization_notes)


@dataclass
class FakeBackend:
    backend_name: str = "fake"
    model_name: str = "fake-model"

    def review(self, prompt):
        return ReframeCriticGeneration(
            draft=_full_draft(prompt.candidate_id),
            input_tokens=10,
            output_tokens=20,
        )


def test_runtime_emits_vector_only_report_without_ranking():
    outcome = ScientificReframeCriticRuntime(FakeBackend()).run(
        shadow=_shadow(),
        evidence=_evidence(),
    )
    assert outcome.report.llm_calls_attempted == 1
    assert outcome.report.llm_calls_succeeded == 1
    assert outcome.report.quality_vector_only is True
    assert outcome.report.candidate_ranking_performed is False
    assert outcome.report.overall_score_computed is False
    assert len(outcome.report.reviews) == 1


@dataclass
class FailingBackend:
    backend_name: str = "fake"
    model_name: str = "fake-model"

    def review(self, prompt):
        raise RuntimeError("critic unavailable")


def test_runtime_keeps_structural_audit_when_llm_critic_fails():
    outcome = ScientificReframeCriticRuntime(FailingBackend()).run(
        shadow=_shadow(),
        evidence=_evidence(),
    )
    review = outcome.report.reviews[0]
    assert outcome.report.llm_calls_attempted == 1
    assert outcome.report.llm_calls_succeeded == 0
    assert review.llm_error_type == "RuntimeError"
    assert review.structural_audit.differential_prediction_count == 1
    assert all(row.rating is None for row in review.dimensions)


def test_structural_audit_tracks_explanatory_span_and_burden_without_scalar():
    review = compile_critic_review(
        candidate=_candidate(),
        evidence=_evidence(),
        draft=_full_draft(),
    )
    audit = review.structural_audit
    assert audit.alternative_explained_count == 2
    assert audit.speculative_construct_count == 1
    assert audit.differential_prediction_count == 1
    assert not hasattr(review, "overall_score")


def test_draft_candidate_id_mismatch_is_normalized_not_rejected():
    review = compile_critic_review(
        candidate=_candidate(),
        evidence=_evidence(),
        draft=_full_draft("wrong-id"),
    )
    assert review.candidate_id == "scientific_reframe:r1"
    assert any("candidate_id did not match" in note for note in review.normalization_notes)
