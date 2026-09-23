from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPolicy,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Policy,
    build_regeneration_unit_v2_freeze,
    execute_regeneration_generation_unit_v2,
)


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="context:1",
        context_sha256="a" * 64,
        source_packet_id="packet:1",
        source_packet_sha256="b" * 64,
        source_report_id="report:1",
        source_report_sha256="c" * 64,
        task_id="task:1",
        question="How are X and Y related?",
        corpus_id="corpus:1",
        domain_profile_id="domain:1",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="statement:1",
                text="X is associated with Y.",
                epistemic_role="reported",
                claim_kind="reported_relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            )
        ],
        policy=HypothesisPolicy(),
    )


def _draft() -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="X-Y relation",
                hypothesis_statement="X modulates Y.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["statement:1"],
                inferential_bridge="The reported association may reflect modulation.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="Y response",
                        expected_direction="qualitative_change",
                        rationale="The bridge predicts a changed Y response.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="Y response",
                        falsifying_outcome="Y response is unchanged across X.",
                    )
                ],
            )
        ]
    )


class _Backend:
    backend_name = "test_backend"
    model_name = "test_model"

    def __init__(self, draft):
        self.draft = draft
        self.generate_calls = 0
        self.repair_calls = 0
        self.last_prompt = None

    def generate(self, prompt):
        self.generate_calls += 1
        self.last_prompt = prompt
        return HypothesisDraftGeneration(
            draft=self.draft,
            input_tokens=10,
            output_tokens=20,
            response_id="response:1",
        )

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        raise AssertionError("repair must never be called by regeneration v2")


def test_policy_separates_regeneration_from_downstream_evaluation() -> None:
    policy = ProspectiveRegenerationUnitV2Policy()
    assert policy.structured_generation_calls_per_hypothesis_max == 1
    assert policy.structured_repair_calls_allowed == 0
    assert policy.full_e2e_rerun_is_regeneration is False
    assert policy.downstream_evaluation_is_separate_stage is True
    assert policy.downstream_evaluation_requires_separately_frozen_budget is True


def test_execution_calls_generate_exactly_once_and_never_repairs() -> None:
    backend = _Backend(_draft())
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
    )
    assert result.status == "GENERATED_AND_COMPILED"
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0
    assert result.generation_calls_attempted == 1
    assert result.repair_calls_attempted == 0


def test_execution_limits_prompt_to_one_hypothesis() -> None:
    backend = _Backend(_draft())
    execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
    )
    assert "at most 1 focused hypotheses" in backend.last_prompt.user_prompt


def test_execution_compiles_deterministically_without_downstream_stages() -> None:
    backend = _Backend(_draft())
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
    )
    assert result.compiled_portfolio is not None
    assert len(result.compiled_portfolio.hypotheses) == 1
    assert result.external_novelty_performed is False
    assert result.n10_performed is False
    assert result.endpoint_binding_performed is False
    assert result.verifier_performed is False
    assert result.downstream_evaluation_performed is False


def test_compile_rejection_does_not_trigger_repair_call() -> None:
    bad = _draft().model_copy(deep=True)
    bad.hypotheses[0].premise_statement_ids = ["statement:missing"]
    backend = _Backend(bad)
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
    )
    assert result.status == "DETERMINISTIC_COMPILE_REJECTED"
    assert "UNKNOWN_PREMISE_STATEMENT" in result.compile_issue_codes
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0


def test_abstention_is_a_valid_single_call_result() -> None:
    backend = _Backend(
        HypothesisPortfolioDraft(
            hypotheses=[],
            abstention_reason="Insufficient evidence for a bounded hypothesis.",
        )
    )
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
    )
    assert result.status == "GENERATED_ABSTENTION_AND_COMPILED"
    assert result.compiled_portfolio is not None
    assert result.compiled_portfolio.hypotheses == []
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0


def test_freeze_requires_clean_worktree_and_records_protocol_fix_only() -> None:
    frozen = build_regeneration_unit_v2_freeze(
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.frozen_before_new_cohort_generation is True
    assert (
        frozen.prior_s135_protocol_failure_used_to_correct_execution_contract
        is True
    )
    assert (
        frozen.prior_s135_scientific_outputs_used_to_change_generation_content
        is False
    )
