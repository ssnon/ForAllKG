from __future__ import annotations

from pathlib import Path

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

    def __init__(self):
        self.generate_calls = 0
        self.repair_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        return HypothesisDraftGeneration(
            draft=_draft(),
            input_tokens=10,
            output_tokens=20,
            response_id="response:1",
        )

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        raise AssertionError("regeneration-v2 must not repair")


def test_unit_still_means_one_generate_zero_repair() -> None:
    backend = _Backend()
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=backend,
        policy=freeze.policy,
    )
    assert result.status == "GENERATED_AND_COMPILED"
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0
    assert result.previous_hypothesis_text_consumed is False
    assert result.novelty_outcome_consumed is False
    assert result.verifier_outcome_consumed is False


def test_unit_performs_no_downstream_scientific_stages() -> None:
    result = execute_regeneration_generation_unit_v2(
        context=_context(),
        backend=_Backend(),
    )
    assert result.retrieval_performed is False
    assert result.semantic_critic_performed is False
    assert result.external_novelty_performed is False
    assert result.n9_performed is False
    assert result.n10_performed is False
    assert result.endpoint_binding_performed is False
    assert result.verifier_performed is False
    assert result.downstream_evaluation_performed is False
