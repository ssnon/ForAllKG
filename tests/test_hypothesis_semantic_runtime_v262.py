from __future__ import annotations

from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReviewDraft,
    SEMANTIC_DIMENSIONS,
)
from dac_her.hypothesis_semantic_llm import HypothesisSemanticGeneration
from dac_her.hypothesis_semantic_runtime import HypothesisSemanticCriticRuntime


class FakeBackend:
    backend_name = "fake_semantic_critic"
    model_name = "fake-model"
    temperature = 0.0
    instructor_mode = "FAKE"
    base_url = "fake://local"
    parse_retries = 0

    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def review(self, prompt):
        self.calls += 1
        return HypothesisSemanticGeneration(
            draft=self.draft,
            input_tokens=20,
            output_tokens=10,
        )


def make_context():
    return HypothesisContext(
        context_id="ctx",
        context_sha256="csha",
        source_packet_id="packet",
        source_packet_sha256="psha",
        source_report_id="report",
        source_report_sha256="rsha",
        task_id="task",
        question="q",
        corpus_id="corpus",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s:1",
                text="Coordination changes adsorption geometry.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["Kiwook_1"],
                scientific_support_node_ids=["n:1"],
                eligible_as_premise=True,
            )
        ],
    )


def make_portfolio(context):
    draft = HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h",
                title="h",
                hypothesis_statement="Coordination may influence adsorption.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["s:1"],
                inferential_bridge="A bounded hypothetical bridge.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p",
                        observable="response",
                        expected_direction="qualitative_change",
                        rationale="linked",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f",
                        observable="response",
                        falsifying_outcome="no response",
                    )
                ],
            )
        ]
    )
    return HypothesisCompiler().compile(context, draft)


def review_draft(portfolio, *, unknown=False):
    hid = "hypothesis:unknown" if unknown else portfolio.hypotheses[0].hypothesis_id
    rows = []
    for dimension in SEMANTIC_DIMENSIONS:
        rows.append(
            HypothesisSemanticDimensionDraft(
                dimension=dimension,
                verdict=(
                    "not_applicable"
                    if dimension == "hypothesis_distinctness"
                    else "pass"
                ),
                rationale="fixture review",
                hypothesis_ids=[] if dimension == "abstention_appropriateness" else [hid],
                statement_ids=["s:1"] if dimension == "premise_fidelity" else [],
            )
        )
    return HypothesisSemanticReviewDraft(
        dimensions=rows,
        overall_summary="fixture semantic review",
    )


def test_runtime_accepts_valid_structured_review():
    context = make_context()
    portfolio = make_portfolio(context)
    backend = FakeBackend(review_draft(portfolio))
    outcome = HypothesisSemanticCriticRuntime(backend).run(context, portfolio)
    assert outcome.accepted
    assert outcome.review is not None
    assert backend.calls == 1
    assert outcome.run_record.failure_stage == "none"
    assert outcome.run_record.input_tokens == 20
    assert outcome.run_record.output_tokens == 10


def test_runtime_rejects_invented_hypothesis_reference():
    context = make_context()
    portfolio = make_portfolio(context)
    backend = FakeBackend(review_draft(portfolio, unknown=True))
    outcome = HypothesisSemanticCriticRuntime(backend).run(context, portfolio)
    assert not outcome.accepted
    assert outcome.review is None
    assert outcome.run_record.failure_stage == "review_validation"
    assert outcome.review_validation_issues


def test_runtime_does_not_call_critic_when_hard_gate_fails():
    context = make_context()
    portfolio = make_portfolio(context)
    card = portfolio.hypotheses[0]
    bad_profile = HypothesisEvidenceProfile(
        premise_count=0,
        gap_count=0,
        source_paper_count=0,
        candidate_premise_count=0,
        reported_premise_count=0,
        synthesis_premise_count=0,
    )
    card = card.model_copy(
        update={
            "premise_statement_ids": [],
            "source_paper_ids": [],
            "evidence_profile": bad_profile,
        }
    )
    portfolio = portfolio.model_copy(update={"hypotheses": [card]})
    backend = FakeBackend(review_draft(make_portfolio(context)))
    outcome = HypothesisSemanticCriticRuntime(backend).run(context, portfolio)
    assert not outcome.accepted
    assert backend.calls == 0
    assert outcome.run_record.failure_stage == "hard_gate"
    assert outcome.run_record.generated is False
