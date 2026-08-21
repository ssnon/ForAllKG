from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.feasibility_intake import FeasibilityIntakeBuilder
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReview,
    SEMANTIC_DIMENSIONS,
)


def _sha(value) -> str:
    payload = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_context() -> HypothesisContext:
    return HypothesisContext(
        domain_profile_id="dac_her",
        context_id="ctx",
        context_sha256="ctxsha",
        source_packet_id="packet",
        source_packet_sha256="packetsha",
        source_report_id="report",
        source_report_sha256="reportsha",
        task_id="task",
        question="How may coordination affect HER?",
        corpus_id="corpus",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:1",
                text="Coordination changes hydrogen adsorption geometry.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["Kiwook_9"],
                scientific_support_node_ids=["n:1"],
                eligible_as_premise=True,
            )
        ],
    )


def make_portfolio(context: HypothesisContext):
    draft = HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="Coordination-mediated adsorption",
                hypothesis_statement="Coordination may alter hydrogen adsorption and HER response.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["stmt:1"],
                inferential_bridge="A bounded coordination-to-adsorption inference.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="hydrogen adsorption",
                        expected_direction="qualitative_change",
                        rationale="Linked to the proposed bridge.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="hydrogen adsorption",
                        falsifying_outcome="No adsorption response to coordination change.",
                    )
                ],
            )
        ]
    )
    return HypothesisCompiler().compile(context, draft)


def make_review(context, portfolio, *, fail_dimension: str | None = None):
    rows = []
    for dimension in SEMANTIC_DIMENSIONS:
        if dimension == "hypothesis_distinctness":
            verdict = "not_applicable"
        elif dimension == fail_dimension:
            verdict = "fail"
        else:
            verdict = "pass"
        rows.append(
            HypothesisSemanticDimensionDraft(
                dimension=dimension,
                verdict=verdict,
                rationale="fixture",
                hypothesis_ids=(
                    [] if dimension == "abstention_appropriateness"
                    else [portfolio.hypotheses[0].hypothesis_id]
                ),
                statement_ids=["stmt:1"] if dimension == "premise_fidelity" else [],
            )
        )
    return HypothesisSemanticReview(
        review_id="review",
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_portfolio_id=portfolio.portfolio_id,
        source_portfolio_sha256=_sha(portfolio),
        source_evaluator_version="fixture",
        source_hard_gate_passed=True,
        critic_prompt_version="fixture",
        critic_prompt_sha256="promptsha",
        dimensions=rows,
        overall_summary="fixture",
    )


def test_intake_preserves_positive_premise_and_lineage():
    context = make_context()
    portfolio = make_portfolio(context)
    review = make_review(context, portfolio)
    intake = FeasibilityIntakeBuilder().build(context, portfolio, review)
    assert intake.hypotheses[0].premises[0].statement_id == "stmt:1"
    assert intake.hypotheses[0].semantic_gate_status == "eligible"
    assert intake.source_portfolio_sha256 == _sha(portfolio)


def test_critical_semantic_fail_routes_to_human_review():
    context = make_context()
    portfolio = make_portfolio(context)
    review = make_review(context, portfolio, fail_dimension="premise_fidelity")
    intake = FeasibilityIntakeBuilder().build(context, portfolio, review)
    assert intake.hypotheses[0].semantic_gate_status == "human_review_required"
