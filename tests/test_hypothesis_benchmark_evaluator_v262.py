from __future__ import annotations

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from dac_her.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)


def context(*, candidate=False, alignment=False):
    row = HypothesisEvidenceStatement(
        statement_id="s:1",
        text="A provisional association links coordination and adsorption." if candidate
        else "Coordination changes adsorption geometry.",
        epistemic_role="evidence_synthesis" if candidate else "reported",
        claim_kind="association" if candidate else "mechanism",
        paper_ids=["Kiwook_1"],
        scientific_support_node_ids=["n:1"],
        alignment_path_ids=["path:a"] if alignment else [],
        requires_verification=candidate,
        eligible_as_premise=True,
    )
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
        evidence_statements=[row],
    )


def draft():
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h",
                title="h",
                hypothesis_statement="Coordination may influence adsorption.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["s:1"],
                inferential_bridge="A hypothetical electronic bridge may connect the observations.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p",
                        observable="adsorption response",
                        expected_direction="qualitative_change",
                        rationale="test",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f",
                        observable="adsorption response",
                        falsifying_outcome="no corresponding response",
                    )
                ],
            )
        ]
    )


def test_valid_portfolio_passes():
    c = context()
    p = HypothesisCompiler().compile(c, draft())
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert report.hard_gate_passed
    assert report.hard_gate_errors == 0


def test_alignment_premise_is_hard_failure():
    c = context(alignment=True)
    p = HypothesisCompiler().compile(c, draft())
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert not report.hard_gate_passed
    assert "ALIGNMENT_USED_AS_SCIENTIFIC_PREMISE" in {
        issue.code for issue in report.hard_gate_issues
    }


def test_zero_premise_final_card_is_hard_failure():
    c = context()
    p = HypothesisCompiler().compile(c, draft())
    profile = HypothesisEvidenceProfile(
        premise_count=0,
        gap_count=0,
        source_paper_count=0,
        candidate_premise_count=0,
        reported_premise_count=0,
        synthesis_premise_count=0,
    )
    card = p.hypotheses[0].model_copy(
        update={
            "premise_statement_ids": [],
            "source_paper_ids": [],
            "candidate_dependency": "none",
            "evidence_profile": profile,
        }
    )
    p = p.model_copy(update={"hypotheses": [card]})
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert not report.hard_gate_passed
    assert "HYPOTHESIS_WITHOUT_ELIGIBLE_PREMISE" in {
        issue.code for issue in report.hard_gate_issues
    }


def test_empty_context_abstention_passes():
    c = HypothesisContext(
        context_id="ctx0",
        context_sha256="csha0",
        source_packet_id="packet0",
        source_packet_sha256="psha0",
        source_report_id="report0",
        source_report_sha256="rsha0",
        task_id="task0",
        question="q",
        corpus_id="corpus",
        evidence_statements=[],
    )
    p = HypothesisPortfolio(
        portfolio_id="p0",
        source_context_id=c.context_id,
        source_context_sha256=c.context_sha256,
        source_report_id=c.source_report_id,
        source_report_sha256=c.source_report_sha256,
        hypotheses=[],
        abstention_reason="No evidence.",
    )
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert report.hard_gate_passed
