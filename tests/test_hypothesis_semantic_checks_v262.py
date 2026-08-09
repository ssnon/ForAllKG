from __future__ import annotations

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from dac_her.hypothesis_compiler import HypothesisCompiler
from dac_her.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    HypothesisRouteContext,
    PredictedObservationDraft,
)


def make_context(rows, *, routes=None):
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
        evidence_statements=rows,
        mechanism_routes=list(routes or []),
    )


def proposal(
    premises,
    *,
    statement,
    bridge,
    observable="response",
    direction="qualitative_change",
):
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h",
                title="h",
                hypothesis_statement=statement,
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=premises,
                inferential_bridge=bridge,
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p",
                        observable=observable,
                        expected_direction=direction,
                        rationale="rationale",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f",
                        observable=observable,
                        falsifying_outcome="counter-observation",
                    )
                ],
            )
        ]
    )


def codes(report):
    return {issue.code for issue in report.diagnostics}


def test_candidate_overclaim_is_diagnostic_not_hard_failure():
    rows = [
        HypothesisEvidenceStatement(
            statement_id="s:c",
            text="A provisional relation is associated with behavior.",
            epistemic_role="evidence_synthesis",
            claim_kind="association",
            paper_ids=["Kiwook_1"],
            scientific_support_node_ids=["n:c"],
            requires_verification=True,
            eligible_as_premise=True,
        )
    ]
    c = make_context(rows)
    p = HypothesisCompiler().compile(
        c,
        proposal(
            ["s:c"],
            statement="The candidate relation proves that the state controls behavior.",
            bridge="The candidate relation demonstrates the mechanism.",
        ),
    )
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert report.hard_gate_passed
    assert "CANDIDATE_OVERCLAIM_LANGUAGE" in codes(report)


def test_alignment_causalization_is_diagnostic():
    rows = [
        HypothesisEvidenceStatement(
            statement_id="s:a",
            text="Water dissociation contributes to HER.",
            epistemic_role="reported",
            claim_kind="mechanism",
            paper_ids=["Kiwook_6"],
            scientific_support_node_ids=["n:a"],
            eligible_as_premise=True,
        ),
        HypothesisEvidenceStatement(
            statement_id="s:b",
            text="Spillover contributes to HER.",
            epistemic_role="reported",
            claim_kind="mechanism",
            paper_ids=["Kiwook_4"],
            scientific_support_node_ids=["n:b"],
            eligible_as_premise=True,
        ),
    ]
    route = HypothesisRouteContext(
        route_id="route:a",
        statement_ids=["s:a", "s:b"],
        paper_ids=["Kiwook_4", "Kiwook_6"],
        structural_type="PATTERN_ALIGNMENT",
        uses_alignment=True,
    )
    c = make_context(rows, routes=[route])
    p = HypothesisCompiler().compile(
        c,
        proposal(
            ["s:a", "s:b"],
            statement="Water dissociation may connect to spillover.",
            bridge="The graph alignment demonstrates that water dissociation causes spillover.",
        ),
    )
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert report.hard_gate_passed
    assert "ALIGNMENT_CAUSALIZATION_LANGUAGE" in codes(report)


def test_nonmonotonic_specificity_is_diagnostic():
    rows = [
        HypothesisEvidenceStatement(
            statement_id="s:dgh",
            text="Intermediate coordination is associated with near-optimal hydrogen adsorption free energy.",
            epistemic_role="reported",
            claim_kind="observation",
            paper_ids=["Kiwook_9"],
            scientific_support_node_ids=["n:dgh"],
            eligible_as_premise=True,
        )
    ]
    c = make_context(rows)
    p = HypothesisCompiler().compile(
        c,
        proposal(
            ["s:dgh"],
            statement="Coordination may tune adsorption.",
            bridge="A proposed electronic bridge may mediate the relation.",
            observable="Hydrogen adsorption free energy across coordination",
            direction="non_monotonic",
        ),
    )
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert report.hard_gate_passed
    assert "UNJUSTIFIED_NON_MONOTONIC_SPECIFICITY" in codes(report)


def test_activity_volcano_can_justify_nonmonotonic_activity_without_warning():
    rows = [
        HypothesisEvidenceStatement(
            statement_id="s:v",
            text="HER exchange current density has a volcano dependence on hydrogen adsorption free energy.",
            epistemic_role="reported",
            claim_kind="observation",
            paper_ids=["Kiwook_9"],
            scientific_support_node_ids=["n:v"],
            eligible_as_premise=True,
        )
    ]
    c = make_context(rows)
    p = HypothesisCompiler().compile(
        c,
        proposal(
            ["s:v"],
            statement="Activity may vary across adsorption regimes.",
            bridge="The reported volcano motivates the activity trend.",
            observable="HER exchange current density across adsorption regimes",
            direction="non_monotonic",
        ),
    )
    report = HypothesisBenchmarkEvaluator().evaluate(c, p)
    assert "UNJUSTIFIED_NON_MONOTONIC_SPECIFICITY" not in codes(report)
