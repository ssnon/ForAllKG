from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dac_her.hypothesis_compiler import HypothesisCompiler
from dac_her.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    HypothesisRouteContext,
    PredictedObservationDraft,
)


def _stable_hex(*parts: object, length: int = 20) -> str:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _context(
    case_id: str,
    statements: list[HypothesisEvidenceStatement],
    *,
    question: str,
    routes: list[HypothesisRouteContext] | None = None,
    partial_blocked: list[str] | None = None,
) -> HypothesisContext:
    token = _stable_hex(case_id)
    return HypothesisContext(
        context_id=f"hypothesis_context:v262:{token}",
        context_sha256=_sha(f"context:{case_id}"),
        source_packet_id=f"packet:v262:{token}",
        source_packet_sha256=_sha(f"packet:{case_id}"),
        source_report_id=f"report:v262:{token}",
        source_report_sha256=_sha(f"report:{case_id}"),
        task_id=f"task:v262:{token}",
        question=question,
        corpus_id="dac_her_v262_fixture",
        evidence_statements=statements,
        mechanism_routes=list(routes or []),
        partial_absence_blocked_paper_ids=list(partial_blocked or []),
    )


def _statement(
    sid: str,
    text: str,
    *,
    paper: str = "Kiwook_9",
    role: str = "reported",
    kind: str = "mechanism",
    premise: bool = True,
    gap: bool = False,
    candidate: bool = False,
    alignment_paths: list[str] | None = None,
    restrictions: list[str] | None = None,
) -> HypothesisEvidenceStatement:
    return HypothesisEvidenceStatement(
        statement_id=sid,
        text=text,
        epistemic_role=role,
        claim_kind=kind,
        paper_ids=[paper],
        scientific_support_node_ids=[f"node:{sid}"],
        scientific_support_edge_ids=[],
        support_path_ids=[],
        alignment_path_ids=list(alignment_paths or []),
        requires_verification=candidate,
        eligible_as_premise=premise,
        eligible_as_gap=gap,
        premise_restrictions=list(restrictions or []),
    )


def _draft(
    local_id: str,
    premise_ids: list[str],
    *,
    statement: str = "The selected evidence may support a testable mechanistic extension.",
    bridge: str = "The proposed bridge is an explicitly hypothetical mechanistic link.",
    gap_ids: list[str] | None = None,
    observable: str = "Catalytic response under the hypothesized mechanism",
    direction: str = "qualitative_change",
) -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id=local_id,
                title=f"Fixture hypothesis {local_id}",
                hypothesis_statement=statement,
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=premise_ids,
                gap_statement_ids=list(gap_ids or []),
                inferential_bridge=bridge,
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable=observable,
                        expected_direction=direction,
                        rationale="The observation follows qualitatively from the proposed bridge.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable=observable,
                        falsifying_outcome="The predicted qualitative response is not observed.",
                    )
                ],
                assumptions=[],
            )
        ],
        abstention_reason=None,
    )


def _replace_card(portfolio: HypothesisPortfolio, **updates) -> HypothesisPortfolio:
    card = portfolio.hypotheses[0].model_copy(update=updates)
    return portfolio.model_copy(update={"hypotheses": [card]})


def build_cases(out_dir: Path) -> list[dict]:
    compiler = HypothesisCompiler()
    cases: list[dict] = []

    def save_case(
        case_id: str,
        category: str,
        description: str,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        expectation: dict,
    ) -> None:
        case_dir = out_dir / "cases" / case_id
        _write(case_dir / "context.json", context)
        _write(case_dir / "portfolio.json", portfolio)
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "description": description,
                "context_path": f"cases/{case_id}/context.json",
                "portfolio_path": f"cases/{case_id}/portfolio.json",
                "expectation": expectation,
            }
        )

    # Canonical: ordinary reported-premise hypothesis.
    c = _context(
        "canonical_valid",
        [_statement("s:reported", "Nitrogen coordination changes local hydrogen adsorption geometry.")],
        question="Can coordination motivate a falsifiable adsorption hypothesis?",
    )
    p = compiler.compile(c, _draft("h-valid", ["s:reported"]))
    save_case(
        "canonical_valid", "canonical", "Valid reported-premise hypothesis.", c, p,
        {"hard_gate_pass": True, "expected_abstention": False},
    )

    # Canonical: confirmed + candidate, with correct propagation.
    c = _context(
        "canonical_candidate",
        [
            _statement("s:reported", "Adsorption changes the local electronic structure."),
            _statement(
                "s:candidate",
                "A provisional electronic-state transition is associated with catalytic behavior.",
                role="evidence_synthesis",
                kind="association",
                candidate=True,
            ),
        ],
        question="Can a provisional electronic-state relation motivate a bounded hypothesis?",
    )
    p = compiler.compile(
        c,
        _draft(
            "h-candidate",
            ["s:reported", "s:candidate"],
            statement="The provisional electronic-state relation may contribute to catalytic behavior.",
            bridge="If the candidate relation holds, electronic restructuring could modify intermediate interactions.",
        ),
    )
    save_case(
        "canonical_candidate", "canonical", "Candidate dependence remains provisional.", c, p,
        {
            "hard_gate_pass": True,
            "expected_abstention": False,
            "forbidden_diagnostic_codes": ["CANDIDATE_OVERCLAIM_LANGUAGE"],
        },
    )

    # Canonical abstention.
    c = _context(
        "canonical_abstention",
        [],
        question="What mechanism follows from the supplied evidence?",
    )
    p = HypothesisPortfolio(
        portfolio_id="hypothesis_portfolio:v262:abstain",
        source_context_id=c.context_id,
        source_context_sha256=c.context_sha256,
        source_report_id=c.source_report_id,
        source_report_sha256=c.source_report_sha256,
        hypotheses=[],
        abstention_reason="No eligible positive premise is supplied.",
    )
    save_case(
        "canonical_abstention", "canonical", "Zero-evidence context should safely abstain.", c, p,
        {"hard_gate_pass": True, "expected_abstention": True},
    )

    # Hard adversarial: ineligible unresolved statement as premise.
    c = _context(
        "adv_ineligible_premise",
        [
            _statement("s:reported", "Coordination changes adsorption geometry."),
            _statement(
                "s:gap",
                "The supplied evidence does not establish charge-transfer mediation.",
                role="unresolved",
                kind="scope_limit",
                premise=False,
                gap=True,
                restrictions=["unresolved_not_positive_premise", "scope_limit_not_positive_premise"],
            ),
        ],
        question="Does charge transfer mediate the coordination effect?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported"]))
    profile = HypothesisEvidenceProfile(
        premise_count=1, gap_count=0, source_paper_count=1,
        candidate_premise_count=0, reported_premise_count=0, synthesis_premise_count=0,
    )
    p = _replace_card(
        base,
        premise_statement_ids=["s:gap"],
        source_paper_ids=["Kiwook_9"],
        evidence_profile=profile,
    )
    save_case(
        "adv_ineligible_premise", "adversarial", "Unresolved gap is used as a positive premise.", c, p,
        {
            "hard_gate_pass": False,
            "required_issue_codes": ["INELIGIBLE_POSITIVE_PREMISE"],
        },
    )

    # Hard adversarial: candidate metadata erased.
    c = _context(
        "adv_candidate_metadata_lost",
        [_statement(
            "s:candidate", "A provisional candidate relation links adsorption and electronic state.",
            role="evidence_synthesis", kind="association", candidate=True
        )],
        question="Can the candidate relation motivate a hypothesis?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:candidate"]))
    p = _replace_card(base, candidate_dependency="none")
    save_case(
        "adv_candidate_metadata_lost", "adversarial", "Candidate dependency metadata is erased.", c, p,
        {
            "hard_gate_pass": False,
            "required_issue_codes": ["CANDIDATE_DEPENDENCY_MISMATCH"],
        },
    )

    # Hard adversarial: partial paper absence claim.
    c = _context(
        "adv_partial_absence",
        [_statement(
            "s:k10", "Axial Co-O coordination is associated with activity and charge transfer.",
            paper="Kiwook_10"
        )],
        question="Could spillover mediate the axial Co-O activity relation?",
        partial_blocked=["Kiwook_10"],
    )
    base = compiler.compile(c, _draft("h-base", ["s:k10"]))
    p = _replace_card(
        base,
        hypothesis_statement="Kiwook_10 does not report hydrogen spillover, so spillover is absent from the study.",
    )
    save_case(
        "adv_partial_absence", "adversarial", "Unsafe paper-level absence claim on partial source.", c, p,
        {
            "hard_gate_pass": False,
            "required_issue_codes": ["PARTIAL_PAPER_ABSENCE_CLAIM"],
        },
    )

    # Hard adversarial: novelty.
    c = _context(
        "adv_novelty",
        [_statement("s:reported", "Coordination changes adsorption geometry.")],
        question="What mechanistic extension follows?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported"]))
    p = _replace_card(base, hypothesis_statement="We propose a novel mechanism linking coordination to adsorption.")
    save_case(
        "adv_novelty", "adversarial", "External novelty claim.", c, p,
        {"hard_gate_pass": False, "required_issue_codes": ["EXTERNAL_NOVELTY_CLAIM"]},
    )

    # Hard adversarial: unsupported number.
    c = _context(
        "adv_numeric",
        [_statement("s:reported", "Coordination changes adsorption geometry.")],
        question="What qualitative prediction follows?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported"]))
    card = base.hypotheses[0]
    pred = card.predicted_observations[0].model_copy(
        update={"rationale": "The proposed mechanism should improve the response by 37%."}
    )
    p = _replace_card(base, predicted_observations=[pred])
    save_case(
        "adv_numeric", "adversarial", "Unsupported quantitative prediction.", c, p,
        {"hard_gate_pass": False, "required_issue_codes": ["UNSUPPORTED_NUMERIC_PREDICTION"]},
    )

    # Hard adversarial: protocol leakage.
    c = _context(
        "adv_protocol",
        [_statement("s:reported", "Coordination changes adsorption geometry.")],
        question="What observation could falsify the mechanism?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported"]))
    p = _replace_card(
        base,
        inferential_bridge="To test the mechanism, anneal the catalyst at 500 C for 2 h before measurement.",
    )
    save_case(
        "adv_protocol", "adversarial", "Experiment protocol leaks into Hypothesis Maker output.", c, p,
        {"hard_gate_pass": False, "required_issue_codes": ["EXPERIMENT_PROTOCOL_LEAKAGE"]},
    )

    # Hard adversarial: alignment-bearing statement used as premise.
    c = _context(
        "adv_alignment_premise",
        [_statement(
            "s:alignment",
            "A registry-aligned concept links water dissociation and spillover.",
            role="evidence_synthesis",
            kind="association",
            alignment_paths=["path:alignment"],
        )],
        question="Does alignment establish a mechanism?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:alignment"]))
    save_case(
        "adv_alignment_premise", "adversarial", "Alignment-dependent statement is used as scientific premise.", c, base,
        {
            "hard_gate_pass": False,
            "required_issue_codes": ["ALIGNMENT_USED_AS_SCIENTIFIC_PREMISE"],
        },
    )

    # Hard adversarial: final portfolio with zero premise IDs.
    c = _context(
        "adv_zero_premise",
        [_statement("s:reported", "Coordination changes adsorption geometry.")],
        question="What follows?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported"]))
    zero_profile = HypothesisEvidenceProfile(
        premise_count=0, gap_count=0, source_paper_count=0,
        candidate_premise_count=0, reported_premise_count=0, synthesis_premise_count=0,
    )
    p = _replace_card(
        base,
        premise_statement_ids=[],
        source_paper_ids=[],
        cross_paper_synthesis=False,
        candidate_dependency="none",
        evidence_profile=zero_profile,
    )
    save_case(
        "adv_zero_premise", "adversarial", "Non-abstaining hypothesis has no positive premise.", c, p,
        {
            "hard_gate_pass": False,
            "required_issue_codes": ["HYPOTHESIS_WITHOUT_ELIGIBLE_PREMISE"],
        },
    )

    # Diagnostic adversarial: candidate overclaim language.
    c = _context(
        "adv_candidate_overclaim",
        [
            _statement("s:reported", "Adsorption changes local electronic structure."),
            _statement(
                "s:candidate", "A provisional relation links the state transition to behavior.",
                role="evidence_synthesis", kind="association", candidate=True
            ),
        ],
        question="Could the candidate relation explain behavior?",
    )
    base = compiler.compile(c, _draft("h-base", ["s:reported", "s:candidate"]))
    p = _replace_card(
        base,
        hypothesis_statement="The candidate relation proves that the electronic transition controls catalytic behavior.",
    )
    save_case(
        "adv_candidate_overclaim", "adversarial", "Candidate-dependent text is phrased as established evidence.", c, p,
        {
            "hard_gate_pass": True,
            "required_diagnostic_codes": ["CANDIDATE_OVERCLAIM_LANGUAGE"],
        },
    )

    # Diagnostic adversarial: alignment causalization without alignment premise.
    c = _context(
        "adv_alignment_causalization",
        [
            _statement("s:water", "Water dissociation contributes to alkaline HER.", paper="Kiwook_6"),
            _statement("s:spill", "Hydrogen spillover contributes to HER.", paper="Kiwook_4"),
        ],
        question="Can water dissociation and spillover be connected?",
        routes=[
            HypothesisRouteContext(
                route_id="route:alignment",
                statement_ids=["s:water", "s:spill"],
                paper_ids=["Kiwook_4", "Kiwook_6"],
                structural_type="PATTERN_ALIGNMENT",
                uses_alignment=True,
            )
        ],
    )
    base = compiler.compile(c, _draft("h-base", ["s:water", "s:spill"]))
    p = _replace_card(
        base,
        inferential_bridge="The graph alignment demonstrates that water dissociation causes hydrogen spillover.",
    )
    save_case(
        "adv_alignment_causalization", "adversarial", "Graph alignment is narrated as causal evidence.", c, p,
        {
            "hard_gate_pass": True,
            "required_diagnostic_codes": ["ALIGNMENT_CAUSALIZATION_LANGUAGE"],
        },
    )

    # Diagnostic adversarial: non-monotonic specificity not stated by premises.
    c = _context(
        "adv_directional_specificity",
        [_statement(
            "s:dgh",
            "Intermediate nitrogen coordination is associated with near-optimal hydrogen adsorption free energy.",
            kind="observation",
        )],
        question="How does adsorption free energy vary across coordination?",
    )
    p = compiler.compile(
        c,
        _draft(
            "h-direction",
            ["s:dgh"],
            statement="Coordination may tune hydrogen adsorption thermodynamics.",
            observable="Hydrogen adsorption free energy across nitrogen coordination",
            direction="non_monotonic",
        ),
    )
    save_case(
        "adv_directional_specificity", "adversarial", "Non-monotonic direction is stronger than supplied premise wording.", c, p,
        {
            "hard_gate_pass": True,
            "required_diagnostic_codes": ["UNJUSTIFIED_NON_MONOTONIC_SPECIFICITY"],
        },
    )

    # Diagnostic adversarial: association -> causal statement without modal qualifier.
    c = _context(
        "adv_causal_strengthening",
        [_statement(
            "s:assoc",
            "Axial coordination is associated with improved activity.",
            role="evidence_synthesis",
            kind="association",
        )],
        question="What relation can be hypothesized?",
    )
    p = compiler.compile(
        c,
        _draft(
            "h-causal",
            ["s:assoc"],
            statement="Axial coordination causes improved catalytic activity.",
            bridge="Axial coordination directly leads to the activity increase.",
        ),
    )
    save_case(
        "adv_causal_strengthening", "adversarial", "Association is strengthened into unqualified causation.", c, p,
        {
            "hard_gate_pass": True,
            "required_diagnostic_codes": ["POSSIBLE_CAUSAL_STRENGTHENING"],
        },
    )

    # Diagnostic adversarial: redundant hypotheses.
    c = _context(
        "adv_redundancy",
        [_statement("s:reported", "Charge redistribution may influence hydrogen adsorption.")],
        question="Generate distinct mechanistic hypotheses.",
    )
    one = _draft(
        "h1", ["s:reported"],
        statement="Charge redistribution may mediate hydrogen adsorption changes.",
        bridge="Charge redistribution may alter the electronic interaction with adsorbed hydrogen.",
    ).hypotheses[0]
    two = one.model_copy(
        update={
            "local_id": "h2",
            "title": "Near-duplicate hypothesis",
            "hypothesis_statement": "Charge redistribution may mediate hydrogen adsorption changes.",
            "inferential_bridge": "Charge redistribution may alter the electronic interaction with adsorbed hydrogen.",
        }
    )
    p = compiler.compile(
        c,
        HypothesisPortfolioDraft(hypotheses=[one, two], abstention_reason=None),
    )
    save_case(
        "adv_redundancy", "adversarial", "Portfolio contains near-duplicate hypotheses.", c, p,
        {
            "hard_gate_pass": True,
            "required_diagnostic_codes": ["HYPOTHESIS_REDUNDANCY"],
        },
    )

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic canonical/adversarial fixtures for Hypothesis Maker v2.6.2."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/hypothesis_v262/generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    cases = build_cases(out_dir)
    suite = {
        "schema_version": "hypothesis-benchmark-suite-v262",
        "suite_id": "hypothesis-v262-deterministic-a1",
        "evaluator_version": "hypothesis-benchmark-evaluator-v2.6.2-a1",
        "cases": cases,
    }
    _write(out_dir / "suite_v262.json", suite)
    print("Hypothesis v2.6.2 fixtures built")
    print("Cases:", len(cases))
    print("Suite:", out_dir / "suite_v262.json")


if __name__ == "__main__":
    main()
