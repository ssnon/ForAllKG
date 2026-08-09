from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisRouteContext,
)
from dac_her.hypothesis_e2_contracts import (
    HypothesisE2ContextCase,
    HypothesisE2ContextManifest,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _statement(
    statement_id: str,
    text: str,
    *,
    role: str = "reported",
    claim_kind: str = "mechanism",
    paper_ids: list[str] | None = None,
    requires_verification: bool = False,
    eligible_as_premise: bool = True,
    eligible_as_gap: bool = False,
    restrictions: list[str] | None = None,
) -> HypothesisEvidenceStatement:
    return HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=text,
        epistemic_role=role,
        claim_kind=claim_kind,
        paper_ids=list(paper_ids or []),
        scientific_support_node_ids=[f"controlled::{statement_id}"],
        scientific_support_edge_ids=[],
        support_path_ids=[],
        alignment_path_ids=[],
        requires_verification=requires_verification,
        eligible_as_premise=eligible_as_premise,
        eligible_as_gap=eligible_as_gap,
        premise_restrictions=list(restrictions or []),
    )


def _context(
    case_id: str,
    question: str,
    statements: list[HypothesisEvidenceStatement],
    *,
    routes: list[HypothesisRouteContext] | None = None,
    partial_absence_blocked_paper_ids: list[str] | None = None,
) -> HypothesisContext:
    token = _sha("e2:" + case_id)[:20]
    return HypothesisContext(
        context_id=f"hypothesis_context:e2:{token}",
        context_sha256=_sha("hypothesis-context-e2-v1:" + case_id),
        source_packet_id=f"packet:e2:{token}",
        source_packet_sha256=_sha("packet-e2:" + case_id),
        source_report_id=f"report:e2:{token}",
        source_report_sha256=_sha("report-e2:" + case_id),
        task_id=f"task:e2:{token}",
        question=question,
        corpus_id="dac_her_v262_e2_controlled_context",
        evidence_statements=statements,
        mechanism_routes=list(routes or []),
        partial_absence_blocked_paper_ids=list(
            partial_absence_blocked_paper_ids or []
        ),
    )


def build_e2_contexts() -> list[tuple[HypothesisE2ContextCase, HypothesisContext]]:
    rows: list[tuple[HypothesisE2ContextCase, HypothesisContext]] = []

    case_id = "e2_candidate_live"
    context = _context(
        case_id,
        (
            "Can a provisional electronic-state relation motivate a bounded "
            "mechanistic hypothesis for catalytic behavior without treating "
            "the provisional relation as established evidence?"
        ),
        [
            _statement(
                "e2:candidate:reported",
                (
                    "Hydrogen adsorption changes the local electronic structure "
                    "at the active site."
                ),
                paper_ids=["E2_Controlled_Candidate"],
            ),
            _statement(
                "e2:candidate:provisional",
                (
                    "A provisional electronic-state transition is associated "
                    "with catalytic response."
                ),
                role="evidence_synthesis",
                claim_kind="association",
                paper_ids=["E2_Controlled_Candidate"],
                requires_verification=True,
            ),
            _statement(
                "e2:candidate:gap",
                (
                    "The supplied context does not establish that the provisional "
                    "electronic-state transition causally controls catalytic response."
                ),
                role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["E2_Controlled_Candidate"],
                eligible_as_premise=False,
                eligible_as_gap=True,
                restrictions=[
                    "unresolved_not_positive_premise",
                    "scope_limit_not_positive_premise",
                ],
            ),
        ],
    )
    rows.append(
        (
            HypothesisE2ContextCase(
                case_id=case_id,
                scenario="candidate",
                description=(
                    "Fresh live Hypothesis Maker output that should preserve "
                    "candidate/provisional dependence."
                ),
                context_path=f"contexts/{case_id}.context.json",
                review_hint=(
                    "Human review focus: candidate_calibration should recognize "
                    "the provisional premise; causal wording must remain hypothetical."
                ),
            ),
            context,
        )
    )

    case_id = "e2_alignment_live"
    context = _context(
        case_id,
        (
            "Can water-dissociation evidence and hydrogen-spillover evidence "
            "motivate a cross-paper hypothesis while treating graph alignment "
            "as navigation rather than causal evidence?"
        ),
        [
            _statement(
                "e2:alignment:water",
                "Water dissociation contributes to alkaline HER behavior.",
                paper_ids=["E2_Controlled_Water"],
            ),
            _statement(
                "e2:alignment:spillover",
                "Hydrogen spillover contributes to HER behavior.",
                paper_ids=["E2_Controlled_Spillover"],
            ),
            _statement(
                "e2:alignment:gap",
                (
                    "The supplied context does not establish a causal coupling "
                    "between water dissociation and hydrogen spillover."
                ),
                role="unresolved",
                claim_kind="scope_limit",
                paper_ids=[
                    "E2_Controlled_Water",
                    "E2_Controlled_Spillover",
                ],
                eligible_as_premise=False,
                eligible_as_gap=True,
                restrictions=[
                    "unresolved_not_positive_premise",
                    "scope_limit_not_positive_premise",
                ],
            ),
        ],
        routes=[
            HypothesisRouteContext(
                route_id="e2:route:alignment",
                statement_ids=[
                    "e2:alignment:water",
                    "e2:alignment:spillover",
                ],
                paper_ids=[
                    "E2_Controlled_Water",
                    "E2_Controlled_Spillover",
                ],
                structural_type="PATTERN_ALIGNMENT",
                uses_alignment=True,
                uses_reverse_navigation=False,
                navigation_heavy=False,
                requires_verification=False,
            )
        ],
    )
    rows.append(
        (
            HypothesisE2ContextCase(
                case_id=case_id,
                scenario="alignment",
                description=(
                    "Fresh live cross-paper output with an alignment-bearing "
                    "route that must not become mechanistic evidence."
                ),
                context_path=f"contexts/{case_id}.context.json",
                review_hint=(
                    "Human review focus: cross_paper_discipline and causal_strengthening "
                    "must distinguish reported premises from alignment/navigation."
                ),
            ),
            context,
        )
    )

    case_id = "e2_partial_live"
    context = _context(
        case_id,
        (
            "What bounded hypothesis can be proposed from positive Kiwook_10 "
            "evidence about axial Co-O coordination, charge transfer, and activity "
            "without making paper-level absence claims?"
        ),
        [
            _statement(
                "e2:partial:k10_positive",
                (
                    "Axial Co-O coordination is associated with catalytic activity "
                    "and charge transfer."
                ),
                claim_kind="association",
                paper_ids=["Kiwook_10"],
            ),
            _statement(
                "e2:partial:gap",
                (
                    "The supplied context does not establish whether hydrogen "
                    "spillover mediates the axial-coordination relation."
                ),
                role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["Kiwook_10"],
                eligible_as_premise=False,
                eligible_as_gap=True,
                restrictions=[
                    "unresolved_not_positive_premise",
                    "scope_limit_not_positive_premise",
                ],
            ),
        ],
        partial_absence_blocked_paper_ids=["Kiwook_10"],
    )
    rows.append(
        (
            HypothesisE2ContextCase(
                case_id=case_id,
                scenario="partial",
                description=(
                    "Fresh live output using positive evidence from a partial "
                    "Kiwook_10 context without unsafe absence claims."
                ),
                context_path=f"contexts/{case_id}.context.json",
                review_hint=(
                    "Human review focus: positive Kiwook_10 evidence is allowed; "
                    "paper-level claims that Kiwook_10 lacks/does not report a relation are unsafe."
                ),
            ),
            context,
        )
    )

    case_id = "e2_abstention_live"
    context = _context(
        case_id,
        (
            "What falsifiable mechanistic hypothesis follows from the supplied "
            "navigation-only context?"
        ),
        [
            _statement(
                "e2:abstention:navigation",
                (
                    "A navigation-only graph note points toward HER-related "
                    "mechanistic concepts but does not assert a scientific relation."
                ),
                role="navigation_note",
                claim_kind="navigation",
                paper_ids=[],
                eligible_as_premise=False,
                eligible_as_gap=False,
                restrictions=["navigation_note_not_positive_premise"],
            )
        ],
    )
    rows.append(
        (
            HypothesisE2ContextCase(
                case_id=case_id,
                scenario="abstention",
                description=(
                    "Fresh live output with no eligible positive premise; "
                    "the Hypothesis Maker should abstain."
                ),
                context_path=f"contexts/{case_id}.context.json",
                review_hint=(
                    "Human review focus: abstention_appropriateness should accept "
                    "abstention and no restricted navigation note may become a premise."
                ),
            ),
            context,
        )
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled E2 contexts for fresh Hypothesis Maker live outputs."
    )
    parser.add_argument(
        "--output-root",
        default="data_dac/hypothesis_e2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).resolve()
    manifest_path = root / "e2_context_manifest.json"
    manifest_cases: list[HypothesisE2ContextCase] = []

    for case, context in build_e2_contexts():
        context_path = root / case.context_path
        _write_json(context_path, context)
        manifest_cases.append(case)

    manifest = HypothesisE2ContextManifest(
        suite_id="hypothesis-v262-e2-controlled-contexts",
        cases=manifest_cases,
    )
    _write_json(manifest_path, manifest)

    print("Hypothesis v2.6.2 E2 controlled contexts built")
    print("Cases:", len(manifest.cases))
    print("Saved:", manifest_path)
    for case in manifest.cases:
        print("-", case.case_id, "[" + case.scenario + "]")


if __name__ == "__main__":
    main()
