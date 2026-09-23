from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotationReport,
    InstructorGroundedIdentityBackend,
    analyze_current_claims,
    run_annotation,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create exact-source grounded constituent annotations for synthetic "
            "atomic prior-art identities, then run a diagnostic-only negative-closure "
            "ablation on current atomic claims only. Stale detail directories are ignored."
        )
    )
    parser.add_argument("--candidate-portfolio", required=True, type=Path)
    parser.add_argument("--atomic-report", required=True, type=Path)
    parser.add_argument("--detail-root", type=Path, default=None)
    parser.add_argument("--annotation-output", required=True, type=Path)
    parser.add_argument("--ablation-output", type=Path, default=None)
    parser.add_argument(
        "--annotation-only",
        action="store_true",
        help=(
            "Materialize exact-source grounded identity annotations only. "
            "This mode is independent of N9/N10 closure detail artifacts and "
            "does not run the diagnostic negative-closure ablation."
        ),
    )
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument(
        "--minimum-negative-abstracts",
        type=int,
        default=3,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    candidates = ProductionFacingScientificCandidatePortfolio.model_validate_json(
        args.candidate_portfolio.read_text(encoding="utf-8")
    )
    atomic_report = AtomicCrossLaneSynthesisReport.model_validate_json(
        args.atomic_report.read_text(encoding="utf-8")
    )

    backend = InstructorGroundedIdentityBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
    )
    annotation_report, system, user = run_annotation(
        candidates=candidates,
        atomic_report=atomic_report,
        backend=backend,
    )

    args.annotation_output.parent.mkdir(parents=True, exist_ok=True)
    args.annotation_output.write_text(
        annotation_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.prompt_output is not None:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(
            "SYSTEM\n======\n"
            + system
            + "\n\nUSER\n====\n"
            + user
            + "\n",
            encoding="utf-8",
        )

    if args.annotation_only:
        print("Grounded identity constituent annotation complete")
        print("Annotation LLM calls:", annotation_report.llm_calls_performed)
        print("Negative-closure ablation performed: false")
        print("N9/N10 closure detail dependency: false")
        print("N9 contract changed: false")
        print("N10 contract changed: false")
        print("Production authority: false")
        print("Production selection changed: false")
        print("Annotation:", args.annotation_output)
        return 0

    if args.detail_root is None:
        raise SystemExit("--detail-root is required unless --annotation-only is set")
    if args.ablation_output is None:
        raise SystemExit("--ablation-output is required unless --annotation-only is set")

    ablation = analyze_current_claims(
        detail_root=args.detail_root,
        atomic_report=atomic_report,
        annotation_report=annotation_report,
        minimum_negative_abstracts=args.minimum_negative_abstracts,
    )
    args.ablation_output.parent.mkdir(parents=True, exist_ok=True)
    args.ablation_output.write_text(
        ablation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Grounded identity constituent shadow complete")
    print("Current atomic claims:", ablation.claim_count)
    print("Stale detail directories ignored: true")
    print("Annotation LLM calls:", annotation_report.llm_calls_performed)
    print(
        "Strict UNASSESSED non-base slots:",
        ablation.strict_unassessed_nonbase_slot_count,
    )
    print(
        "Grounded-constituent NOT_FOUND non-base slots:",
        ablation.grounded_constituent_not_found_nonbase_slot_count,
    )
    print(
        "FULL_RELATION NOT_FOUND with grounded constituents:",
        ablation.grounded_constituent_full_relation_not_found_count,
    )

    for annotation in annotation_report.annotations:
        print()
        print("Identity:", annotation.claim_id)
        print(" ", annotation.identity_term)
        for group in annotation.groups:
            print(
                "  group:",
                group.label,
                "=>",
                [span.exact_source_text for span in group.spans],
            )

    for claim in ablation.claims:
        print()
        print("Claim:", claim.claim_id)
        for slot in claim.slots:
            if slot.slot == "BASE_RELATION":
                continue
            print(
                " ",
                slot.slot,
                "| material=",
                slot.material_abstract_work_count,
                "| eligible strict/grounded=",
                slot.strict_negative_eligible_count,
                "/",
                slot.grounded_constituent_negative_eligible_count,
                "| state=",
                slot.strict_state,
                "->",
                slot.grounded_constituent_shadow_state,
            )

    print()
    print("Negative closure only: true")
    print("Positive evidence semantics changed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production authority: false")
    print("Production selection changed: false")
    print("Annotation:", args.annotation_output)
    print("Ablation:", args.ablation_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
