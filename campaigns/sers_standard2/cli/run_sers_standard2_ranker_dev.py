from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.ranker_dev_validation import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    read_json,
    run_ranker_validation,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    diagnostic_root = args.diagnostic_root.expanduser().resolve()
    spec_root = (
        args.spec_root
        if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        repo_root=ROOT,
        diagnostic_root=diagnostic_root,
        spec_path=spec_root / "ranker_spec.json",
    )
    if issues:
        print("ranker-only DEV preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    marker_path = spec_root / "SPEC_FREEZE_PASS.json"
    if not marker_path.is_file():
        print("ranker-only DEV preflight: FAIL")
        print(" - SPEC_FREEZE_PASS missing")
        print("Network calls:", 0)
        return 2
    marker = read_json(marker_path)
    if (
        marker.get("status") != "spec_freeze_pass"
        or marker.get("spec_id") != spec.get("spec_id")
    ):
        print("ranker-only DEV preflight: FAIL")
        print(" - spec freeze marker mismatch")
        print("Network calls:", 0)
        return 2

    if output_root.exists():
        print("ranker-only DEV preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Network calls:", 0)
        return 2

    print("SERS Ranker-only DEV Validation")
    print("Spec ID:", spec["spec_id"])
    print("Canonical works:", spec["canonical_work_count"])
    print("Claims:", spec["claim_count"])
    print("Core claims:", spec["core_claim_count"])
    print("Domain profile:", spec["ranker"]["domain_profile_id"])
    print("Embedding model:", spec["ranker"]["embed_model"])
    print("Device:", spec["ranker"]["device"])
    print("Top-N:", spec["ranker"]["max_ranked_works_per_claim"])
    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Claim review:", False)
    print("Novelty verdict:", False)

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    report, audit_md = run_ranker_validation(
        repo_root=ROOT,
        diagnostic_root=diagnostic_root,
        spec=spec,
    )

    if report["mechanical_outcome"] != "RANKER_MECHANICAL_DEV_PASS":
        print()
        print("ranker-only DEV validation: MECHANICAL FAIL")
        print("Summary:", report["summary"])
        for row in report["claim_reports"]:
            failed = [
                key
                for key, value in row["checks"].items()
                if not value
            ]
            if failed:
                print(
                    " -",
                    row["claim_id"],
                    row["importance"],
                    "failed:",
                    failed,
                    "candidate_pool=",
                    row["candidate_pool_count"],
                    "topn_abstracts=",
                    row["topn_abstract_count"],
                )
        print("Write performed: False")
        return 2

    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_root / "ranker_report.json",
        report,
    )
    atomic_text(
        output_root / "human_relevance_audit.md",
        audit_md,
    )
    atomic_json(
        output_root / "MECHANICAL_PASS.json",
        {
            "status": "mechanical_pass",
            "run_id": report["run_id"],
            "run_sha256": report["run_sha256"],
            "scientific_relevance_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "automatic_claim_level_review_authorized": False,
            "network_calls": 0,
            "llm_calls": 0,
        },
    )

    print()
    print("ranker-only DEV validation: COMPLETE")
    print("Run ID:", report["run_id"])
    print("Mechanical outcome:", report["mechanical_outcome"])
    print(
        "Scientific relevance:",
        report["scientific_relevance_outcome"],
    )
    print("Summary:", report["summary"])
    print()
    for index, row in enumerate(
        report["claim_reports"],
        start=1,
    ):
        print(
            f"[{index}] {row['importance']} "
            f"{row['kind']} | "
            f"pool={row['candidate_pool_count']} "
            f"topN_abs={row['topn_abstract_count']}/{row['topn_count']} "
            f"top1={row['top1_relevance_score']:.4f}"
        )
        print("   CLAIM:", row["claim_text"][:220])
        for work in row["top_ranked_works"][:3]:
            print(
                f"   #{work['rank']} "
                f"rel={work['relevance_score']:.4f} "
                f"sem={work['semantic_similarity']:.4f} "
                f"lex={work['lexical_coverage']:.3f} "
                f"domain={work['reaction_domain_relevance']:.3f} "
                f"scope={work['catalyst_scope_relevance']:.3f} "
                f"abs={work['abstract_available']} | "
                f"{work['title'][:150]}"
            )
        print()

    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Automatic claim-level review authorized:", False)
    print(
        "Human audit:",
        output_root / "human_relevance_audit.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
