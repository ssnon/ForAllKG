from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.pre_n10_exploratory_novelty_v1 import (
    build_pre_n10_exploratory_novelty_plan_v1,
    build_pre_n10_exploratory_novelty_report_v1,
    write_exact_or_validate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exploratory external prior-art search for strict-ready "
            "NOVELTY_BEARING claim subsets from semantically admissible "
            "regenerated hypotheses that remain PRE_N10_INTERVENTION_REQUIRED. "
            "This lane creates no N9/N10/certification/production authority."
        )
    )
    parser.add_argument(
        "--regeneration-reentry-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--provider-plan",
        required=True,
        type=Path,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    provider_file = args.provider_plan.expanduser().resolve()
    if not provider_file.is_file():
        raise ValueError(
            "missing frozen provider plan: " + str(provider_file)
        )
    if not os.environ.get(args.api_key_env) and not args.dry_run:
        raise ValueError(
            "required API key environment variable is not set: "
            + args.api_key_env
        )

    root = args.output_root.expanduser().resolve()
    plan = build_pre_n10_exploratory_novelty_plan_v1(
        reentry_report_path=args.regeneration_reentry_report,
        output_root=root,
    )
    plan_path = root / "exploratory_novelty.plan.json"
    report_path = root / "exploratory_novelty.report.json"

    write_exact_or_validate(plan_path, plan)

    print("Pre-N10 exploratory novelty")
    print("Plan:", plan.plan_id)
    print("Eligible lineages:", plan.lineage_count)
    print("Selected strict-ready novelty-bearing claims:", plan.selected_claim_count)
    print("Strict certification contract changed: false")
    print("Incomplete claims promoted: false")
    print("N9/N10/V_post authority created: false")

    for row in plan.lineages:
        print()
        print(
            "Lineage:",
            row.source_hypothesis_id,
            "->",
            row.regenerated_hypothesis_id,
        )
        print("  selected:", row.selected_claim_ids)
        print("  excluded:", row.excluded_claim_ids)

    if args.dry_run:
        print()
        print("Dry-run only: external prior-art search performed: false")
        return 0

    if report_path.exists():
        raise ValueError(
            "exploratory novelty report is write-once: "
            + str(report_path)
        )

    lineage_reports = []

    for row in plan.lineages:
        lineage_root = (
            root
            / "lineage"
            / row.source_hypothesis_id.replace(":", "_").replace("/", "_")
        )
        prefix = lineage_root / "external_novelty"
        external_report_path = prefix.with_suffix(".report.json")
        external_prior_path = prefix.with_suffix(".prior_art.json")
        external_plan_path = prefix.with_suffix(".claims_queries.json")

        existing = [
            str(path)
            for path in (
                external_report_path,
                external_prior_path,
                external_plan_path,
            )
            if path.exists()
        ]
        if existing:
            raise ValueError(
                "exploratory external outputs are write-once; existing="
                + repr(existing)
            )

        argv = [
            sys.executable,
            "-m",
            "scripts.discovery.run_external_novelty",
            "--portfolio",
            row.portfolio_path,
            "--domain-profile",
            row.domain_profile_id,
            "--model",
            args.model,
            "--api-key-env",
            args.api_key_env,
            "--provider-plan",
            str(provider_file),
            "--results-per-query",
            str(args.results_per_query),
            "--max-ranked-works",
            "8",
            "--reuse-query-plan",
            row.exploratory_query_plan_path,
            "--output-prefix",
            str(prefix),
        ]
        if args.base_url:
            argv += ["--base-url", args.base_url]
        if args.save_prompts:
            argv += ["--save-prompts"]

        print()
        print(
            "Running exploratory external novelty:",
            row.source_hypothesis_id,
        )
        subprocess.run(
            argv,
            check=True,
            env=os.environ.copy(),
        )

        if not external_report_path.is_file():
            raise ValueError(
                "external novelty completed without report: "
                + str(external_report_path)
            )
        report = ExternalNoveltyReport.model_validate_json(
            external_report_path.read_text(encoding="utf-8")
        )
        lineage_reports.append(
            (row, report, external_report_path)
        )

        card = [
            card
            for card in report.cards
            if card.hypothesis_id == row.regenerated_hypothesis_id
        ][0]
        print("  external status:", card.status)
        for review in card.claim_reviews:
            if review.claim_id in set(row.selected_claim_ids):
                print(
                    "   ",
                    review.claim_id,
                    "->",
                    review.status,
                )

    final_report = build_pre_n10_exploratory_novelty_report_v1(
        plan=plan,
        lineage_reports=lineage_reports,
    )
    write_exact_or_validate(report_path, final_report)

    print()
    print("Exploratory novelty complete")
    print("Report:", final_report.report_id)
    print("Lineages:", final_report.lineage_count)
    print("Selected claims:", final_report.selected_claim_count)
    print("External statuses:", final_report.external_status_counts)
    print("Novelty certification authority: false")
    print("Candidate survival authority: false")
    print("N9 performed: false")
    print("N10 performed: false")
    print("V_post performed: false")
    print("Output:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
