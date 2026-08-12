from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.broad_extraction_diagnostics import (
    write_broad_extraction_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Broad-KG extraction failures, validation issues, LLM "
            "recovery usage, token usage, and mechanism yield."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", default="catalysis_mechanism")
    parser.add_argument("--data-root", default="data_broad")
    parser.add_argument("--paper-ids", nargs="+", required=True)
    parser.add_argument(
        "--preflight-outlier",
        action="append",
        default=[],
        help=(
            "Paper ID excluded by the current pipeline before extraction due "
            "to the abstract-length guard. Historical runs are ignored for "
            "current-run efficiency accounting. Repeat as needed."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.domain_profile != "catalysis_mechanism":
        raise ValueError(
            "Broad extraction diagnostics are reserved for the "
            "catalysis_mechanism domain profile."
        )
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_root / "pipeline_runs" / args.corpus_id / "diagnostics"
    )
    report_path, rows_path, issue_path = write_broad_extraction_diagnostics(
        data_root=data_root,
        paper_ids=args.paper_ids,
        output_dir=output_dir,
        preflight_outlier_ids=args.preflight_outlier,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("Broad extraction diagnostics complete")
    print("Requested papers:", report["requested_paper_count"])
    print(
        "Graph-usable papers:",
        report["graph_usable_paper_count"],
        f"({report['graph_usable_paper_fraction']:.1%})",
    )
    print("Materialization statuses:", report["materialization_status_counts"])
    print("LLM calls:", report["llm_calls"])
    print("Total tokens:", report["total_tokens"])
    print(
        "Wasted calls/tokens:",
        f"{report['wasted_call_fraction']:.1%}",
        f"{report['wasted_token_fraction']:.1%}",
    )
    print("Direct mechanism edges:", report["direct_mechanism_edges"])
    print(
        "Current/stale projections:",
        report["projection_paper_count"],
        report.get("stale_projection_count", 0),
    )
    print("Preflight abstract outliers:", report.get("preflight_outlier_count", 0))
    print("Top terminal validation issues:")
    top_issues = sorted(
        report["terminal_validation_issue_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    )[:10]
    for code, count in top_issues:
        print(f"  {code}: {count}")
    print("Top relation endpoint mismatch patterns:")
    for row in report.get("relation_mismatch_patterns", [])[:10]:
        expected = ",".join(row.get("expected_types") or [])
        print(
            "  ",
            f"{row.get('relation')} {row.get('side')} ",
            f"actual={row.get('actual_type')} expected=[{expected}] ",
            f"count={row.get('count')}",
            sep="",
        )
    print("Top isolated-node patterns:")
    for row in report.get("isolated_node_patterns", [])[:10]:
        print(
            "  ",
            f"collection={row.get('node_collection')} ",
            f"actual={row.get('actual_type')} ",
            f"count={row.get('count')}",
            sep="",
        )
    print("Report:", report_path)
    print("Per-paper rows:", rows_path)
    print("Issue counts:", issue_path)


if __name__ == "__main__":
    main()
