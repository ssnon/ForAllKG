from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.broad_binding_sidecar import load_bindings, run_bound_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Broad extraction diagnostics against exact historical attempts "
            "captured in an explicit binding sidecar. No latest-pointer fallback is used."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--bindings-file", required=True)
    parser.add_argument("--data-root", default="data_broad")
    parser.add_argument("--paper-ids", nargs="+", required=True)
    parser.add_argument("--preflight-outlier", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    bindings_file = Path(args.bindings_file)
    if not bindings_file.is_absolute():
        bindings_file = (PROJECT_ROOT / bindings_file).resolve()
    _, bindings = load_bindings(bindings_file)
    output_dir = Path(args.output_dir) if args.output_dir else (
        data_root / "pipeline_runs" / args.corpus_id / "bound_diagnostics"
    )
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    report_path, rows_path, issues_path, missing = run_bound_diagnostics(
        data_root=data_root,
        paper_ids=args.paper_ids,
        bindings=bindings,
        output_dir=output_dir,
        preflight_outlier_ids=args.preflight_outlier,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("Bound Broad extraction diagnostics complete")
    print("Bindings file:", bindings_file)
    print("Bound papers:", report.get("bound_paper_count", 0), "/", report.get("requested_paper_count", 0))
    print("Missing bindings:", missing)
    print("Graph-usable papers:", report.get("graph_usable_paper_count"), f"({report.get('graph_usable_paper_fraction', 0.0):.1%})")
    print("Materialization statuses:", report.get("materialization_status_counts"))
    print("LLM calls:", report.get("llm_calls"))
    print("Total tokens:", report.get("total_tokens"))
    print("Wasted calls/tokens:", f"{report.get('wasted_call_fraction', 0.0):.1%}", f"{report.get('wasted_token_fraction', 0.0):.1%}")
    print("Direct mechanism edges:", report.get("direct_mechanism_edges"))
    print("Current/stale projections:", report.get("projection_paper_count"), report.get("stale_projection_count", 0))
    print("Report:", report_path)
    print("Per-paper rows:", rows_path)
    print("Issue counts:", issues_path)


if __name__ == "__main__":
    main()
