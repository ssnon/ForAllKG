from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dac_her.literature_discovery import load_query_plan
from dac_her.literature_discovery.abstract_packages import build_abstract_packages
from dac_her.literature_discovery.selection import (
    read_candidates_jsonl,
    select_literature,
    write_selection_artifacts,
)
from dac_her.literature_discovery.selection_plan import load_selection_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_CONFIG = PROJECT_ROOT / "configs" / "literature" / "broad_catalysis_v1.yaml"
DEFAULT_SELECTION_CONFIG = (
    PROJECT_ROOT / "configs" / "literature" / "broad_catalysis_selection_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a balanced broad-catalysis abstract corpus from discovery "
            "candidates and emit GraphAgents-compatible abstract packages."
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--query-config", default=str(DEFAULT_QUERY_CONFIG))
    parser.add_argument("--selection-config", default=str(DEFAULT_SELECTION_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-packages",
        action="store_true",
        help="Write selected/rejected/report artifacts without document packages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_plan = load_query_plan(args.query_config)
    selection_plan = load_selection_plan(args.selection_config)
    candidates = read_candidates_jsonl(args.candidates)

    result = select_literature(
        candidates,
        query_plan=query_plan,
        selection_plan=selection_plan,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "data_broad"
        / "selection"
        / selection_plan.plan_id
        / timestamp
    )
    selected_path, rejected_path, report_path = write_selection_artifacts(
        result,
        output_dir=output_dir,
        candidates_path=args.candidates,
        query_plan=query_plan,
        selection_plan=selection_plan,
    )

    papers_yaml = None
    package_manifest = None
    if not args.no_packages:
        papers_yaml, package_manifest = build_abstract_packages(
            result.selected,
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
        )

    print("Literature selection complete")
    print("Selection plan:", selection_plan.plan_id)
    print("Input candidates:", len(candidates))
    print("Selected:", result.selected_count)
    print("Target:", result.target_count)
    print("Selected JSONL:", selected_path)
    print("Rejected JSONL:", rejected_path)
    print("Report:", report_path)
    if papers_yaml:
        print("Papers config:", papers_yaml)
    if package_manifest:
        print("Package manifest:", package_manifest)


if __name__ == "__main__":
    main()
