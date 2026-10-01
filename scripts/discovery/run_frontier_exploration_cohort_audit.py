#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipeline_core.discovery.frontier_exploration_audit import (
    FrontierExplorationAudit,
    build_frontier_exploration_cohort_audit,
)


def _case_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--case must use CASE_ID=/path/to/frontier_exploration.audit.json"
        )
    case_id, raw_path = value.split("=", 1)
    case_id = case_id.strip()
    raw_path = raw_path.strip()
    if not case_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "--case requires non-empty CASE_ID and path"
        )
    return case_id, Path(raw_path)


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple Exploration Frontier audits into a frozen "
            "cohort diagnostic. No case is ranked and no winner is selected."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_case_arg,
        metavar="CASE_ID=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    audits = {}
    for case_id, path in args.case:
        if case_id in audits:
            raise ValueError(f"duplicate cohort case id: {case_id}")
        audits[case_id] = FrontierExplorationAudit.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    cohort = build_frontier_exploration_cohort_audit(audits=audits)
    _write_json(args.output, cohort)

    print("Exploration Frontier cohort audit complete")
    print("cases:", cohort.case_count)
    print("raw ideas:", cohort.total_raw_idea_count)
    print(
        "cases with HO cap reached:",
        cohort.case_count_with_topology_cap_reached,
    )
    print(
        "cases with external-only primitives:",
        cohort.case_count_with_open_world_exclusive_primitives,
    )
    print(
        "cases with external primitive topology reuse:",
        cohort.case_count_with_open_world_topology_reuse,
    )
    print(
        "cases with interpretive branches:",
        cohort.case_count_with_interpretive_branches,
    )
    print(
        "cases with candidate-driven tensions:",
        cohort.case_count_with_candidate_driven_tensions,
    )
    print(
        "cases with verified depth 3:",
        cohort.case_count_with_verified_depth_3,
    )
    print("CROSS_CASE_WINNER_SELECTED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
