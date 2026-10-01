from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.idea_evolution_audit import (
    IdeaEvolutionAudit,
    build_idea_evolution_cohort_audit,
)


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use CASE_ID=PATH")
    case_id, raw_path = value.split("=", 1)
    case_id = case_id.strip()
    if not case_id:
        raise argparse.ArgumentTypeError("case ID may not be empty")
    return case_id, Path(raw_path).expanduser()


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate Idea Evolution audits across frozen cases without ranking "
            "cases or selecting a winner."
        )
    )
    p.add_argument("--case", action="append", required=True, type=_case)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    rows = []
    for case_id, path in args.case:
        audit = IdeaEvolutionAudit.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        rows.append((case_id, audit))

    report = build_idea_evolution_cohort_audit(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Idea Evolution cohort audit complete")
    print("cases:", report.case_count)
    print("total evolved ideas:", report.total_evolution_idea_count)
    print(
        "cases with cross-source/new-backbone/candidate/reframe:",
        report.case_count_with_cross_source_transition,
        "/",
        report.case_count_with_new_backbone_transition,
        "/",
        report.case_count_with_candidate_interpretation,
        "/",
        report.case_count_with_reframe,
    )
    print(
        "cases with any conceptual transition:",
        report.case_count_with_any_conceptual_transition,
    )
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
