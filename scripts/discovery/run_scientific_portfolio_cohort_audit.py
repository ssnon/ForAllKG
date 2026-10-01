from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_portfolio_audit import (
    ScientificPortfolioAudit,
    build_scientific_portfolio_cohort_audit,
)


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use CASE_ID=/path/to/audit.json")
    key, path = value.split("=", 1)
    if not key.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--case requires non-empty CASE_ID and path")
    return key.strip(), Path(path.strip()).expanduser()


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate Scientific Portfolio Selection audits across a frozen cohort.")
    p.add_argument("--case", action="append", required=True, type=_case)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    seen = set()
    for case_id, path in args.case:
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        rows.append((case_id, ScientificPortfolioAudit.model_validate_json(path.read_text(encoding="utf-8"))))
    cohort = build_scientific_portfolio_cohort_audit(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cohort.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print("Scientific Portfolio cohort audit complete")
    print("cases:", cohort.case_count)
    print("retained:", cohort.total_retained_candidate_count)
    print("materialized hypotheses:", cohort.total_materialized_hypothesis_count)
    print("cases ready for downstream verification:", cohort.case_count_ready_for_downstream_verification)
    print("CROSS_CASE_WINNER_SELECTED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
