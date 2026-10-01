
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.residual_authority_readiness_v7_shadow import (
    build_authority_readiness_v7,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregation-a", required=True, type=Path)
    p.add_argument("--aggregation-b", required=True, type=Path)
    p.add_argument("--stability", required=True, type=Path)
    p.add_argument("--cohort-audit-a", required=True, type=Path)
    p.add_argument("--cohort-audit-b", required=True, type=Path)
    p.add_argument("--candidate-memory", required=True, type=Path)
    p.add_argument("--selection-audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = build_authority_readiness_v7(
        aggregation_a=load(args.aggregation_a),
        aggregation_b=load(args.aggregation_b),
        stability=load(args.stability),
        cohort_audit_a=load(args.cohort_audit_a),
        cohort_audit_b=load(args.cohort_audit_b),
        candidate_memory=load(args.candidate_memory),
        selection_audit=load(args.selection_audit),
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("===== RESIDUAL AUTHORITY READINESS V7 SHADOW =====")
    print(
        "Selection completeness:",
        report["candidate_selection_completeness_pass"],
    )
    print(
        "Sentinel:",
        report["candidate_selection_sentinel_relationship"],
    )
    print(
        "Base ready:",
        report["base_authority_ready_candidate_count"],
    )
    print(
        "V7 ready:",
        report["authority_ready_candidate_count"],
    )
    for row in report["rows"]:
        print(
            row["hypothesis_id"],
            "|", row["claim_id"],
            "|", row["state"],
        )
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
