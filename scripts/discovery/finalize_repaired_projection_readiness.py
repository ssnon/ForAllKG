from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.repaired_projection_readiness_accounting import (
    build_accounting_report,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze final P06-P10 R1 pre-verifier readiness accounting. "
            "This is read-only: it performs no repair, no endpoint binding, "
            "no retrieval, no verifier call, and no novelty assessment."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--repair-plan", required=True, type=Path)
    parser.add_argument("--repair-execution", required=True, type=Path)
    parser.add_argument("--reentry-report", required=True, type=Path)
    parser.add_argument("--provenance-retry-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "projection readiness accounting is write-once; "
            "use a fresh output path"
        )

    report = build_accounting_report(
        campaign_root=args.campaign_root,
        repair_plan=_load(args.repair_plan),
        repair_execution=_load(args.repair_execution),
        reentry=_load(args.reentry_report),
        provenance_retry=_load(args.provenance_retry_report),
    )
    write_json_exclusive(output, report)

    print("Repaired projection readiness accounting frozen")
    print("Report:", report.report_id)
    print("Endpoint-ready after R1:", report.endpoint_ready_case_ids)
    print(
        "Source-binding abstained:",
        report.source_binding_abstained_case_ids,
    )
    print("Full-verifier ready after R1:", report.full_verifier_ready_case_ids)
    print("Dispositions:", report.disposition_counts)
    print()
    for row in report.cases:
        print(
            row.case_id,
            "|",
            row.final_disposition,
            "| endpoint-ready=",
            row.endpoint_ready_after_r1,
            "| projected=",
            row.projected_claim_count,
        )
        if row.source_binding_reason_codes:
            print("  source-binding reasons:", row.source_binding_reason_codes)
    print()
    print("Novelty verdict count: 0")
    print("Second repair attempts: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
