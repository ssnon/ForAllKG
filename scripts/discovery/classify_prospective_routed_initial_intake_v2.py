from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_initial_intake_v2 import (
    build_initial_intake_v2,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether a P16-P20 initial E2E run produced the two "
            "relational binding inputs required to enter the pre-verifier. "
            "This is a deterministic accounting guard that ports the existing "
            "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS semantics; it performs no "
            "scientific execution."
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("initial intake-v2 report is write-once")

    report = build_initial_intake_v2(
        case_id=args.case_id,
        run_dir=args.run_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(output, report)

    print("Prospective routed initial intake-v2 classified")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Manifest status:", report.main_e2e_manifest_status)
    print("Alpha4 hypotheses:", report.alpha4_hypothesis_count)
    print(
        "N10 candidate/certification:",
        report.n10_candidate_portfolio_present,
        "/",
        report.n10_certification_present,
    )
    print("Disposition:", report.disposition)
    if report.reason_codes:
        print("Reasons:", report.reason_codes)
    print("Binding plan should run:", str(report.binding_plan_should_run).lower())
    print("Gate v2 should run:", str(report.gate_v2_should_run).lower())
    print("Routed dispatch should run:", str(report.routed_dispatch_should_run).lower())
    print("LLM calls performed: 0")
    print("Scientific mutation performed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
