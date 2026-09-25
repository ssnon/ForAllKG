from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_downstream_handoff_v1 import (
    PreN10DownstreamHandoffReportV1,
)
from pipeline_core.discovery.pre_n10_relational_binding_bridge_v1 import (
    build_pre_n10_relational_binding_bridge_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Translate the unified pre-N10 external/N9/N10 shadow population "
            "into per-lineage frozen RelationalAtomicBindingPlan artifacts. "
            "N10 certification class is recorded but does not filter binding "
            "reachability."
        )
    )
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--external-n10-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    handoff = PreN10DownstreamHandoffReportV1.model_validate_json(
        args.handoff.expanduser().resolve().read_text(encoding="utf-8")
    )
    report = build_pre_n10_relational_binding_bridge_v1(
        handoff=handoff,
        external_n10_report_path=args.external_n10_report,
        output_root=args.output_root,
    )

    print("Pre-N10 relational binding bridge complete")
    print("Report:", report.report_id)
    print("Lineages:", report.lineage_count)
    print(
        "Binding-ready/not-ready:",
        report.binding_ready_lineage_count,
        "/",
        report.not_binding_ready_lineage_count,
    )
    print("N10 statuses:", report.n10_certification_status_counts)
    print("Binding statuses:", report.binding_status_counts)
    print("Endpoint binding performed: false")
    print("Verifier performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
