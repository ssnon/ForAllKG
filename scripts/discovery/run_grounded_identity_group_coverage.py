from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotationReport,
)
from pipeline_core.discovery.reframing.grounded_identity_group_coverage import (
    build_group_coverage_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose grounded-identity negative-closure failure into "
            "per-constituent and intersection coverage over already-reviewed "
            "material abstracts. No retrieval, review, or production semantics "
            "are changed."
        )
    )
    parser.add_argument("--atomic-report", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--detail-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()

    atomic_report = AtomicCrossLaneSynthesisReport.model_validate_json(
        args.atomic_report.read_text(encoding="utf-8")
    )
    annotation = GroundedIdentityAnnotationReport.model_validate_json(
        args.annotation.read_text(encoding="utf-8")
    )

    report = build_group_coverage_report(
        detail_root=args.detail_root,
        atomic_report=atomic_report,
        annotation_report=annotation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Grounded identity group coverage diagnostic complete")
    print("Current claims:", report.claim_count)
    print("Stale detail directories ignored: true")

    for claim in report.claims:
        print()
        print("Claim:", claim.claim_id)
        print("Identity:", claim.identity_term)

        for slot in claim.slots:
            print()
            print(
                " ",
                slot.slot,
                "| material abstracts =",
                slot.material_abstract_work_count,
            )
            for group in slot.groups:
                print(
                    "   group:",
                    group.group_label,
                    "| spans=",
                    group.source_spans,
                    "| matches=",
                    group.matching_material_work_count,
                )
            print(
                "   all groups together:",
                slot.all_groups_matching_work_count,
            )
            print(
                "   signatures:",
                slot.group_match_signature_counts,
            )

    print()
    print("Retrieval changed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
