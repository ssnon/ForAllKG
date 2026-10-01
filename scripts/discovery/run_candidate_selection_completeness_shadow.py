
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.candidate_selection_completeness_shadow import (
    audit_candidate_selection,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selection-plan", required=True, type=Path)
    p.add_argument("--prior-art-packet", required=True, type=Path)
    p.add_argument("--sentinel", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = audit_candidate_selection(
        selection_plan=load(args.selection_plan),
        packet=load(args.prior_art_packet),
        sentinel=load(args.sentinel),
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

    print("===== CANDIDATE SELECTION COMPLETENESS SHADOW =====")
    print("PASS:", report["pass"])
    print("Failures:", report["failure_count"])
    print("Warnings:", report["warning_count"])
    print(
        "Sentinel relationship:",
        report["sentinel_relationship"],
    )
    print(
        "Sentinel selected:",
        report["sentinel_selected_by_tiered_plan"],
    )
    for failure in report["failures"]:
        print("FAIL:", failure)
    for warning in report["warnings"]:
        print("WARN:", warning)
    print("SHADOW_ONLY=True")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
