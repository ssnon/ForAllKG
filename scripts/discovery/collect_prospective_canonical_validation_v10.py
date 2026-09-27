from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_canonical_validation_v10 import (
    ProspectiveCanonicalValidationCollectorFreezeV10,
    collect_prospective_canonical_validation_v10,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-freeze", required=True, type=Path)
    args = parser.parse_args()

    freeze_path = args.collector_freeze.expanduser().resolve()
    frozen = ProspectiveCanonicalValidationCollectorFreezeV10.model_validate_json(
        freeze_path.read_text(encoding="utf-8")
    )

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    )
    if dirty:
        raise ValueError(
            "v10 collection requires a fully clean worktree including untracked files"
        )
    if head != frozen.collector_repository_head_sha:
        raise ValueError("v10 collection requires exact frozen HEAD")

    output = Path(frozen.validation_output_path)
    if output.exists():
        raise ValueError("v10 validation output is write-once")

    report = collect_prospective_canonical_validation_v10(frozen)
    write_json_exclusive(output, report)

    print("P54-P58 v10 canonical validation collected")
    print("Report:", report.report_id)
    print("Cases:", report.case_count)
    print(
        "Terminal before V_pre:",
        report.terminal_before_initial_vpre_case_count,
    )
    print("Pre-N10 terminal:", report.pre_n10_terminal_case_count)
    print("Bridge reached:", report.bridge_reached_case_count)
    print("V_post reached:", report.vpost_reached_case_count)
    print(
        "Fresh legacy-authority violations:",
        report.fresh_legacy_authority_violation_count,
    )
    print(
        "V_post source-binding modes:",
        report.vpost_source_binding_mode_counts,
    )
    print("Certification decisions:", report.certification_decision_counts)
    print("LLM calls: 0")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
