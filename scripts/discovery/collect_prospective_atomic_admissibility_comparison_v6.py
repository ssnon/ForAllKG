from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v6 import (
    ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV6,
    collect_prospective_atomic_admissibility_comparison_v6,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-freeze", required=True, type=Path)
    args = parser.parse_args()

    freeze_path = args.collector_freeze.expanduser().resolve()
    frozen = ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV6.model_validate_json(
        freeze_path.read_text(encoding="utf-8")
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    )
    if dirty:
        raise ValueError("v6 collection requires clean tracked worktree")
    if head != frozen.collector_repository_head_sha:
        raise ValueError("v6 collection requires exact frozen HEAD")

    output = Path(frozen.comparison_output_path)
    if output.exists():
        raise ValueError("v6 comparison output is write-once")

    report = collect_prospective_atomic_admissibility_comparison_v6(frozen)
    write_json_exclusive(output, report)
    print("P34-P38 v6 comparison collected")
    print("Report:", report.report_id)
    print("Cases:", report.case_count)
    print("Terminal cases:", report.terminal_before_initial_vpre_case_count)
    print("Claim comparisons:", report.claim_comparison_count)
    print("Pair counts:", report.pair_counts)
    print("Reason transitions:", report.reason_transition_counts)
    print("Neutral blockers:", report.neutral_blocker_dimension_counts)
    print("LLM calls: 0")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
