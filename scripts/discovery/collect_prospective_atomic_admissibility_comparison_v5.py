from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_comparison_collector_v5 import (
    ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5,
    collect_prospective_atomic_admissibility_comparison_v5,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return head, dirty


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the frozen P29-P33 legacy-vs-neutral Pre-N10 comparison. "
            "This command performs no LLM calls and mutates no campaign artifact."
        )
    )
    parser.add_argument("--collector-freeze", required=True, type=Path)
    args = parser.parse_args()

    freeze_path = args.collector_freeze.expanduser().resolve()
    if not freeze_path.is_file():
        raise ValueError("missing collector freeze: " + str(freeze_path))

    frozen = (
        ProspectiveAtomicAdmissibilityComparisonCollectorFreezeV5.model_validate_json(
            freeze_path.read_text(encoding="utf-8")
        )
    )

    head, dirty = _git_state()
    if dirty:
        raise ValueError("comparison collection requires clean tracked worktree")
    if head != frozen.collector_repository_head_sha:
        raise ValueError(
            "comparison collection must run at exact frozen collector HEAD"
        )

    output_path = Path(frozen.comparison_output_path)
    if output_path.exists():
        raise ValueError("comparison output is write-once")

    report = collect_prospective_atomic_admissibility_comparison_v5(frozen)
    write_json_exclusive(output_path, report)

    print("P29-P33 atomic admissibility comparison collected")
    print("Report:", report.report_id)
    print("Cases:", report.case_count)
    print(
        "Terminal before initial V_pre:",
        report.terminal_before_initial_vpre_case_count,
    )
    print("Claim comparisons:", report.claim_comparison_count)
    print("Pair counts:", report.pair_counts)
    print("Reason transitions:", report.reason_transition_counts)
    print("Neutral blocker counts:", report.neutral_blocker_dimension_counts)
    print("LLM calls: 0")
    print("Campaign artifact mutation: false")
    print("Scientific validation authority: false")
    print("Production selection authority: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
