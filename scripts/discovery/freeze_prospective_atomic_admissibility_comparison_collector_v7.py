from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    ProspectiveAtomicAdmissibilityExecutionPlanV7,
    build_prospective_atomic_admissibility_comparison_collector_freeze_v7,
    sha256_file,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def git_state() -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    )
    return head, dirty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--comparison-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan_path = args.execution_plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    comparison = args.comparison_output.expanduser().resolve()
    if output.exists():
        raise ValueError("v7 collector freeze is write-once")
    if comparison.exists():
        raise ValueError("v7 comparison output already exists")

    plan = ProspectiveAtomicAdmissibilityExecutionPlanV7.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    head, dirty = git_state()

    frozen = build_prospective_atomic_admissibility_comparison_collector_freeze_v7(
        execution_plan=plan,
        execution_plan_file_sha256=sha256_file(plan_path),
        collector_repository_head_sha=head,
        repository_worktree_dirty=dirty,
        comparison_output_path=comparison,
    )
    write_json_exclusive(output, frozen)
    print("P39-P43 v7 comparison collector frozen")
    print("Freeze:", frozen.freeze_id)
    print("HEAD:", frozen.collector_repository_head_sha)
    print("Case denominator:", frozen.case_count)
    print("Full worktree clean: true")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
