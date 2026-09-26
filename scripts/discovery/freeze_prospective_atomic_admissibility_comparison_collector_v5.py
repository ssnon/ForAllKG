from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_comparison_collector_v5 import (
    build_prospective_atomic_admissibility_comparison_collector_freeze_v5,
    sha256_file,
)
from pipeline_core.discovery.prospective_atomic_admissibility_execution_plan_v5 import (
    ProspectiveAtomicAdmissibilityExecutionPlanV5,
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


def _require_ancestor(older: str, newer: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(label + " HEAD is not an ancestor of current HEAD")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the P29-P33 legacy-vs-neutral comparison collector before "
            "any P29-P33 execution. The collector population, artifact discovery "
            "rules, terminal-case accounting, and four-way taxonomy become "
            "write-once here."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--comparison-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan_path = args.execution_plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    comparison_output = args.comparison_output.expanduser().resolve()

    if not plan_path.is_file():
        raise ValueError("missing P29-P33 execution plan: " + str(plan_path))
    if output_path.exists():
        raise ValueError("comparison collector freeze is write-once")
    if comparison_output.exists():
        raise ValueError(
            "comparison output must not exist before collector freeze"
        )

    plan = ProspectiveAtomicAdmissibilityExecutionPlanV5.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    head, dirty = _git_state()
    _require_ancestor(
        plan.execution_plan_repository_head_sha,
        head,
        "P29-P33 execution plan",
    )

    frozen = (
        build_prospective_atomic_admissibility_comparison_collector_freeze_v5(
            execution_plan=plan,
            execution_plan_file_sha256=sha256_file(plan_path),
            collector_repository_head_sha=head,
            repository_tracked_worktree_dirty=dirty,
            comparison_output_path=comparison_output,
        )
    )
    write_json_exclusive(output_path, frozen)

    print("Prospective atomic admissibility comparison collector frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.collector_repository_head_sha)
    print("Cases:", frozen.case_ids)
    print("Case denominator fixed:", frozen.case_count)
    print("Claim population:", frozen.claim_population_policy)
    print("Terminal-before-decomposition retained: true")
    print("Result-conditioned artifact selection: false")
    print("LLM calls allowed: false")
    print("Campaign execution may begin after this freeze: true")
    print("Comparison output:", frozen.comparison_output_path)
    print("Freeze output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
