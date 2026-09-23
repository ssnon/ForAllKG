from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_relational_execution_plan import (
    ProspectiveRelationalExecutionSettings,
    build_prospective_relational_execution_plan,
)
from pipeline_core.discovery.prospective_source_task_freeze import (
    ProspectiveSourceTaskCampaignFreeze,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, bool(tracked.strip())


def _require_ancestor(older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "source-task freeze repository HEAD is not an ancestor "
            "of the execution-plan repository HEAD"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze identical prospective execution semantics for P06-P10 "
            "before the first source task is executed."
        )
    )
    parser.add_argument("--source-freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.source_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise ValueError("missing source-task freeze: " + str(source_path))
    if output_path.exists():
        raise ValueError(
            "prospective execution plan is write-once; use a fresh output path"
        )

    source = ProspectiveSourceTaskCampaignFreeze.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()
    _require_ancestor(source.repository_head_sha, head)

    plan = build_prospective_relational_execution_plan(
        source_freeze=source,
        source_freeze_sha256=_sha256_file(source_path),
        execution_plan_repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
        settings=ProspectiveRelationalExecutionSettings(),
    )
    write_json_exclusive(output_path, plan)

    print("Prospective relational execution plan frozen")
    print("Plan:", plan.plan_id)
    print("Source-task freeze:", plan.source_task_freeze_id)
    print("Source freeze HEAD:", plan.source_task_freeze_repository_head_sha)
    print("Execution-plan HEAD:", plan.execution_plan_repository_head_sha)
    print("Tracked worktree dirty: false")
    print("Cases:", plan.case_ids)
    print()
    print("Provider request:", plan.settings.provider_request)
    print("Results/query:", plan.settings.results_per_query)
    print("Post-generation N10 authority: certification_only")
    print("Structural selection rule:", plan.settings.hypothesis_selection_rule)
    print("Post-case adaptation allowed: false")
    print("Case replacement allowed: false")
    print()
    for row in plan.cases:
        print(row.case_id)
        print("  run_dir:", row.run_dir)
        print("  main E2E:", " ".join(row.main_e2e_argv))
    print()
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
