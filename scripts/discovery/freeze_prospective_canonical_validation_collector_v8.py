from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_canonical_validation_v8 import (
    ProspectiveCanonicalValidationExecutionPlanV8,
    build_prospective_canonical_validation_collector_freeze_v8,
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
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan_path = args.execution_plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validation = args.validation_output.expanduser().resolve()
    if output.exists():
        raise ValueError("v8 collector freeze is write-once")
    if validation.exists():
        raise ValueError("v8 validation output already exists")

    plan = ProspectiveCanonicalValidationExecutionPlanV8.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    head, dirty = git_state()

    frozen = build_prospective_canonical_validation_collector_freeze_v8(
        execution_plan=plan,
        execution_plan_file_sha256=sha256_file(plan_path),
        collector_repository_head_sha=head,
        repository_worktree_dirty=dirty,
        validation_output_path=validation,
    )
    write_json_exclusive(output, frozen)
    print("P44-P48 v8 canonical validation collector frozen")
    print("Freeze:", frozen.freeze_id)
    print("HEAD:", frozen.collector_repository_head_sha)
    print("Case denominator:", frozen.case_count)
    print("Full worktree clean: true")
    print("Outputs observed before freeze: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
