from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.prospective_novelty_validation_execution import (
    ProspectiveNoveltyExecutionPlan,
)


def _write_exclusive(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(
            "execution record is write-once: " + str(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a frozen prospective novelty-validation plan. Existing "
            "non-empty scientific run directories are never overwritten."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Optional source_case_id filter. Repeat to execute multiple cases. "
            "Default: all frozen cases."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    plan = ProspectiveNoveltyExecutionPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )

    selected = [
        row
        for row in plan.cases
        if not args.case_id
        or row.source_case_id in set(args.case_id)
    ]

    if args.case_id:
        found = {row.source_case_id for row in selected}
        missing = sorted(set(args.case_id) - found)
        if missing:
            raise ValueError(
                "unknown case-id(s): " + repr(missing)
            )

    print("Prospective novelty-validation execution")
    print("Plan:", plan.plan_id)
    print("Cases selected:", len(selected))
    print("Dry run:", bool(args.dry_run))

    failures = 0

    for row in selected:
        run_dir = Path(row.run_dir)
        record = run_dir.parent / (
            row.source_case_id.lower()
            + ".execution.json"
        )

        command = [
            sys.executable,
            *row.command_argv,
        ]

        print()
        print("=" * 88)
        print(row.source_case_id)
        print("=" * 88)
        print("$", " ".join(command))

        if args.dry_run:
            continue

        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                "prospective case run directory is already non-empty; "
                "refusing overwrite/reuse: "
                + str(run_dir)
            )

        completed = subprocess.run(
            command,
            check=False,
        )

        _write_exclusive(
            record,
            {
                "schema_version":
                    "prospective-novelty-validation-execution-record-v1",
                "plan_id":
                    plan.plan_id,
                "prospective_case_id":
                    row.prospective_case_id,
                "source_case_id":
                    row.source_case_id,
                "run_dir":
                    row.run_dir,
                "returncode":
                    int(completed.returncode),
                "completed_successfully":
                    completed.returncode == 0,
                "scientific_result_interpreted":
                    False,
                "rerun_automatically_authorized":
                    False,
            },
        )

        if completed.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                return int(completed.returncode or 1)

    print()
    print("Execution complete")
    print("Failures:", failures)
    print("Scientific result interpreted by runner: false")
    print("Automatic rerun authorized: false")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
