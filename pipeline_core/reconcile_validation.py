from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pipeline_core.reconcile_runtime import (
    StageState,
    StrictRunState,
    read_json,
)


ActivePayloadIssue = Callable[
    [dict[str, Any]],
    str | None,
]

CompatibilityCheck = Callable[
    [
        dict[str, Any],
        dict[str, Any],
    ],
    tuple[bool, str],
]


def validate_strict_run(
    *,
    run_dir: str | Path,
    current: dict[str, Any],
    active_payload_issue: ActivePayloadIssue,
    compatibility_check: CompatibilityCheck,
) -> StrictRunState:
    run_dir = Path(run_dir)

    run_path = (
        run_dir / "run.json"
    )

    active_path = (
        run_dir
        / "active_chunks.json"
    )

    run_meta = read_json(
        run_path
    )

    active = read_json(
        active_path
    )

    run_id = run_dir.name

    if not run_meta:
        return StrictRunState(
            StageState.pending(
                "strict run.json "
                "missing/invalid",
                run_path,
            ),
            run_id,
            run_dir,
        )

    if not active:
        return StrictRunState(
            StageState.pending(
                "active_chunks.json "
                "missing/invalid",
                active_path,
            ),
            run_id,
            run_dir,
        )

    if (
        str(
            run_meta.get(
                "run_id"
            )
            or ""
        )
        != run_id
    ):
        return StrictRunState(
            StageState.pending(
                "strict run directory/"
                "metadata mismatch",
                run_path,
            ),
            run_id,
            run_dir,
        )

    if (
        str(
            active.get(
                "run_id"
            )
            or ""
        )
        != run_id
    ):
        return StrictRunState(
            StageState.pending(
                "strict active/run "
                "metadata mismatch",
                active_path,
            ),
            run_id,
            run_dir,
        )

    chunks = active.get(
        "chunks"
    )

    if (
        not isinstance(
            chunks,
            list,
        )
        or not chunks
    ):
        return StrictRunState(
            StageState.pending(
                "strict run has no "
                "active chunks",
                active_path,
            ),
            run_id,
            run_dir,
        )

    issue = active_payload_issue(
        active
    )

    if issue is not None:
        return StrictRunState(
            StageState.pending(
                issue,
                active_path,
            ),
            run_id,
            run_dir,
        )

    compatible, reason = (
        compatibility_check(
            run_meta,
            current,
        )
    )

    if not compatible:
        return StrictRunState(
            StageState.pending(
                reason,
                run_path,
                run_meta,
            ),
            run_id,
            run_dir,
        )

    return StrictRunState(
        StageState.ready(
            reason,
            run_dir,
            run_meta,
        ),
        run_id,
        run_dir,
    )
