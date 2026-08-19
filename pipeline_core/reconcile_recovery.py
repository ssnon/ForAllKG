from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from pipeline_core.reconcile_runtime import (
    StageState,
    StrictRunState,
    read_json,
)


StrictRunValidator = Callable[
    [Path],
    StrictRunState,
]


def run_directories_newest_first(
    runs_root: str | Path,
) -> list[Path]:
    runs_root = Path(
        runs_root
    )

    if not runs_root.is_dir():
        return []

    return sorted(
        (
            path
            for path
            in runs_root.iterdir()
            if path.is_dir()
        ),
        key=lambda path:
            path.stat().st_mtime_ns,
        reverse=True,
    )


def discover_strict_run_candidates(
    *,
    paper_root: str | Path,
) -> tuple[list[Path], str]:
    paper_root = Path(
        paper_root
    )

    pointer_path = (
        paper_root
        / "latest_run.json"
    )

    pointer = read_json(
        pointer_path
    )

    candidates: list[Path] = []

    pointer_reason = (
        "no latest strict run"
    )

    if pointer:
        run_id = str(
            pointer.get(
                "run_id"
            )
            or ""
        ).strip()

        if run_id:
            candidates.append(
                paper_root
                / "runs"
                / run_id
            )

            pointer_reason = (
                f"latest pointer="
                f"{run_id}"
            )

        else:
            pointer_reason = (
                "latest_run.json "
                "has no run_id"
            )

    others = (
        run_directories_newest_first(
            paper_root / "runs"
        )
    )

    seen = {
        path.resolve()
        for path in candidates
    }

    candidates.extend(
        path
        for path in others
        if path.resolve()
        not in seen
    )

    return (
        candidates,
        pointer_reason,
    )


def select_compatible_strict_run(
    *,
    candidates: Sequence[Path],
    pointer_reason: str,
    validate_run: StrictRunValidator,
) -> StrictRunState:
    if not candidates:
        return StrictRunState(
            StageState.pending(
                pointer_reason
            )
        )

    failures: list[str] = []

    for index, run_dir in enumerate(
        candidates
    ):
        state = validate_run(
            Path(run_dir)
        )

        if state.stage.valid:
            if index == 0:
                return state

            return StrictRunState(
                StageState.ready(
                    "recovered compatible "
                    "usable strict run "
                    "from runs/*; "
                    f"{pointer_reason}; "
                    f"selected={state.run_id}",
                    state.run_dir,
                    state.stage.metadata,
                ),
                state.run_id,
                state.run_dir,
            )

        failures.append(
            f"{Path(run_dir).name}: "
            f"{state.stage.reason}"
        )

    detail = "; ".join(
        failures[:3]
    )

    if len(failures) > 3:
        detail += (
            f"; +{len(failures) - 3} "
            "older run(s)"
        )

    return StrictRunState(
        StageState.pending(
            "no compatible usable "
            "strict run exists"
            + (
                f" ({detail})"
                if detail
                else ""
            )
        )
    )


def first_usable_strict_run(
    *,
    candidates: Sequence[Path],
    validate_run: StrictRunValidator,
) -> StrictRunState | None:
    for run_dir in candidates:
        state = validate_run(
            Path(run_dir)
        )

        if state.stage.valid:
            return state

    return None
