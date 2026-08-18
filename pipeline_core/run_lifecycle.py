"""Shared run-family and attempt lifecycle runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_core.serialization_primitives import read_json, write_json


def paper_output_root(
    project_root: str | Path,
    paper_id: str,
    *,
    data_root: str | Path,
) -> Path:
    data_root_path = Path(data_root)
    if not data_root_path.is_absolute():
        data_root_path = Path(project_root).resolve() / data_root_path
    return data_root_path.resolve() / "extracted" / paper_id


def run_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    *,
    data_root: str | Path,
) -> Path:
    return (
        paper_output_root(
            project_root,
            paper_id,
            data_root=data_root,
        )
        / "runs"
        / run_id
    )


def attempt_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    attempt_id: str,
    *,
    data_root: str | Path,
) -> Path:
    return (
        run_directory(
            project_root,
            paper_id,
            run_id,
            data_root=data_root,
        )
        / "attempts"
        / attempt_id
    )


def _latest_attempt_from_family(
    run_dir: Path,
) -> Path:
    pointer_path = run_dir / "latest_attempt.json"

    if not pointer_path.exists():
        return run_dir

    pointer = read_json(pointer_path)
    raw = pointer.get("attempt_directory")

    if not raw:
        return run_dir

    path = Path(str(raw))
    return path if path.exists() else run_dir


def write_latest_attempt_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    attempt_id: str,
    data_root: str | Path,
    attempt_layout_version: str,
    updated_at_utc: str,
) -> Path:
    run_id = str(run_metadata["run_id"])

    family_dir = run_directory(
        project_root,
        paper_id,
        run_id,
        data_root=data_root,
    )

    concrete_dir = attempt_directory(
        project_root,
        paper_id,
        run_id,
        attempt_id,
        data_root=data_root,
    )

    return write_json(
        family_dir / "latest_attempt.json",
        {
            "paper_id": paper_id,
            "run_id": run_id,
            "run_fingerprint": run_metadata["run_fingerprint"],
            "attempt_layout_version": attempt_layout_version,
            "attempt_id": attempt_id,
            "attempt_directory": str(concrete_dir),
            "updated_at_utc": updated_at_utc,
        },
    )


def write_latest_run_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    data_root: str | Path,
    attempt_layout_version: str,
    updated_at_utc: str,
    attempt_id: str | None = None,
) -> Path:
    root = paper_output_root(
        project_root,
        paper_id,
        data_root=data_root,
    )

    run_id = str(run_metadata["run_id"])

    family_dir = run_directory(
        project_root,
        paper_id,
        run_id,
        data_root=data_root,
    )

    payload: dict[str, Any] = {
        "paper_id": paper_id,
        "run_id": run_id,
        "run_fingerprint": run_metadata["run_fingerprint"],
        "run_directory": str(family_dir),
        "updated_at_utc": updated_at_utc,
    }

    if attempt_id:
        payload["attempt_layout_version"] = attempt_layout_version
        payload["attempt_id"] = attempt_id
        payload["attempt_directory"] = str(
            attempt_directory(
                project_root,
                paper_id,
                run_id,
                attempt_id,
                data_root=data_root,
            )
        )

    return write_json(
        root / "latest_run.json",
        payload,
    )


def resolve_run_directory(
    *,
    project_root: str | Path,
    paper_id: str,
    run_id: str | None,
    data_root: str | Path,
    attempt_id: str | None = None,
) -> Path:
    pointer: dict[str, Any] | None = None

    if run_id:
        family_dir = run_directory(
            project_root,
            paper_id,
            run_id,
            data_root=data_root,
        )
    else:
        pointer_path = (
            paper_output_root(
                project_root,
                paper_id,
                data_root=data_root,
            )
            / "latest_run.json"
        )

        if not pointer_path.exists():
            raise FileNotFoundError(
                "No latest run pointer found for "
                f"{paper_id!r}: {pointer_path}"
            )

        pointer = read_json(pointer_path)
        family_dir = Path(pointer["run_directory"])

    if attempt_id:
        path = family_dir / "attempts" / attempt_id
    elif pointer and pointer.get("attempt_directory"):
        candidate = Path(str(pointer["attempt_directory"]))
        path = (
            candidate
            if candidate.exists()
            else _latest_attempt_from_family(family_dir)
        )
    else:
        path = _latest_attempt_from_family(family_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Run directory not found: {path}"
        )

    return path.resolve()
