from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence


Mode = Literal[
    "evidence",
    "mechanism",
    "exploratory",
]

FreshnessPolicy = Literal[
    "source",
    "semantic",
    "full",
]

StageName = Literal[
    "strict",
    "strict_graph",
    "bridge",
    "projection",
    "corpus",
    "navigation",
    "index",
]


class ReconcileError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    return (
        value
        if isinstance(value, dict)
        else None
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_component(
    value: str,
) -> str:
    return (
        value
        .replace("/", "_")
        .replace("\\", "_")
        .strip()
        or "paper"
    )


def same_path(
    left: str | Path,
    right: str | Path,
) -> bool:
    try:
        return (
            Path(left).resolve()
            == Path(right).resolve()
        )
    except Exception:
        return str(left) == str(right)


@dataclass(frozen=True)
class StageState:
    valid: bool
    reason: str
    path: Path | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def ready(
        cls,
        reason: str,
        path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StageState":
        return cls(
            True,
            reason,
            path,
            metadata,
        )

    @classmethod
    def pending(
        cls,
        reason: str,
        path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StageState":
        return cls(
            False,
            reason,
            path,
            metadata,
        )


@dataclass(frozen=True)
class StrictRunState:
    stage: StageState
    run_id: str | None = None
    run_dir: Path | None = None


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: str | Path,
    label: str,
    log_dir: str | Path,
    heartbeat_seconds: float,
    dry_run: bool = False,
) -> bool:
    print(
        f"[reconcile] {label} | start",
        flush=True,
    )
    print(
        "[reconcile]   $ "
        + shlex.join(list(command)),
        flush=True,
    )

    if dry_run:
        return True

    cwd = Path(cwd)
    log_dir = Path(log_dir)

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stdout_path = (
        log_dir / "stdout.log"
    )
    stderr_path = (
        log_dir / "stderr.log"
    )

    started = time.monotonic()

    with (
        stdout_path.open(
            "w",
            encoding="utf-8",
        ) as stdout,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr,
    ):
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        while True:
            try:
                code = process.wait(
                    timeout=(
                        heartbeat_seconds
                        or None
                    )
                )
                break

            except subprocess.TimeoutExpired:
                print(
                    f"[reconcile] {label} "
                    "| still running | "
                    "elapsed="
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )

    elapsed = (
        time.monotonic() - started
    )

    if code == 0:
        print(
            f"[reconcile] {label} "
            f"| passed | {elapsed:.1f}s",
            flush=True,
        )
        return True

    print(
        f"[reconcile] {label} "
        f"| failed({code}) "
        f"| {elapsed:.1f}s "
        f"| stderr={stderr_path}",
        flush=True,
    )

    return False
