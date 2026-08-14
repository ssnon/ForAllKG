from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dac_her.corpus_acquisition.access_contracts import (
    AccessResolution,
    SourceArtifact,
)


def safe_state_name(work_id: str) -> str:
    import hashlib

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", work_id).strip("_")
    digest = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:56]}__{digest}.json"


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_work_state(
    path: Path,
) -> tuple[AccessResolution, SourceArtifact] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid work state: {path}")
    resolution = AccessResolution.model_validate(
        loaded["access_resolution"]
    )
    artifact = SourceArtifact.model_validate(
        loaded["main_artifact"]
    )
    return resolution, artifact


def write_work_state(
    *,
    path: Path,
    resolution: AccessResolution,
    artifact: SourceArtifact,
) -> None:
    atomic_write_json(
        path,
        {
            "work_id": resolution.work_id,
            "access_resolution": resolution.model_dump(mode="json"),
            "main_artifact": artifact.model_dump(mode="json"),
        },
    )


def write_jsonl(
    path: Path,
    rows: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    tmp.replace(path)
