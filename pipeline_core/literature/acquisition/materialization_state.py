from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_core.literature.acquisition.materialization_contracts import MaterializedDocument


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


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
                ) + "\n"
            )
    tmp.replace(path)


def state_path(
    root: Path,
    paper_id: str,
) -> Path:
    return root / f"{paper_id}.json"


def load_state(
    path: Path,
) -> list[MaterializedDocument] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid M4 state: {path}")
    return [
        MaterializedDocument.model_validate(row)
        for row in loaded.get("documents", [])
    ]


def write_state(
    *,
    path: Path,
    documents: list[MaterializedDocument],
    source_artifact_ids: list[str],
    source_artifact_sha256: dict[str, str | None],
) -> None:
    atomic_write_json(
        path,
        {
            "paper_id": (
                documents[0].paper_id
                if documents
                else path.stem
            ),
            "source_artifact_ids": source_artifact_ids,
            "source_artifact_sha256": source_artifact_sha256,
            "documents": [
                row.model_dump(mode="json")
                for row in documents
            ],
        },
    )


def state_matches_sources(
    *,
    path: Path,
    artifacts,
) -> bool:
    if not path.exists():
        return False
    loaded = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = sorted(
        artifact.artifact_id for artifact in artifacts
    )
    expected_sha = {
        artifact.artifact_id: artifact.sha256
        for artifact in artifacts
    }
    return (
        sorted(loaded.get("source_artifact_ids", []))
        == expected_ids
        and loaded.get("source_artifact_sha256", {})
        == expected_sha
    )
