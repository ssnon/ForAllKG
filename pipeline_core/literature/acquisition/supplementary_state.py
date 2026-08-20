from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pipeline_core.literature.acquisition.access_contracts import SourceArtifact
from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryDiscovery


def access_resolution_sha256(resolution) -> str:
    import hashlib

    payload = json.dumps(
        resolution.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_state_name(work_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", work_id).strip("_")
    digest = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:56]}__{digest}.json"


def atomic_write_json(path: Path, value: Any) -> None:
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


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    tmp.replace(path)


def load_state(
    path: Path,
) -> tuple[SupplementaryDiscovery, list[SourceArtifact]] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    discovery = SupplementaryDiscovery.model_validate(
        loaded["supplementary_discovery"]
    )
    artifacts = [
        SourceArtifact.model_validate(row)
        for row in loaded.get("artifacts", [])
    ]
    return discovery, artifacts


def write_state(
    *,
    path: Path,
    discovery: SupplementaryDiscovery,
    artifacts: list[SourceArtifact],
    main_access_sha256: str | None,
) -> None:
    atomic_write_json(
        path,
        {
            "work_id": discovery.work_id,
            "main_access_sha256": main_access_sha256,
            "supplementary_discovery": discovery.model_dump(mode="json"),
            "artifacts": [
                row.model_dump(mode="json") for row in artifacts
            ],
        },
    )



def state_matches_main_access(
    *,
    path: Path,
    main_access_sha256: str | None,
) -> bool:
    if not path.exists():
        return False
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded.get("main_access_sha256") == main_access_sha256
