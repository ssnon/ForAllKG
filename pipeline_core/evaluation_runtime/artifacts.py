from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Serialize a value using the repository's canonical JSON convention."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    """Return SHA256 of canonical JSON encoded as UTF-8."""
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return SHA256 of file bytes without loading the whole file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json_without_fields(
    payload: Mapping[str, Any],
    *fields: str,
) -> str:
    """Canonical JSON SHA after removing named top-level fields."""
    value = dict(payload)
    for field in fields:
        value.pop(field, None)
    return sha256_json(value)


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Read UTF-8 JSON and fail closed unless its root is an object."""
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value
