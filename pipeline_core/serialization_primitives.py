"""Strict shared hashing and JSON serialization primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_text(
    text: str,
) -> str:
    return sha256_bytes(
        text.encode("utf-8")
    )


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_json(
    path: str | Path,
    payload: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def read_json(
    path: str | Path,
) -> Any:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )
