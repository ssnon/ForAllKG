from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
)
from dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
)


def _sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(
    path: str | Path | None,
) -> str:
    if path is None:
        return ""

    path = Path(path)

    if (
        not path.exists()
        or not path.is_file()
    ):
        return ""

    return _sha256_bytes(
        path.read_bytes()
    )


def stable_json_hash(
    value: Any,
) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return _sha256_bytes(payload)


def _created_at_utc() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _hash_files_by_name(
    paths: Iterable[str | Path],
) -> dict[str, str]:
    result: dict[str, str] = {}

    normalized = sorted(
        (
            Path(path).resolve()
            for path in paths
        ),
        key=lambda item: str(item),
    )

    for path in normalized:
        name = path.name

        if name in result:
            raise ValueError(
                "Duplicate fingerprint "
                f"filename: {name!r}"
            )

        result[name] = file_sha256(path)

    return result


def compute_bridge_extraction_metadata(
    *,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    model: str,
    provider: str | None,
    strict_chunk_paths: Iterable[
        str | Path
    ],
    source_chunk_paths: Iterable[
        str | Path
    ],
    implementation_paths: Iterable[
        str | Path
    ],
    runtime_options: (
        dict[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    """
    Compute the identity of raw LLM extraction.

    This fingerprint intentionally excludes:
    - Bridge policy
    - deterministic relation repairs
    - canonical graph
    - Bridge graph materialization
    """
    strict_run_dir = Path(
        strict_run_dir
    ).resolve()

    payload = {
        "bridge_prompt_version": (
            BRIDGE_PROMPT_VERSION
        ),
        "model": model,
        "provider": provider or "",
        "strict_run_id": str(
            active_payload.get(
                "run_id",
                "",
            )
        ),
        "strict_run_fingerprint": str(
            active_payload.get(
                "run_fingerprint",
                "",
            )
        ),
        "strict_chunks": (
            _hash_files_by_name(
                strict_chunk_paths
            )
        ),
        "source_chunks": (
            _hash_files_by_name(
                source_chunk_paths
            )
        ),
        "implementation": (
            _hash_files_by_name(
                implementation_paths
            )
        ),
        "runtime_options": (
            runtime_options or {}
        ),
    }

    fingerprint = stable_json_hash(
        payload
    )

    return {
        "bridge_extraction_id": (
            fingerprint[:16]
        ),
        "bridge_extraction_fingerprint": (
            fingerprint
        ),
        "strict_run_directory": str(
            strict_run_dir
        ),
        "created_at_utc": (
            _created_at_utc()
        ),
        **payload,
    }


def compute_bridge_policy_run_metadata(
    *,
    strict_run_dir: str | Path,
    extraction_metadata: dict[
        str,
        Any,
    ],
    raw_chunk_paths: Iterable[
        str | Path
    ],
    canonical_graph_path: (
        str | Path | None
    ),
    implementation_paths: Iterable[
        str | Path
    ],
) -> dict[str, Any]:
    """
    Compute the identity of one policy application.

    This fingerprint depends on the frozen raw
    extraction, current Bridge policy, deterministic
    repair implementation, and canonical graph.
    """
    strict_run_dir = Path(
        strict_run_dir
    ).resolve()

    payload = {
        "bridge_extraction_id": str(
            extraction_metadata[
                "bridge_extraction_id"
            ]
        ),
        "bridge_extraction_fingerprint": str(
            extraction_metadata[
                "bridge_extraction_fingerprint"
            ]
        ),
        "bridge_policy_version": (
            BRIDGE_POLICY_VERSION
        ),
        "raw_chunks": (
            _hash_files_by_name(
                raw_chunk_paths
            )
        ),
        "canonical_graph_sha256": (
            file_sha256(
                canonical_graph_path
            )
        ),
        "implementation": (
            _hash_files_by_name(
                implementation_paths
            )
        ),
    }

    fingerprint = stable_json_hash(
        payload
    )

    return {
        "bridge_policy_run_id": (
            fingerprint[:16]
        ),
        "bridge_policy_run_fingerprint": (
            fingerprint
        ),
        "strict_run_directory": str(
            strict_run_dir
        ),
        "created_at_utc": (
            _created_at_utc()
        ),
        **payload,
    }


def bridge_extraction_directory(
    strict_run_dir: str | Path,
    extraction_id: str,
) -> Path:
    return (
        Path(strict_run_dir)
        / "bridge_extractions"
        / extraction_id
    )


def bridge_policy_run_directory(
    strict_run_dir: str | Path,
    policy_run_id: str,
) -> Path:
    return (
        Path(strict_run_dir)
        / "bridge_policy_runs"
        / policy_run_id
    )


# ============================================================
# Legacy compatibility
# ============================================================

def compute_bridge_run_metadata(
    *,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    model: str,
    provider: str | None,
    strict_chunk_paths: Iterable[
        str | Path
    ],
    source_chunk_paths: Iterable[
        str | Path
    ],
    canonical_graph_path: (
        str | Path | None
    ),
    implementation_paths: Iterable[
        str | Path
    ],
) -> dict[str, Any]:
    """
    Legacy combined fingerprint retained while older
    run directories and scripts remain readable.
    """
    strict_run_dir = Path(
        strict_run_dir
    ).resolve()

    payload = {
        "bridge_prompt_version": (
            BRIDGE_PROMPT_VERSION
        ),
        "bridge_policy_version": (
            BRIDGE_POLICY_VERSION
        ),
        "model": model,
        "provider": provider or "",
        "strict_run_id": str(
            active_payload.get(
                "run_id",
                "",
            )
        ),
        "strict_run_fingerprint": str(
            active_payload.get(
                "run_fingerprint",
                "",
            )
        ),
        "strict_chunks": (
            _hash_files_by_name(
                strict_chunk_paths
            )
        ),
        "source_chunks": (
            _hash_files_by_name(
                source_chunk_paths
            )
        ),
        "canonical_graph_sha256": (
            file_sha256(
                canonical_graph_path
            )
        ),
        "implementation": (
            _hash_files_by_name(
                implementation_paths
            )
        ),
    }

    fingerprint = stable_json_hash(
        payload
    )

    return {
        "bridge_run_id": (
            fingerprint[:16]
        ),
        "bridge_run_fingerprint": (
            fingerprint
        ),
        "strict_run_directory": str(
            strict_run_dir
        ),
        "created_at_utc": (
            _created_at_utc()
        ),
        **payload,
    }


def bridge_run_directory(
    strict_run_dir: str | Path,
    bridge_run_id: str,
) -> Path:
    return (
        Path(strict_run_dir)
        / "bridge_runs"
        / bridge_run_id
    )
