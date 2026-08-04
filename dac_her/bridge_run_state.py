from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from dac_her.bridge_policy import BRIDGE_POLICY_VERSION
from dac_her.bridge_prompts import BRIDGE_PROMPT_VERSION


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    return _sha256_bytes(path.read_bytes())


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def compute_bridge_run_metadata(
    *,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    model: str,
    provider: str | None,
    strict_chunk_paths: Iterable[str | Path],
    source_chunk_paths: Iterable[str | Path],
    canonical_graph_path: str | Path | None,
    implementation_paths: Iterable[str | Path],
) -> dict[str, Any]:
    strict_run_dir = Path(strict_run_dir).resolve()
    strict_chunks = {
        str(Path(path).resolve()): file_sha256(path)
        for path in sorted(map(Path, strict_chunk_paths), key=lambda item: str(item))
    }
    source_chunks = {
        str(Path(path).resolve()): file_sha256(path)
        for path in sorted(map(Path, source_chunk_paths), key=lambda item: str(item))
    }
    implementation = {
        str(Path(path).name): file_sha256(path)
        for path in sorted(map(Path, implementation_paths), key=lambda item: str(item))
    }

    fingerprint_payload = {
        "bridge_prompt_version": BRIDGE_PROMPT_VERSION,
        "bridge_policy_version": BRIDGE_POLICY_VERSION,
        "model": model,
        "provider": provider or "",
        "strict_run_id": str(active_payload.get("run_id", "")),
        "strict_run_fingerprint": str(active_payload.get("run_fingerprint", "")),
        "strict_chunks": strict_chunks,
        "source_chunks": source_chunks,
        "canonical_graph_sha256": file_sha256(canonical_graph_path),
        "implementation": implementation,
    }
    fingerprint = stable_json_hash(fingerprint_payload)
    return {
        "bridge_run_id": fingerprint[:16],
        "bridge_run_fingerprint": fingerprint,
        "strict_run_directory": str(strict_run_dir),
        **fingerprint_payload,
    }


def bridge_run_directory(
    strict_run_dir: str | Path,
    bridge_run_id: str,
) -> Path:
    return Path(strict_run_dir) / "bridge_runs" / bridge_run_id
