from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dac_her.fresh_c_acquisition import (
    canonical_json,
    load_and_validate_protocol,
    sha256_file,
    sha256_json,
)
from scripts.verify_sers_fresh_c_acquisition_protocol_v1 import verify


DEFAULT_PROTOCOL = Path(
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "evaluation/sers_fresh_c/"
    "c0_1a_protocol_preregistration_freeze_v1"
)

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_acquisition.py",
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json",
    "scripts/verify_sers_fresh_c_acquisition_protocol_v1.py",
    "scripts/freeze_sers_fresh_c_acquisition_protocol_v1.py",
    "scripts/verify_sers_fresh_c_acquisition_protocol_freeze_v1.py",
    "tests/test_sers_fresh_c_acquisition_protocol_v1.py",
    "dac_her/corpus_acquisition/catalog_expansion.py",
    "dac_her/corpus_acquisition/oa_resolution.py",
    "dac_her/corpus_acquisition/artifact_acquisition.py",
    "scripts/acquire_corpus_sources.py",
    "configs/acquisition/source_access_default_v1.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the non-activating Fresh-C C0.1A protocol "
            "preregistration. All critical code must already be tracked "
            "and byte-identical to HEAD."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    args = parse_args()
    root = Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel")
    )
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else root / args.protocol
    )
    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)

    # Tracked tree/index must be clean. Untracked backup patches are allowed.
    tracked_worktree = subprocess.run(
        ["git", "diff", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode
    tracked_index = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode
    if tracked_worktree != 0 or tracked_index != 0:
        raise RuntimeError(
            "Tracked tree/index is not clean; refuse protocol freeze."
        )

    source_commit = _git(root, "rev-parse", "HEAD")
    component_sha256: dict[str, str] = {}
    for relative in CRITICAL_COMPONENTS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            committed = _git_bytes(root, source_commit, relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Critical Fresh-C preregistration component is not "
                f"tracked in HEAD: {relative}"
            ) from exc
        working_sha = sha256_file(path)
        committed_sha = _sha256_bytes(committed)
        if working_sha != committed_sha:
            raise RuntimeError(
                "Critical component differs from HEAD: "
                f"{relative}"
            )
        component_sha256[relative] = working_sha

    body: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-acquisition-protocol-"
            "preregistration-freeze-v1"
        ),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "source_code_commit": source_commit,
        "critical_component_sha256": component_sha256,
        "freeze_kind": "PREREGISTRATION_ONLY",
        "stage": "C0.1A",
        "activation_preconditions_required": (
            protocol.activation_preconditions_required
        ),
        "activation_preconditions_satisfied": False,
        "fresh_c_stage_activated": False,
        "live_discovery_started": False,
        "live_selection_started": False,
        "live_acquisition_started": False,
        "content_sealed": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    identity_sha = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_acquisition_protocol_preregistration_freeze_v1:"
        + identity_sha[:20]
    )
    body["manifest_sha256"] = _payload_sha(
        body,
        "manifest_sha256",
    )

    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    manifest_path = output / "freeze_manifest.json"
    ready_path = output / "FREEZE_READY.json"
    if manifest_path.exists() or ready_path.exists():
        raise FileExistsError(
            "Fresh-C preregistration freeze artifacts already exist; "
            "refuse overwrite."
        )

    _atomic_json(manifest_path, body)
    ready = {
        "schema_version": (
            "sers-fresh-c-acquisition-protocol-"
            "preregistration-ready-v1"
        ),
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "fresh_c_stage_activated": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    _atomic_json(ready_path, ready)

    print("Fresh-C C0.1A protocol preregistration freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Fresh-C stage activated: False")
    print("Live discovery started: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
