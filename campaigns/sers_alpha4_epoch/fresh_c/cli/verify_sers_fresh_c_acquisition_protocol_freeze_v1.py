from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    load_and_validate_protocol,
    sha256_file,
    sha256_json,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_acquisition_protocol_v1 import (
    CRITICAL_COMPONENTS,
    DEFAULT_OUTPUT_DIR,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_acquisition_protocol_v1 import verify


DEFAULT_PROTOCOL = Path(
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the non-activating Fresh-C C0.1A protocol "
            "preregistration freeze."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--freeze-dir",
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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
    freeze_dir = (
        args.freeze_dir
        if args.freeze_dir.is_absolute()
        else root / args.freeze_dir
    )
    manifest_path = freeze_dir / "freeze_manifest.json"
    ready_path = freeze_dir / "FREEZE_READY.json"

    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)
    manifest = _read_json(manifest_path)
    ready = _read_json(ready_path)

    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Freeze protocol ID mismatch.")
    if manifest.get("protocol_sha256") != protocol.protocol_sha256:
        raise ValueError("Freeze protocol SHA mismatch.")
    if manifest.get("freeze_kind") != "PREREGISTRATION_ONLY":
        raise ValueError("Freeze kind drifted.")
    expected_manifest_sha = _payload_sha(
        manifest,
        "manifest_sha256",
    )
    if manifest.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError("Freeze manifest SHA drifted.")

    source_commit = str(manifest.get("source_code_commit") or "")
    if not source_commit:
        raise ValueError("Freeze lacks source_code_commit.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    historical_hashes = manifest.get("critical_component_sha256")
    if not isinstance(historical_hashes, dict) or not historical_hashes:
        raise ValueError("Freeze critical-component hash map missing.")

    # Historical freeze identity is bound to the paths that existed at the
    # recorded source commit.  Current campaign relocation must not rewrite
    # or reinterpret those frozen path identities.
    for historical_path, expected_sha in sorted(historical_hashes.items()):
        if not isinstance(historical_path, str) or not historical_path.strip():
            raise ValueError("Freeze critical-component path is invalid.")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ValueError(
                "Freeze critical-component SHA is invalid: "
                f"{historical_path}"
            )

        committed = _git_bytes(
            root,
            source_commit,
            historical_path,
        )
        committed_sha = _sha256_bytes(committed)

        if expected_sha != committed_sha:
            raise ValueError(
                "Historical source-commit component hash mismatch: "
                f"{historical_path}"
            )

    # The relocated campaign implementation is a replay/verification surface,
    # not part of the historical freeze identity.  Require it to exist, but do
    # not compare its current bytes against historical source-commit hashes.
    missing_current_components = [
        relative
        for relative in CRITICAL_COMPONENTS
        if not (root / relative).is_file()
    ]
    if missing_current_components:
        raise FileNotFoundError(
            "Current relocated critical component missing: "
            + ", ".join(sorted(missing_current_components))
        )

    false_fields = (
        "activation_preconditions_satisfied",
        "fresh_c_stage_activated",
        "live_discovery_started",
        "live_selection_started",
        "live_acquisition_started",
        "content_sealed",
        "fresh_reserve_c_consumed",
        "semantic_read_performed",
        "automatic_next_stage_authorized",
    )
    for field in false_fields:
        if manifest.get(field) is not False:
            raise ValueError(f"Freeze safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("Freeze must have zero network calls.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("Freeze must have zero LLM calls.")
    if manifest.get("stop") is not True:
        raise ValueError("Freeze STOP guard drifted.")

    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("READY freeze ID mismatch.")
    if ready.get("manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("READY manifest SHA mismatch.")
    if ready.get("fresh_c_stage_activated") is not False:
        raise ValueError("READY unexpectedly activates Fresh C.")
    if ready.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("READY unexpectedly consumes Fresh C.")
    if ready.get("automatic_next_stage_authorized") is not False:
        raise ValueError("READY unexpectedly authorizes next stage.")
    if ready.get("stop") is not True:
        raise ValueError("READY STOP guard drifted.")

    print("Fresh-C C0.1A preregistration freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Fresh-C stage activated: False")
    print("Live discovery started: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
