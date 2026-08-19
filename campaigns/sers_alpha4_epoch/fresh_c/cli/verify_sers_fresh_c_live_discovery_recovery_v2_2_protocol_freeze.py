from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_2 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    V21_FAILED_PATH,
    V21_STARTED_PATH,
    load_and_validate_protocol,
    validate_v21_failed_epoch,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_live_discovery_recovery_v2_2_protocol import (
    CRITICAL_COMPONENTS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol_freeze",
        ],
        cwd=root,
        check=True,
    )
    validate_v21_failed_epoch(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    manifest = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    ready = _read(root / DEFAULT_FREEZE_DIR / "FREEZE_READY.json")

    if manifest.get("protocol_id") != p.protocol_id:
        raise ValueError("v2.2 freeze protocol ID mismatch.")
    if manifest.get("protocol_sha256") != p.protocol_sha256:
        raise ValueError("v2.2 freeze protocol SHA mismatch.")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise ValueError("v2.2 freeze manifest SHA drifted.")
    if manifest.get("parent_v21_network_epoch_started") is not True:
        raise ValueError("v2.2 parent network-start flag drifted.")
    if manifest.get("parent_v21_failed_epoch_preserved") is not True:
        raise ValueError("v2.2 parent preservation flag drifted.")
    if manifest.get("parent_v21_success_artifacts_absent") is not True:
        raise ValueError("v2.2 parent success-artifact guard drifted.")
    if manifest.get("compatibility_change_only") is not True:
        raise ValueError("v2.2 compatibility-only flag drifted.")
    if manifest.get("scientific_search_transport_semantics_changed") is not False:
        raise ValueError("v2.2 semantics changed.")
    if manifest.get("recovery_live_discovery_authorized") is not False:
        raise ValueError("v2.2 unexpectedly authorized.")
    if manifest.get("recovery_live_discovery_started") is not False:
        raise ValueError("v2.2 unexpectedly started.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.2 consumed Fresh C.")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("v2.2 freeze used network.")
    if manifest.get("stop") is not True:
        raise ValueError("v2.2 STOP drifted.")

    source_commit = manifest["source_code_commit"]
    for relative, field in (
        (V21_STARTED_PATH, "parent_v21_started_file_sha256"),
        (V21_FAILED_PATH, "parent_v21_failed_file_sha256"),
    ):
        current_sha = sha256_file(root / relative)
        if manifest.get(field) != current_sha:
            raise ValueError(f"v2.2 parent marker SHA drifted: {relative}")
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative.as_posix()}"],
            cwd=root,
        )
        if hashlib.sha256(committed).hexdigest() != current_sha:
            raise ValueError(f"v2.2 parent marker not frozen: {relative}")

    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("v2.2 critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"v2.2 frozen source hash mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"v2.2 current source drifted: {relative}")

    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("v2.2 ready ID mismatch.")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("v2.2 ready SHA mismatch.")

    print("Fresh-C C0.1C-v2.2 compatibility-repair freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Parent v2.1 attempt: {manifest['parent_v21_attempt_id']}")
    print("Parent v2.1 network epoch started: True")
    print("Parent v2.1 failed epoch preserved: True")
    print("Compatibility change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Recovery live discovery ready: True")
    print("Recovery live discovery authorized: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
