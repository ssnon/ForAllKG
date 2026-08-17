from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_live_discovery_recovery_v2_1 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V2_FREEZE_COMMIT,
    assert_parent_v2_never_started_network,
    load_and_validate_protocol,
)
from scripts.freeze_sers_fresh_c_live_discovery_recovery_v2_1_protocol import (
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
            "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_protocol_freeze",
        ],
        cwd=root,
        check=True,
    )
    assert_parent_v2_never_started_network(root)

    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    manifest = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    ready = _read(root / DEFAULT_FREEZE_DIR / "FREEZE_READY.json")

    if manifest.get("protocol_id") != p.protocol_id:
        raise ValueError("v2.1 freeze protocol ID mismatch.")
    if manifest.get("protocol_sha256") != p.protocol_sha256:
        raise ValueError("v2.1 freeze protocol SHA mismatch.")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise ValueError("v2.1 freeze manifest SHA drifted.")
    if manifest.get("parent_v2_freeze_commit") != EXPECTED_V2_FREEZE_COMMIT:
        raise ValueError("v2.1 parent freeze commit mismatch.")
    if manifest.get("parent_v2_network_epoch_started") is not False:
        raise ValueError("v2.1 incorrectly marks parent network started.")
    if manifest.get("harness_change_only") is not True:
        raise ValueError("v2.1 harness-only flag drifted.")
    if manifest.get("scientific_search_transport_semantics_changed") is not False:
        raise ValueError("v2.1 semantics changed.")
    if manifest.get("recovery_live_discovery_authorized") is not False:
        raise ValueError("v2.1 unexpectedly authorized.")
    if manifest.get("recovery_live_discovery_started") is not False:
        raise ValueError("v2.1 unexpectedly started.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.1 unexpectedly consumed Fresh C.")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("v2.1 freeze used network.")
    if manifest.get("stop") is not True:
        raise ValueError("v2.1 STOP drifted.")

    source_commit = manifest["source_code_commit"]
    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("v2.1 critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"v2.1 frozen source hash mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"v2.1 current source drifted: {relative}")

    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("v2.1 ready ID mismatch.")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("v2.1 ready SHA mismatch.")

    print("Fresh-C C0.1C-v2.1 harness-repair freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Parent v2 network epoch started: False")
    print("Harness change only: True")
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
