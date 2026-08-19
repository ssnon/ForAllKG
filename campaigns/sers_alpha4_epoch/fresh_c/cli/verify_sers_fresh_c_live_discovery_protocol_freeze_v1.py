from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_DISCOVERY_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_C01B_FREEZE_COMMIT,
    EXPECTED_C01B_LOCK_ID,
    EXPECTED_C01B_LOCK_SHA256,
    EXPECTED_HISTORICAL_LEDGER_ID,
    EXPECTED_HISTORICAL_LEDGER_SHA256,
    load_and_validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_live_discovery_protocol_v1 import CRITICAL_COMPONENTS
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_protocol_v1 import verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Fresh-C C0.1C live-discovery protocol freeze."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_DISCOVERY_FREEZE_DIR)
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _payload_sha(payload: dict, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    args = parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    freeze_dir = args.freeze_dir if args.freeze_dir.is_absolute() else root / args.freeze_dir

    subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_activation_readiness_v1"],
        cwd=root,
        check=True,
    )
    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)
    manifest = _read(freeze_dir / "freeze_manifest.json")
    ready = _read(freeze_dir / "FREEZE_READY.json")

    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("C0.1C freeze protocol ID mismatch.")
    if manifest.get("protocol_sha256") != protocol.protocol_sha256:
        raise ValueError("C0.1C freeze protocol SHA mismatch.")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise ValueError("C0.1C freeze manifest SHA drifted.")
    if manifest.get("c0_1b_lock_id") != EXPECTED_C01B_LOCK_ID:
        raise ValueError("C0.1B lock ID mismatch in C0.1C freeze.")
    if manifest.get("c0_1b_lock_sha256") != EXPECTED_C01B_LOCK_SHA256:
        raise ValueError("C0.1B lock SHA mismatch in C0.1C freeze.")
    if manifest.get("c0_1b_freeze_commit") != EXPECTED_C01B_FREEZE_COMMIT:
        raise ValueError("C0.1B freeze commit mismatch.")
    if manifest.get("historical_ledger_id") != EXPECTED_HISTORICAL_LEDGER_ID:
        raise ValueError("Historical ledger ID mismatch in freeze.")
    if manifest.get("historical_ledger_sha256") != EXPECTED_HISTORICAL_LEDGER_SHA256:
        raise ValueError("Historical ledger SHA mismatch in freeze.")

    source_commit = str(manifest.get("source_code_commit") or "")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    hashes = manifest.get("critical_component_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C0.1C critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"C0.1C frozen source hash mismatch: {relative}")
        current = root / relative
        if sha256_file(current) != expected:
            raise ValueError(f"C0.1C current critical component drifted: {relative}")

    for field in (
        "live_discovery_authorized",
        "live_discovery_started",
        "fresh_reserve_c_consumed",
        "semantic_read_performed",
        "automatic_c0_1d_transition_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C0.1C freeze safety field drifted: {field}")
    if manifest.get("live_discovery_ready") is not True:
        raise ValueError("C0.1C freeze is not ready.")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C0.1C freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C0.1C freeze used LLM.")
    if manifest.get("stop") is not True:
        raise ValueError("C0.1C freeze STOP drifted.")

    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("C0.1C FREEZE_READY ID mismatch.")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("C0.1C FREEZE_READY SHA mismatch.")
    if ready.get("live_discovery_authorized") is not False:
        raise ValueError("C0.1C FREEZE_READY unexpectedly authorizes discovery.")
    if ready.get("live_discovery_started") is not False:
        raise ValueError("C0.1C FREEZE_READY unexpectedly starts discovery.")
    if ready.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("C0.1C FREEZE_READY unexpectedly consumes Fresh C.")

    print("Fresh-C C0.1C live-discovery protocol freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Live discovery ready: True")
    print("Live discovery authorized: False")
    print("Live discovery started: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("Fresh Reserve C consumed: False")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
