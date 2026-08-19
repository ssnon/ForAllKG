from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V1_FAILED_ATTEMPT_ID,
    V1_FAILED_PATH,
    V1_STARTED_PATH,
    load_and_validate_protocol,
    validate_v1_failed_epoch,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_live_discovery_recovery_v2_protocol import (
    CRITICAL_COMPONENTS,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_protocol import (
    verify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify C0.1C-v2 recovery protocol freeze."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=DEFAULT_FREEZE_DIR,
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_fresh_c_live_discovery_protocol_freeze_v1",
        ],
        cwd=root,
        check=True,
    )
    validate_v1_failed_epoch(root)
    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)

    manifest = _read(freeze_dir / "freeze_manifest.json")
    ready = _read(freeze_dir / "FREEZE_READY.json")

    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Recovery-v2 freeze protocol ID mismatch.")
    if manifest.get("protocol_sha256") != protocol.protocol_sha256:
        raise ValueError("Recovery-v2 freeze protocol SHA mismatch.")
    if manifest.get("manifest_sha256") != _payload_sha(
        manifest,
        "manifest_sha256",
    ):
        raise ValueError("Recovery-v2 freeze manifest SHA drifted.")
    if (
        manifest.get("recovery_parent_attempt_id")
        != EXPECTED_V1_FAILED_ATTEMPT_ID
    ):
        raise ValueError("Recovery-v2 parent attempt mismatch.")

    source_commit = str(manifest.get("source_code_commit") or "")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    for relative, field in (
        (V1_STARTED_PATH, "recovery_parent_started_file_sha256"),
        (V1_FAILED_PATH, "recovery_parent_failed_file_sha256"),
    ):
        current = root / relative
        current_sha = sha256_file(current)
        if manifest.get(field) != current_sha:
            raise ValueError(
                f"Recovery parent marker SHA drifted: {relative}"
            )
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative.as_posix()}"],
            cwd=root,
        )
        if hashlib.sha256(committed).hexdigest() != current_sha:
            raise ValueError(
                f"Recovery parent marker not frozen in source commit: {relative}"
            )

    hashes = manifest.get("critical_component_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("Recovery-v2 critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(
                f"Recovery-v2 frozen source hash mismatch: {relative}"
            )
        if sha256_file(root / relative) != expected:
            raise ValueError(
                f"Recovery-v2 current source drifted: {relative}"
            )

    if manifest.get("recovery_parent_failed_epoch_preserved") is not True:
        raise ValueError("Recovery parent preservation flag drifted.")
    for field in (
        "search_queries_changed_from_v1",
        "provider_set_changed_from_v1",
        "search_depth_changed_from_v1",
        "historical_ledger_changed_from_v1",
        "target_count_changed_from_v1",
        "blind_ordering_changed_from_v1",
        "scientific_selection_semantics_changed_from_v1",
        "recovery_live_discovery_authorized",
        "recovery_live_discovery_started",
        "fresh_reserve_c_consumed",
        "semantic_read_performed",
        "automatic_c0_1d_transition_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"Recovery-v2 safety field drifted: {field}")
    if manifest.get("recovery_live_discovery_ready") is not True:
        raise ValueError("Recovery-v2 is not ready.")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("Recovery-v2 freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("Recovery-v2 freeze used LLM.")
    if manifest.get("stop") is not True:
        raise ValueError("Recovery-v2 STOP drifted.")

    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("Recovery-v2 FREEZE_READY ID mismatch.")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Recovery-v2 FREEZE_READY SHA mismatch.")
    if ready.get("recovery_live_discovery_authorized") is not False:
        raise ValueError("Recovery-v2 unexpectedly authorized.")
    if ready.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("Recovery-v2 unexpectedly consumed Fresh C.")

    print("Fresh-C C0.1C-v2 recovery freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(
        "Recovery parent attempt: "
        f"{EXPECTED_V1_FAILED_ATTEMPT_ID}"
    )
    print("Recovery parent failed epoch preserved: True")
    print("Scientific/search semantics changed: False")
    print("Recovery live discovery ready: True")
    print("Recovery live discovery authorized: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
