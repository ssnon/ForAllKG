from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_live_discovery_recovery_v2_2 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V21_FREEZE_COMMIT,
    V21_FAILED_PATH,
    V21_STARTED_PATH,
    load_and_validate_protocol,
    validate_v21_failed_epoch,
)
from scripts.verify_sers_fresh_c_live_discovery_recovery_v2_2_protocol import verify

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_live_discovery_recovery_v2_2.py",
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_2_protocol.json",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_2_protocol.py",
    "scripts/freeze_sers_fresh_c_live_discovery_recovery_v2_2_protocol.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_2_protocol_freeze.py",
    "scripts/run_sers_fresh_c_live_discovery_recovery_v2_2.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_2_result.py",
    "tests/test_sers_fresh_c_live_discovery_recovery_v2_2.py",
    "dac_her/fresh_c_live_discovery_recovery_v2_1.py",
    "dac_her/fresh_c_live_discovery_recovery_v2.py",
    "dac_her/fresh_c_live_discovery.py",
    "dac_her/fresh_c_acquisition.py",
    "dac_her/fresh_c_activation.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _require_tracked_exact(root: Path, relative: Path, commit: str) -> str:
    current = root / relative
    if not current.exists():
        raise FileNotFoundError(current)
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(current) != expected:
        raise RuntimeError(f"Parent v2.1 artifact drifted: {relative}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Tracked worktree dirty; refuse v2.2 freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Index dirty; refuse v2.2 freeze.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol_freeze",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_V21_FREEZE_COMMIT, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    validate_v21_failed_epoch(root)
    p = verify(
        args.protocol if args.protocol.is_absolute() else root / args.protocol
    )

    source_commit = _git(root, "rev-parse", "HEAD")
    started_sha = _require_tracked_exact(root, V21_STARTED_PATH, source_commit)
    failed_sha = _require_tracked_exact(root, V21_FAILED_PATH, source_commit)

    hashes = {}
    for relative in CRITICAL_COMPONENTS:
        current = root / relative
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(current) != expected:
            raise RuntimeError(f"v2.2 critical component drifted: {relative}")
        hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-live-discovery-recovery-v2-2-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "parent_v21_attempt_id": p.parent_v21_attempt_id,
        "parent_v21_started_file_sha256": started_sha,
        "parent_v21_failed_file_sha256": failed_sha,
        "parent_v21_network_epoch_started": True,
        "parent_v21_failed_epoch_preserved": True,
        "parent_v21_success_artifacts_absent": True,
        "compatibility_change_only": True,
        "scientific_search_transport_semantics_changed": False,
        "critical_component_sha256": hashes,
        "recovery_live_discovery_ready": True,
        "recovery_live_discovery_authorized": False,
        "recovery_live_discovery_started": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_live_discovery_recovery_v2_2_freeze_v1:"
        + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = out / "freeze_manifest.json"
    ready = out / "FREEZE_READY.json"
    if manifest.exists() or ready.exists():
        raise FileExistsError("v2.2 freeze artifacts already exist.")
    _atomic(manifest, body)
    _atomic(
        ready,
        {
            "schema_version": "sers-fresh-c-live-discovery-recovery-v2-2-ready-v1",
            "freeze_id": body["freeze_id"],
            "manifest_sha256": body["manifest_sha256"],
            "parent_v21_network_epoch_started": True,
            "parent_v21_failed_epoch_preserved": True,
            "compatibility_change_only": True,
            "recovery_live_discovery_ready": True,
            "recovery_live_discovery_authorized": False,
            "fresh_reserve_c_consumed": False,
            "stop": True,
        },
    )

    print("Fresh-C C0.1C-v2.2 compatibility-repair freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Parent v2.1 attempt: {p.parent_v21_attempt_id}")
    print("Parent v2.1 network epoch started: True")
    print("Parent v2.1 failed epoch preserved: True")
    print("Parent v2.1 success artifacts absent: True")
    print("Compatibility change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Recovery live discovery ready: True")
    print("Recovery live discovery authorized: False")
    print("Network calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
