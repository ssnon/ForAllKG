from __future__ import annotations

import argparse
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
from scripts.verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol import verify


CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_live_discovery_recovery_v2_1.py",
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_1_protocol.json",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol.py",
    "scripts/freeze_sers_fresh_c_live_discovery_recovery_v2_1_protocol.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol_freeze.py",
    "scripts/run_sers_fresh_c_live_discovery_recovery_v2_1.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_1_result.py",
    "tests/test_sers_fresh_c_live_discovery_recovery_v2_1.py",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Tracked worktree dirty; refuse v2.1 freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Index dirty; refuse v2.1 freeze.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_protocol_freeze",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_V2_FREEZE_COMMIT, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    assert_parent_v2_never_started_network(root)
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    p = verify(protocol_path)

    source_commit = _git(root, "rev-parse", "HEAD")
    hashes = {}
    for relative in CRITICAL_COMPONENTS:
        current = root / relative
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(current) != expected:
            raise RuntimeError(f"v2.1 critical component drifted: {relative}")
        hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-live-discovery-recovery-v2-1-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "parent_v2_freeze_commit": EXPECTED_V2_FREEZE_COMMIT,
        "parent_v2_network_epoch_started": False,
        "parent_v2_failure_kind": "pre_network_argparse_harness_mismatch",
        "harness_change_only": True,
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
        "sers_fresh_c_live_discovery_recovery_v2_1_freeze_v1:"
        + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = output / "freeze_manifest.json"
    ready = output / "FREEZE_READY.json"
    if manifest.exists() or ready.exists():
        raise FileExistsError("v2.1 freeze artifacts already exist.")
    _atomic(manifest, body)
    _atomic(
        ready,
        {
            "schema_version": "sers-fresh-c-live-discovery-recovery-v2-1-ready-v1",
            "freeze_id": body["freeze_id"],
            "manifest_sha256": body["manifest_sha256"],
            "parent_v2_network_epoch_started": False,
            "harness_change_only": True,
            "recovery_live_discovery_ready": True,
            "recovery_live_discovery_authorized": False,
            "fresh_reserve_c_consumed": False,
            "stop": True,
        },
    )

    print("Fresh-C C0.1C-v2.1 harness-repair freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Parent v2 network epoch started: False")
    print("Harness change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Recovery live discovery ready: True")
    print("Recovery live discovery authorized: False")
    print("Network calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
