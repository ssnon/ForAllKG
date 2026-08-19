from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V22_FREEZE_COMMIT,
    V22_DIAGNOSTICS_PATH,
    V22_FAILED_PATH,
    V22_STARTED_PATH,
    load_and_validate_protocol,
    validate_v22_failed_epoch,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery_recovery_v2_3.py",
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_3_protocol.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_3_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_live_discovery_recovery_v2_3_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_3_protocol_freeze.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_live_discovery_recovery_v2_3.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_3_result.py",
    "tests/test_sers_fresh_c_live_discovery_recovery_v2_3.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery_recovery_v2_2.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery_recovery_v2.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_acquisition.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_activation.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _payload_sha(payload, field):
    v = dict(payload)
    v.pop(field, None)
    return sha256_json(v)


def _tracked_sha(root: Path, path: Path, commit: str) -> str:
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{path.as_posix()}"], cwd=root
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(root / path) != expected:
        raise RuntimeError(f"Parent artifact drifted: {path}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse v2.3 freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse v2.3 freeze.")

    subprocess.run(
        [sys.executable, "-m",
         "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_2_protocol_freeze"],
        cwd=root, check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_V22_FREEZE_COMMIT, "HEAD"],
        cwd=root, check=True, stdout=subprocess.DEVNULL,
    )
    validate_v22_failed_epoch(root)
    p = load_and_validate_protocol(
        args.protocol if args.protocol.is_absolute() else root / args.protocol
    )

    source_commit = _git(root, "rev-parse", "HEAD")
    parent_hashes = {
        str(V22_STARTED_PATH): _tracked_sha(root, V22_STARTED_PATH, source_commit),
        str(V22_FAILED_PATH): _tracked_sha(root, V22_FAILED_PATH, source_commit),
        str(V22_DIAGNOSTICS_PATH): _tracked_sha(root, V22_DIAGNOSTICS_PATH, source_commit),
    }
    hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"v2.3 component drifted: {relative}")
        hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-authenticated-recovery-v2-3-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "parent_v22_attempt_id": p.parent_v22_attempt_id,
        "parent_v22_artifact_sha256": parent_hashes,
        "parent_v22_http_429_confirmed": True,
        "parent_v22_crossref_4_of_4_success": True,
        "semantic_scholar_api_key_required": True,
        "credential_value_persisted": False,
        "authenticated_transport_is_only_material_change": True,
        "scientific_search_selection_semantics_changed": False,
        "transport_pacing_retry_changed": False,
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
        "sers_fresh_c_authenticated_recovery_v2_3_freeze_v1:" + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if output.exists():
        raise FileExistsError("v2.3 freeze dir already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "semantic_scholar_api_key_required": True,
        "recovery_live_discovery_ready": True,
        "recovery_live_discovery_authorized": False,
        "fresh_reserve_c_consumed": False,
        "stop": True,
    })

    print("Fresh-C C0.1C-v2.3 authenticated recovery freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Parent v2.2 HTTP 429 confirmed: True")
    print("Parent v2.2 Crossref 4/4 successful: True")
    print("Semantic Scholar API key required: True")
    print("Authenticated transport is only material change: True")
    print("Scientific/search/selection semantics changed: False")
    print("Transport pacing/retry changed: False")
    print("Network calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
