from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_C01B_DIR,
    DEFAULT_DISCOVERY_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_C01B_FREEZE_COMMIT,
    EXPECTED_C01B_LOCK_ID,
    EXPECTED_C01B_LOCK_SHA256,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    EXPECTED_HISTORICAL_LEDGER_ID,
    EXPECTED_HISTORICAL_LEDGER_SHA256,
    load_and_validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_protocol_v1 import verify


CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery.py",
    "dac_her/sers_fresh_c_live_discovery_protocol_v1.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_live_discovery_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_protocol_freeze_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_live_discovery_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_result_v1.py",
    "tests/test_sers_fresh_c_live_discovery_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_acquisition.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_activation.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_activation_readiness_v1.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Fresh-C C0.1C live-discovery execution components. "
            "This performs zero network calls and does not authorize "
            "live discovery."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DISCOVERY_FREEZE_DIR
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    args = parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Tracked worktree is dirty; refuse C0.1C freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Index is dirty; refuse C0.1C freeze.")

    subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_activation_readiness_v1"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_C01B_FREEZE_COMMIT, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)

    c01b_lock_path = root / DEFAULT_C01B_DIR / "activation_readiness_lock.json"
    c01b_lock = json.loads(c01b_lock_path.read_text(encoding="utf-8"))
    if c01b_lock.get("lock_id") != EXPECTED_C01B_LOCK_ID:
        raise ValueError("C0.1B lock ID drifted.")
    if c01b_lock.get("lock_sha256") != EXPECTED_C01B_LOCK_SHA256:
        raise ValueError("C0.1B lock SHA drifted.")
    if c01b_lock.get("historical_canonical_identity_count") != EXPECTED_HISTORICAL_IDENTITY_COUNT:
        raise ValueError("C0.1B historical identity count drifted.")

    ledger_path = root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("ledger_id") != EXPECTED_HISTORICAL_LEDGER_ID:
        raise ValueError("Historical ledger ID drifted.")
    if ledger.get("ledger_sha256") != EXPECTED_HISTORICAL_LEDGER_SHA256:
        raise ValueError("Historical ledger semantic SHA drifted.")
    if len(ledger.get("canonical_ids") or []) != EXPECTED_HISTORICAL_IDENTITY_COUNT:
        raise ValueError("Historical ledger count drifted.")

    source_commit = _git(root, "rev-parse", "HEAD")
    component_sha256: dict[str, str] = {}
    for relative in CRITICAL_COMPONENTS:
        current = root / relative
        if not current.exists():
            raise FileNotFoundError(current)
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"], cwd=root
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"C0.1C critical component not tracked in HEAD: {relative}"
            ) from exc
        committed_sha = hashlib.sha256(committed).hexdigest()
        current_sha = sha256_file(current)
        if committed_sha != current_sha:
            raise RuntimeError(f"C0.1C critical component differs from HEAD: {relative}")
        component_sha256[relative] = current_sha

    body: dict[str, Any] = {
        "schema_version": "sers-fresh-c-live-discovery-protocol-freeze-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "source_code_commit": source_commit,
        "c0_1b_lock_id": EXPECTED_C01B_LOCK_ID,
        "c0_1b_lock_sha256": EXPECTED_C01B_LOCK_SHA256,
        "c0_1b_freeze_commit": EXPECTED_C01B_FREEZE_COMMIT,
        "historical_ledger_id": EXPECTED_HISTORICAL_LEDGER_ID,
        "historical_ledger_sha256": EXPECTED_HISTORICAL_LEDGER_SHA256,
        "historical_identity_count": EXPECTED_HISTORICAL_IDENTITY_COUNT,
        "critical_component_sha256": component_sha256,
        "live_discovery_ready": True,
        "live_discovery_authorized": False,
        "live_discovery_started": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }
    identity_sha = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_live_discovery_protocol_freeze_v1:" + identity_sha[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    manifest_path = output / "freeze_manifest.json"
    ready_path = output / "FREEZE_READY.json"
    if manifest_path.exists() or ready_path.exists():
        raise FileExistsError("C0.1C freeze artifacts already exist; refuse overwrite.")
    _atomic_json(manifest_path, body)
    _atomic_json(
        ready_path,
        {
            "schema_version": "sers-fresh-c-live-discovery-ready-v1",
            "freeze_id": body["freeze_id"],
            "manifest_sha256": body["manifest_sha256"],
            "live_discovery_ready": True,
            "live_discovery_authorized": False,
            "live_discovery_started": False,
            "fresh_reserve_c_consumed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        },
    )

    print("Fresh-C C0.1C live-discovery protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Live discovery ready: True")
    print("Live discovery authorized: False")
    print("Live discovery started: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
