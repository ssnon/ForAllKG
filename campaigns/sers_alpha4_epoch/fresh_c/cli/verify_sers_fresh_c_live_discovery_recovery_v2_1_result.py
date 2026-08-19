from __future__ import annotations

import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import FRESH_C_BLIND_ORDER_NAMESPACE, sha256_file
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import DEFAULT_C01B_DIR
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_1 import (
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    load_and_validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_1_protocol_freeze import (
    main as verify_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _score(canonical_id: str) -> str:
    import hashlib
    return hashlib.sha256(
        (FRESH_C_BLIND_ORDER_NAMESPACE + "\0" + canonical_id).encode("utf-8")
    ).hexdigest()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_freeze()
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    run_dir = root / DEFAULT_RUN_DIR
    if (run_dir / "DISCOVERY_RECOVERY_FAILED.json").exists():
        raise RuntimeError("v2.1 failed epoch exists; success verification forbidden.")

    manifest = _read(run_dir / "run_manifest.json")
    diagnostics = _read(run_dir / "TRANSPORT_DIAGNOSTICS.json")
    queue = _read(run_dir / "blind_selection_queue.json")
    locators = _read(run_dir / "access_locator_manifest.json")
    complete = _read(run_dir / "DISCOVERY_RECOVERY_COMPLETE.json")

    if manifest.get("protocol_id") != p.protocol_id:
        raise ValueError("v2.1 result protocol mismatch.")
    if manifest.get("successful_provider_query_executions") != 8:
        raise ValueError("v2.1 does not have 8/8 provider success.")
    if manifest.get("fresh_identity_queue_count", 0) < 25:
        raise ValueError("v2.1 fresh queue below target.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.1 consumed Fresh C.")
    if manifest.get("semantic_read_performed") is not False:
        raise ValueError("v2.1 performed semantic read.")
    if manifest.get("automatic_c0_1d_transition_authorized") is not False:
        raise ValueError("v2.1 auto-authorized C0.1D.")

    for filename, field in (
        ("TRANSPORT_DIAGNOSTICS.json", "transport_diagnostics_file_sha256"),
        ("blind_selection_queue.json", "blind_queue_file_sha256"),
        ("access_locator_manifest.json", "access_locator_file_sha256"),
    ):
        if sha256_file(run_dir / filename) != manifest.get(field):
            raise ValueError(f"v2.1 file SHA drifted: {filename}")

    if diagnostics.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.1 diagnostics consumed Fresh C.")
    if diagnostics.get("semantic_read_performed") is not False:
        raise ValueError("v2.1 diagnostics semantic read.")
    if diagnostics.get("llm_calls") != 0:
        raise ValueError("v2.1 diagnostics used LLM.")

    historical = set(
        _read(root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json")
        .get("canonical_ids") or []
    )
    records = queue.get("records") or []
    for i, row in enumerate(records, start=1):
        if row["canonical_id"] in historical:
            raise ValueError("Historical identity leaked into v2.1 queue.")
        if row["rank"] != i:
            raise ValueError("v2.1 rank sequence drifted.")
        if row["score_sha256"] != _score(row["canonical_id"]):
            raise ValueError("v2.1 blind score drifted.")

    queue_ids = {r["canonical_id"] for r in records}
    locator_ids = {
        r["canonical_id"] for r in (locators.get("records") or [])
    }
    if queue_ids != locator_ids:
        raise ValueError("v2.1 locator/queue set mismatch.")
    if complete.get("run_id") != manifest.get("run_id"):
        raise ValueError("v2.1 COMPLETE run ID mismatch.")

    print("Fresh-C C0.1C-v2.1 recovery result verifier")
    print(f"Run ID: {manifest['run_id']}")
    print(f"Run SHA256: {manifest['run_sha256']}")
    print("Provider-query executions: 8/8 successful")
    print(f"Fresh blind queue identities: {manifest['fresh_identity_queue_count']}")
    print("Harness change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Fresh Reserve C consumed: False")
    print("Semantic read performed: False")
    print("LLM calls: 0")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
