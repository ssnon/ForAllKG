from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    sha256_file,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import DEFAULT_C01B_DIR
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_4 import (
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    load_and_validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_4_protocol_freeze import (
    main as verify_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    v = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v, dict):
        raise ValueError(f"Expected object: {path}")
    return v


def _score(canonical_id: str) -> str:
    return hashlib.sha256(
        (FRESH_C_BLIND_ORDER_NAMESPACE + "\0" + canonical_id).encode("utf-8")
    ).hexdigest()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_freeze()
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    run_dir = root / DEFAULT_RUN_DIR

    if (run_dir / "DISCOVERY_RECOVERY_FAILED.json").exists():
        raise RuntimeError("v2.4 failed epoch exists; success verification forbidden.")

    m = _read(run_dir / "run_manifest.json")
    d = _read(run_dir / "TRANSPORT_DIAGNOSTICS.json")
    q = _read(run_dir / "blind_selection_queue.json")
    l = _read(run_dir / "access_locator_manifest.json")
    c = _read(run_dir / "DISCOVERY_RECOVERY_COMPLETE.json")

    if m["protocol_id"] != p.protocol_id:
        raise ValueError("v2.4 result protocol mismatch.")
    if m.get("successful_provider_query_executions") != 8:
        raise ValueError("v2.4 does not have 8/8 provider success.")
    if m.get("fresh_identity_queue_count", 0) < 25:
        raise ValueError("v2.4 fresh queue below target.")
    if m.get("provider_universe_changed") is not True:
        raise ValueError("v2.4 provider change missing.")
    if m.get("scientific_selection_semantics_changed") is not False:
        raise ValueError("v2.4 scientific selection semantics changed.")
    if m.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.4 consumed Fresh C.")
    if m.get("semantic_read_performed") is not False:
        raise ValueError("v2.4 semantic read occurred.")

    for filename, field in (
        ("TRANSPORT_DIAGNOSTICS.json", "transport_diagnostics_file_sha256"),
        ("blind_selection_queue.json", "blind_queue_file_sha256"),
        ("access_locator_manifest.json", "access_locator_file_sha256"),
    ):
        if sha256_file(run_dir / filename) != m[field]:
            raise ValueError(f"v2.4 file SHA drifted: {filename}")

    executions = d.get("provider_executions") or []
    providers = [row.get("provider") for row in executions]
    if providers.count("openalex") != 4 or providers.count("crossref") != 4:
        raise ValueError("v2.4 diagnostics provider counts drifted.")
    if any(row.get("success") is not True for row in executions):
        raise ValueError("v2.4 diagnostics contain provider failure.")
    if d.get("credential_values_persisted") is not False:
        raise ValueError("v2.4 persisted credential values.")
    if d.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.4 diagnostics consumed Fresh C.")

    historical = set(
        _read(root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json")
        .get("canonical_ids") or []
    )
    records = q.get("records") or []
    for index, row in enumerate(records, start=1):
        cid = row["canonical_id"]
        if cid in historical:
            raise ValueError("Historical identity leaked into v2.4 queue.")
        if row["rank"] != index:
            raise ValueError("v2.4 blind rank drifted.")
        if row["score_sha256"] != _score(cid):
            raise ValueError("v2.4 blind score drifted.")

    queue_ids = {row["canonical_id"] for row in records}
    locator_ids = {
        row["canonical_id"] for row in (l.get("records") or [])
    }
    if queue_ids != locator_ids:
        raise ValueError("v2.4 queue/locator identity mismatch.")
    if c["run_id"] != m["run_id"] or c["run_sha256"] != m["run_sha256"]:
        raise ValueError("v2.4 COMPLETE mismatch.")

    print("Fresh-C C0.1C-v2.4 OpenAlex+Crossref result verifier")
    print(f"Run ID: {m['run_id']}")
    print(f"Run SHA256: {m['run_sha256']}")
    print("Provider-query executions: 8/8 successful")
    print(f"Fresh blind queue identities: {m['fresh_identity_queue_count']}")
    print("Provider substitution: semantic_scholar -> openalex")
    print("Provider universe changed: True")
    print("Scientific selection semantics changed: False")
    print("Fresh Reserve C consumed: False")
    print("Semantic read performed: False")
    print("LLM calls: 0")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
