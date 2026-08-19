from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    V22_DIAGNOSTICS_PATH,
    V22_FAILED_PATH,
    V22_STARTED_PATH,
    load_and_validate_protocol,
    validate_v22_failed_epoch,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_live_discovery_recovery_v2_3_protocol import CRITICAL_COMPONENTS


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    v = json.loads(path.read_text())
    if not isinstance(v, dict):
        raise ValueError(f"Expected object: {path}")
    return v


def _payload_sha(payload, field):
    v = dict(payload)
    v.pop(field, None)
    return sha256_json(v)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    subprocess.run(
        [sys.executable, "-m",
         "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_2_protocol_freeze"],
        cwd=root, check=True,
    )
    validate_v22_failed_epoch(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    m = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    r = _read(root / DEFAULT_FREEZE_DIR / "FREEZE_READY.json")

    if m["protocol_id"] != p.protocol_id or m["protocol_sha256"] != p.protocol_sha256:
        raise ValueError("v2.3 freeze protocol mismatch.")
    if m["manifest_sha256"] != _payload_sha(m, "manifest_sha256"):
        raise ValueError("v2.3 freeze SHA drifted.")
    if m.get("parent_v22_http_429_confirmed") is not True:
        raise ValueError("v2.3 missing parent 429 confirmation.")
    if m.get("parent_v22_crossref_4_of_4_success") is not True:
        raise ValueError("v2.3 missing Crossref confirmation.")
    if m.get("semantic_scholar_api_key_required") is not True:
        raise ValueError("v2.3 API key requirement drifted.")
    if m.get("credential_value_persisted") is not False:
        raise ValueError("v2.3 persisted credential value.")
    if m.get("scientific_search_selection_semantics_changed") is not False:
        raise ValueError("v2.3 scientific/search semantics changed.")
    if m.get("transport_pacing_retry_changed") is not False:
        raise ValueError("v2.3 pacing/retry changed.")
    if m.get("recovery_live_discovery_authorized") is not False:
        raise ValueError("v2.3 unexpectedly authorized.")
    if m.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.3 consumed Fresh C.")
    if m.get("network_calls_during_freeze") != 0:
        raise ValueError("v2.3 freeze used network.")

    source_commit = m["source_code_commit"]
    for relative in (V22_STARTED_PATH, V22_FAILED_PATH, V22_DIAGNOSTICS_PATH):
        current = sha256_file(root / relative)
        if m["parent_v22_artifact_sha256"][str(relative)] != current:
            raise ValueError(f"Parent v2.2 artifact SHA drifted: {relative}")
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative.as_posix()}"], cwd=root
        )
        if hashlib.sha256(committed).hexdigest() != current:
            raise ValueError(f"Parent v2.2 artifact not frozen: {relative}")

    hashes = m["critical_component_sha256"]
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected or sha256_file(root / relative) != expected:
            raise ValueError(f"v2.3 component drifted: {relative}")

    if r["freeze_id"] != m["freeze_id"] or r["manifest_sha256"] != m["manifest_sha256"]:
        raise ValueError("v2.3 READY mismatch.")

    print("Fresh-C C0.1C-v2.3 authenticated recovery freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {m['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Parent v2.2 HTTP 429 confirmed: True")
    print("Semantic Scholar API key required: True")
    print("Authenticated transport is only material change: True")
    print("Scientific/search/selection semantics changed: False")
    print("Transport pacing/retry changed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
