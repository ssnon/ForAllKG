from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_live_discovery_recovery_v2_4 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    V22_DIAGNOSTICS_PATH,
    V22_FAILED_PATH,
    V22_STARTED_PATH,
    load_and_validate_protocol,
    validate_v22_failure,
    validate_v23_frozen_unexecuted,
)
from scripts.freeze_sers_fresh_c_live_discovery_recovery_v2_4_protocol import (
    CRITICAL_COMPONENTS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    v = json.loads(path.read_text(encoding="utf-8"))
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
         "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_3_protocol_freeze"],
        cwd=root, check=True,
    )
    validate_v22_failure(root)
    validate_v23_frozen_unexecuted(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    m = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    r = _read(root / DEFAULT_FREEZE_DIR / "FREEZE_READY.json")

    if m["protocol_id"] != p.protocol_id or m["protocol_sha256"] != p.protocol_sha256:
        raise ValueError("v2.4 freeze protocol mismatch.")
    if m["manifest_sha256"] != _payload_sha(m, "manifest_sha256"):
        raise ValueError("v2.4 freeze SHA drifted.")
    if m.get("v23_frozen_but_unexecuted") is not True:
        raise ValueError("v2.4 v2.3 lineage drifted.")
    if m.get("provider_universe_changed") is not True:
        raise ValueError("v2.4 provider change not recorded.")
    for field in (
        "frozen_queries_changed",
        "historical_ledger_changed",
        "target_count_changed",
        "blind_ordering_changed",
        "hypothesis_aware_selection_added",
        "scientific_selection_semantics_changed",
        "recovery_live_discovery_authorized",
        "recovery_live_discovery_started",
        "fresh_reserve_c_consumed",
        "semantic_read_performed",
        "automatic_c0_1d_transition_authorized",
    ):
        if m.get(field) is not False:
            raise ValueError(f"v2.4 safety field drifted: {field}")
    if m.get("openalex_api_key_required") is not True:
        raise ValueError("v2.4 OpenAlex key requirement drifted.")
    if m.get("credential_value_persisted") is not False:
        raise ValueError("v2.4 credential value persisted.")
    if m.get("network_calls_during_freeze") != 0:
        raise ValueError("v2.4 freeze used network.")

    source_commit = m["source_code_commit"]
    hashes = m["critical_component_sha256"]
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected or sha256_file(root / relative) != expected:
            raise ValueError(f"v2.4 component drifted: {relative}")

    if r["freeze_id"] != m["freeze_id"] or r["manifest_sha256"] != m["manifest_sha256"]:
        raise ValueError("v2.4 READY mismatch.")

    print("Fresh-C C0.1C-v2.4 OpenAlex+Crossref substitution freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {m['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("v2.3 frozen but unexecuted: True")
    print("Provider substitution: semantic_scholar -> openalex")
    print("Provider universe changed: True")
    print("Frozen queries changed: False")
    print("Blind ordering changed: False")
    print("Scientific selection semantics changed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
