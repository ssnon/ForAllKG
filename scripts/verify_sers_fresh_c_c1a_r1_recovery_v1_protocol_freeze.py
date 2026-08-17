from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
    mutool_fingerprint,
    validate_failed_c1a_state,
    validate_pdfminer_version,
)
from scripts.freeze_sers_fresh_c_c1a_r1_recovery_v1_protocol import (
    CRITICAL_COMPONENTS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object: {path}")
    return raw


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    validate_failed_c1a_state(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    pdfminer = validate_pdfminer_version()
    mutool = mutool_fingerprint()
    manifest = _read(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    ready = _read(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")

    if manifest["protocol_id"] != p.protocol_id:
        raise ValueError("C1A-R1 freeze protocol ID mismatch.")
    if manifest["protocol_sha256"] != p.protocol_sha256:
        raise ValueError("C1A-R1 freeze protocol SHA mismatch.")
    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C1A-R1 freeze manifest SHA drifted.")
    if manifest["pdfminer_six_version"] != pdfminer:
        raise ValueError("C1A-R1 pdfminer version drifted.")
    if manifest["mutool_sha256"] != mutool["sha256"]:
        raise ValueError("C1A-R1 mutool binary SHA drifted.")
    if manifest["mutool_version_output"] != mutool["version_output"]:
        raise ValueError("C1A-R1 mutool version drifted.")

    if manifest.get("fresh_reserve_c_already_consumed") is not True:
        raise ValueError("C1A-R1 must remain post-consumption.")
    for field in (
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "live_recovery_authorized",
        "live_recovery_started",
        "automatic_c1b_transition_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C1A-R1 safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1A-R1 freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1A-R1 freeze used LLM.")

    source_commit = manifest["source_code_commit"]
    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C1A-R1 critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"C1A-R1 frozen hash mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"C1A-R1 current source drifted: {relative}")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1A-R1 READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C1A-R1 READY manifest SHA mismatch.")

    print("Fresh-C C1A-R1 recovery protocol freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Source identities: exact same frozen 25")
    print("Fresh Reserve C already consumed: True")
    print(f"pdfminer.six: {pdfminer}")
    print("mutool binary SHA/version: CURRENT")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
