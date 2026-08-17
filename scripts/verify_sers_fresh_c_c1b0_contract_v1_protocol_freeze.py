from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from dac_her.fresh_c_c1b0_contract_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    load_object,
    sha256_file,
    validate_protocol,
)
from scripts.freeze_sers_fresh_c_c1b0_contract_v1_protocol import CRITICAL_COMPONENTS


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    manifest = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")

    if manifest["protocol_id"] != p["protocol_id"]:
        raise ValueError("C1B.0 freeze protocol ID mismatch.")
    if manifest["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("C1B.0 freeze protocol SHA mismatch.")
    tmp = dict(manifest)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.0 freeze manifest SHA drifted.")

    source_commit = manifest["source_code_commit"]
    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C1B.0 critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"C1B.0 frozen component mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"C1B.0 current component drifted: {relative}")

    for field in (
        "fresh_c_scientific_text_semantic_read_performed",
        "c1b0_audit_authorized",
        "c1b1_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C1B.0 safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1B.0 protocol freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1B.0 protocol freeze used LLM.")
    if manifest.get("stop") is not True:
        raise ValueError("C1B.0 protocol freeze STOP drifted.")
    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1B.0 READY freeze ID mismatch.")

    print("Fresh-C C1B.0 input-contract protocol freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Fresh-C scientific text semantic read performed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("C1B.1 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
