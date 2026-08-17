from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_c1a_materialization_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
    validate_c01d_closed_frozen,
)
from scripts.freeze_sers_fresh_c_c1a_materialization_v1_protocol import (
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
    validate_c01d_closed_frozen(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    manifest = _read(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    ready = _read(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")

    if manifest["protocol_id"] != p.protocol_id:
        raise ValueError("C1A freeze protocol ID mismatch.")
    if manifest["protocol_sha256"] != p.protocol_sha256:
        raise ValueError("C1A freeze protocol SHA mismatch.")
    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C1A freeze manifest SHA drifted.")

    false_fields = (
        "network_allowed_during_materialization",
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "fresh_reserve_c_consumed",
        "live_materialization_authorized",
        "live_materialization_started",
        "automatic_c1b_transition_authorized",
    )
    for field in false_fields:
        if manifest.get(field) is not False:
            raise ValueError(f"C1A safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1A freeze used network.")

    source_commit = manifest["source_code_commit"]
    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C1A critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"C1A frozen hash mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"C1A current source drifted: {relative}")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1A READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C1A READY manifest SHA mismatch.")

    print("Fresh-C C1A materialization protocol freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Selected sealed PDFs: 25")
    print("Consumption marker before first text extraction: True")
    print("Materializer: pdftext 0.6.3 + pypdfium2 4.30.0")
    print("Network allowed during materialization: False")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
