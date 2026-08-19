from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_content_acquisition_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
    validate_upstream_v24,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_content_acquisition_v1_protocol import (
    CRITICAL_COMPONENTS,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    validate_upstream_v24(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    manifest = _read(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    ready = _read(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json"
    )

    if manifest["protocol_id"] != p.protocol_id:
        raise ValueError("C0.1D freeze protocol ID mismatch.")
    if manifest["protocol_sha256"] != p.protocol_sha256:
        raise ValueError("C0.1D freeze protocol SHA mismatch.")
    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C0.1D freeze manifest SHA drifted.")
    for field in (
        "live_acquisition_authorized",
        "live_acquisition_started",
        "scientific_metadata_inspection_allowed",
        "pdf_text_extraction_allowed",
        "paywall_bypass_allowed",
        "fresh_reserve_c_consumed",
        "automatic_c1_transition_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C0.1D safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C0.1D freeze used network.")

    source_commit = manifest["source_code_commit"]
    hashes = manifest["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C0.1D critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != expected:
            raise ValueError(f"C0.1D frozen hash mismatch: {relative}")
        if sha256_file(root / relative) != expected:
            raise ValueError(f"C0.1D current source drifted: {relative}")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C0.1D READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C0.1D READY SHA mismatch.")

    print("Fresh-C C0.1D blind OA acquisition protocol freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Blind queue count: {manifest['blind_queue_count']}")
    print(f"Target verified PDFs: {manifest['target_verified_pdf_count']}")
    print("Scientific metadata inspection allowed: False")
    print("PDF text extraction allowed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
