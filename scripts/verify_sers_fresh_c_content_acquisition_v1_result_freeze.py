from __future__ import annotations

import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_content_acquisition_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    load_json_object,
)
from scripts.verify_sers_fresh_c_content_acquisition_v1_result import (
    main as verify_result,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    manifest = load_json_object(
        root / DEFAULT_RESULT_FREEZE_DIR / "freeze_manifest.json"
    )
    ready = load_json_object(
        root / DEFAULT_RESULT_FREEZE_DIR / "FREEZE_READY.json"
    )
    selected = load_json_object(
        root / DEFAULT_RUN_DIR / "selected_reserve_c.json"
    )

    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C0.1D result freeze SHA drifted.")
    if manifest.get("selected_verified_pdf_count") != 25:
        raise ValueError("C0.1D result freeze count drifted.")
    if manifest.get("reserve_c_identity_selection_finalized") is not True:
        raise ValueError("C0.1D result freeze selection not finalized.")
    if manifest.get("reserve_c_content_sealed") is not True:
        raise ValueError("C0.1D result freeze content not sealed.")
    if manifest.get("semantic_read_performed") is not False:
        raise ValueError("C0.1D result freeze semantic read drifted.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("C0.1D result freeze consumed Fresh C.")
    if manifest.get("c1_authorized") is not False:
        raise ValueError("C0.1D result freeze unexpectedly authorized C1.")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C0.1D result freeze used network.")

    expected = {
        row["canonical_id"]: row["artifact_sha256"]
        for row in selected["records"]
    }
    if manifest.get("selected_pdf_sha256") != expected:
        raise ValueError("C0.1D result freeze PDF hash map drifted.")
    for row in selected["records"]:
        path = root / row["local_path"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("C0.1D sealed PDF SHA drifted after freeze.")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C0.1D result READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C0.1D result READY SHA mismatch.")

    print("Fresh-C C0.1D result freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source run ID: {manifest['source_run_id']}")
    print("Selected verified OA PDFs: 25")
    print(f"Content seal SHA256: {manifest['content_seal_sha256']}")
    print("Reserve-C identity selection finalized: True")
    print("Reserve-C content sealed: True")
    print("Semantic read performed: False")
    print("Fresh Reserve C consumed: False")
    print("C1 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
