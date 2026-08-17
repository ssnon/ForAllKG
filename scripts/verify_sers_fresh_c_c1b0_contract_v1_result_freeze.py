from __future__ import annotations

import subprocess
from pathlib import Path

from dac_her.fresh_c_c1b0_contract_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    canonical_json_sha256,
    load_object,
)
from scripts.verify_sers_fresh_c_c1b0_contract_v1_result import main as verify_result


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    manifest = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(manifest)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.0 result freeze SHA drifted.")
    if manifest.get("source_identity_count") != 25:
        raise ValueError("C1B.0 result freeze corpus count drifted.")
    for field in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "c1b1_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C1B.0 result freeze safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1B.0 result freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1B.0 result freeze used LLM.")
    if manifest.get("stop") is not True:
        raise ValueError("C1B.0 result freeze STOP drifted.")
    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1B.0 result READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C1B.0 result READY SHA mismatch.")
    print("Fresh-C C1B.0 input-contract result freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Contract ID: {manifest['contract_id']}")
    print("Exact materialized corpus: 25/25")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("C1B.1 authorized: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
