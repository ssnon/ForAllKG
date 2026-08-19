from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    load_json_object,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1a_r1_recovery_v1_result import (
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
    corpus = load_json_object(
        root / DEFAULT_RUN_DIR / "recovered_corpus_manifest.json"
    )

    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C1A-R1 result freeze SHA drifted.")
    if manifest.get("source_identity_count") != 25:
        raise ValueError("C1A-R1 result freeze count drifted.")
    if manifest.get("fresh_reserve_c_already_consumed") is not True:
        raise ValueError("C1A-R1 result freeze lost consumed state.")
    for field in (
        "identity_replacement_performed",
        "redownload_performed",
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "negative_absence_inference_from_any_single_paper_allowed",
        "c1b_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C1A-R1 result freeze safety field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1A-R1 result freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1A-R1 result freeze used LLM.")

    expected_text = {
        row["canonical_id"]: row["materialized_text_sha256"]
        for row in corpus["records"]
    }
    if manifest.get("materialized_text_sha256") != expected_text:
        raise ValueError("C1A-R1 result freeze text hash map drifted.")
    for row in corpus["records"]:
        text_path = root / row["materialized_text_path"]
        if sha256_file(text_path) != row["materialized_text_sha256"]:
            raise ValueError("C1A-R1 frozen text SHA drifted.")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1A-R1 result READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C1A-R1 result READY SHA mismatch.")

    print("Fresh-C C1A-R1 recovery result freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source run ID: {manifest['source_run_id']}")
    print("Materialized same frozen sources: 25/25")
    print(f"Direct original materializations: {manifest['direct_original_count']}")
    print(
        "Structurally repaired derivatives: "
        f"{manifest['structurally_repaired_derivative_count']}"
    )
    print("Fresh Reserve C already consumed: True")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("C1B authorized: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
