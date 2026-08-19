from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    load_json_object,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1a_materialization_v1_result import (
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
        root / DEFAULT_RUN_DIR / "materialized_corpus_manifest.json"
    )

    if manifest["manifest_sha256"] != _payload_sha(
        manifest, "manifest_sha256"
    ):
        raise ValueError("C1A result freeze SHA drifted.")
    if manifest.get("materialized_pdf_count") != 25:
        raise ValueError("C1A result freeze count drifted.")
    for field in (
        "fresh_reserve_c_consumed",
        "consumption_irreversible",
        "pdf_text_extraction_performed",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"C1A result freeze true field drifted: {field}")
    for field in (
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "external_literature_lookup_performed",
        "c1b_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"C1A result freeze false field drifted: {field}")
    if manifest.get("network_calls_during_freeze") != 0:
        raise ValueError("C1A result freeze used network.")
    if manifest.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1A result freeze used LLM.")

    expected_text = {
        row["canonical_id"]: row["materialized_text_sha256"]
        for row in corpus["records"]
    }
    if manifest.get("materialized_text_sha256") != expected_text:
        raise ValueError("C1A result freeze text hash map drifted.")
    for row in corpus["records"]:
        text_path = root / row["materialized_text_path"]
        if sha256_file(text_path) != row["materialized_text_sha256"]:
            raise ValueError("C1A frozen materialized text SHA drifted.")

    if ready["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("C1A result READY freeze ID mismatch.")
    if ready["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("C1A result READY SHA mismatch.")

    print("Fresh-C C1A materialization result freeze verifier")
    print(f"Freeze ID: {manifest['freeze_id']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Source run ID: {manifest['source_run_id']}")
    print("Materialized sealed PDFs: 25/25")
    print("Fresh Reserve C consumed: True")
    print("Consumption irreversible: True")
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
