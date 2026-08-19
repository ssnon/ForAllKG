from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    DEFAULT_RUN_DIR,
    load_json_object,
    validate_c01d_closed_frozen,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1a_materialization_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_protocol_freeze()
    upstream = validate_c01d_closed_frozen(root)
    run_dir = root / DEFAULT_RUN_DIR

    if (run_dir / "C1A_MATERIALIZATION_FAILED.json").exists():
        raise RuntimeError("C1A failed epoch exists; success verification forbidden.")

    marker = load_json_object(run_dir / "RESERVE_C_CONSUMPTION_STARTED.json")
    corpus = load_json_object(run_dir / "materialized_corpus_manifest.json")
    run = load_json_object(run_dir / "run_manifest.json")
    complete = load_json_object(run_dir / "C1A_MATERIALIZATION_COMPLETE.json")

    if marker.get("fresh_reserve_c_consumed") is not True:
        raise ValueError("C1A consumption marker does not mark consumed.")
    if marker.get("consumption_irreversible") is not True:
        raise ValueError("C1A consumption marker is not irreversible.")
    if corpus.get("materialized_pdf_count") != 25:
        raise ValueError("C1A corpus count drifted.")
    if run.get("materialized_pdf_count") != 25:
        raise ValueError("C1A run count drifted.")

    upstream_map = {
        row["canonical_id"]: row for row in upstream["records"]
    }
    records = corpus.get("records") or []
    if len(records) != 25:
        raise ValueError("C1A materialized records length drifted.")
    seen = set()
    for row in records:
        cid = row["canonical_id"]
        if cid in seen or cid not in upstream_map:
            raise ValueError("C1A identity set drifted.")
        seen.add(cid)
        if row["source_pdf_sha256"] != upstream_map[cid]["source_pdf_sha256"]:
            raise ValueError("C1A source PDF hash drifted.")
        text_path = root / row["materialized_text_path"]
        pages_path = root / row["pages_manifest_path"]
        if sha256_file(text_path) != row["materialized_text_sha256"]:
            raise ValueError("C1A text SHA drifted.")
        if sha256_file(pages_path) != row["pages_manifest_sha256"]:
            raise ValueError("C1A pages manifest SHA drifted.")
        if row["character_count"] <= 0:
            raise ValueError("C1A empty materialized text.")

    true_fields = (
        "fresh_reserve_c_consumed",
        "consumption_irreversible",
        "pdf_text_extraction_performed",
    )
    for field in true_fields:
        if run.get(field) is not True:
            raise ValueError(f"C1A expected true field drifted: {field}")

    false_fields = (
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "hypothesis_state_mutation_performed",
        "external_literature_lookup_performed",
        "automatic_c1b_transition_authorized",
    )
    for field in false_fields:
        if run.get(field) is not False:
            raise ValueError(f"C1A expected false field drifted: {field}")
    if run.get("network_calls") != 0 or run.get("llm_calls") != 0:
        raise ValueError("C1A unexpectedly used network or LLM.")

    if complete["run_id"] != run["run_id"]:
        raise ValueError("C1A COMPLETE run ID mismatch.")
    if complete["run_sha256"] != run["run_sha256"]:
        raise ValueError("C1A COMPLETE run SHA mismatch.")

    print("Fresh-C C1A local materialization result verifier")
    print(f"Run ID: {run['run_id']}")
    print(f"Run SHA256: {run['run_sha256']}")
    print(f"Materialized corpus SHA256: {run['materialized_corpus_sha256']}")
    print("Materialized sealed PDFs: 25/25")
    print("Fresh Reserve C consumed: True")
    print("Consumption irreversible: True")
    print("PDF text extraction performed: True")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("External literature lookup performed: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("Automatic C1B transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
