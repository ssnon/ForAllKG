from __future__ import annotations

import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file
from dac_her.fresh_c_c1a_materialization_v1 import validate_c01d_closed_frozen
from dac_her.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_RUN_DIR,
    load_json_object,
    validate_failed_c1a_state,
)
from scripts.verify_sers_fresh_c_c1a_r1_recovery_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_protocol_freeze()
    validate_failed_c1a_state(root)
    upstream = validate_c01d_closed_frozen(root)
    run_dir = root / DEFAULT_RUN_DIR

    if (run_dir / "C1A_R1_RECOVERY_FAILED.json").exists():
        raise RuntimeError("C1A-R1 failed epoch exists; success verification forbidden.")

    marker = load_json_object(run_dir / "C1A_R1_RECOVERY_STARTED.json")
    corpus = load_json_object(run_dir / "recovered_corpus_manifest.json")
    run = load_json_object(run_dir / "run_manifest.json")
    complete = load_json_object(run_dir / "C1A_R1_RECOVERY_COMPLETE.json")

    if marker.get("fresh_reserve_c_already_consumed") is not True:
        raise ValueError("C1A-R1 marker lost consumed state.")
    if marker.get("this_is_new_fresh_c_consumption") is not False:
        raise ValueError("C1A-R1 must not claim a new Fresh-C consumption.")

    if corpus.get("materialized_source_count") != 25:
        raise ValueError("C1A-R1 corpus count drifted.")
    records = corpus.get("records") or []
    if len(records) != 25:
        raise ValueError("C1A-R1 records length drifted.")

    upstream_map = {row["canonical_id"]: row for row in upstream["records"]}
    seen = set()
    repaired = 0
    for row in records:
        cid = row["canonical_id"]
        if cid in seen or cid not in upstream_map:
            raise ValueError("C1A-R1 identity set drifted.")
        seen.add(cid)
        if row["source_pdf_sha256"] != upstream_map[cid]["source_pdf_sha256"]:
            raise ValueError("C1A-R1 source PDF hash drifted.")
        if row.get("negative_absence_inference_allowed") is not False:
            raise ValueError("C1A-R1 negative absence policy drifted.")
        text_path = root / row["materialized_text_path"]
        pages_path = root / row["pages_manifest_path"]
        if sha256_file(text_path) != row["materialized_text_sha256"]:
            raise ValueError("C1A-R1 materialized text SHA drifted.")
        if sha256_file(pages_path) != row["pages_manifest_sha256"]:
            raise ValueError("C1A-R1 pages manifest SHA drifted.")
        if row["page_count"] <= 0 or row["document_nonwhitespace_count"] <= 0:
            raise ValueError("C1A-R1 empty materialization.")
        if row["materialization_mode"] == "STRUCTURALLY_REPAIRED_DERIVATIVE":
            repaired += 1
            derivative = row.get("repaired_derivative")
            if not isinstance(derivative, dict):
                raise ValueError("C1A-R1 repaired derivative provenance missing.")
            path = root / derivative["path"]
            if sha256_file(path) != derivative["derivative_sha256"]:
                raise ValueError("C1A-R1 repaired derivative SHA drifted.")
            if derivative.get("original_source_overwritten") is not False:
                raise ValueError("C1A-R1 original source overwrite detected.")
            if derivative.get("completeness_claim_allowed") is not False:
                raise ValueError("C1A-R1 repaired completeness policy drifted.")

    if seen != set(upstream_map):
        raise ValueError("C1A-R1 source identity set is not exact.")
    if repaired != corpus["structurally_repaired_derivative_count"]:
        raise ValueError("C1A-R1 repaired count drifted.")

    for field in (
        "fresh_reserve_c_already_consumed",
        "consumption_irreversible",
        "pdf_text_extraction_performed",
    ):
        if corpus.get(field) is not True:
            raise ValueError(f"C1A-R1 expected true field drifted: {field}")
    for field in (
        "this_is_new_fresh_c_consumption",
        "identity_replacement_performed",
        "redownload_performed",
        "prior_failed_outputs_reused",
        "scientific_reviewer_read_performed",
        "scientific_adjudication_performed",
        "hypothesis_state_mutation_performed",
        "positive_evidence_promotion_performed",
        "external_literature_lookup_performed",
        "automatic_c1b_transition_authorized",
    ):
        if corpus.get(field) is not False:
            raise ValueError(f"C1A-R1 expected false field drifted: {field}")
    if corpus.get("network_calls") != 0 or corpus.get("llm_calls") != 0:
        raise ValueError("C1A-R1 unexpectedly used network or LLM.")

    if complete["run_id"] != run["run_id"]:
        raise ValueError("C1A-R1 COMPLETE run ID mismatch.")
    if complete["run_sha256"] != run["run_sha256"]:
        raise ValueError("C1A-R1 COMPLETE run SHA mismatch.")

    print("Fresh-C C1A-R1 recovery result verifier")
    print(f"Run ID: {run['run_id']}")
    print(f"Run SHA256: {run['run_sha256']}")
    print(f"Recovered corpus SHA256: {run['recovered_corpus_sha256']}")
    print("Materialized same frozen sources: 25/25")
    print(f"Direct original materializations: {run['direct_original_count']}")
    print(f"Structurally repaired derivatives: {run['structurally_repaired_derivative_count']}")
    print("Fresh Reserve C already consumed: True")
    print("Identity replacement performed: False")
    print("Redownload performed: False")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("Negative absence inference from any single paper: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("Automatic C1B transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
