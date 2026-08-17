from __future__ import annotations

import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_json
from dac_her.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    load_json_object,
)
from scripts.verify_sers_fresh_c_c1a_r1_recovery_v1_result import (
    main as verify_result,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    run_dir = root / DEFAULT_RUN_DIR
    run = load_json_object(run_dir / "run_manifest.json")
    corpus = load_json_object(run_dir / "recovered_corpus_manifest.json")

    source_pdf_hashes = {
        row["canonical_id"]: row["source_pdf_sha256"]
        for row in corpus["records"]
    }
    text_hashes = {
        row["canonical_id"]: row["materialized_text_sha256"]
        for row in corpus["records"]
    }
    repair_hashes = {
        row["canonical_id"]: row["repaired_derivative"]["derivative_sha256"]
        for row in corpus["records"]
        if row["materialization_mode"] == "STRUCTURALLY_REPAIRED_DERIVATIVE"
    }

    body = {
        "schema_version": "sers-fresh-c-c1a-r1-recovery-result-freeze-v1",
        "source_run_id": run["run_id"],
        "source_run_sha256": run["run_sha256"],
        "recovered_corpus_sha256": run["recovered_corpus_sha256"],
        "source_identity_count": 25,
        "source_pdf_sha256": source_pdf_hashes,
        "materialized_text_sha256": text_hashes,
        "repaired_derivative_sha256": repair_hashes,
        "direct_original_count": run["direct_original_count"],
        "structurally_repaired_derivative_count": run[
            "structurally_repaired_derivative_count"
        ],
        "fresh_reserve_c_already_consumed": True,
        "consumption_irreversible": True,
        "identity_replacement_performed": False,
        "redownload_performed": False,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "negative_absence_inference_from_any_single_paper_allowed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "c1b_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1a_r1_recovery_result_freeze_v1:" + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = root / DEFAULT_RESULT_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1A-R1 result freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "fresh_reserve_c_already_consumed": True,
        "scientific_adjudication_performed": False,
        "c1b_authorized": False,
        "stop": True,
    })

    print("Fresh-C C1A-R1 recovery result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source run ID: {body['source_run_id']}")
    print("Materialized same frozen sources: 25/25")
    print(f"Direct original materializations: {body['direct_original_count']}")
    print(
        "Structurally repaired derivatives: "
        f"{body['structurally_repaired_derivative_count']}"
    )
    print("Fresh Reserve C already consumed: True")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("C1B authorized: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
