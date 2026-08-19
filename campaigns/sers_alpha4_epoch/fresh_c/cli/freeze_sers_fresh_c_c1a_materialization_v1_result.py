from __future__ import annotations

import json
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
    corpus = load_json_object(run_dir / "materialized_corpus_manifest.json")
    marker = load_json_object(run_dir / "RESERVE_C_CONSUMPTION_STARTED.json")

    text_hashes = {
        row["canonical_id"]: row["materialized_text_sha256"]
        for row in corpus["records"]
    }
    source_pdf_hashes = {
        row["canonical_id"]: row["source_pdf_sha256"]
        for row in corpus["records"]
    }

    body = {
        "schema_version": "sers-fresh-c-c1a-materialization-result-freeze-v1",
        "source_run_id": run["run_id"],
        "source_run_sha256": run["run_sha256"],
        "materialized_corpus_sha256": run["materialized_corpus_sha256"],
        "consumption_marker_sha256": sha256_file(
            run_dir / "RESERVE_C_CONSUMPTION_STARTED.json"
        ),
        "materialized_pdf_count": 25,
        "source_pdf_sha256": source_pdf_hashes,
        "materialized_text_sha256": text_hashes,
        "fresh_reserve_c_consumed": True,
        "consumption_irreversible": True,
        "pdf_text_extraction_performed": True,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "external_literature_lookup_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "c1b_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1a_materialization_result_freeze_v1:" + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = root / DEFAULT_RESULT_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1A result freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "fresh_reserve_c_consumed": True,
        "consumption_irreversible": True,
        "scientific_adjudication_performed": False,
        "c1b_authorized": False,
        "stop": True,
    })

    print("Fresh-C C1A materialization result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source run ID: {body['source_run_id']}")
    print("Materialized sealed PDFs: 25/25")
    print("Fresh Reserve C consumed: True")
    print("Consumption irreversible: True")
    print("Scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("C1B authorized: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
