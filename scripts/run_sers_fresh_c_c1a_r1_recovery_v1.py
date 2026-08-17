from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_c1a_materialization_v1 import validate_c01d_closed_frozen
from dac_her.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    extract_pdfminer_pages,
    load_and_validate_protocol,
    load_json_object,
    mutool_executable,
    mutool_fingerprint,
    render_page_bounded_text,
    repair_with_mutool,
    validate_failed_c1a_state,
    validate_pdfminer_version,
)
from scripts.verify_sers_fresh_c_c1a_r1_recovery_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-C C1A-R1 post-consumption structural recovery over the "
            "same frozen 25 source PDFs. This is not a new Fresh-C opening."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--confirm-post-consumption-recovery", action="store_true")
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _preflight(root: Path) -> dict[str, Any]:
    verify_protocol_freeze()
    validate_failed_c1a_state(root)
    upstream = validate_c01d_closed_frozen(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    pdfminer = validate_pdfminer_version()
    mutool = mutool_fingerprint()
    freeze = load_json_object(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    if mutool["sha256"] != freeze["mutool_sha256"]:
        raise RuntimeError("C1A-R1 mutool binary drifted since freeze.")
    if mutool["version_output"] != freeze["mutool_version_output"]:
        raise RuntimeError("C1A-R1 mutool version drifted since freeze.")

    run_dir = root / DEFAULT_RUN_DIR
    empty = not run_dir.exists() or not any(run_dir.iterdir())
    return {
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "source_identity_count": len(upstream["records"]),
        "all_source_pdf_hashes_current": True,
        "original_failed_c1a_consumed": True,
        "original_failed_c1a_rerun_allowed": False,
        "recovery_run_dir_empty": empty,
        "pdfminer_six_version": pdfminer,
        "mutool_sha256_current": True,
        "mutool_version_current": True,
        "identity_replacement_allowed": False,
        "redownload_allowed": False,
        "prior_failed_outputs_reused": False,
        "external_literature_lookup_allowed": False,
        "network_calls_during_preflight": 0,
        "llm_calls_during_preflight": 0,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "automatic_c1b_transition_authorized": False,
        "authorized": False,
        "stop": True,
    }


def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["recovery_run_dir_empty"]:
        raise RuntimeError("C1A-R1 recovery epoch already exists.")
    print("Fresh-C C1A-R1 guarded post-consumption recovery preflight")
    for key, value in state.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["recovery_run_dir_empty"]:
        raise RuntimeError("C1A-R1 recovery epoch already started; rerun forbidden.")

    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    upstream = validate_c01d_closed_frozen(root)
    freeze = load_json_object(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    mutool = mutool_executable()

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    marker_path = run_dir / "C1A_R1_RECOVERY_STARTED.json"
    failed_path = run_dir / "C1A_R1_RECOVERY_FAILED.json"
    output_root = run_dir / "materialized"
    output_root.mkdir(parents=True, exist_ok=False)

    _atomic(marker_path, {
        "schema_version": "sers-fresh-c-c1a-r1-recovery-started-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "protocol_freeze_id": freeze["freeze_id"],
        "source_identity_count": 25,
        "fresh_reserve_c_already_consumed": True,
        "consumption_irreversible": True,
        "this_is_new_fresh_c_consumption": False,
        "same_recovery_epoch_rerun_allowed": False,
        "identity_replacement_allowed": False,
        "redownload_allowed": False,
        "prior_failed_outputs_reused": False,
        "network_calls": 0,
        "llm_calls": 0,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "automatic_c1b_transition_authorized": False,
        "stop": True,
    })

    records = []
    try:
        for source in upstream["records"]:
            index = source["reserve_index"]
            source_path = root / source["source_path"]
            paper_dir = output_root / f"reserve_c_{index:03d}"
            paper_dir.mkdir(parents=True, exist_ok=False)

            mode = "DIRECT_ORIGINAL"
            repaired = None
            direct_error_type = None
            direct_error_summary = None

            try:
                pages = extract_pdfminer_pages(source_path)
                effective_path = source_path
            except Exception as exc:
                mode = "STRUCTURALLY_REPAIRED_DERIVATIVE"
                direct_error_type = type(exc).__name__
                direct_error_summary = str(exc)[:500]

                derivative = paper_dir / "repaired_derivative.pdf"
                repair_log = paper_dir / "mutool_clean.log"
                repaired = repair_with_mutool(
                    binary=mutool,
                    source=source_path,
                    derivative=derivative,
                    log_path=repair_log,
                )
                pages = extract_pdfminer_pages(derivative)
                effective_path = derivative

            text = render_page_bounded_text(pages)
            text_path = paper_dir / "full_text.txt"
            text_path.write_text(text, encoding="utf-8")

            page_rows = []
            for page_number, page in enumerate(pages, start=1):
                page_rows.append({
                    "page_number": page_number,
                    "text": page,
                    "text_sha256": sha256_json({"text": page}),
                    "character_count": len(page),
                    "nonwhitespace_count": len("".join(page.split())),
                })
            pages_manifest = {
                "schema_version": "sers-fresh-c-c1a-r1-pages-v1",
                "reserve_index": index,
                "canonical_id": source["canonical_id"],
                "source_pdf_sha256": source["source_pdf_sha256"],
                "materialization_mode": mode,
                "page_count": len(pages),
                "pages": page_rows,
                "fresh_reserve_c_already_consumed": True,
                "scientific_reviewer_read_performed": False,
                "scientific_adjudication_performed": False,
                "negative_absence_inference_allowed": False,
            }
            pages_path = paper_dir / "pages.json"
            _atomic(pages_path, pages_manifest)

            record = {
                "reserve_index": index,
                "canonical_id": source["canonical_id"],
                "source_path": source["source_path"],
                "source_pdf_sha256": source["source_pdf_sha256"],
                "materialization_mode": mode,
                "primary_extractor": p.primary_extractor,
                "direct_extraction_error_type": direct_error_type,
                "direct_extraction_error_summary": direct_error_summary,
                "effective_pdf_path": str(effective_path.relative_to(root)),
                "materialized_text_path": str(text_path.relative_to(root)),
                "materialized_text_sha256": sha256_file(text_path),
                "pages_manifest_path": str(pages_path.relative_to(root)),
                "pages_manifest_sha256": sha256_file(pages_path),
                "page_count": len(pages),
                "pages_with_nonwhitespace": sum(
                    row["nonwhitespace_count"] > 0 for row in page_rows
                ),
                "document_nonwhitespace_count": sum(
                    row["nonwhitespace_count"] for row in page_rows
                ),
                "positive_evidence_use_allowed_later": True,
                "negative_absence_inference_allowed": False,
                "completeness_claim_allowed": mode == "DIRECT_ORIGINAL",
                "network_calls": 0,
                "llm_calls": 0,
            }
            if repaired is not None:
                record["repaired_derivative"] = {
                    **repaired,
                    "path": str(effective_path.relative_to(root)),
                    "original_source_pdf_sha256": source["source_pdf_sha256"],
                    "original_source_overwritten": False,
                    "completeness_claim_allowed": False,
                }
            records.append(record)

        if len(records) != 25:
            raise RuntimeError("C1A-R1 did not materialize exactly 25 sources.")

        direct_count = sum(
            row["materialization_mode"] == "DIRECT_ORIGINAL"
            for row in records
        )
        repaired_count = sum(
            row["materialization_mode"] == "STRUCTURALLY_REPAIRED_DERIVATIVE"
            for row in records
        )

        corpus = {
            "schema_version": "sers-fresh-c-c1a-r1-recovered-corpus-v1",
            "protocol_id": p.protocol_id,
            "protocol_sha256": p.protocol_sha256,
            "protocol_freeze_id": freeze["freeze_id"],
            "source_identity_count": 25,
            "materialized_source_count": 25,
            "direct_original_count": direct_count,
            "structurally_repaired_derivative_count": repaired_count,
            "records": records,
            "fresh_reserve_c_already_consumed": True,
            "consumption_irreversible": True,
            "this_is_new_fresh_c_consumption": False,
            "identity_replacement_performed": False,
            "redownload_performed": False,
            "prior_failed_outputs_reused": False,
            "pdf_text_extraction_performed": True,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "hypothesis_state_mutation_performed": False,
            "positive_evidence_promotion_performed": False,
            "negative_absence_inference_from_any_single_paper_allowed": False,
            "external_literature_lookup_performed": False,
            "network_calls": 0,
            "llm_calls": 0,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        }
        corpus["corpus_sha256"] = _payload_sha(corpus, "corpus_sha256")
        corpus_path = run_dir / "recovered_corpus_manifest.json"
        _atomic(corpus_path, corpus)

        run = {
            "schema_version": "sers-fresh-c-c1a-r1-recovery-run-v1",
            "protocol_id": p.protocol_id,
            "protocol_sha256": p.protocol_sha256,
            "protocol_freeze_id": freeze["freeze_id"],
            "recovery_started_marker_sha256": sha256_file(marker_path),
            "recovered_corpus_path": str(corpus_path.relative_to(root)),
            "recovered_corpus_file_sha256": sha256_file(corpus_path),
            "recovered_corpus_sha256": corpus["corpus_sha256"],
            "materialized_source_count": 25,
            "direct_original_count": direct_count,
            "structurally_repaired_derivative_count": repaired_count,
            "fresh_reserve_c_already_consumed": True,
            "consumption_irreversible": True,
            "this_is_new_fresh_c_consumption": False,
            "identity_replacement_performed": False,
            "redownload_performed": False,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "external_literature_lookup_performed": False,
            "network_calls": 0,
            "llm_calls": 0,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        }
        ident = sha256_json(run)
        run["run_id"] = "sers_fresh_c_c1a_r1_recovery_run_v1:" + ident[:20]
        run["run_sha256"] = _payload_sha(run, "run_sha256")
        run_path = run_dir / "run_manifest.json"
        _atomic(run_path, run)

        _atomic(run_dir / "C1A_R1_RECOVERY_COMPLETE.json", {
            "schema_version": "sers-fresh-c-c1a-r1-recovery-complete-v1",
            "run_id": run["run_id"],
            "run_sha256": run["run_sha256"],
            "recovered_corpus_sha256": corpus["corpus_sha256"],
            "materialized_source_count": 25,
            "direct_original_count": direct_count,
            "structurally_repaired_derivative_count": repaired_count,
            "fresh_reserve_c_already_consumed": True,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        })

        print("Fresh-C C1A-R1 recovery complete")
        print(f"Run ID: {run['run_id']}")
        print(f"Run SHA256: {run['run_sha256']}")
        print(f"Recovered corpus SHA256: {corpus['corpus_sha256']}")
        print("Materialized same frozen sources: 25/25")
        print(f"Direct original materializations: {direct_count}")
        print(f"Structurally repaired derivatives: {repaired_count}")
        print("Fresh Reserve C already consumed: True")
        print("This is new Fresh-C consumption: False")
        print("Identity replacement performed: False")
        print("Redownload performed: False")
        print("Scientific reviewer read performed: False")
        print("Scientific adjudication performed: False")
        print("Network calls: 0")
        print("LLM calls: 0")
        print("Automatic C1B transition authorized: False")
        print("STOP: True")
        return 0

    except Exception as exc:
        _atomic(failed_path, {
            "schema_version": "sers-fresh-c-c1a-r1-recovery-failed-v1",
            "exception_type": type(exc).__name__,
            "exception_summary": str(exc)[:500],
            "fresh_reserve_c_already_consumed": True,
            "consumption_irreversible": True,
            "same_recovery_epoch_rerun_allowed": False,
            "identity_replacement_allowed": False,
            "redownload_allowed": False,
            "completed_source_count_before_failure": len(records),
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "network_calls": 0,
            "llm_calls": 0,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        })
        raise


def main() -> int:
    args = parse_args()
    if args.preflight:
        return preflight()
    if args.confirm_post_consumption_recovery:
        return execute()
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
