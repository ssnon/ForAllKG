from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_materialization_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    load_and_validate_protocol,
    load_json_object,
    materialize_pdf_pages,
    render_page_bounded_text,
    validate_c01d_closed_frozen,
    validate_package_versions,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1a_materialization_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Guarded Fresh-C C1A irreversible local PDF text materialization. "
            "Writing the consumption marker irrevocably consumes Fresh C."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument(
        "--confirm-irrevocable-reserve-c-consumption",
        action="store_true",
    )
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
    upstream = validate_c01d_closed_frozen(root)
    versions = validate_package_versions()
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    freeze = load_json_object(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    run_dir = root / DEFAULT_RUN_DIR
    empty = not run_dir.exists() or not any(run_dir.iterdir())

    return {
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "c01d_result_freeze_id": p.c01d_result_freeze_id,
        "c01d_content_seal_sha256": p.c01d_content_seal_sha256,
        "selected_pdf_count": len(upstream["records"]),
        "pdftext_version": versions["pdftext"],
        "pypdfium2_version": versions["pypdfium2"],
        "all_source_pdf_hashes_current": True,
        "all_source_pdf_magic_current": True,
        "run_dir_empty": empty,
        "network_allowed_during_materialization": False,
        "external_literature_lookup_allowed": False,
        "llm_calls_during_preflight": 0,
        "fresh_reserve_c_consumed": False,
        "consumption_marker_written": False,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "authorized": False,
        "stop": True,
    }


def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("C1A run directory not empty; same epoch forbidden.")
    print("Fresh-C C1A guarded irreversible-consumption preflight")
    for key, value in state.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("C1A epoch already started; rerun forbidden.")

    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    upstream = validate_c01d_closed_frozen(root)
    freeze = load_json_object(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    marker_path = run_dir / "RESERVE_C_CONSUMPTION_STARTED.json"
    failed_path = run_dir / "C1A_MATERIALIZATION_FAILED.json"
    materialized_root = run_dir / "materialized"
    materialized_root.mkdir(parents=True, exist_ok=False)

    # IRREVERSIBLE BOUNDARY: this marker is written immediately before the
    # first PDF text extraction. From this point onward Fresh C is consumed,
    # including on any later exception.
    _atomic(marker_path, {
        "schema_version": "sers-fresh-c-reserve-c-consumption-started-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "protocol_freeze_id": freeze["freeze_id"],
        "c01d_result_freeze_id": p.c01d_result_freeze_id,
        "c01d_content_seal_sha256": p.c01d_content_seal_sha256,
        "selected_pdf_count": 25,
        "fresh_reserve_c_consumed": True,
        "consumption_irreversible": True,
        "same_epoch_rerun_allowed": False,
        "failure_restores_freshness": False,
        "network_allowed": False,
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

            pages = materialize_pdf_pages(source_path)
            text = render_page_bounded_text(pages)
            paper_dir = materialized_root / f"reserve_c_{index:03d}"
            paper_dir.mkdir(parents=True, exist_ok=False)
            text_path = paper_dir / "full_text.txt"
            text_path.write_text(text, encoding="utf-8")

            page_manifest = {
                "schema_version": "sers-fresh-c-c1a-pages-v1",
                "reserve_index": index,
                "canonical_id": source["canonical_id"],
                "source_pdf_sha256": source["source_pdf_sha256"],
                "page_count": len(pages),
                "pages": [
                    {
                        "page_number": page_index,
                        "text": page,
                        "text_sha256": sha256_json({"text": page}),
                        "character_count": len(page),
                    }
                    for page_index, page in enumerate(pages, start=1)
                ],
                "fresh_reserve_c_consumed": True,
                "scientific_reviewer_read_performed": False,
                "scientific_adjudication_performed": False,
            }
            pages_path = paper_dir / "pages.json"
            _atomic(pages_path, page_manifest)

            records.append({
                "reserve_index": index,
                "canonical_id": source["canonical_id"],
                "source_path": source["source_path"],
                "source_pdf_sha256": source["source_pdf_sha256"],
                "materialized_text_path": str(text_path.relative_to(root)),
                "materialized_text_sha256": sha256_file(text_path),
                "pages_manifest_path": str(pages_path.relative_to(root)),
                "pages_manifest_sha256": sha256_file(pages_path),
                "page_count": len(pages),
                "character_count": len(text),
                "materializer": p.materializer,
                "network_calls": 0,
                "llm_calls": 0,
            })

        if len(records) != 25:
            raise RuntimeError("C1A did not materialize exactly 25 sealed PDFs.")

        corpus_manifest = {
            "schema_version": "sers-fresh-c-c1a-materialized-corpus-v1",
            "protocol_id": p.protocol_id,
            "protocol_sha256": p.protocol_sha256,
            "protocol_freeze_id": freeze["freeze_id"],
            "c01d_result_freeze_id": p.c01d_result_freeze_id,
            "c01d_content_seal_sha256": p.c01d_content_seal_sha256,
            "materializer": p.materializer,
            "pdftext_version": p.pdftext_version,
            "pypdfium2_version": p.pypdfium2_version,
            "materialized_pdf_count": 25,
            "records": records,
            "fresh_reserve_c_consumed": True,
            "consumption_irreversible": True,
            "pdf_text_extraction_performed": True,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "hypothesis_state_mutation_performed": False,
            "external_literature_lookup_performed": False,
            "network_calls": 0,
            "llm_calls": 0,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        }
        corpus_manifest["corpus_sha256"] = _payload_sha(
            corpus_manifest, "corpus_sha256"
        )
        corpus_path = run_dir / "materialized_corpus_manifest.json"
        _atomic(corpus_path, corpus_manifest)

        run_manifest = {
            "schema_version": "sers-fresh-c-c1a-materialization-run-v1",
            "protocol_id": p.protocol_id,
            "protocol_sha256": p.protocol_sha256,
            "protocol_freeze_id": freeze["freeze_id"],
            "consumption_marker_path": str(marker_path.relative_to(root)),
            "consumption_marker_sha256": sha256_file(marker_path),
            "materialized_corpus_path": str(corpus_path.relative_to(root)),
            "materialized_corpus_file_sha256": sha256_file(corpus_path),
            "materialized_corpus_sha256": corpus_manifest["corpus_sha256"],
            "materialized_pdf_count": 25,
            "fresh_reserve_c_consumed": True,
            "consumption_irreversible": True,
            "pdf_text_extraction_performed": True,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "hypothesis_state_mutation_performed": False,
            "external_literature_lookup_performed": False,
            "network_calls": 0,
            "llm_calls": 0,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        }
        ident = sha256_json(run_manifest)
        run_manifest["run_id"] = (
            "sers_fresh_c_c1a_materialization_run_v1:" + ident[:20]
        )
        run_manifest["run_sha256"] = _payload_sha(
            run_manifest, "run_sha256"
        )
        run_manifest_path = run_dir / "run_manifest.json"
        _atomic(run_manifest_path, run_manifest)
        _atomic(run_dir / "C1A_MATERIALIZATION_COMPLETE.json", {
            "schema_version": "sers-fresh-c-c1a-materialization-complete-v1",
            "run_id": run_manifest["run_id"],
            "run_sha256": run_manifest["run_sha256"],
            "materialized_corpus_sha256": corpus_manifest["corpus_sha256"],
            "materialized_pdf_count": 25,
            "fresh_reserve_c_consumed": True,
            "scientific_reviewer_read_performed": False,
            "scientific_adjudication_performed": False,
            "automatic_c1b_transition_authorized": False,
            "stop": True,
        })

        print("Fresh-C C1A local materialization complete")
        print(f"Run ID: {run_manifest['run_id']}")
        print(f"Run SHA256: {run_manifest['run_sha256']}")
        print(f"Materialized corpus SHA256: {corpus_manifest['corpus_sha256']}")
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
        return 0

    except Exception as exc:
        _atomic(failed_path, {
            "schema_version": "sers-fresh-c-c1a-materialization-failed-v1",
            "exception_type": type(exc).__name__,
            "fresh_reserve_c_consumed": True,
            "consumption_irreversible": True,
            "same_epoch_rerun_allowed": False,
            "failure_restores_freshness": False,
            "identity_replacement_allowed": False,
            "materialized_pdf_count_before_failure": len(records),
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
    if args.confirm_irrevocable_reserve_c_consumption:
        return execute()
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
