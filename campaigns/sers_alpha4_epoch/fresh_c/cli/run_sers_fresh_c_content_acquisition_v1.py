from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dac_her.corpus_acquisition.artifact_acquisition import MainArtifactDownloader
from dac_her.corpus_acquisition.oa_resolution import OpenAccessResolver
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_content_acquisition_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    load_and_validate_protocol,
    load_json_object,
    locator_record_to_minimal_work,
    require_credentials,
    seal_payload,
    source_policy,
    validate_upstream_v24,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_content_acquisition_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Guarded Fresh-C C0.1D blind OA PDF acquisition. "
            "No PDF text extraction or scientific read is performed."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--confirm-live-content-acquisition", action="store_true")
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
    upstream = validate_upstream_v24(root)
    require_credentials()
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    freeze = load_json_object(
        root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json"
    )
    run_dir = root / DEFAULT_RUN_DIR
    empty = not run_dir.exists() or not any(run_dir.iterdir())

    # Validate that every frozen identity has a locator record without reading
    # scientific title/abstract fields.
    for canonical_id in upstream["queue_ids"]:
        locator_record_to_minimal_work(
            upstream["locator_map"][canonical_id]
        )

    return {
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "upstream_v24_run_id": p.upstream_v24_run_id,
        "blind_queue_count": len(upstream["queue_ids"]),
        "target_verified_pdf_count": p.target_successful_pdf_count,
        "maximum_identity_attempts": p.maximum_identity_attempts,
        "openalex_api_key_present": bool(os.getenv("OPENALEX_API_KEY")),
        "unpaywall_email_or_fallback_present": bool(
            os.getenv("UNPAYWALL_EMAIL") or os.getenv("CROSSREF_MAILTO")
        ),
        "run_dir_empty": empty,
        "manual_candidate_replacement_allowed": False,
        "hypothesis_aware_selection_allowed": False,
        "scientific_metadata_inspection_allowed": False,
        "pdf_text_extraction_allowed": False,
        "fresh_reserve_c_consumed": False,
        "network_calls_during_preflight": 0,
        "llm_calls_during_preflight": 0,
        "authorized": False,
        "stop": True,
    }


def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("C0.1D run directory not empty; same epoch forbidden.")
    print("Fresh-C C0.1D guarded blind OA content-acquisition preflight")
    for key, value in state.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("C0.1D epoch already started; rerun forbidden.")

    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    upstream = validate_upstream_v24(root)
    policy = source_policy(p)
    resolver = OpenAccessResolver(policy)
    downloader = MainArtifactDownloader(policy)

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    started_path = run_dir / "CONTENT_ACQUISITION_STARTED.json"
    failed_path = run_dir / "CONTENT_ACQUISITION_FAILED.json"
    attempts_path = run_dir / "identity_attempts.jsonl"
    selected_path = run_dir / "selected_reserve_c.json"
    seal_path = run_dir / "content_seal.json"

    _atomic(started_path, {
        "schema_version": "sers-fresh-c-content-acquisition-started-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_v24_run_id": p.upstream_v24_run_id,
        "blind_queue_count": p.upstream_blind_queue_count,
        "target_verified_pdf_count": p.target_successful_pdf_count,
        "network_boundary_opened": True,
        "same_epoch_rerun_allowed": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "pdf_text_extraction_performed": False,
    })

    selected = []
    attempted_rows = []
    try:
        for rank, canonical_id in enumerate(upstream["queue_ids"], start=1):
            if len(selected) >= p.target_successful_pdf_count:
                break
            locator = upstream["locator_map"][canonical_id]
            work = locator_record_to_minimal_work(locator)
            resolution = resolver.resolve(work)
            artifact = downloader.acquire(
                work=work,
                resolution=resolution,
                output_root=run_dir,
            )

            row = {
                "blind_rank": rank,
                "canonical_id": canonical_id,
                "access_status": resolution.status,
                "artifact_status": artifact.status,
                "selected_location_id": artifact.selected_location_id,
                "resolver_attempt_statuses": [
                    {
                        "resolver": attempt.resolver,
                        "status": attempt.status,
                    }
                    for attempt in resolution.resolver_attempts
                ],
                "attempted_download_location_count": artifact.attempted_location_count,
                "download_failure_codes": [
                    attempt.error_code
                    for attempt in artifact.download_attempts
                    if attempt.status == "failed"
                ],
                "fresh_reserve_c_consumed": False,
                "semantic_read_performed": False,
            }
            attempted_rows.append(row)
            with attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

            if artifact.status != "downloaded":
                continue
            if not artifact.local_path or not artifact.sha256:
                raise RuntimeError("Downloaded artifact missing path or SHA256.")
            local_path = Path(artifact.local_path)
            if not local_path.is_absolute():
                local_path = root / local_path
            if not local_path.exists():
                raise RuntimeError("Downloaded artifact path missing.")
            if sha256_file(local_path) != artifact.sha256:
                raise RuntimeError("Downloaded artifact SHA256 verification failed.")
            with local_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise RuntimeError("Downloaded artifact PDF magic drifted.")

            selected.append({
                "reserve_index": len(selected) + 1,
                "blind_rank": rank,
                "canonical_id": canonical_id,
                "artifact_sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
                "local_path": str(local_path.relative_to(root)),
                "selected_location_id": artifact.selected_location_id,
                "source_resolver": next(
                    (
                        location.resolver
                        for location in resolution.locations
                        if location.location_id == artifact.selected_location_id
                    ),
                    None,
                ),
            })

        if len(selected) != p.target_successful_pdf_count:
            raise RuntimeError(
                "C0.1D exhausted frozen blind queue before obtaining 25 "
                "verified OA PDFs."
            )

        selected_payload = {
            "schema_version": "sers-fresh-c-selected-reserve-c-v1",
            "selection_rule": p.selection_rule,
            "selected_count": len(selected),
            "attempted_identity_count": len(attempted_rows),
            "records": selected,
            "manual_candidate_replacement_performed": False,
            "scientific_metadata_inspection_performed": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "pdf_text_extraction_performed": False,
            "llm_calls": 0,
        }
        selected_payload["selection_sha256"] = _payload_sha(
            selected_payload, "selection_sha256"
        )
        _atomic(selected_path, selected_payload)

        seal = seal_payload(selected)
        _atomic(seal_path, seal)

        manifest = {
            "schema_version": "sers-fresh-c-content-acquisition-run-v1",
            "protocol_id": p.protocol_id,
            "protocol_sha256": p.protocol_sha256,
            "upstream_v24_run_id": p.upstream_v24_run_id,
            "upstream_v24_run_sha256": p.upstream_v24_run_sha256,
            "blind_queue_count": p.upstream_blind_queue_count,
            "attempted_identity_count": len(attempted_rows),
            "selected_verified_pdf_count": len(selected),
            "selected_reserve_c_path": str(selected_path.relative_to(root)),
            "selected_reserve_c_file_sha256": sha256_file(selected_path),
            "content_seal_path": str(seal_path.relative_to(root)),
            "content_seal_file_sha256": sha256_file(seal_path),
            "content_seal_sha256": seal["content_seal_sha256"],
            "identity_attempts_path": str(attempts_path.relative_to(root)),
            "identity_attempts_file_sha256": sha256_file(attempts_path),
            "reserve_c_identity_selection_finalized": True,
            "reserve_c_content_sealed": True,
            "manual_candidate_replacement_performed": False,
            "hypothesis_aware_selection_performed": False,
            "scientific_metadata_inspection_performed": False,
            "pdf_text_extraction_performed": False,
            "semantic_read_performed": False,
            "paywall_bypass_attempted": False,
            "positive_evidence_promotion_performed": False,
            "fresh_reserve_c_consumed": False,
            "llm_calls": 0,
            "automatic_c1_transition_authorized": False,
            "stop": True,
        }
        ident = sha256_json(manifest)
        manifest["run_id"] = (
            "sers_fresh_c_content_acquisition_run_v1:" + ident[:20]
        )
        manifest["run_sha256"] = _payload_sha(manifest, "run_sha256")
        _atomic(run_dir / "run_manifest.json", manifest)
        _atomic(run_dir / "CONTENT_ACQUISITION_COMPLETE.json", {
            "schema_version": "sers-fresh-c-content-acquisition-complete-v1",
            "run_id": manifest["run_id"],
            "run_sha256": manifest["run_sha256"],
            "selected_verified_pdf_count": len(selected),
            "content_seal_sha256": seal["content_seal_sha256"],
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c1_transition_authorized": False,
            "stop": True,
        })

        print("Fresh-C C0.1D blind OA content acquisition complete")
        print(f"Run ID: {manifest['run_id']}")
        print(f"Run SHA256: {manifest['run_sha256']}")
        print(f"Attempted blind identities: {len(attempted_rows)}")
        print(f"Selected verified OA PDFs: {len(selected)}")
        print(f"Content seal SHA256: {seal['content_seal_sha256']}")
        print("Manual replacement performed: False")
        print("Scientific metadata inspection performed: False")
        print("PDF text extraction performed: False")
        print("Semantic read performed: False")
        print("Fresh Reserve C consumed: False")
        print("LLM calls: 0")
        print("Automatic C1 transition authorized: False")
        print("STOP: True")
        return 0

    except Exception as exc:
        _atomic(failed_path, {
            "schema_version": "sers-fresh-c-content-acquisition-failed-v1",
            "exception_type": type(exc).__name__,
            "network_boundary_opened": True,
            "same_epoch_rerun_allowed": False,
            "new_protocol_epoch_required": True,
            "failure_authorizes_manual_replacement": False,
            "failure_authorizes_scientific_tuning": False,
            "downloaded_pdf_count_before_failure": len(selected),
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "pdf_text_extraction_performed": False,
            "automatic_c1_transition_authorized": False,
            "stop": True,
        })
        raise


def main() -> int:
    args = parse_args()
    if args.preflight:
        return preflight()
    if args.confirm_live_content_acquisition:
        return execute()
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
