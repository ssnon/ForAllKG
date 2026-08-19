from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    sha256_file,
    sha256_json,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_C01B_DIR,
    DEFAULT_PROTOCOL_PATH as V1_PROTOCOL_PATH,
    build_fresh_queue,
    load_and_validate_protocol as load_v1_protocol,
    make_access_locator_payload,
    make_blind_queue_payload,
    make_catalog_queries,
    project_packet_to_identity_only,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2 import (
    DiagnosticSemanticScholarCatalogProvider,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_2 import (
    make_transport_diagnostics_payload_v2_2,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    EXPECTED_V22_FAILED_ATTEMPT_ID,
    require_api_key_presence,
    validate_v22_failed_epoch,
    load_and_validate_protocol,
)
from dac_her.literature_catalog import (
    CrossrefCatalogProvider,
    LiteratureCatalogRetriever,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_3_protocol_freeze import (
    main as verify_recovery_freeze,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run guarded Fresh-C C0.1C-v2.2 metadata recovery. "
            "This is a new one-shot network epoch and still does not "
            "consume or semantically read Fresh Reserve C."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument(
        "--confirm-live-discovery-recovery-v2-3",
        action="store_true",
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _attempt_id(protocol_id: str, freeze_id: str) -> str:
    raw = (
        protocol_id + "\0" + freeze_id + "\0recovery-v2.3"
    ).encode("utf-8")
    return (
        "sers_fresh_c_live_discovery_recovery_attempt_v2_3:"
        + hashlib.sha256(raw).hexdigest()[:20]
    )


def _load_freeze(root: Path) -> dict[str, Any]:
    return _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")


def _preflight(root: Path) -> dict[str, Any]:
    verify_recovery_freeze()
    validate_v22_failed_epoch(root)
    require_api_key_presence()

    protocol = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    freeze = _load_freeze(root)
    run_dir = root / DEFAULT_RUN_DIR
    run_dir_empty = not run_dir.exists() or not any(run_dir.iterdir())

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "recovery_parent_attempt_id": EXPECTED_V22_FAILED_ATTEMPT_ID,
        "recovery_parent_failed_epoch_preserved": True,
        "providers": ",".join(protocol.providers),
        "broad_queries": len(protocol.broad_queries),
        "results_per_query_provider": protocol.results_per_query,
        "historical_identity_count": protocol.historical_identity_count,
        "target_acquired_papers": protocol.target_acquired_papers,
        "semantic_scholar_api_key_present": True,
        "crossref_mailto_present": bool(
            os.getenv("CROSSREF_MAILTO")
        ),
        "run_dir_empty": run_dir_empty,
        "scientific_selection_semantics_changed": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_preflight": 0,
        "llm_calls_during_preflight": 0,
        "authorized": False,
        "stop": True,
    }


def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    result = _preflight(root)
    if not result["run_dir_empty"]:
        raise RuntimeError(
            "Recovery-v2 run directory is not empty; same epoch rerun forbidden."
        )
    print("Fresh-C C0.1C-v2.2 guarded recovery preflight")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError(
            "Recovery-v2 epoch already started; rerun forbidden."
        )

    require_api_key_presence()
    protocol = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    v1_protocol = load_v1_protocol(root / V1_PROTOCOL_PATH)
    freeze = _load_freeze(root)
    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)

    attempt_id = _attempt_id(
        protocol.protocol_id,
        freeze["freeze_id"],
    )
    started_path = run_dir / "DISCOVERY_RECOVERY_STARTED.json"
    failed_path = run_dir / "DISCOVERY_RECOVERY_FAILED.json"
    diagnostics_path = run_dir / "TRANSPORT_DIAGNOSTICS.json"

    started = {
        "schema_version": (
            "sers-fresh-c-live-discovery-recovery-started-v2-2"
        ),
        "attempt_id": attempt_id,
        "recovery_parent_attempt_id": EXPECTED_V22_FAILED_ATTEMPT_ID,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_boundary_opened": True,
        "same_recovery_epoch_rerun_allowed": False,
        "failure_authorizes_query_or_selection_tuning": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
    }
    _atomic_json(started_path, started)

    s2 = DiagnosticSemanticScholarCatalogProvider()
    crossref = CrossrefCatalogProvider()
    retriever = LiteratureCatalogRetriever(
        [s2, crossref],
        results_per_query=protocol.results_per_query,
    )
    queries = make_catalog_queries(v1_protocol)

    try:
        outcome = retriever.retrieve(
            profile_id="sers_fresh_c_broad_domain_v1",
            queries=queries,
        )
        packet = outcome.packet

        diagnostics = make_transport_diagnostics_payload_v2_2(
            protocol_id=protocol.protocol_id,
            parent_attempt_id=EXPECTED_V22_FAILED_ATTEMPT_ID,
            broad_queries=protocol.broad_queries,
            executions=packet.executions,
            semantic_scholar_attempts=s2.attempt_diagnostics,
        )
        _atomic_json(diagnostics_path, diagnostics)

        failures = [row for row in packet.executions if not row.success]
        if len(packet.executions) != protocol.expected_provider_query_executions:
            raise RuntimeError(
                "Recovery-v2 provider-query execution count incomplete: "
                f"{len(packet.executions)} != "
                f"{protocol.expected_provider_query_executions}"
            )
        if failures:
            labels = [
                f"{row.provider}:{row.query_id}"
                for row in failures
            ]
            raise RuntimeError(
                "Recovery-v2 requires all frozen provider-query "
                "executions to succeed; failed="
                + ",".join(labels)
            )

        projection = project_packet_to_identity_only(packet)
        ledger_path = (
            root
            / DEFAULT_C01B_DIR
            / "historical_exclusion_ledger.json"
        )
        ledger = _read(ledger_path)
        queue, locators, historical_excluded = build_fresh_queue(
            projection=projection,
            historical_ledger=ledger,
        )
        if len(queue) < protocol.target_acquired_papers:
            raise RuntimeError(
                "Recovery-v2 fresh identity queue is smaller than "
                f"frozen target {protocol.target_acquired_papers}; "
                "new protocol epoch required."
            )

        queue_payload = make_blind_queue_payload(
            protocol=protocol,
            queue=queue,
        )
        locator_payload = make_access_locator_payload(
            protocol=protocol,
            locators=locators,
        )
        queue_path = run_dir / "blind_selection_queue.json"
        locator_path = run_dir / "access_locator_manifest.json"
        _atomic_json(queue_path, queue_payload)
        _atomic_json(locator_path, locator_payload)

        run_body: dict[str, Any] = {
            "schema_version": (
                "sers-fresh-c-live-discovery-recovery-run-v2-2"
            ),
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "freeze_id": freeze["freeze_id"],
            "freeze_manifest_sha256": freeze["manifest_sha256"],
            "attempt_id": attempt_id,
            "recovery_parent_attempt_id": EXPECTED_V22_FAILED_ATTEMPT_ID,
            "searched_at_utc": packet.searched_at_utc,
            "providers": protocol.providers,
            "broad_query_count": len(protocol.broad_queries),
            "results_per_query": protocol.results_per_query,
            "expected_provider_query_executions": (
                protocol.expected_provider_query_executions
            ),
            "observed_provider_query_executions": len(packet.executions),
            "successful_provider_query_executions": sum(
                1 for row in packet.executions if row.success
            ),
            "raw_work_count": packet.raw_work_count,
            "catalog_canonical_work_count": packet.canonical_work_count,
            "projected_unique_identity_count": len(
                projection.identity_records
            ),
            "identity_duplicate_merge_count": (
                projection.duplicate_merge_count
            ),
            "ambiguous_identity_excluded_count": (
                projection.ambiguous_identity_excluded_count
            ),
            "historical_excluded_identity_count": historical_excluded,
            "fresh_identity_queue_count": len(queue),
            "target_acquired_papers": protocol.target_acquired_papers,
            "transport_diagnostics_path": str(
                diagnostics_path.relative_to(root)
            ),
            "transport_diagnostics_file_sha256": sha256_file(
                diagnostics_path
            ),
            "blind_queue_path": str(queue_path.relative_to(root)),
            "blind_queue_file_sha256": sha256_file(queue_path),
            "access_locator_path": str(locator_path.relative_to(root)),
            "access_locator_file_sha256": sha256_file(locator_path),
            "raw_catalog_packet_persisted": False,
            "scientific_metadata_fields_persisted": False,
            "scientific_selection_semantics_changed_from_v1": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "llm_calls": 0,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        }
        identity_sha = sha256_json(run_body)
        run_body["run_id"] = (
            "sers_fresh_c_live_discovery_recovery_run_v2_2:"
            + identity_sha[:20]
        )
        run_body["run_sha256"] = _payload_sha(
            run_body,
            "run_sha256",
        )
        _atomic_json(run_dir / "run_manifest.json", run_body)
        _atomic_json(
            run_dir / "DISCOVERY_RECOVERY_COMPLETE.json",
            {
                "schema_version": (
                    "sers-fresh-c-live-discovery-recovery-complete-v2-2"
                ),
                "attempt_id": attempt_id,
                "run_id": run_body["run_id"],
                "run_sha256": run_body["run_sha256"],
                "fresh_identity_queue_count": len(queue),
                "target_acquired_papers": protocol.target_acquired_papers,
                "fresh_reserve_c_consumed": False,
                "semantic_read_performed": False,
                "automatic_c0_1d_transition_authorized": False,
                "stop": True,
            },
        )

        print("Fresh-C C0.1C-v2 guarded live metadata recovery")
        print(f"Attempt ID: {attempt_id}")
        print(f"Run ID: {run_body['run_id']}")
        print(f"Run SHA256: {run_body['run_sha256']}")
        print(
            "Provider-query executions: "
            f"{len(packet.executions)}/{len(packet.executions)} successful"
        )
        print(f"Raw metadata works: {packet.raw_work_count}")
        print(
            "Catalog canonical works: "
            f"{packet.canonical_work_count}"
        )
        print(
            "Projected unique identities: "
            f"{len(projection.identity_records)}"
        )
        print(
            "Ambiguous identities excluded: "
            f"{projection.ambiguous_identity_excluded_count}"
        )
        print(
            "Historical identities excluded: "
            f"{historical_excluded}"
        )
        print(f"Fresh blind queue identities: {len(queue)}")
        print(
            "Semantic Scholar API key present: "
            f"{bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY'))}"
        )
        print("Scientific selection semantics changed: False")
        print("Raw catalog packet persisted: False")
        print("Fresh Reserve C consumed: False")
        print("Semantic read performed: False")
        print("LLM calls: 0")
        print("Automatic C0.1D transition authorized: False")
        print("STOP: True")
        return 0
    except Exception as exc:
        diagnostics_sha = (
            sha256_file(diagnostics_path)
            if diagnostics_path.exists()
            else None
        )
        _atomic_json(
            failed_path,
            {
                "schema_version": (
                    "sers-fresh-c-live-discovery-recovery-failed-v2-2"
                ),
                "attempt_id": attempt_id,
                "recovery_parent_attempt_id": (
                    EXPECTED_V22_FAILED_ATTEMPT_ID
                ),
                "exception_type": type(exc).__name__,
                "transport_diagnostics_file_sha256": diagnostics_sha,
                "network_boundary_opened": True,
                "same_recovery_epoch_rerun_allowed": False,
                "new_protocol_epoch_required": True,
                "failure_authorizes_query_or_selection_tuning": False,
                "fresh_reserve_c_consumed": False,
                "semantic_read_performed": False,
                "automatic_c0_1d_transition_authorized": False,
                "stop": True,
            },
        )
        raise


def main() -> int:
    args = parse_args()
    if args.preflight:
        return preflight()
    if args.confirm_live_discovery_recovery_v2_3:
        return execute()
    raise RuntimeError("Unreachable argument state.")


if __name__ == "__main__":
    raise SystemExit(main())
