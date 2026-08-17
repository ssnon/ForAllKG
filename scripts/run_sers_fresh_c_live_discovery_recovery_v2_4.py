from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_live_discovery import (
    DEFAULT_C01B_DIR,
    DEFAULT_PROTOCOL_PATH as V1_PROTOCOL_PATH,
    build_fresh_queue,
    load_and_validate_protocol as load_v1_protocol,
    make_access_locator_payload,
    make_blind_queue_payload,
    make_catalog_queries,
    project_packet_to_identity_only,
)
from dac_her.fresh_c_live_discovery_recovery_v2_4 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    EXPECTED_V22_FAILED_ATTEMPT_ID,
    DiagnosticOpenAlexCatalogProvider,
    load_and_validate_protocol,
    make_transport_diagnostics_payload_v2_4,
    require_openalex_api_key,
    validate_v22_failure,
    validate_v23_frozen_unexecuted,
)
from dac_her.literature_catalog import (
    CrossrefCatalogProvider,
    LiteratureCatalogRetriever,
)
from scripts.verify_sers_fresh_c_live_discovery_recovery_v2_4_protocol_freeze import (
    main as verify_freeze,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run guarded Fresh-C C0.1C-v2.4 OpenAlex+Crossref metadata "
            "discovery. This substitutes provider universe only for frozen "
            "transport-availability reasons and does not consume Fresh C."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument(
        "--confirm-live-discovery-recovery-v2-4",
        action="store_true",
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _attempt_id(protocol_id: str, freeze_id: str) -> str:
    raw = (protocol_id + "\0" + freeze_id + "\0openalex-v2.4").encode("utf-8")
    return (
        "sers_fresh_c_openalex_crossref_attempt_v2_4:"
        + hashlib.sha256(raw).hexdigest()[:20]
    )


def _preflight(root: Path) -> dict[str, Any]:
    verify_freeze()
    validate_v22_failure(root)
    validate_v23_frozen_unexecuted(root)
    require_openalex_api_key()

    protocol = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    freeze = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    run_dir = root / DEFAULT_RUN_DIR
    empty = not run_dir.exists() or not any(run_dir.iterdir())
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "v22_failed_attempt_id": EXPECTED_V22_FAILED_ATTEMPT_ID,
        "v23_frozen_but_unexecuted": True,
        "providers": ",".join(protocol.providers),
        "provider_universe_changed": True,
        "broad_queries": len(protocol.broad_queries),
        "results_per_query_provider": protocol.results_per_query,
        "historical_identity_count": protocol.historical_identity_count,
        "target_acquired_papers": protocol.target_acquired_papers,
        "openalex_api_key_present": True,
        "crossref_mailto_present": bool(os.getenv("CROSSREF_MAILTO")),
        "run_dir_empty": empty,
        "frozen_queries_changed": False,
        "blind_ordering_changed": False,
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
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("v2.4 run directory not empty; rerun forbidden.")
    print("Fresh-C C0.1C-v2.4 guarded OpenAlex+Crossref preflight")
    for key, value in state.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["run_dir_empty"]:
        raise RuntimeError("v2.4 epoch already started; rerun forbidden.")

    require_openalex_api_key()
    protocol = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    v1_protocol = load_v1_protocol(root / V1_PROTOCOL_PATH)
    freeze = _read(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    attempt_id = _attempt_id(protocol.protocol_id, freeze["freeze_id"])

    started_path = run_dir / "DISCOVERY_RECOVERY_STARTED.json"
    failed_path = run_dir / "DISCOVERY_RECOVERY_FAILED.json"
    diagnostics_path = run_dir / "TRANSPORT_DIAGNOSTICS.json"

    _atomic(started_path, {
        "schema_version": "sers-fresh-c-openalex-crossref-started-v2-4",
        "attempt_id": attempt_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_universe_changed": True,
        "network_boundary_opened": True,
        "same_recovery_epoch_rerun_allowed": False,
        "failure_authorizes_query_or_selection_tuning": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
    })

    openalex = DiagnosticOpenAlexCatalogProvider()
    crossref = CrossrefCatalogProvider()
    retriever = LiteratureCatalogRetriever(
        [openalex, crossref],
        results_per_query=protocol.results_per_query,
    )
    queries = make_catalog_queries(v1_protocol)

    try:
        packet = retriever.retrieve(
            profile_id="sers_fresh_c_broad_domain_v1",
            queries=queries,
        ).packet

        diagnostics = make_transport_diagnostics_payload_v2_4(
            protocol=protocol,
            executions=packet.executions,
            openalex_attempts=openalex.attempt_diagnostics,
        )
        _atomic(diagnostics_path, diagnostics)

        failures = [row for row in packet.executions if not row.success]
        if len(packet.executions) != protocol.expected_provider_query_executions:
            raise RuntimeError(
                "v2.4 provider-query execution count incomplete: "
                f"{len(packet.executions)} != "
                f"{protocol.expected_provider_query_executions}"
            )
        if failures:
            labels = [f"{row.provider}:{row.query_id}" for row in failures]
            raise RuntimeError(
                "v2.4 requires every frozen provider-query execution "
                "to succeed; failed=" + ",".join(labels)
            )

        projection = project_packet_to_identity_only(packet)
        ledger = _read(
            root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json"
        )
        queue, locators, historical_excluded = build_fresh_queue(
            projection=projection,
            historical_ledger=ledger,
        )
        if len(queue) < protocol.target_acquired_papers:
            raise RuntimeError(
                "v2.4 fresh identity queue below frozen target 25."
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
        _atomic(queue_path, queue_payload)
        _atomic(locator_path, locator_payload)

        run_body: dict[str, Any] = {
            "schema_version": "sers-fresh-c-openalex-crossref-run-v2-4",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "freeze_id": freeze["freeze_id"],
            "freeze_manifest_sha256": freeze["manifest_sha256"],
            "attempt_id": attempt_id,
            "providers": protocol.providers,
            "provider_universe_changed": True,
            "provider_substitution_from": "semantic_scholar",
            "provider_substitution_to": "openalex",
            "broad_query_count": len(protocol.broad_queries),
            "results_per_query": protocol.results_per_query,
            "observed_provider_query_executions": len(packet.executions),
            "successful_provider_query_executions": sum(
                1 for row in packet.executions if row.success
            ),
            "raw_work_count": packet.raw_work_count,
            "catalog_canonical_work_count": packet.canonical_work_count,
            "projected_unique_identity_count": len(projection.identity_records),
            "identity_duplicate_merge_count": projection.duplicate_merge_count,
            "ambiguous_identity_excluded_count": (
                projection.ambiguous_identity_excluded_count
            ),
            "historical_excluded_identity_count": historical_excluded,
            "fresh_identity_queue_count": len(queue),
            "target_acquired_papers": protocol.target_acquired_papers,
            "transport_diagnostics_path": str(
                diagnostics_path.relative_to(root)
            ),
            "transport_diagnostics_file_sha256": sha256_file(diagnostics_path),
            "blind_queue_path": str(queue_path.relative_to(root)),
            "blind_queue_file_sha256": sha256_file(queue_path),
            "access_locator_path": str(locator_path.relative_to(root)),
            "access_locator_file_sha256": sha256_file(locator_path),
            "raw_catalog_packet_persisted": False,
            "scientific_metadata_fields_persisted": False,
            "hypothesis_aware_selection_added": False,
            "scientific_selection_semantics_changed": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "llm_calls": 0,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        }
        ident = sha256_json(run_body)
        run_body["run_id"] = (
            "sers_fresh_c_openalex_crossref_run_v2_4:" + ident[:20]
        )
        run_body["run_sha256"] = _payload_sha(run_body, "run_sha256")
        _atomic(run_dir / "run_manifest.json", run_body)
        _atomic(run_dir / "DISCOVERY_RECOVERY_COMPLETE.json", {
            "schema_version": "sers-fresh-c-openalex-crossref-complete-v2-4",
            "attempt_id": attempt_id,
            "run_id": run_body["run_id"],
            "run_sha256": run_body["run_sha256"],
            "fresh_identity_queue_count": len(queue),
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        })

        print("Fresh-C C0.1C-v2.4 guarded OpenAlex+Crossref live discovery")
        print(f"Attempt ID: {attempt_id}")
        print(f"Run ID: {run_body['run_id']}")
        print(f"Run SHA256: {run_body['run_sha256']}")
        print("Provider-query executions: 8/8 successful")
        print(f"Raw metadata works: {packet.raw_work_count}")
        print(f"Projected unique identities: {len(projection.identity_records)}")
        print(f"Historical identities excluded: {historical_excluded}")
        print(f"Fresh blind queue identities: {len(queue)}")
        print("Provider universe changed: True")
        print("Scientific selection semantics changed: False")
        print("Raw catalog packet persisted: False")
        print("Fresh Reserve C consumed: False")
        print("Semantic read performed: False")
        print("LLM calls: 0")
        print("Automatic C0.1D transition authorized: False")
        print("STOP: True")
        return 0
    except Exception as exc:
        _atomic(failed_path, {
            "schema_version": "sers-fresh-c-openalex-crossref-failed-v2-4",
            "attempt_id": attempt_id,
            "exception_type": type(exc).__name__,
            "network_boundary_opened": True,
            "same_recovery_epoch_rerun_allowed": False,
            "new_protocol_epoch_required": True,
            "failure_authorizes_query_or_selection_tuning": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        })
        raise


def main() -> int:
    args = parse_args()
    if args.preflight:
        return preflight()
    if args.confirm_live_discovery_recovery_v2_4:
        return execute()
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
