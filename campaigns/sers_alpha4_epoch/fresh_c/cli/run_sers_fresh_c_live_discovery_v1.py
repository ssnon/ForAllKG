from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_C01B_DIR,
    DEFAULT_DISCOVERY_FREEZE_DIR,
    DEFAULT_DISCOVERY_RUN_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    EXPECTED_HISTORICAL_LEDGER_ID,
    EXPECTED_HISTORICAL_LEDGER_SHA256,
    TARGET_ACQUIRED_PAPERS,
    LiveDiscoveryManifest,
    assert_complete_execution,
    build_fresh_queue,
    load_and_validate_protocol,
    make_access_locator_payload,
    make_blind_queue_payload,
    make_catalog_queries,
    project_packet_to_identity_only,
    summarize_query_executions,
)
from dac_her.literature_catalog import (
    CrossrefCatalogProvider,
    LiteratureCatalogRetriever,
    SemanticScholarCatalogProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded Fresh-C C0.1C live metadata discovery. "
            "--preflight performs zero network calls. Actual discovery "
            "requires --confirm-live-discovery."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--confirm-live-discovery", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--freeze-dir", type=Path, default=DEFAULT_DISCOVERY_FREEZE_DIR
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_DISCOVERY_RUN_DIR)
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _consumption_name_hits(root: Path) -> list[str]:
    base = root / "evaluation/sers_fresh_c"
    if not base.exists():
        return []
    hits: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if "consum" in name:
            hits.append(path.relative_to(root).as_posix())
    return sorted(hits)


def preflight(
    *,
    root: Path,
    protocol_path: Path,
    freeze_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Tracked worktree is dirty; live discovery refused.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode != 0:
        raise RuntimeError("Index is dirty; live discovery refused.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_protocol_freeze_v1",
            "--protocol",
            str(protocol_path),
            "--freeze-dir",
            str(freeze_dir),
        ],
        cwd=root,
        check=True,
    )
    protocol = load_and_validate_protocol(protocol_path)
    freeze = _read(freeze_dir / "freeze_manifest.json")

    ledger_path = root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json"
    ledger = _read(ledger_path)
    if ledger.get("ledger_id") != EXPECTED_HISTORICAL_LEDGER_ID:
        raise ValueError("Historical ledger ID drifted before discovery.")
    if ledger.get("ledger_sha256") != EXPECTED_HISTORICAL_LEDGER_SHA256:
        raise ValueError("Historical ledger SHA drifted before discovery.")
    if len(ledger.get("canonical_ids") or []) != EXPECTED_HISTORICAL_IDENTITY_COUNT:
        raise ValueError("Historical ledger identity count drifted before discovery.")

    if run_dir.exists():
        existing = sorted(path.name for path in run_dir.iterdir())
        detail = ",".join(existing) if existing else "<empty-directory-exists>"
        raise RuntimeError(
            "C0.1C run directory already exists; same-epoch execution refused: "
            + detail
        )
    consumption_hits = _consumption_name_hits(root)
    if consumption_hits:
        raise RuntimeError(
            "Fresh-C consumption-named artifacts already exist; discovery refused: "
            + ",".join(consumption_hits)
        )

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze["manifest_sha256"],
        "historical_ledger_id": ledger["ledger_id"],
        "historical_identity_count": len(ledger["canonical_ids"]),
        "providers": protocol.providers,
        "broad_queries": len(protocol.broad_queries),
        "results_per_query_provider": protocol.results_per_query,
        "expected_provider_query_executions": (
            protocol.expected_provider_query_executions
        ),
        "max_raw_metadata_rows": protocol.max_raw_metadata_rows,
        "target_acquired_papers": protocol.target_acquired_papers,
        "run_dir_empty": True,
        "consumption_artifact_name_hits": 0,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_preflight": 0,
        "llm_calls_during_preflight": 0,
        "preflight": "PASS",
    }


def _print_preflight(result: Mapping[str, Any]) -> None:
    print("Fresh-C C0.1C guarded live-discovery preflight")
    for key, value in result.items():
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        print(f"{key}: {value}")
    print("Live discovery authorized: False")
    print("STOP: True")


def _started_payload(
    *,
    preflight_result: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sers-fresh-c-live-discovery-started-v1",
        "started_at_utc": _utc_now(),
        "protocol_id": preflight_result["protocol_id"],
        "protocol_sha256": preflight_result["protocol_sha256"],
        "discovery_freeze_id": preflight_result["freeze_id"],
        "discovery_freeze_manifest_sha256": (
            preflight_result["freeze_manifest_sha256"]
        ),
        "explicit_live_discovery_confirmation_received": True,
        "network_boundary_opened": True,
        "same_epoch_rerun_allowed": False,
        "failure_authorizes_query_or_selection_tuning": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "llm_calls_before_network_boundary": 0,
    }
    identity = sha256_json(body)
    body["attempt_id"] = "sers_fresh_c_live_discovery_attempt_v1:" + identity[:20]
    body["marker_sha256"] = _payload_sha(body, "marker_sha256")
    return body


def execute(
    *,
    root: Path,
    protocol_path: Path,
    freeze_dir: Path,
    run_dir: Path,
) -> int:
    # Final value-blind revalidation immediately before the network boundary.
    pf = preflight(
        root=root,
        protocol_path=protocol_path,
        freeze_dir=freeze_dir,
        run_dir=run_dir,
    )
    protocol = load_and_validate_protocol(protocol_path)
    freeze = _read(freeze_dir / "freeze_manifest.json")
    ledger_path = root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json"
    ledger = _read(ledger_path)

    run_dir.mkdir(parents=True, exist_ok=False)
    started_path = run_dir / "DISCOVERY_STARTED.json"
    started = _started_payload(preflight_result=pf)
    _atomic_json(started_path, started)

    try:
        queries = make_catalog_queries(protocol)
        providers = [
            SemanticScholarCatalogProvider(),
            CrossrefCatalogProvider(),
        ]
        retriever = LiteratureCatalogRetriever(
            providers,
            results_per_query=protocol.results_per_query,
        )
        outcome = retriever.retrieve(
            profile_id="sers_fresh_c_broad_domain_v1",
            queries=queries,
        )
        packet = outcome.packet

        if packet.providers_requested != protocol.providers:
            raise RuntimeError("Provider order/identity drifted in catalog packet.")
        if len(packet.queries) != len(protocol.broad_queries):
            raise RuntimeError("Catalog packet query count drifted.")
        for observed, expected in zip(packet.queries, queries, strict=True):
            if observed != expected:
                raise RuntimeError("Catalog packet query contract drifted.")
        if packet.raw_work_count > protocol.max_raw_metadata_rows:
            raise RuntimeError("Catalog raw work count exceeded frozen budget.")

        assert_complete_execution(
            protocol=protocol,
            executions=packet.executions,
        )
        projection = project_packet_to_identity_only(packet)
        queue, fresh_locators, historical_excluded = build_fresh_queue(
            projection=projection,
            historical_ledger=ledger,
        )
        if len(queue) < TARGET_ACQUIRED_PAPERS:
            raise RuntimeError(
                "Fresh identity queue is smaller than frozen target acquired "
                f"paper count: {len(queue)} < {TARGET_ACQUIRED_PAPERS}."
            )

        queue_payload = make_blind_queue_payload(
            protocol=protocol,
            queue=queue,
        )
        locator_payload = make_access_locator_payload(
            protocol=protocol,
            locators=fresh_locators,
        )
        queue_path = run_dir / "blind_selection_queue.json"
        locator_path = run_dir / "access_locator_manifest.json"
        _atomic_json(queue_path, queue_payload)
        _atomic_json(locator_path, locator_payload)

        execution_rows = summarize_query_executions(packet.executions)
        body: dict[str, Any] = {
            "schema_version": "sers-fresh-c-live-discovery-run-v1",
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.protocol_sha256,
            "discovery_freeze_id": freeze["freeze_id"],
            "discovery_freeze_manifest_sha256": freeze["manifest_sha256"],
            "c0_1b_lock_id": protocol.c0_1b_lock_id,
            "c0_1b_lock_sha256": protocol.c0_1b_lock_sha256,
            "historical_ledger_id": ledger["ledger_id"],
            "historical_ledger_sha256": ledger["ledger_sha256"],
            "historical_identity_count": len(ledger["canonical_ids"]),
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
            "query_executions": [
                row.model_dump(mode="json") for row in execution_rows
            ],
            "raw_work_count": packet.raw_work_count,
            "catalog_canonical_work_count": packet.canonical_work_count,
            "projected_unique_identity_count": len(projection.identity_records),
            "identity_duplicate_merge_count": projection.duplicate_merge_count,
            "ambiguous_identity_excluded_count": (
                projection.ambiguous_identity_excluded_count
            ),
            "historical_excluded_identity_count": historical_excluded,
            "fresh_identity_queue_count": len(queue),
            "target_acquired_papers": TARGET_ACQUIRED_PAPERS,
            "blind_queue_path": str(queue_path.relative_to(root)),
            "blind_queue_file_sha256": sha256_file(queue_path),
            "access_locator_path": str(locator_path.relative_to(root)),
            "access_locator_file_sha256": sha256_file(locator_path),
            "raw_catalog_packet_persisted": False,
            "scientific_metadata_fields_persisted": False,
            "title_persisted": False,
            "abstract_persisted": False,
            "citation_count_persisted": False,
            "scientific_fields_used_for_ordering": False,
            "llm_calls": 0,
            "network_used": True,
            "exact_http_request_count_known": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        }
        identity = sha256_json(body)
        body["run_id"] = "sers_fresh_c_live_discovery_run_v1:" + identity[:20]
        body["run_sha256"] = _payload_sha(body, "run_sha256")
        manifest = LiveDiscoveryManifest.model_validate(body)
        manifest_path = run_dir / "live_discovery_manifest.json"
        _atomic_json(manifest_path, manifest.model_dump(mode="json"))

        complete = {
            "schema_version": "sers-fresh-c-live-discovery-complete-v1",
            "attempt_id": started["attempt_id"],
            "run_id": manifest.run_id,
            "run_sha256": manifest.run_sha256,
            "fresh_identity_queue_count": len(queue),
            "target_acquired_papers": TARGET_ACQUIRED_PAPERS,
            "same_epoch_rerun_allowed": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        }
        complete["marker_sha256"] = _payload_sha(complete, "marker_sha256")
        _atomic_json(run_dir / "DISCOVERY_COMPLETE.json", complete)

        print("Fresh-C C0.1C guarded live metadata discovery")
        print(f"Attempt ID: {started['attempt_id']}")
        print(f"Run ID: {manifest.run_id}")
        print(f"Run SHA256: {manifest.run_sha256}")
        print(f"Raw metadata works: {manifest.raw_work_count}")
        print(
            "Catalog canonical works: "
            f"{manifest.catalog_canonical_work_count}"
        )
        print(
            "Projected unique identities: "
            f"{manifest.projected_unique_identity_count}"
        )
        print(
            "Ambiguous identities excluded: "
            f"{manifest.ambiguous_identity_excluded_count}"
        )
        print(
            "Historical identities excluded: "
            f"{manifest.historical_excluded_identity_count}"
        )
        print(f"Fresh blind queue identities: {manifest.fresh_identity_queue_count}")
        print("Target acquired papers for C0.1D: 25")
        print("Provider-query executions: 8/8 successful")
        print("Raw catalog packet persisted: False")
        print("Scientific metadata fields persisted: False")
        print("LLM calls: 0")
        print("Fresh Reserve C consumed: False")
        print("Semantic read performed: False")
        print("Automatic C0.1D transition authorized: False")
        print("STOP: True")
        return 0
    except Exception as exc:
        failed = {
            "schema_version": "sers-fresh-c-live-discovery-failed-v1",
            "attempt_id": started["attempt_id"],
            "failed_at_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "network_boundary_opened": True,
            "same_epoch_rerun_allowed": False,
            "new_protocol_epoch_required": True,
            "failure_authorizes_query_or_selection_tuning": False,
            "fresh_reserve_c_consumed": False,
            "semantic_read_performed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        }
        failed["marker_sha256"] = _payload_sha(failed, "marker_sha256")
        _atomic_json(run_dir / "DISCOVERY_FAILED.json", failed)
        raise


def main() -> int:
    args = parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    protocol_path = _resolve(root, args.protocol)
    freeze_dir = _resolve(root, args.freeze_dir)
    run_dir = _resolve(root, args.run_dir)

    if args.preflight:
        result = preflight(
            root=root,
            protocol_path=protocol_path,
            freeze_dir=freeze_dir,
            run_dir=run_dir,
        )
        _print_preflight(result)
        return 0
    return execute(
        root=root,
        protocol_path=protocol_path,
        freeze_dir=freeze_dir,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
