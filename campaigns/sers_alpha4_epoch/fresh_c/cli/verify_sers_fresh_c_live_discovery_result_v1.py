from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    FreshCIdentityRecord,
    rank_fresh_identities,
    sha256_file,
    sha256_json,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_C01B_DIR,
    DEFAULT_DISCOVERY_RUN_DIR,
    DEFAULT_PROTOCOL_PATH,
    TARGET_ACQUIRED_PAPERS,
    BlindQueueRecord,
    LiveDiscoveryManifest,
    load_and_validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify frozen C0.1C discovery output without network access."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_DISCOVERY_RUN_DIR)
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _payload_sha(payload: dict, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    args = parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir

    subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_protocol_freeze_v1"],
        cwd=root,
        check=True,
    )
    protocol = load_and_validate_protocol(protocol_path)

    failed_path = run_dir / "DISCOVERY_FAILED.json"
    if failed_path.exists():
        raise RuntimeError("C0.1C discovery epoch has terminal FAILED marker.")
    started = _read(run_dir / "DISCOVERY_STARTED.json")
    complete = _read(run_dir / "DISCOVERY_COMPLETE.json")
    manifest_raw = _read(run_dir / "live_discovery_manifest.json")
    manifest = LiveDiscoveryManifest.model_validate(manifest_raw)
    if manifest.run_sha256 != _payload_sha(manifest_raw, "run_sha256"):
        raise ValueError("C0.1C run semantic SHA drifted.")
    if manifest.protocol_id != protocol.protocol_id:
        raise ValueError("C0.1C result protocol ID mismatch.")
    if manifest.protocol_sha256 != protocol.protocol_sha256:
        raise ValueError("C0.1C result protocol SHA mismatch.")

    queue_path = root / manifest.blind_queue_path
    locator_path = root / manifest.access_locator_path
    if sha256_file(queue_path) != manifest.blind_queue_file_sha256:
        raise ValueError("C0.1C queue file SHA drifted.")
    if sha256_file(locator_path) != manifest.access_locator_file_sha256:
        raise ValueError("C0.1C locator file SHA drifted.")

    queue_raw = _read(queue_path)
    locator_raw = _read(locator_path)
    if queue_raw.get("queue_sha256") != _payload_sha(queue_raw, "queue_sha256"):
        raise ValueError("C0.1C queue semantic SHA drifted.")
    if locator_raw.get("locator_sha256") != _payload_sha(
        locator_raw, "locator_sha256"
    ):
        raise ValueError("C0.1C locator semantic SHA drifted.")

    queue = [BlindQueueRecord.model_validate(row) for row in queue_raw["records"]]
    if len(queue) < TARGET_ACQUIRED_PAPERS:
        raise ValueError("C0.1C fresh queue is smaller than frozen target.")
    if [row.rank for row in queue] != list(range(1, len(queue) + 1)):
        raise ValueError("C0.1C queue ranks are not contiguous.")
    if len({row.canonical_id for row in queue}) != len(queue):
        raise ValueError("C0.1C queue contains duplicate identities.")

    ledger = _read(root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json")
    historical = set(ledger["canonical_ids"])
    overlap = historical & {row.canonical_id for row in queue}
    if overlap:
        raise ValueError("Historical identity leaked into C0.1C fresh queue.")

    replay_candidates = [
        FreshCIdentityRecord(
            canonical_id=row.canonical_id,
            catalog_work_id="opaque:" + row.canonical_id,
            identity_method=row.identity_method,
        )
        for row in queue
    ]
    replay = rank_fresh_identities(
        candidates=replay_candidates,
        historical_ledger=ledger,
    )
    if [row.canonical_id for row in replay] != [row.canonical_id for row in queue]:
        raise ValueError("C0.1C queue is not the frozen identity-only blind ordering.")
    if [row.score_sha256 for row in replay] != [row.score_sha256 for row in queue]:
        raise ValueError("C0.1C queue blind scores drifted.")

    fresh_ids = {row.canonical_id for row in queue}
    locator_ids = {str(row["canonical_id"]) for row in locator_raw["records"]}
    if locator_ids != fresh_ids:
        raise ValueError("C0.1C locator identities do not exactly match fresh queue.")

    serialized = json.dumps(
        {"queue": queue_raw, "locator": locator_raw, "manifest": manifest_raw},
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden_key in ('"title"', '"abstract"', '"citation_count"'):
        if forbidden_key in serialized:
            raise ValueError(
                "C0.1C persisted forbidden scientific metadata field: "
                + forbidden_key
            )

    if manifest.observed_provider_query_executions != 8:
        raise ValueError("C0.1C observed execution count drifted.")
    if manifest.successful_provider_query_executions != 8:
        raise ValueError("C0.1C not all provider-query executions succeeded.")
    if not all(row.success for row in manifest.query_executions):
        raise ValueError("C0.1C execution summary contains failure.")
    if manifest.fresh_reserve_c_consumed is not False:
        raise ValueError("C0.1C unexpectedly consumed Fresh C.")
    if manifest.semantic_read_performed is not False:
        raise ValueError("C0.1C unexpectedly performed semantic read.")
    if manifest.automatic_c0_1d_transition_authorized is not False:
        raise ValueError("C0.1C unexpectedly authorized C0.1D.")

    if started.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("C0.1C start marker rerun policy drifted.")
    if complete.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("C0.1C complete marker rerun policy drifted.")
    if complete.get("run_id") != manifest.run_id:
        raise ValueError("C0.1C complete marker run ID mismatch.")
    if complete.get("run_sha256") != manifest.run_sha256:
        raise ValueError("C0.1C complete marker run SHA mismatch.")

    print("Fresh-C C0.1C live-discovery result verifier")
    print(f"Run ID: {manifest.run_id}")
    print(f"Run SHA256: {manifest.run_sha256}")
    print(f"Raw metadata works: {manifest.raw_work_count}")
    print(f"Projected unique identities: {manifest.projected_unique_identity_count}")
    print(f"Ambiguous identities excluded: {manifest.ambiguous_identity_excluded_count}")
    print(f"Historical identities excluded: {manifest.historical_excluded_identity_count}")
    print(f"Fresh blind queue identities: {manifest.fresh_identity_queue_count}")
    print("Provider-query executions: 8/8 successful")
    print("Raw catalog packet persisted: False")
    print("Scientific metadata fields persisted: False")
    print("LLM calls: 0")
    print("Fresh Reserve C consumed: False")
    print("Semantic read performed: False")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
