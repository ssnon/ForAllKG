from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from dac_her.fresh_c_acquisition import (
    load_and_validate_protocol,
    sha256_file,
    sha256_json,
)
from dac_her.alpha4c5f2_reserve import (
    RESERVE_A_COUNT,
    RESERVE_B_COUNT,
)
from dac_her.fresh_c_activation import (
    ActivationReadinessLock,
    C00_DISPOSITION,
    C01B_SEMANTICS_ID,
    EXPECTED_C01A_FREEZE_COMMIT,
    EXPECTED_C01A_FREEZE_ID,
    EXPECTED_C01A_FREEZE_MANIFEST_SHA256,
    EXPECTED_C01A_PROTOCOL_ID,
    EXPECTED_C01A_PROTOCOL_SHA256,
    EXPECTED_C01A_SOURCE_COMMIT,
    EXPECTED_I0_FREEZE_ID,
    EXPECTED_I0_MANIFEST_SHA256,
    make_search_budget,
    make_target_count_policy,
    validate_historical_sweep_artifacts,
)
from scripts.build_sers_fresh_c_historical_exclusion_ledger_v1 import (
    DEFAULT_OUTPUT_DIR,
)

DEFAULT_C01A_PROTOCOL = Path(
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json"
)

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_activation.py",
    "scripts/build_sers_fresh_c_historical_exclusion_ledger_v1.py",
    "scripts/verify_sers_fresh_c_historical_exclusion_ledger_v1.py",
    "scripts/freeze_sers_fresh_c_activation_readiness_v1.py",
    "scripts/verify_sers_fresh_c_activation_readiness_v1.py",
    "tests/test_sers_fresh_c_activation_readiness_v1.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
    "dac_her/alpha4c5f2_reserve.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Fresh-C C0.1B activation readiness. "
            "This does not authorize or run live discovery."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--confirm-c0-0-new-fresh-acquisition-required",
        action="store_true",
        help=(
            "Attest the completed read-only C0.0 lineage audit "
            "disposition NEW_FRESH_ACQUISITION_REQUIRED."
        ),
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


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


def _payload_sha(
    payload: Mapping[str, Any],
    field: str,
) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def main() -> int:
    args = parse_args()
    if not args.confirm_c0_0_new_fresh_acquisition_required:
        raise RuntimeError(
            "C0.0 disposition attestation missing; refuse readiness lock."
        )

    root = Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel")
    )
    if subprocess.run(
        ["git", "diff", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            "Tracked worktree is dirty; refuse C0.1B readiness freeze."
        )
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            "Index is dirty; refuse C0.1B readiness freeze."
        )

    # I0 and C0.1A must still verify immediately before readiness freeze.
    subprocess.run(
        [
            "python",
            "-m",
            "scripts.verify_sers_i0_integrated_orchestration_freeze_v1",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "scripts.verify_sers_fresh_c_acquisition_protocol_freeze_v1",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_C01A_FREEZE_COMMIT,
            "HEAD",
        ],
        cwd=root,
        check=True,
    )
    if RESERVE_A_COUNT != 25 or RESERVE_B_COUNT != 25:
        raise ValueError(
            "Pre-existing Reserve A/B cardinality basis drifted."
        )

    protocol = load_and_validate_protocol(
        root / DEFAULT_C01A_PROTOCOL
    )
    if protocol.protocol_id != EXPECTED_C01A_PROTOCOL_ID:
        raise ValueError("C0.1A protocol ID drifted.")
    if protocol.protocol_sha256 != EXPECTED_C01A_PROTOCOL_SHA256:
        raise ValueError("C0.1A protocol SHA drifted.")

    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    sweep_path = output / "historical_identity_sweep_manifest.json"
    ledger_path = output / "historical_exclusion_ledger.json"
    manifest, ledger = validate_historical_sweep_artifacts(
        root=root,
        manifest_path=sweep_path,
        ledger_path=ledger_path,
    )

    source_commit = _git(root, "rev-parse", "HEAD")
    hashes: dict[str, str] = {}
    for relative in CRITICAL_COMPONENTS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=root,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"C0.1B critical component not tracked in HEAD: {relative}"
            ) from exc
        committed_sha = __import__("hashlib").sha256(
            committed
        ).hexdigest()
        current_sha = sha256_file(path)
        if committed_sha != current_sha:
            raise RuntimeError(
                f"C0.1B critical component differs from HEAD: {relative}"
            )
        hashes[relative] = current_sha

    attestation = {
        "schema_version": "sers-fresh-c-c0-0-disposition-attestation-v1",
        "disposition": C00_DISPOSITION,
        "basis": (
            "completed_read_only_lineage_audit_no_eligible_existing_"
            "fresh_reserve"
        ),
        "operator_attested": True,
        "scientific_values_read_for_attestation": False,
        "fresh_reserve_c_consumed": False,
    }
    attestation["attestation_sha256"] = _payload_sha(
        attestation,
        "attestation_sha256",
    )
    attestation_path = output / "C0_0_DISPOSITION_ATTESTATION.json"

    lock_body: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-activation-readiness-lock-v1"
        ),
        "semantics_id": C01B_SEMANTICS_ID,
        "source_commit": source_commit,
        "i0_freeze_id": EXPECTED_I0_FREEZE_ID,
        "i0_manifest_sha256": EXPECTED_I0_MANIFEST_SHA256,
        "c0_0_disposition": C00_DISPOSITION,
        "c0_0_operator_attested": True,
        "c0_1a_protocol_id": EXPECTED_C01A_PROTOCOL_ID,
        "c0_1a_protocol_sha256": EXPECTED_C01A_PROTOCOL_SHA256,
        "c0_1a_freeze_id": EXPECTED_C01A_FREEZE_ID,
        "c0_1a_freeze_manifest_sha256": (
            EXPECTED_C01A_FREEZE_MANIFEST_SHA256
        ),
        "c0_1a_source_commit": EXPECTED_C01A_SOURCE_COMMIT,
        "c0_1a_freeze_commit": EXPECTED_C01A_FREEZE_COMMIT,
        "historical_sweep_manifest_path": str(
            sweep_path.relative_to(root)
        ),
        "historical_sweep_manifest_sha256": sha256_file(
            sweep_path
        ),
        "historical_exclusion_ledger_path": str(
            ledger_path.relative_to(root)
        ),
        "historical_exclusion_ledger_sha256": sha256_file(
            ledger_path
        ),
        "historical_canonical_identity_count": len(
            ledger.canonical_ids
        ),
        "search_budget": make_search_budget().model_dump(mode="json"),
        "target_count_policy": (
            make_target_count_policy().model_dump(mode="json")
        ),
        "fresh_c_stage_activated": False,
        "live_discovery_ready": True,
        "live_discovery_authorized": False,
        "live_discovery_started": False,
        "live_selection_started": False,
        "live_acquisition_started": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_lock": 0,
        "llm_calls_during_lock": 0,
        "automatic_next_stage_authorized": False,
        "stop": True,
        "critical_component_sha256": hashes,
    }
    identity_sha = sha256_json(lock_body)
    lock_body["lock_id"] = (
        "sers_fresh_c_activation_readiness_lock_v1:"
        + identity_sha[:20]
    )
    lock_body["lock_sha256"] = _payload_sha(
        lock_body,
        "lock_sha256",
    )
    lock = ActivationReadinessLock.model_validate(lock_body)

    lock_path = output / "activation_readiness_lock.json"
    ready_path = output / "READINESS_LOCKED.json"
    for path in (attestation_path, lock_path, ready_path):
        if path.exists():
            raise FileExistsError(
                f"C0.1B readiness artifact already exists: {path}"
            )

    _atomic_json(attestation_path, attestation)
    _atomic_json(lock_path, lock.model_dump(mode="json"))
    _atomic_json(
        ready_path,
        {
            "schema_version": (
                "sers-fresh-c-activation-readiness-ready-v1"
            ),
            "lock_id": lock.lock_id,
            "lock_sha256": lock.lock_sha256,
            "historical_exclusion_ledger_sha256": (
                lock.historical_exclusion_ledger_sha256
            ),
            "target_acquired_papers": 25,
            "results_per_query": 100,
            "max_raw_metadata_rows": 800,
            "live_discovery_ready": True,
            "live_discovery_authorized": False,
            "fresh_reserve_c_consumed": False,
            "automatic_next_stage_authorized": False,
            "stop": True,
        },
    )

    print("Fresh-C C0.1B activation readiness freeze")
    print(f"Lock ID: {lock.lock_id}")
    print(f"Lock SHA256: {lock.lock_sha256}")
    print(f"Source commit: {source_commit}")
    print(
        "Historical canonical identities: "
        f"{lock.historical_canonical_identity_count}"
    )
    print("Providers: semantic_scholar,crossref")
    print("Broad queries: 4")
    print("Results per query/provider: 100")
    print("Provider-query executions: 8")
    print("Maximum raw metadata rows: 800")
    print("Fresh-C target acquired papers: 25")
    print("Live discovery ready: True")
    print("Live discovery authorized: False")
    print("Live discovery started: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during lock: 0")
    print("LLM calls during lock: 0")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
