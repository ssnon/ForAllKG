from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_activation import (
    ActivationReadinessLock,
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
from campaigns.sers_alpha4_epoch.fresh_c.cli.build_sers_fresh_c_historical_exclusion_ledger_v1 import (
    DEFAULT_OUTPUT_DIR,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_activation_readiness_v1 import (
    CRITICAL_COMPONENTS,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Fresh-C C0.1B activation readiness. "
            "This does not authorize or run live discovery."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _read_json(path: Path) -> dict:
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
    root = Path(
        _git(Path.cwd(), "rev-parse", "--show-toplevel")
    )
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )

    subprocess.run(
        [
            "python",
            "-m",
            "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_i0_integrated_orchestration_freeze_v1",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_acquisition_protocol_freeze_v1",
        ],
        cwd=root,
        check=True,
    )

    sweep_path = output / "historical_identity_sweep_manifest.json"
    ledger_path = output / "historical_exclusion_ledger.json"
    _, ledger = validate_historical_sweep_artifacts(
        root=root,
        manifest_path=sweep_path,
        ledger_path=ledger_path,
    )

    attestation = _read_json(
        output / "C0_0_DISPOSITION_ATTESTATION.json"
    )
    if attestation.get("operator_attested") is not True:
        raise ValueError("C0.0 attestation is not affirmative.")
    if attestation.get("scientific_values_read_for_attestation") is not False:
        raise ValueError("C0.0 attestation scientific-read flag drifted.")
    if attestation.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("C0.0 attestation consumes Fresh C.")
    if attestation.get("attestation_sha256") != _payload_sha(
        attestation,
        "attestation_sha256",
    ):
        raise ValueError("C0.0 attestation SHA drifted.")

    lock_path = output / "activation_readiness_lock.json"
    lock_raw = _read_json(lock_path)
    lock = ActivationReadinessLock.model_validate(lock_raw)
    if lock.lock_sha256 != _payload_sha(lock_raw, "lock_sha256"):
        raise ValueError("Activation readiness lock SHA drifted.")

    if lock.i0_freeze_id != EXPECTED_I0_FREEZE_ID:
        raise ValueError("I0 freeze ID mismatch.")
    if lock.i0_manifest_sha256 != EXPECTED_I0_MANIFEST_SHA256:
        raise ValueError("I0 freeze SHA mismatch.")
    if lock.c0_1a_protocol_id != EXPECTED_C01A_PROTOCOL_ID:
        raise ValueError("C0.1A protocol ID mismatch.")
    if lock.c0_1a_protocol_sha256 != EXPECTED_C01A_PROTOCOL_SHA256:
        raise ValueError("C0.1A protocol SHA mismatch.")
    if lock.c0_1a_freeze_id != EXPECTED_C01A_FREEZE_ID:
        raise ValueError("C0.1A freeze ID mismatch.")
    if (
        lock.c0_1a_freeze_manifest_sha256
        != EXPECTED_C01A_FREEZE_MANIFEST_SHA256
    ):
        raise ValueError("C0.1A freeze SHA mismatch.")
    if lock.c0_1a_source_commit != EXPECTED_C01A_SOURCE_COMMIT:
        raise ValueError("C0.1A source commit mismatch.")
    if lock.c0_1a_freeze_commit != EXPECTED_C01A_FREEZE_COMMIT:
        raise ValueError("C0.1A freeze commit mismatch.")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_C01A_FREEZE_COMMIT,
            lock.source_commit,
        ],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    if sha256_file(sweep_path) != lock.historical_sweep_manifest_sha256:
        raise ValueError("Historical sweep file SHA mismatch.")
    if sha256_file(ledger_path) != lock.historical_exclusion_ledger_sha256:
        raise ValueError("Historical ledger file SHA mismatch.")
    if len(ledger.canonical_ids) != lock.historical_canonical_identity_count:
        raise ValueError("Historical identity count mismatch.")

    if lock.search_budget != make_search_budget():
        raise ValueError("Search budget drifted.")
    if lock.target_count_policy != make_target_count_policy():
        raise ValueError("Target count policy drifted.")

    source_commit = lock.source_commit
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if set(lock.critical_component_sha256) != set(CRITICAL_COMPONENTS):
        raise ValueError("C0.1B critical component set drifted.")
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=root,
        )
        expected = hashlib.sha256(committed).hexdigest()
        if lock.critical_component_sha256[relative] != expected:
            raise ValueError(
                f"C0.1B source component hash mismatch: {relative}"
            )
        current = root / relative
        if not current.exists():
            raise FileNotFoundError(current)
        if sha256_file(current) != expected:
            raise ValueError(
                f"C0.1B current critical component drifted: {relative}"
            )

    ready = _read_json(output / "READINESS_LOCKED.json")
    if ready.get("lock_id") != lock.lock_id:
        raise ValueError("READINESS lock ID mismatch.")
    if ready.get("lock_sha256") != lock.lock_sha256:
        raise ValueError("READINESS lock SHA mismatch.")
    if ready.get("target_acquired_papers") != 25:
        raise ValueError("READINESS target count drifted.")
    if ready.get("results_per_query") != 100:
        raise ValueError("READINESS search depth drifted.")
    if ready.get("max_raw_metadata_rows") != 800:
        raise ValueError("READINESS search budget drifted.")
    if ready.get("live_discovery_ready") is not True:
        raise ValueError("READINESS is not ready.")
    if ready.get("live_discovery_authorized") is not False:
        raise ValueError("READINESS unexpectedly authorizes discovery.")
    if ready.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("READINESS unexpectedly consumes Fresh C.")
    if ready.get("automatic_next_stage_authorized") is not False:
        raise ValueError("READINESS unexpectedly authorizes next stage.")
    if ready.get("stop") is not True:
        raise ValueError("READINESS STOP guard drifted.")

    print("Fresh-C C0.1B activation readiness verifier")
    print(f"Lock ID: {lock.lock_id}")
    print(f"Lock SHA256: {lock.lock_sha256}")
    print(f"Source commit: {lock.source_commit}")
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
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
