from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from dac_her.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    sha256_file,
)
from dac_her.fresh_c_live_discovery import DEFAULT_C01B_DIR
from dac_her.fresh_c_live_discovery_recovery_v2 import (
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    EXPECTED_V1_FAILED_ATTEMPT_ID,
    load_and_validate_protocol,
    validate_v1_failed_epoch,
)
from scripts.verify_sers_fresh_c_live_discovery_recovery_v2_protocol_freeze import (
    main as verify_recovery_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _blind_score(canonical_id: str) -> str:
    import hashlib
    return hashlib.sha256(
        (
            FRESH_C_BLIND_ORDER_NAMESPACE
            + "\0"
            + canonical_id
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_recovery_freeze()
    validate_v1_failed_epoch(root)
    protocol = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    run_dir = root / DEFAULT_RUN_DIR

    if (run_dir / "DISCOVERY_RECOVERY_FAILED.json").exists():
        raise RuntimeError(
            "Recovery-v2 failed epoch exists; success verification forbidden."
        )

    manifest = _read(run_dir / "run_manifest.json")
    diagnostics = _read(run_dir / "TRANSPORT_DIAGNOSTICS.json")
    queue = _read(run_dir / "blind_selection_queue.json")
    locators = _read(run_dir / "access_locator_manifest.json")
    complete = _read(run_dir / "DISCOVERY_RECOVERY_COMPLETE.json")

    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Recovery result protocol ID mismatch.")
    if manifest.get("recovery_parent_attempt_id") != EXPECTED_V1_FAILED_ATTEMPT_ID:
        raise ValueError("Recovery result parent attempt mismatch.")
    if manifest.get("successful_provider_query_executions") != 8:
        raise ValueError("Recovery result does not have 8/8 successes.")
    if manifest.get("fresh_identity_queue_count", 0) < 25:
        raise ValueError("Recovery result fresh queue below target 25.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("Recovery result consumed Fresh C.")
    if manifest.get("semantic_read_performed") is not False:
        raise ValueError("Recovery result performed semantic read.")
    if manifest.get("scientific_selection_semantics_changed_from_v1") is not False:
        raise ValueError("Recovery result changed selection semantics.")
    if manifest.get("automatic_c0_1d_transition_authorized") is not False:
        raise ValueError("Recovery result auto-authorized C0.1D.")

    if sha256_file(run_dir / "TRANSPORT_DIAGNOSTICS.json") != manifest.get(
        "transport_diagnostics_file_sha256"
    ):
        raise ValueError("Recovery diagnostics file SHA drifted.")
    if sha256_file(run_dir / "blind_selection_queue.json") != manifest.get(
        "blind_queue_file_sha256"
    ):
        raise ValueError("Recovery blind queue file SHA drifted.")
    if sha256_file(run_dir / "access_locator_manifest.json") != manifest.get(
        "access_locator_file_sha256"
    ):
        raise ValueError("Recovery locator file SHA drifted.")

    forbidden_diag_keys = {
        "query_text",
        "title",
        "abstract",
        "citation_count",
        "scientific_response_body",
        "response_body",
        "api_key",
        "mailto_value",
    }
    serialized_diag = json.dumps(
        diagnostics,
        ensure_ascii=False,
        sort_keys=True,
    )
    for key in forbidden_diag_keys:
        if f'"{key}"' in serialized_diag:
            raise ValueError(
                f"Recovery diagnostics contains forbidden key: {key}"
            )
    if diagnostics.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("Recovery diagnostics consumed Fresh C.")
    if diagnostics.get("semantic_read_performed") is not False:
        raise ValueError("Recovery diagnostics performed semantic read.")
    if diagnostics.get("llm_calls") != 0:
        raise ValueError("Recovery diagnostics used LLM.")

    ledger = _read(
        root / DEFAULT_C01B_DIR / "historical_exclusion_ledger.json"
    )
    historical = set(ledger.get("canonical_ids") or [])
    records = queue.get("records") or []
    if len(records) != manifest.get("fresh_identity_queue_count"):
        raise ValueError("Recovery queue count mismatch.")
    for index, row in enumerate(records, start=1):
        canonical_id = row["canonical_id"]
        if canonical_id in historical:
            raise ValueError("Historical identity leaked into Fresh-C queue.")
        if row["rank"] != index:
            raise ValueError("Recovery blind rank sequence drifted.")
        if row["score_sha256"] != _blind_score(canonical_id):
            raise ValueError("Recovery blind score drifted.")

    locator_ids = {
        row["canonical_id"]
        for row in (locators.get("records") or [])
    }
    queue_ids = {row["canonical_id"] for row in records}
    if locator_ids != queue_ids:
        raise ValueError("Recovery locator/queue identity set mismatch.")

    if complete.get("run_id") != manifest.get("run_id"):
        raise ValueError("Recovery COMPLETE run ID mismatch.")
    if complete.get("run_sha256") != manifest.get("run_sha256"):
        raise ValueError("Recovery COMPLETE run SHA mismatch.")
    if complete.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("Recovery COMPLETE consumed Fresh C.")

    print("Fresh-C C0.1C-v2 recovery result verifier")
    print(f"Run ID: {manifest['run_id']}")
    print(f"Run SHA256: {manifest['run_sha256']}")
    print(
        "Recovery parent attempt: "
        f"{EXPECTED_V1_FAILED_ATTEMPT_ID}"
    )
    print("Provider-query executions: 8/8 successful")
    print(
        "Fresh blind queue identities: "
        f"{manifest['fresh_identity_queue_count']}"
    )
    print("Scientific selection semantics changed: False")
    print("Scientific metadata persisted in diagnostics: False")
    print("Fresh Reserve C consumed: False")
    print("Semantic read performed: False")
    print("LLM calls: 0")
    print("Automatic C0.1D transition authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
