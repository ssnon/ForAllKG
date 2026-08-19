from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V1_FAILED_ATTEMPT_ID,
    V1_FAILED_PATH,
    V1_STARTED_PATH,
    load_and_validate_protocol,
    validate_v1_failed_epoch,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_live_discovery_recovery_v2_protocol import (
    verify,
)


CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery_recovery_v2.py",
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_protocol.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_live_discovery_recovery_v2_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_protocol_freeze.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_live_discovery_recovery_v2.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_live_discovery_recovery_v2_result.py",
    "tests/test_sers_fresh_c_live_discovery_recovery_v2.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_live_discovery.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_acquisition.py",
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_activation.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze C0.1C-v2 transport-recovery execution components. "
            "The v1 failed epoch must already be tracked and immutable."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FREEZE_DIR,
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


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _tracked_and_identical(root: Path, relative: Path, commit: str) -> str:
    rel = relative.as_posix()
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:{rel}"],
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Required recovery-parent artifact is not tracked in HEAD: {rel}"
        ) from exc
    current = root / relative
    if not current.exists():
        raise FileNotFoundError(current)
    committed_sha = hashlib.sha256(committed).hexdigest()
    current_sha = sha256_file(current)
    if committed_sha != current_sha:
        raise RuntimeError(
            f"Recovery-parent artifact differs from HEAD: {rel}"
        )
    return current_sha


def main() -> int:
    args = parse_args()
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else root / args.protocol
    )
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )

    if subprocess.run(
        ["git", "diff", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError("Tracked worktree dirty; refuse recovery-v2 freeze.")
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise RuntimeError("Index dirty; refuse recovery-v2 freeze.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_fresh_c_live_discovery_protocol_freeze_v1",
        ],
        cwd=root,
        check=True,
    )
    validate_v1_failed_epoch(root)
    verify(protocol_path)
    protocol = load_and_validate_protocol(protocol_path)

    source_commit = _git(root, "rev-parse", "HEAD")
    started_sha = _tracked_and_identical(
        root,
        V1_STARTED_PATH,
        source_commit,
    )
    failed_sha = _tracked_and_identical(
        root,
        V1_FAILED_PATH,
        source_commit,
    )

    component_sha256: dict[str, str] = {}
    for relative in CRITICAL_COMPONENTS:
        current = root / relative
        if not current.exists():
            raise FileNotFoundError(current)
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=root,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Recovery-v2 critical component not tracked: {relative}"
            ) from exc
        committed_sha = hashlib.sha256(committed).hexdigest()
        current_sha = sha256_file(current)
        if committed_sha != current_sha:
            raise RuntimeError(
                f"Recovery-v2 component differs from HEAD: {relative}"
            )
        component_sha256[relative] = current_sha

    body: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-live-discovery-recovery-v2-protocol-freeze-v1"
        ),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "source_code_commit": source_commit,
        "recovery_parent_attempt_id": EXPECTED_V1_FAILED_ATTEMPT_ID,
        "recovery_parent_started_file_sha256": started_sha,
        "recovery_parent_failed_file_sha256": failed_sha,
        "recovery_parent_failed_epoch_preserved": True,
        "critical_component_sha256": component_sha256,
        "search_queries_changed_from_v1": False,
        "provider_set_changed_from_v1": False,
        "search_depth_changed_from_v1": False,
        "historical_ledger_changed_from_v1": False,
        "target_count_changed_from_v1": False,
        "blind_ordering_changed_from_v1": False,
        "scientific_selection_semantics_changed_from_v1": False,
        "recovery_live_discovery_ready": True,
        "recovery_live_discovery_authorized": False,
        "recovery_live_discovery_started": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }
    identity_sha = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_live_discovery_recovery_v2_protocol_freeze_v1:"
        + identity_sha[:20]
    )
    body["manifest_sha256"] = _payload_sha(
        body,
        "manifest_sha256",
    )

    manifest_path = output / "freeze_manifest.json"
    ready_path = output / "FREEZE_READY.json"
    if manifest_path.exists() or ready_path.exists():
        raise FileExistsError(
            "Recovery-v2 freeze artifacts already exist; refuse overwrite."
        )
    _atomic_json(manifest_path, body)
    _atomic_json(
        ready_path,
        {
            "schema_version": (
                "sers-fresh-c-live-discovery-recovery-v2-ready-v1"
            ),
            "freeze_id": body["freeze_id"],
            "manifest_sha256": body["manifest_sha256"],
            "recovery_parent_attempt_id": EXPECTED_V1_FAILED_ATTEMPT_ID,
            "recovery_live_discovery_ready": True,
            "recovery_live_discovery_authorized": False,
            "recovery_live_discovery_started": False,
            "fresh_reserve_c_consumed": False,
            "automatic_c0_1d_transition_authorized": False,
            "stop": True,
        },
    )

    print("Fresh-C C0.1C-v2 recovery protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(
        "Recovery parent attempt: "
        f"{EXPECTED_V1_FAILED_ATTEMPT_ID}"
    )
    print("Recovery parent failed epoch preserved: True")
    print("Scientific/search semantics changed: False")
    print("Recovery live discovery ready: True")
    print("Recovery live discovery authorized: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
