from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dac_her.fresh_c_acquisition import sha256_file, sha256_json
from dac_her.fresh_c_live_discovery_recovery_v2_4 import (
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    V22_DIAGNOSTICS_PATH,
    V22_FAILED_PATH,
    V22_STARTED_PATH,
    load_and_validate_protocol,
    validate_v22_failure,
    validate_v23_frozen_unexecuted,
)

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_live_discovery_recovery_v2_4.py",
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_4_protocol.json",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_4_protocol.py",
    "scripts/freeze_sers_fresh_c_live_discovery_recovery_v2_4_protocol.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_4_protocol_freeze.py",
    "scripts/run_sers_fresh_c_live_discovery_recovery_v2_4.py",
    "scripts/verify_sers_fresh_c_live_discovery_recovery_v2_4_result.py",
    "tests/test_sers_fresh_c_live_discovery_recovery_v2_4.py",
    "dac_her/fresh_c_live_discovery_recovery_v2_3.py",
    "dac_her/fresh_c_live_discovery_recovery_v2_2.py",
    "dac_her/fresh_c_live_discovery.py",
    "dac_her/fresh_c_acquisition.py",
    "dac_her/fresh_c_activation.py",
    "dac_her/literature_catalog.py",
    "dac_her/literature_catalog_contracts.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _read(path: Path):
    v = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v, dict):
        raise ValueError(f"Expected object: {path}")
    return v


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _payload_sha(payload, field):
    v = dict(payload)
    v.pop(field, None)
    return sha256_json(v)


def _tracked_sha(root: Path, relative: Path, commit: str) -> str:
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(root / relative) != expected:
        raise RuntimeError(f"Frozen parent artifact drifted: {relative}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    args = parser.parse_args()

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse v2.4 freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse v2.4 freeze.")

    subprocess.run(
        [sys.executable, "-m",
         "scripts.verify_sers_fresh_c_live_discovery_recovery_v2_3_protocol_freeze"],
        cwd=root, check=True,
    )
    v22 = validate_v22_failure(root)
    v23 = validate_v23_frozen_unexecuted(root)
    p = load_and_validate_protocol(
        args.protocol if args.protocol.is_absolute() else root / args.protocol
    )

    source_commit = _git(root, "rev-parse", "HEAD")
    parent_v22_hashes = {
        str(V22_STARTED_PATH): _tracked_sha(root, V22_STARTED_PATH, source_commit),
        str(V22_FAILED_PATH): _tracked_sha(root, V22_FAILED_PATH, source_commit),
        str(V22_DIAGNOSTICS_PATH): _tracked_sha(root, V22_DIAGNOSTICS_PATH, source_commit),
    }

    from dac_her.fresh_c_live_discovery_recovery_v2_3 import (
        DEFAULT_FREEZE_DIR as V23_FREEZE_DIR,
    )
    v23_manifest_rel = V23_FREEZE_DIR / "freeze_manifest.json"
    v23_ready_rel = V23_FREEZE_DIR / "FREEZE_READY.json"
    parent_v23_hashes = {
        str(v23_manifest_rel): _tracked_sha(root, v23_manifest_rel, source_commit),
        str(v23_ready_rel): _tracked_sha(root, v23_ready_rel, source_commit),
    }

    hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"v2.4 component drifted: {relative}")
        hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-openalex-crossref-substitution-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "v22_failed_attempt_id": p.v22_failed_attempt_id,
        "v22_artifact_sha256": parent_v22_hashes,
        "v22_semantic_scholar_429_confirmed": True,
        "v22_crossref_4_of_4_success": True,
        "v23_freeze_id": v23["manifest"]["freeze_id"],
        "v23_freeze_manifest_sha256": v23["manifest"]["manifest_sha256"],
        "v23_artifact_sha256": parent_v23_hashes,
        "v23_frozen_but_unexecuted": True,
        "provider_substitution": {
            "from": "semantic_scholar",
            "to": "openalex",
            "reason": "transport_availability_only_after_repeated_http_429",
        },
        "provider_universe_changed": True,
        "frozen_queries_changed": False,
        "historical_ledger_changed": False,
        "target_count_changed": False,
        "blind_ordering_changed": False,
        "hypothesis_aware_selection_added": False,
        "scientific_selection_semantics_changed": False,
        "openalex_api_key_required": True,
        "credential_value_persisted": False,
        "critical_component_sha256": hashes,
        "recovery_live_discovery_ready": True,
        "recovery_live_discovery_authorized": False,
        "recovery_live_discovery_started": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_openalex_crossref_substitution_freeze_v1:"
        + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if output.exists():
        raise FileExistsError("v2.4 freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "v23_frozen_but_unexecuted": True,
        "provider_universe_changed": True,
        "recovery_live_discovery_ready": True,
        "recovery_live_discovery_authorized": False,
        "fresh_reserve_c_consumed": False,
        "stop": True,
    })

    print("Fresh-C C0.1C-v2.4 OpenAlex+Crossref substitution freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("v2.2 Semantic Scholar HTTP 429 confirmed: True")
    print("v2.2 Crossref 4/4 successful: True")
    print("v2.3 frozen but unexecuted: True")
    print("Provider substitution: semantic_scholar -> openalex")
    print("Provider universe changed: True")
    print("Frozen queries changed: False")
    print("Blind ordering changed: False")
    print("Scientific selection semantics changed: False")
    print("OpenAlex API key required: True")
    print("Network calls during freeze: 0")
    print("Fresh Reserve C consumed: False")
    print("STOP: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
