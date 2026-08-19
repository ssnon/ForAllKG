from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_file, sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1a_r1_recovery_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    FAILED_AUDIT,
    FAILED_CONSUMPTION_MARKER,
    FAILED_MARKER,
    load_and_validate_protocol,
    mutool_fingerprint,
    validate_failed_c1a_state,
    validate_pdfminer_version,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1a_r1_recovery_v1.py",
    "dac_her/sers_fresh_c_c1a_r1_recovery_v1_protocol.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_r1_recovery_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1a_r1_recovery_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_r1_recovery_v1_protocol_freeze.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_c1a_r1_recovery_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_r1_recovery_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1a_r1_recovery_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1a_r1_recovery_v1_result_freeze.py",
    "tests/test_sers_fresh_c_c1a_r1_recovery_v1.py",
    "requirements.txt",
)

PARENT_AUDIT_ARTIFACTS = (
    FAILED_CONSUMPTION_MARKER,
    FAILED_MARKER,
    FAILED_AUDIT,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _payload_sha(payload, field):
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _tracked_sha(root: Path, relative: Path, commit: str) -> str:
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=root
    )
    expected = hashlib.sha256(committed).hexdigest()
    if sha256_file(root / relative) != expected:
        raise RuntimeError(f"C1A-R1 parent artifact drifted: {relative}")
    return expected


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse C1A-R1 protocol freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C1A-R1 protocol freeze.")

    validate_failed_c1a_state(root)
    p = load_and_validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    pdfminer = validate_pdfminer_version()
    mutool = mutool_fingerprint()
    source_commit = _git(root, "rev-parse", "HEAD")

    parent_hashes = {
        str(relative): _tracked_sha(root, relative, source_commit)
        for relative in PARENT_AUDIT_ARTIFACTS
    }

    component_hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"C1A-R1 component drifted: {relative}")
        component_hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-c1a-r1-recovery-protocol-freeze-v1",
        "protocol_id": p.protocol_id,
        "protocol_sha256": p.protocol_sha256,
        "source_code_commit": source_commit,
        "parent_failed_epoch_artifact_sha256": parent_hashes,
        "critical_component_sha256": component_hashes,
        "source_identity_count": 25,
        "fresh_reserve_c_already_consumed": True,
        "consumption_irreversible": True,
        "prior_failed_outputs_reused": False,
        "pdfminer_six_version": pdfminer,
        "mutool_path": mutool["path"],
        "mutool_sha256": mutool["sha256"],
        "mutool_version_output": mutool["version_output"],
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "scientific_reviewer_read_performed": False,
        "scientific_adjudication_performed": False,
        "live_recovery_ready": True,
        "live_recovery_authorized": False,
        "live_recovery_started": False,
        "automatic_c1b_transition_authorized": False,
        "stop": True,
    }
    ident = sha256_json(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1a_r1_recovery_protocol_freeze_v1:" + ident[:20]
    )
    body["manifest_sha256"] = _payload_sha(body, "manifest_sha256")

    output = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1A-R1 protocol freeze directory already exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "fresh_reserve_c_already_consumed": True,
        "live_recovery_ready": True,
        "live_recovery_authorized": False,
        "scientific_adjudication_performed": False,
        "stop": True,
    })

    print("Fresh-C C1A-R1 recovery protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Source identities: exact same frozen 25")
    print("Fresh Reserve C already consumed: True")
    print(f"pdfminer.six: {pdfminer}")
    print(f"mutool SHA256: {mutool['sha256']}")
    print("mutool version captured: True")
    print("Prior failed outputs reused: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Scientific adjudication performed: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
