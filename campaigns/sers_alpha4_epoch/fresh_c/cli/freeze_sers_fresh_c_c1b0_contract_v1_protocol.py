from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b0_contract_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    sha256_file,
    validate_protocol,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1b0_contract_v1.py",
    "dac_her/sers_fresh_c_c1b0_contract_v1_protocol.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b0_contract_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b0_contract_v1_protocol.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b0_contract_v1_protocol_freeze.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_c1b0_contract_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b0_contract_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b0_contract_v1_result.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b0_contract_v1_result_freeze.py",
    "tests/test_sers_fresh_c_c1b0_contract_v1.py",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse C1B.0 protocol freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C1B.0 protocol freeze.")

    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    source_commit = _git(root, "rev-parse", "HEAD")

    component_hashes = {}
    for relative in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root
        )
        expected = hashlib.sha256(committed).hexdigest()
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"C1B.0 component drifted: {relative}")
        component_hashes[relative] = expected

    body = {
        "schema_version": "sers-fresh-c-c1b0-contract-protocol-freeze-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "source_code_commit": source_commit,
        "critical_component_sha256": component_hashes,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "c1b0_audit_ready": True,
        "c1b0_audit_authorized": False,
        "c1b1_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b0_contract_protocol_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)

    output = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1B.0 protocol freeze directory exists.")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "c1b0_audit_ready": True,
        "c1b0_audit_authorized": False,
        "c1b1_authorized": False,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "stop": True,
    })
    print("Fresh-C C1B.0 input-contract protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print("Fresh-C scientific text semantic read performed: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("C1B.1 authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
