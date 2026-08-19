import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    validate_parent_freeze,
    validate_protocol,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1b1_r1_transport_qualification_v1.py",
    "dac_her/sers_fresh_c_c1b1_r1_transport_protocol_v1.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_r1_transport_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b1_r1_transport_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_r1_transport_protocol_freeze_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_c1b1_r1_transport_qualification_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_r1_transport_result_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b1_r1_transport_result_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_r1_transport_result_freeze_v1.py",
    "tests/test_sers_fresh_c_c1b1_r1_transport_qualification_v1.py",
)

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse transport protocol freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse transport protocol freeze")
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_parent_freeze(root)
    source_commit = _git(root, "rev-parse", "HEAD")
    hashes = {}
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Transport component drifted: {rel}")
        hashes[rel] = sha
    body = {
        "schema_version": "sers-fresh-c-c1b1-r1-transport-protocol-freeze-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "parent_freeze_id": p["parent_freeze_id"],
        "parent_freeze_sha256": p["parent_freeze_sha256"],
        "source_code_commit": source_commit,
        "critical_component_sha256": hashes,
        "corrected_transport_semantics": p["corrected_transport_semantics"],
        "base_url": p["base_url"],
        "reviewer_model": p["reviewer_model"],
        "openai_package_version": importlib.metadata.version("openai"),
        "pydantic_package_version": importlib.metadata.version("pydantic"),
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "qualification_authorized": False,
        "c1b2_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b1_r1_transport_protocol_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)
    output = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if output.exists():
        raise FileExistsError("Transport protocol freeze directory exists")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "qualification_authorized": False,
        "c1b2_authorized": False,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "stop": True,
    })
    print("Fresh-C C1B.1-R1 transport protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Corrected transport: {body['corrected_transport_semantics']}")
    print(f"Base URL: {body['base_url']}")
    print(f"Reviewer model: {body['reviewer_model']}")
    print("Fresh-C scientific text read: False")
    print("Scientific adjudication: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Qualification authorized: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
