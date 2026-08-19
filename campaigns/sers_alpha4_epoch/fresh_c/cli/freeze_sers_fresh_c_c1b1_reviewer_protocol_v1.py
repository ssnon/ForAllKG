import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    validate_model_env,
    validate_protocol,
    validate_source_c1b0_freeze,
)

CRITICAL_COMPONENTS = (
    "campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1b1_reviewer_contract_v1.py",
    "dac_her/sers_fresh_c_c1b1_reviewer_protocol_v1.json",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_reviewer_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b1_reviewer_protocol_v1.py",
    "campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b1_reviewer_protocol_freeze_v1.py",
    "tests/test_sers_fresh_c_c1b1_reviewer_contract_v1.py",
)

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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
        raise RuntimeError("Tracked worktree dirty; refuse C1B.1 protocol freeze.")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C1B.1 protocol freeze.")

    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_source_c1b0_freeze(root)
    model = validate_model_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must exist before reviewer protocol freeze")

    source_commit = _git(root, "rev-parse", "HEAD")
    component_hashes = {}
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        committed_sha = hashlib.sha256(committed).hexdigest()
        if _file_sha(root / rel) != committed_sha:
            raise RuntimeError(f"C1B.1 component drifted: {rel}")
        component_hashes[rel] = committed_sha

    openai_version = importlib.metadata.version("openai")
    pydantic_version = importlib.metadata.version("pydantic")
    body = {
        "schema_version": "sers-fresh-c-c1b1-reviewer-protocol-freeze-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "source_c1b0_result_freeze_id": p["source_c1b0_result_freeze_id"],
        "source_c1b0_result_freeze_sha256": p["source_c1b0_result_freeze_sha256"],
        "source_code_commit": source_commit,
        "critical_component_sha256": component_hashes,
        "reviewer_backend": p["reviewer_backend"],
        "reviewer_model": model,
        "reviewer_model_env": p["reviewer_model_env"],
        "api_key_env": p["api_key_env"],
        "api_key_present_at_freeze": True,
        "openai_package_version": openai_version,
        "pydantic_package_version": pydantic_version,
        "temperature": p["temperature"],
        "paper_review_max_tokens": p["paper_review_max_tokens"],
        "final_adjudication_max_tokens": p["final_adjudication_max_tokens"],
        "paper_review_system_prompt_sha256": p["paper_review_system_prompt_sha256"],
        "final_adjudicator_system_prompt_sha256": p["final_adjudicator_system_prompt_sha256"],
        "paper_review_schema_sha256": p["paper_review_schema_sha256"],
        "final_adjudication_schema_sha256": p["final_adjudication_schema_sha256"],
        "paper_review_order": p["paper_review_order"],
        "maximum_llm_calls_future_c1b2": p["maximum_llm_calls"],
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "c1b2_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b1_reviewer_protocol_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)

    output = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if output.exists():
        raise FileExistsError("C1B.1 reviewer protocol freeze directory exists")
    _atomic(output / "freeze_manifest.json", body)
    _atomic(output / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "reviewer_model": model,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "c1b2_authorized": False,
        "stop": True,
    })
    print("Fresh-C C1B.1 scientific-reviewer protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Reviewer backend: {body['reviewer_backend']}")
    print(f"Reviewer model: {model}")
    print(f"openai package: {openai_version}")
    print(f"pydantic package: {pydantic_version}")
    print("API key present: True")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("C1B.2 authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
