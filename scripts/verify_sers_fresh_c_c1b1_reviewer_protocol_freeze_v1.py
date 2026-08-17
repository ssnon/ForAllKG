import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    load_object,
    validate_model_env,
    validate_protocol,
    validate_source_c1b0_freeze,
)
from scripts.freeze_sers_fresh_c_c1b1_reviewer_protocol_v1 import CRITICAL_COMPONENTS

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_source_c1b0_freeze(root)
    m = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(m)
    stored_sha = tmp.pop("manifest_sha256")
    if stored_sha != canonical_json_sha256(tmp):
        raise ValueError("C1B.1 freeze manifest SHA drifted")
    if m["protocol_id"] != p["protocol_id"] or m["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("C1B.1 protocol binding drifted")
    if validate_model_env() != m["reviewer_model"]:
        raise ValueError("C1B.1 reviewer model env no longer matches freeze")
    if importlib.metadata.version("openai") != m["openai_package_version"]:
        raise ValueError("C1B.1 openai package drifted")
    if importlib.metadata.version("pydantic") != m["pydantic_package_version"]:
        raise ValueError("C1B.1 pydantic package drifted")

    source_commit = m["source_code_commit"]
    hashes = m["critical_component_sha256"]
    if set(hashes) != set(CRITICAL_COMPONENTS):
        raise ValueError("C1B.1 component set drifted")
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if hashes[rel] != sha:
            raise ValueError(f"C1B.1 frozen component mismatch: {rel}")
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f"C1B.1 current component drifted: {rel}")

    for field in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "c1b2_authorized",
    ):
        if m.get(field) is not False:
            raise ValueError(f"C1B.1 safety field drifted: {field}")
    if m.get("network_calls_during_freeze") != 0 or m.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1B.1 freeze used network or LLM")
    if m.get("stop") is not True:
        raise ValueError("C1B.1 STOP drifted")
    if ready["freeze_id"] != m["freeze_id"] or ready["manifest_sha256"] != stored_sha:
        raise ValueError("C1B.1 FREEZE_READY mismatch")

    print("Fresh-C C1B.1 scientific-reviewer protocol freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored_sha}")
    print(f"Source code commit: {source_commit}")
    print(f"Reviewer model: {m['reviewer_model']}")
    print("Prompt/schema hashes: CURRENT")
    print("Exact paper review order 1..25: CURRENT")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("C1B.2 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
