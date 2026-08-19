import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    load_object,
    validate_parent_freeze,
    validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_c1b1_r1_transport_protocol_v1 import CRITICAL_COMPONENTS

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_parent_freeze(root)
    m = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(m)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Transport protocol freeze SHA drifted")
    if m["protocol_id"] != p["protocol_id"] or m["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Transport protocol binding drifted")
    if importlib.metadata.version("openai") != m["openai_package_version"]:
        raise ValueError("openai package drifted")
    if importlib.metadata.version("pydantic") != m["pydantic_package_version"]:
        raise ValueError("pydantic package drifted")
    source_commit = m["source_code_commit"]
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if m["critical_component_sha256"].get(rel) != sha:
            raise ValueError(f"Frozen component mismatch: {rel}")
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Current component drifted: {rel}")
    if ready["freeze_id"] != m["freeze_id"] or ready["manifest_sha256"] != stored:
        raise ValueError("Transport FREEZE_READY mismatch")
    for key in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "qualification_authorized",
        "c1b2_authorized",
    ):
        if m.get(key) is not False:
            raise ValueError(f"Safety field drifted: {key}")
    if m["network_calls_during_freeze"] != 0 or m["llm_calls_during_freeze"] != 0:
        raise ValueError("Transport freeze unexpectedly used network/LLM")
    if m["stop"] is not True:
        raise ValueError("STOP drifted")
    print("Fresh-C C1B.1-R1 transport protocol freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored}")
    print(f"Corrected transport: {m['corrected_transport_semantics']}")
    print(f"Base URL: {m['base_url']}")
    print(f"Reviewer model: {m['reviewer_model']}")
    print("Fresh-C scientific text read: False")
    print("Scientific adjudication: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("C1B.2 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
