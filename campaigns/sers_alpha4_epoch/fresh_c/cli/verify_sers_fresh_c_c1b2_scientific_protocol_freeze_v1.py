from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_SCHEMA_QUALIFICATION_DIR,
    build_target_boundaries,
    canonical_json_sha256,
    load_object,
    schema_qualification_valid,
    sha256_file,
    validate_corpus_metadata,
    validate_frozen_lineage,
    validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_c1b2_scientific_protocol_v1 import CRITICAL_COMPONENTS

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    lineage = validate_frozen_lineage(root)
    targets = build_target_boundaries(lineage["r2_report"])
    records = validate_corpus_metadata(root, parse_pages=False)

    qpath = root / DEFAULT_SCHEMA_QUALIFICATION_DIR / "qualification_result.json"
    qualification = load_object(qpath)
    schema_qualification_valid(qualification)

    m = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(m)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.2 freeze manifest SHA drifted")
    if m["protocol_id"] != p["protocol_id"] or m["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("C1B.2 protocol binding drifted")
    if m["exact_schema_qualification_id"] != qualification["qualification_id"]:
        raise ValueError("C1B.2 qualification ID binding drifted")
    if m["exact_schema_qualification_sha256"] != qualification["qualification_sha256"]:
        raise ValueError("C1B.2 qualification SHA binding drifted")
    if m["exact_schema_qualification_file_sha256"] != sha256_file(qpath):
        raise ValueError("C1B.2 qualification file hash drifted")
    if m["target_boundaries_sha256"] != canonical_json_sha256(targets):
        raise ValueError("C1B.2 target-boundary hash drifted")
    if m["transport_schema_adapter_id"] != p["transport_schema_adapter_id"]:
        raise ValueError("C1B.2 transport schema adapter drifted")
    if m["paper_review_transport_schema_sha256"] != p["paper_review_transport_schema_sha256"]:
        raise ValueError("C1B.2 paper transport schema SHA drifted")
    if m["final_adjudication_transport_schema_sha256"] != p["final_adjudication_transport_schema_sha256"]:
        raise ValueError("C1B.2 final transport schema SHA drifted")
    if m["source_identity_count"] != len(records) or len(records) != 25:
        raise ValueError("C1B.2 source identity count drifted")
    if importlib.metadata.version("openai") != m["openai_package_version"]:
        raise ValueError("C1B.2 openai package drifted")
    if importlib.metadata.version("pydantic") != m["pydantic_package_version"]:
        raise ValueError("C1B.2 pydantic package drifted")

    source_commit = m["source_code_commit"]
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if m["critical_component_sha256"].get(rel) != sha:
            raise ValueError(f"C1B.2 frozen component mismatch: {rel}")
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f"C1B.2 current component drifted: {rel}")

    if ready["freeze_id"] != m["freeze_id"] or ready["manifest_sha256"] != stored:
        raise ValueError("C1B.2 FREEZE_READY mismatch")
    for key in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "live_execution_authorized",
        "automatic_post_c1b2_transition_allowed",
    ):
        if m.get(key) is not False:
            raise ValueError(f"C1B.2 safety field drifted: {key}")
    if m.get("live_execution_ready") is not True:
        raise ValueError("C1B.2 live readiness drifted")
    if m.get("network_calls_during_freeze") != 0 or m.get("llm_calls_during_freeze") != 0:
        raise ValueError("C1B.2 freeze unexpectedly used network/LLM")
    if m.get("stop") is not True:
        raise ValueError("C1B.2 STOP drifted")

    print("Fresh-C C1B.2 scientific-adjudication protocol freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored}")
    print(f"Source code commit: {source_commit}")
    print("Raw C1B.1 reviewer schemas unchanged: CURRENT")
    print("OpenAI-strict paper/final transport schemas: CURRENT")
    print("Target-boundary hash: CURRENT")
    print("Exact 25 source hashes: CURRENT")
    print(f"Reviewer model: {m['reviewer_model']}")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("Live execution ready: True")
    print("Live execution authorized: False")
    print("Automatic post-C1B.2 transition: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
