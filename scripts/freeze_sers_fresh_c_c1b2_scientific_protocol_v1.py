from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_SCHEMA_QUALIFICATION_DIR,
    atomic_json,
    build_target_boundaries,
    canonical_json_sha256,
    load_object,
    schema_qualification_valid,
    sha256_file,
    validate_corpus_metadata,
    validate_frozen_lineage,
    validate_protocol,
)

CRITICAL_COMPONENTS = (
    "dac_her/fresh_c_c1b2_scientific_adjudication_v1.py",
    "dac_her/sers_fresh_c_c1b2_scientific_protocol_v1.json",
    "scripts/verify_sers_fresh_c_c1b2_scientific_protocol_v1.py",
    "scripts/run_sers_fresh_c_c1b2_scientific_adjudication_v1.py",
    "scripts/freeze_sers_fresh_c_c1b2_scientific_protocol_v1.py",
    "scripts/verify_sers_fresh_c_c1b2_scientific_protocol_freeze_v1.py",
    "scripts/verify_sers_fresh_c_c1b2_scientific_result_v1.py",
    "scripts/freeze_sers_fresh_c_c1b2_scientific_result_v1.py",
    "scripts/verify_sers_fresh_c_c1b2_scientific_result_freeze_v1.py",
    "tests/test_sers_fresh_c_c1b2_scientific_adjudication_v1.py",
)

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse C1B.2 protocol freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse C1B.2 protocol freeze")

    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    lineage = validate_frozen_lineage(root)
    targets = build_target_boundaries(lineage["r2_report"])
    # Hash-only metadata validation; do not parse scientific page text.
    records = validate_corpus_metadata(root, parse_pages=False)

    qpath = root / DEFAULT_SCHEMA_QUALIFICATION_DIR / "qualification_result.json"
    qualification = load_object(qpath)
    schema_qualification_valid(qualification)
    if qualification.get("protocol_id") != p["protocol_id"]:
        raise ValueError("Exact-schema qualification protocol ID mismatch")
    if qualification.get("protocol_sha256") != p["protocol_sha256"]:
        raise ValueError("Exact-schema qualification protocol SHA mismatch")

    source_commit = _git(root, "rev-parse", "HEAD")
    component_hashes = {}
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"C1B.2 component drifted: {rel}")
        component_hashes[rel] = sha

    body = {
        "schema_version": "sers-fresh-c-c1b2-scientific-protocol-freeze-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "source_code_commit": source_commit,
        "critical_component_sha256": component_hashes,
        "exact_schema_qualification_id": qualification["qualification_id"],
        "exact_schema_qualification_sha256": qualification["qualification_sha256"],
        "exact_schema_qualification_file_sha256": sha256_file(qpath),
        "exact_schema_qualification_network_calls": 2,
        "exact_schema_qualification_llm_calls": 2,
        "target_boundaries_sha256": canonical_json_sha256(targets),
        "source_identity_count": len(records),
        "reviewer_model": p["reviewer_model"],
        "base_url": p["base_url"],
        "openai_package_version": importlib.metadata.version("openai"),
        "pydantic_package_version": importlib.metadata.version("pydantic"),
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "live_execution_ready": True,
        "live_execution_authorized": False,
        "automatic_post_c1b2_transition_allowed": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b2_scientific_protocol_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)

    out = root / DEFAULT_PROTOCOL_FREEZE_DIR
    if out.exists():
        raise FileExistsError("C1B.2 protocol freeze directory already exists")
    atomic_json(out / "freeze_manifest.json", body)
    atomic_json(out / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "live_execution_ready": True,
        "live_execution_authorized": False,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "stop": True,
    })

    print("Fresh-C C1B.2 scientific-adjudication protocol freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source code commit: {source_commit}")
    print(f"Exact-schema qualification ID: {qualification['qualification_id']}")
    print("Exact paper/final schemas qualified: True/True")
    print("Exact frozen scientific targets: 2")
    print("Exact frozen Fresh-C papers: 25")
    print(f"Reviewer model: {p['reviewer_model']}")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls during freeze: 0")
    print("LLM calls during freeze: 0")
    print("Live execution ready: True")
    print("Live execution authorized: False")
    print("Automatic post-C1B.2 transition: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
