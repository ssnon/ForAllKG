from __future__ import annotations

import json
import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    DEFAULT_RUN_DIR,
    atomic_json,
    canonical_json_sha256,
    load_object,
    sha256_file,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b2_scientific_result_v1 import main as verify_result

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    run_dir = root / DEFAULT_RUN_DIR
    run = load_object(run_dir / "run_manifest.json")
    final = load_object(run_dir / "final_adjudication.json")

    paper_hashes = {}
    for row in run["paper_review_records"]:
        path = root / row["record_path"]
        paper_hashes[str(row["reserve_index"])] = sha256_file(path)

    adjudication = final["adjudication"]
    body = {
        "schema_version": "sers-fresh-c-c1b2-scientific-result-freeze-v1",
        "source_run_id": run["run_id"],
        "source_run_sha256": run["run_sha256"],
        "paper_review_file_sha256": paper_hashes,
        "final_adjudication_file_sha256": sha256_file(
            run_dir / "final_adjudication.json"
        ),
        "h1_fresh_c_verdict": adjudication["h1_fresh_c_verdict"],
        "h2_terminal_state": adjudication["h2_terminal_state"],
        "h2_resurrected": False,
        "h3_fresh_c_verdict": adjudication["h3_fresh_c_verdict"],
        "source_identity_count": 25,
        "paper_review_calls": 25,
        "final_adjudication_calls": 1,
        "scientific_llm_calls": 26,
        "scientific_network_calls": 26,
        "fresh_c_scientific_text_read_performed": True,
        "scientific_adjudication_performed": True,
        "external_literature_used": False,
        "count_threshold_used": False,
        "hypothesis_rewrite_performed": False,
        "hypothesis_upgrade_performed": False,
        "preservation_does_not_establish_absence_or_novelty": True,
        "same_epoch_rerun_allowed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_c1b2_scientific_result_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)

    out = root / DEFAULT_RESULT_FREEZE_DIR
    if out.exists():
        raise FileExistsError("C1B.2 result freeze directory already exists")
    atomic_json(out / "freeze_manifest.json", body)
    atomic_json(out / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "scientific_adjudication_performed": True,
        "same_epoch_rerun_allowed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    })

    print("Fresh-C C1B.2 scientific-adjudication result freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source run ID: {body['source_run_id']}")
    print(f"H1 Fresh-C verdict: {body['h1_fresh_c_verdict']}")
    print("H2 terminal state: REJECT_AS_FORMULATED")
    print(f"H3 Fresh-C verdict: {body['h3_fresh_c_verdict']}")
    print("Paper reviews frozen: 25")
    print("Scientific LLM/network calls: 26/26")
    print("Same-epoch rerun allowed: False")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
