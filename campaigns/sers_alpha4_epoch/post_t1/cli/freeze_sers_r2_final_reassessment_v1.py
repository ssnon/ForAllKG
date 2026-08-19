from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from campaigns.sers_alpha4_epoch.post_t1.cli.run_sers_r2_final_reassessment_v1 import (
    COMPLETE_PATH,
    REPORT_PATH,
    ROOT,
    R0_FREEZE_MANIFEST,
    R0_FREEZE_READY,
    SPEC_PATH,
    canonical,
    git_bytes_at,
    git_text,
    rel,
    sha256_bytes,
    sha256_file,
    tracked_at,
)

FREEZE_ROOT = ROOT / "evaluation/sers_novelty_gap/r2_final_reassessment_freeze_v1"
VERIFY_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_r2_final_reassessment_v1.py"
RUN_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/run_sers_r2_final_reassessment_v1.py"
FREEZE_CREATE_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/freeze_sers_r2_final_reassessment_v1.py"
FREEZE_VERIFY_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_r2_final_reassessment_freeze_v1.py"
TEST_PATH = ROOT / "tests/test_sers_r2_final_reassessment_v1.py"


def main() -> int:
    if FREEZE_ROOT.exists():
        print("SERS R2 final reassessment freeze creation: FAIL")
        print(" - freeze root already exists")
        return 2

    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    if tracked_dirty:
        print("SERS R2 final reassessment freeze creation: FAIL")
        print(" - tracked working tree/index is not clean")
        return 2

    verify = subprocess.run([sys.executable, "-m", "scripts.verify_sers_r2_final_reassessment_v1"], cwd=ROOT, text=True)
    if verify.returncode != 0:
        return 2

    source_commit = git_text("rev-parse", "HEAD")
    for path in [REPORT_PATH, COMPLETE_PATH, SPEC_PATH, RUN_PATH, VERIFY_PATH, FREEZE_CREATE_PATH, FREEZE_VERIFY_PATH, TEST_PATH, R0_FREEZE_MANIFEST, R0_FREEZE_READY]:
        rp = rel(path)
        if not tracked_at("HEAD", rp):
            print("SERS R2 final reassessment freeze creation: FAIL")
            print(" - critical file is not tracked in source commit:", rp)
            return 2
        if git_bytes_at("HEAD", rp) != path.read_bytes():
            print("SERS R2 final reassessment freeze creation: FAIL")
            print(" - critical file differs from source commit:", rp)
            return 2

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    critical_paths = [REPORT_PATH, COMPLETE_PATH, SPEC_PATH, RUN_PATH, VERIFY_PATH, FREEZE_CREATE_PATH, FREEZE_VERIFY_PATH, TEST_PATH, R0_FREEZE_MANIFEST, R0_FREEZE_READY]
    critical_hashes = {rel(path): sha256_file(path) for path in critical_paths}
    payload = {
        "schema_version": "sers-r2-final-reassessment-freeze-v1",
        "source_branch": git_text("branch", "--show-current"),
        "source_r2_report_commit": source_commit,
        "r2_report_id": report["report_id"],
        "r2_report_sha256": report["report_sha256"],
        "source_r0_freeze_id": report["source_lineage"]["source_r0_freeze_id"],
        "source_r0_manifest_sha256": report["source_lineage"]["source_r0_manifest_sha256"],
        "source_r2_spec_id": report["source_lineage"]["source_r2_spec_id"],
        "source_r2_spec_sha256": report["source_lineage"]["source_r2_spec_sha256"],
        "critical_file_sha256": critical_hashes,
        "primary_remaining_candidate_hypothesis_id": report["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"],
        "secondary_bounded_extension_hypothesis_id": report["portfolio_decision"]["secondary_bounded_extension_hypothesis_id"],
        "rejected_as_formulated_hypothesis_ids": report["portfolio_decision"]["rejected_as_formulated_hypothesis_ids"],
        "scientific_reviewer_llm_used": True,
        "human_scientist_reviewer_present": False,
        "runtime_llm_calls": 0,
        "runtime_network_calls": 0,
        "r1_executed": False,
        "hypothesis_rewrite_called": False,
        "r2_complete": True,
        "i0_started": False,
        "fresh_reserve_c_consumed": False,
        "fresh_reserve_c_authorized": False,
        "automatic_next_stage_authorized": False,
        "network_calls_during_freeze_creation": 0,
        "stop_after_freeze": True,
    }
    manifest_sha = sha256_bytes(canonical(payload).encode("utf-8"))
    manifest = dict(payload)
    manifest["freeze_id"] = "sers_r2_final_reassessment_freeze_v1:" + manifest_sha[:20]
    manifest["manifest_sha256"] = manifest_sha
    ready = {
        "schema_version": "sers-r2-final-reassessment-freeze-ready-v1",
        "freeze_id": manifest["freeze_id"],
        "manifest_sha256": manifest_sha,
        "source_r2_report_commit": source_commit,
        "ready": True,
        "r2_complete": True,
        "i0_started": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }

    FREEZE_ROOT.mkdir(parents=True, exist_ok=False)
    (FREEZE_ROOT / "freeze_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (FREEZE_ROOT / "FREEZE_READY.json").write_text(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("SERS R2 final reassessment freeze creation: PASS")
    print("Freeze ID:", manifest["freeze_id"])
    print("Manifest SHA256:", manifest_sha)
    print("Source R2 report commit:", source_commit)
    print("Primary remaining candidate:", manifest["primary_remaining_candidate_hypothesis_id"])
    print("Network calls during freeze creation:", 0)
    print("R1 executed:", False)
    print("I0 started:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
