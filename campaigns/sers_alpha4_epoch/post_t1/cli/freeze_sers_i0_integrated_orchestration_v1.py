from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from campaigns.sers_alpha4_epoch.post_t1.cli.run_sers_i0_integrated_orchestration_v1 import (
    COMPLETE_PATH,
    FREEZE_CREATE_PATH,
    FREEZE_VERIFY_PATH,
    GAP_PLAN_PATH,
    HANDOFF_PATH,
    R0_ADJ_PATH,
    R0_FREEZE_MANIFEST,
    R0_FREEZE_READY,
    R2_COMPLETE_PATH,
    R2_FREEZE_MANIFEST,
    R2_FREEZE_READY,
    R2_REPORT_PATH,
    ROOT,
    RUN_PATH,
    SPEC_PATH,
    T0_FREEZE_MANIFEST,
    T1_FREEZE_MANIFEST,
    TEST_PATH,
    VERIFY_PATH,
    canonical,
    git_bytes_at,
    git_text,
    rel,
    sha256_bytes,
    sha256_file,
    tracked_at,
)

FREEZE_ROOT = ROOT / "evaluation/sers_novelty_gap/i0_integrated_orchestration_freeze_v1"
MANIFEST_PATH = FREEZE_ROOT / "freeze_manifest.json"
READY_PATH = FREEZE_ROOT / "FREEZE_READY.json"


def _atomic_write_freeze(manifest: dict, ready: dict) -> None:
    parent = FREEZE_ROOT.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_root = parent / f".{FREEZE_ROOT.name}.tmp"
    if tmp_root.exists():
        raise ValueError("I0 freeze temporary root already exists")
    tmp_root.mkdir()
    try:
        (tmp_root / MANIFEST_PATH.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (tmp_root / READY_PATH.name).write_text(
            json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_root.replace(FREEZE_ROOT)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise


def main() -> int:
    if FREEZE_ROOT.exists():
        print("SERS I0 integrated orchestration freeze creation: FAIL")
        print(" - freeze root already exists")
        return 2

    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    if tracked_dirty:
        print("SERS I0 integrated orchestration freeze creation: FAIL")
        print(" - tracked working tree/index is not clean")
        return 2

    verify = subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_i0_integrated_orchestration_v1"],
        cwd=ROOT,
        text=True,
    )
    if verify.returncode != 0:
        return 2

    source_commit = git_text("rev-parse", "HEAD")
    critical_paths = [
        HANDOFF_PATH,
        COMPLETE_PATH,
        SPEC_PATH,
        RUN_PATH,
        VERIFY_PATH,
        FREEZE_CREATE_PATH,
        FREEZE_VERIFY_PATH,
        TEST_PATH,
        R2_FREEZE_MANIFEST,
        R2_FREEZE_READY,
        R2_REPORT_PATH,
        R2_COMPLETE_PATH,
        R0_FREEZE_MANIFEST,
        R0_FREEZE_READY,
        R0_ADJ_PATH,
        T1_FREEZE_MANIFEST,
        T0_FREEZE_MANIFEST,
        GAP_PLAN_PATH,
    ]
    for path in critical_paths:
        rp = rel(path)
        if not tracked_at("HEAD", rp):
            print("SERS I0 integrated orchestration freeze creation: FAIL")
            print(" - critical file is not tracked in source commit:", rp)
            return 2
        if git_bytes_at("HEAD", rp) != path.read_bytes():
            print("SERS I0 integrated orchestration freeze creation: FAIL")
            print(" - critical file differs from source commit:", rp)
            return 2

    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))
    critical_hashes = {rel(path): sha256_file(path) for path in critical_paths}
    payload = {
        "schema_version": "sers-i0-integrated-orchestration-freeze-v1",
        "source_branch": git_text("branch", "--show-current"),
        "source_i0_handoff_commit": source_commit,
        "i0_orchestration_id": handoff["orchestration_id"],
        "i0_orchestration_sha256": handoff["orchestration_sha256"],
        "source_i0_spec_id": handoff["source_lineage"]["source_i0_spec_id"],
        "source_i0_spec_sha256": handoff["source_lineage"]["source_i0_spec_sha256"],
        "source_i0_code_commit": handoff["source_lineage"]["source_i0_code_commit"],
        "source_r2_freeze_commit": handoff["source_lineage"]["source_r2_freeze_commit"],
        "source_r2_freeze_id": handoff["source_lineage"]["source_r2_freeze_id"],
        "source_r2_manifest_sha256": handoff["source_lineage"]["source_r2_manifest_sha256"],
        "source_r2_report_commit": handoff["source_lineage"]["source_r2_report_commit"],
        "source_r2_report_id": handoff["source_lineage"]["source_r2_report_id"],
        "source_r2_report_sha256": handoff["source_lineage"]["source_r2_report_sha256"],
        "critical_file_sha256": critical_hashes,
        "primary_remaining_candidate_hypothesis_id": handoff[
            "frozen_r2_portfolio_decision"
        ]["primary_remaining_candidate_hypothesis_id"],
        "secondary_bounded_extension_hypothesis_id": handoff[
            "frozen_r2_portfolio_decision"
        ]["secondary_bounded_extension_hypothesis_id"],
        "rejected_as_formulated_hypothesis_ids": handoff[
            "frozen_r2_portfolio_decision"
        ]["rejected_as_formulated_hypothesis_ids"],
        "upstream_scientific_reviewer_llm_used": True,
        "human_scientist_reviewer_present": False,
        "i0_runtime_llm_calls": 0,
        "i0_runtime_network_calls": 0,
        "scientific_reassessment_performed": False,
        "new_scientific_judgment_performed": False,
        "new_retrieval_performed": False,
        "ranker_called": False,
        "claim_reviewer_called": False,
        "hypothesis_rewrite_called": False,
        "r1_executed": False,
        "i0_complete": True,
        "fresh_reserve_c_readiness_assessed": False,
        "fresh_reserve_c_authorized": False,
        "fresh_reserve_c_consumed": False,
        "fresh_reserve_c_marker_write_allowed": False,
        "holdout_execution_authorized": False,
        "automatic_next_stage_authorized": False,
        "network_calls_during_freeze_creation": 0,
        "stop_after_freeze": True,
    }
    manifest_sha = sha256_bytes(canonical(payload).encode("utf-8"))
    manifest = dict(payload)
    manifest["freeze_id"] = "sers_i0_integrated_orchestration_freeze_v1:" + manifest_sha[:20]
    manifest["manifest_sha256"] = manifest_sha
    ready = {
        "schema_version": "sers-i0-integrated-orchestration-freeze-ready-v1",
        "freeze_id": manifest["freeze_id"],
        "manifest_sha256": manifest_sha,
        "source_i0_handoff_commit": source_commit,
        "ready": True,
        "i0_complete": True,
        "fresh_reserve_c_readiness_assessed": False,
        "fresh_reserve_c_authorized": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }

    try:
        _atomic_write_freeze(manifest, ready)
    except Exception as exc:
        print("SERS I0 integrated orchestration freeze creation: FAIL")
        print(" - output write:", exc)
        return 2

    print("SERS I0 integrated orchestration freeze creation: PASS")
    print("Freeze ID:", manifest["freeze_id"])
    print("Manifest SHA256:", manifest_sha)
    print("Source I0 handoff commit:", source_commit)
    print(
        "Primary remaining candidate:",
        manifest["primary_remaining_candidate_hypothesis_id"],
    )
    print("Network calls during freeze creation:", 0)
    print("Scientific reassessment performed:", False)
    print("R1 executed:", False)
    print("Fresh Reserve C readiness assessed:", False)
    print("Fresh Reserve C authorized:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
