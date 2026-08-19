from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from campaigns.sers_alpha4_epoch.post_t1.cli.freeze_sers_r2_final_reassessment_v1 import FREEZE_ROOT
from campaigns.sers_alpha4_epoch.post_t1.cli.run_sers_r2_final_reassessment_v1 import ROOT, canonical, git_bytes_at, rel, sha256_bytes, sha256_file, tracked_at

MANIFEST_PATH = FREEZE_ROOT / "freeze_manifest.json"
READY_PATH = FREEZE_ROOT / "FREEZE_READY.json"


def main() -> int:
    base = subprocess.run([sys.executable, "-m", "scripts.verify_sers_r2_final_reassessment_v1"], cwd=ROOT, text=True)
    if base.returncode != 0:
        return 2

    issues: list[str] = []
    if not MANIFEST_PATH.is_file():
        issues.append("R2 freeze manifest missing")
    if not READY_PATH.is_file():
        issues.append("R2 FREEZE_READY missing")
    if issues:
        print("SERS R2 final reassessment freeze verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ready = json.loads(READY_PATH.read_text(encoding="utf-8"))
    payload = dict(manifest)
    freeze_id = payload.pop("freeze_id", None)
    manifest_sha = payload.pop("manifest_sha256", None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    if manifest_sha != recomputed:
        issues.append("R2 freeze manifest SHA mismatch")
    if freeze_id != "sers_r2_final_reassessment_freeze_v1:" + recomputed[:20]:
        issues.append("R2 freeze ID mismatch")

    for rp, expected_sha in manifest.get("critical_file_sha256", {}).items():
        path = ROOT / rp
        if not path.is_file():
            issues.append(f"critical file missing:{rp}")
        elif sha256_file(path) != expected_sha:
            issues.append(f"critical file hash mismatch:{rp}")

    if ready.get("ready") is not True or ready.get("stop") is not True:
        issues.append("R2 FREEZE_READY ready/STOP invalid")
    for key in ["freeze_id", "manifest_sha256", "source_r2_report_commit"]:
        expected = manifest.get(key) if key != "source_r2_report_commit" else manifest.get("source_r2_report_commit")
        if ready.get(key) != expected:
            issues.append(f"R2 FREEZE_READY mismatch:{key}")
    if ready.get("r2_complete") is not True:
        issues.append("R2 FREEZE_READY completion flag false")
    for key in ["i0_started", "fresh_reserve_c_consumed", "automatic_next_stage_authorized"]:
        if ready.get(key) is not False:
            issues.append(f"R2 FREEZE_READY guard changed:{key}")

    required_false = [
        "human_scientist_reviewer_present",
        "r1_executed",
        "hypothesis_rewrite_called",
        "i0_started",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in required_false:
        if manifest.get(key) is not False:
            issues.append(f"R2 freeze guard must be false:{key}")
    if manifest.get("scientific_reviewer_llm_used") is not True:
        issues.append("R2 scientific reviewer LLM usage must remain explicit")
    if manifest.get("runtime_llm_calls") != 0 or manifest.get("runtime_network_calls") != 0:
        issues.append("R2 runtime call count changed")
    if manifest.get("network_calls_during_freeze_creation") != 0:
        issues.append("R2 freeze creation network count changed")
    if manifest.get("r2_complete") is not True or manifest.get("stop_after_freeze") is not True:
        issues.append("R2 freeze completion/STOP invalid")

    source_commit = manifest.get("source_r2_report_commit")
    if not isinstance(source_commit, str):
        issues.append("R2 source report commit missing")
    else:
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0:
            issues.append("R2 source report commit not ancestor of HEAD")
        for rp in [
            "evaluation/sers_novelty_gap/r2_final_reassessment_run_v1/r2_report.json",
            "evaluation/sers_novelty_gap/r2_final_reassessment_run_v1/R2_COMPLETE.json",
        ]:
            if not tracked_at(source_commit, rp):
                issues.append(f"R2 source report commit missing tracked artifact:{rp}")
            else:
                expected_sha = manifest.get("critical_file_sha256", {}).get(rp)
                if expected_sha and sha256_bytes(git_bytes_at(source_commit, rp)) != expected_sha:
                    issues.append(f"R2 source report commit artifact hash mismatch:{rp}")

    if issues:
        print("SERS R2 final reassessment freeze verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("SERS R2 final reassessment freeze verification: PASS")
    print("Freeze ID:", freeze_id)
    print("Manifest SHA256:", manifest_sha)
    print("Source R2 report commit:", source_commit)
    print("Primary remaining candidate:", manifest["primary_remaining_candidate_hypothesis_id"])
    print("Scientific reviewer LLM used:", True)
    print("Human scientist reviewer present:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("R1 executed:", False)
    print("I0 started:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
