from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
ADJUDICATION_ROOT = (
    ROOT
    / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_v1"
)
FREEZE_ROOT = (
    ROOT
    / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_freeze_v1"
)
ADJUDICATION_PATH = ADJUDICATION_ROOT / "adjudication.json"
REVIEW_PATH = ADJUDICATION_ROOT / "SCIENTIFIC_REVIEW.md"
VERIFY_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_r0_manual_scientific_adjudication_v1.py"
FREEZE_VERIFY_PATH = (
    ROOT
    / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_r0_manual_scientific_adjudication_freeze_v1.py"
)
TEST_PATH = ROOT / "tests/test_sers_r0_manual_scientific_adjudication_v1.py"
FREEZE_CREATE_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/freeze_sers_r0_manual_scientific_adjudication_v1.py"
T1_FREEZE_PATH = (
    ROOT
    / "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_freeze_v2"
    / "freeze_manifest.json"
)
GAP_PLAN_PATH = (
    ROOT
    / "evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1"
    / "novelty_gap_plan.json"
)

EXPECTED_BRANCH = "feat/SERS-targeted-retrieval-live-dev"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> int:
    if FREEZE_ROOT.exists():
        print("SERS R0 manual adjudication freeze creation: FAIL")
        print(" - freeze root already exists")
        return 2

    branch = _git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        print("SERS R0 manual adjudication freeze creation: FAIL")
        print(" - unexpected branch:", branch)
        return 2

    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    if tracked_dirty:
        print("SERS R0 manual adjudication freeze creation: FAIL")
        print(" - tracked working tree/index is not clean")
        return 2

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_r0_manual_scientific_adjudication_v1",
        ],
        cwd=ROOT,
        text=True,
    )
    if verify.returncode != 0:
        return 2

    source_commit = _git("rev-parse", "HEAD")
    adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))

    critical_paths = [
        ADJUDICATION_PATH,
        REVIEW_PATH,
        VERIFY_PATH,
        FREEZE_VERIFY_PATH,
        FREEZE_CREATE_PATH,
        TEST_PATH,
        T1_FREEZE_PATH,
        GAP_PLAN_PATH,
    ]
    critical_hashes = {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in critical_paths
    }

    payload = {
        "schema_version": "sers-r0-manual-scientific-adjudication-freeze-v1",
        "source_branch": branch,
        "source_adjudication_commit": source_commit,
        "adjudication_id": adjudication["adjudication_id"],
        "adjudication_sha256": adjudication["adjudication_sha256"],
        "source_r0_2_commit": adjudication["source_lineage"]["source_r0_2_commit"],
        "source_gap_plan_id": adjudication["source_lineage"]["gap_plan_id"],
        "source_gap_plan_sha256": adjudication["source_lineage"]["gap_plan_sha256"],
        "source_t1_run_id": adjudication["source_lineage"]["t1_run_id"],
        "source_t1_freeze_id": adjudication["source_lineage"]["t1_freeze_id"],
        "source_t1_manifest_sha256": adjudication["source_lineage"]["t1_manifest_sha256"],
        "critical_file_sha256": critical_hashes,
        "scientific_reviewer_mode": adjudication["reviewer"]["mode"],
        "scientific_reviewer_model": adjudication["reviewer"]["model"],
        "human_scientist_reviewer_present": False,
        "scientific_reviewer_llm_used": True,
        "deterministic_r0_router_llm_calls": 0,
        "r0_scientific_adjudication_complete": True,
        "hypothesis_rewrite_called": False,
        "r1_authorized_for_any_hypothesis": False,
        "r2_started": False,
        "fresh_reserve_c_consumed": False,
        "fresh_reserve_c_authorized": False,
        "automatic_next_stage_authorized": False,
        "network_calls_during_freeze_creation": 0,
        "stop_after_freeze": True,
    }
    manifest_sha = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    manifest = dict(payload)
    manifest["freeze_id"] = (
        "sers_r0_manual_scientific_adjudication_freeze_v1:"
        + manifest_sha[:20]
    )
    manifest["manifest_sha256"] = manifest_sha

    ready = {
        "schema_version": "sers-r0-manual-scientific-adjudication-freeze-ready-v1",
        "freeze_id": manifest["freeze_id"],
        "manifest_sha256": manifest_sha,
        "source_adjudication_commit": source_commit,
        "ready": True,
        "automatic_next_stage_authorized": False,
        "fresh_reserve_c_consumed": False,
        "stop": True,
    }

    FREEZE_ROOT.mkdir(parents=True, exist_ok=False)
    (FREEZE_ROOT / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (FREEZE_ROOT / "FREEZE_READY.json").write_text(
        json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("SERS R0 manual adjudication freeze creation: PASS")
    print("Freeze ID:", manifest["freeze_id"])
    print("Manifest SHA256:", manifest_sha)
    print("Source adjudication commit:", source_commit)
    print("Network calls during freeze creation:", 0)
    print("R1 authorized:", False)
    print("R2 started:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
