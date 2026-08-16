from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.verify_sers_novelty_gap_g0_g2_production_freeze_v1 import (
    DEFAULT_FREEZE_ROOT,
    verify_freeze,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "feat/SERS-novelty-gap-next"
FREEZE_BRANCH = "freeze/SERS-novelty-gap-g0-g2-v1-20260817"
COMMIT_MESSAGE = "freeze SERS novelty gap G0-G2 production v1"

COMMIT_PATHS = (
    "dac_her/novelty_gap_analysis.py",
    "tests/test_sers_novelty_gap_query_compaction_production_v2.py",
    "scripts/run_sers_novelty_gap_g0_g2_production_integration_v2.py",
    "scripts/verify_sers_novelty_gap_g0_g2_production_integration_v2.py",
    "scripts/verify_sers_novelty_gap_g0_g2_production_freeze_v1.py",
    "scripts/finalize_sers_novelty_gap_g0_g2_production_freeze_v1.py",
    "evaluation/sers_novelty_gap/g0_g2_production_freeze_v1",
)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-freeze-commit",
        action="store_true",
    )
    args = parser.parse_args()
    if not args.confirm_freeze_commit:
        parser.error("--confirm-freeze-commit is required")

    branch = git("branch", "--show-current").stdout.strip()
    if branch != EXPECTED_BRANCH:
        print("Freeze finalization: FAIL")
        print(" - expected branch:", EXPECTED_BRANCH)
        print(" - observed branch:", branch)
        return 2

    ok, issues, meta = verify_freeze(ROOT, DEFAULT_FREEZE_ROOT)
    if not ok:
        print("Freeze finalization: FAIL")
        print(" - offline freeze verification failed")
        for issue in issues:
            print("   -", issue)
        return 2

    allowed_tracked = {
        "dac_her/novelty_gap_analysis.py",
    }
    tracked_changes = {
        line.strip()
        for line in git("diff", "--name-only").stdout.splitlines()
        if line.strip()
    }
    if not tracked_changes.issubset(allowed_tracked):
        print("Freeze finalization: FAIL")
        print(" - unexpected tracked changes:")
        for path in sorted(tracked_changes - allowed_tracked):
            print("   -", path)
        return 2
    if "dac_her/novelty_gap_analysis.py" not in tracked_changes:
        print("Freeze finalization: FAIL")
        print(" - production novelty_gap_analysis.py is not modified")
        return 2

    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        print("Freeze finalization: FAIL")
        print(" - staged changes already exist")
        return 2

    for path in COMMIT_PATHS:
        if path.startswith("evaluation/"):
            result = git("add", "-f", "--", path, check=False)
        else:
            result = git("add", "--", path, check=False)
        if result.returncode != 0:
            print("Freeze finalization: FAIL")
            print(" - failed to stage:", path)
            print(result.stderr)
            git("reset", check=False)
            return 2

    staged = {
        line.strip()
        for line in git(
            "diff",
            "--cached",
            "--name-only",
        ).stdout.splitlines()
        if line.strip()
    }
    expected_prefix = (
        "evaluation/sers_novelty_gap/"
        "g0_g2_production_freeze_v1/"
    )
    required_exact = {
        path for path in COMMIT_PATHS
        if not path.startswith("evaluation/")
    }
    if not required_exact.issubset(staged):
        print("Freeze finalization: FAIL")
        print(" - required staged paths missing")
        git("reset", check=False)
        return 2
    if not any(path.startswith(expected_prefix) for path in staged):
        print("Freeze finalization: FAIL")
        print(" - freeze evidence bundle not staged")
        git("reset", check=False)
        return 2

    diff_check = subprocess.run(
        ["git", "diff", "--cached", "--check"],
        cwd=ROOT,
        text=True,
    )
    if diff_check.returncode != 0:
        print("Freeze finalization: FAIL")
        print(" - staged diff whitespace check failed")
        git("reset", check=False)
        return 2

    commit = git("commit", "-m", COMMIT_MESSAGE, check=False)
    if commit.returncode != 0:
        print("Freeze finalization: FAIL")
        print(commit.stdout)
        print(commit.stderr)
        git("reset", check=False)
        return 2

    commit_sha = git("rev-parse", "HEAD").stdout.strip()

    existing = git(
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{FREEZE_BRANCH}",
        check=False,
    )
    if existing.returncode == 0:
        existing_sha = git(
            "rev-parse",
            FREEZE_BRANCH,
        ).stdout.strip()
        if existing_sha != commit_sha:
            print("Freeze finalization: FAIL")
            print(" - freeze branch already exists at another commit")
            return 2
    else:
        git("branch", FREEZE_BRANCH, commit_sha)

    ok, issues, meta = verify_freeze(ROOT, DEFAULT_FREEZE_ROOT)
    if not ok:
        print("Freeze finalization: FAIL AFTER COMMIT")
        for issue in issues:
            print(" -", issue)
        return 2

    print("Freeze finalization: PASS")
    print("Commit:", commit_sha)
    print("Freeze branch:", FREEZE_BRANCH)
    print("Freeze ID:", meta["freeze_id"])
    print("Manifest SHA256:", meta["manifest_sha256"])
    print("Remote push performed:", False)
    print("Targeted retrieval called:", False)
    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
