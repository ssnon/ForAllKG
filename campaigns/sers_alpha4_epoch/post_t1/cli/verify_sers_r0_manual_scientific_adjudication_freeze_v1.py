from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FREEZE_ROOT = (
    ROOT
    / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_freeze_v1"
)
MANIFEST_PATH = FREEZE_ROOT / "freeze_manifest.json"
READY_PATH = FREEZE_ROOT / "FREEZE_READY.json"


CURRENT_REPLAY_CRITICAL_FILES = {
    "scripts/freeze_sers_r0_manual_scientific_adjudication_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "freeze_sers_r0_manual_scientific_adjudication_v1.py"
    ),
    "scripts/verify_sers_r0_manual_scientific_adjudication_freeze_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "verify_sers_r0_manual_scientific_adjudication_freeze_v1.py"
    ),
    "scripts/verify_sers_r0_manual_scientific_adjudication_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "verify_sers_r0_manual_scientific_adjudication_v1.py"
    ),
}

CURRENT_REGRESSION_SURFACE = (
    "tests/test_sers_r0_manual_scientific_adjudication_v1.py"
,)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    base = subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_r0_manual_scientific_adjudication_v1",
        ],
        cwd=ROOT,
        text=True,
    )
    if base.returncode != 0:
        return 2

    issues: list[str] = []
    if not MANIFEST_PATH.is_file():
        issues.append("freeze manifest missing")
    if not READY_PATH.is_file():
        issues.append("FREEZE_READY missing")
    if issues:
        print("SERS R0 manual adjudication freeze verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ready = json.loads(READY_PATH.read_text(encoding="utf-8"))

    payload = dict(manifest)
    freeze_id = payload.pop("freeze_id", None)
    manifest_sha = payload.pop("manifest_sha256", None)
    recomputed = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    if manifest_sha != recomputed:
        issues.append("freeze manifest SHA mismatch")
    expected_freeze_id = (
        "sers_r0_manual_scientific_adjudication_freeze_v1:"
        + recomputed[:20]
    )
    if freeze_id != expected_freeze_id:
        issues.append("freeze ID mismatch")

    critical_hashes = manifest.get("critical_file_sha256", {})
    historical_source_commit = manifest.get("source_adjudication_commit")

    if not isinstance(critical_hashes, dict) or not critical_hashes:
        issues.append("critical-file hash map missing")

    if not isinstance(historical_source_commit, str) or not historical_source_commit:
        issues.append("historical source adjudication commit missing")

    if (
        isinstance(critical_hashes, dict)
        and critical_hashes
        and isinstance(historical_source_commit, str)
        and historical_source_commit
    ):
        # Historical freeze identity belongs to the original source commit and
        # original path vocabulary.  Repository relocation must not rewrite it.
        for historical_path, expected_sha in sorted(critical_hashes.items()):
            try:
                committed = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{historical_source_commit}:{historical_path}",
                    ],
                    cwd=ROOT,
                )
            except subprocess.CalledProcessError:
                issues.append(
                    f"historical critical file missing:{historical_path}"
                )
                continue

            observed_sha = hashlib.sha256(committed).hexdigest()

            if observed_sha != expected_sha:
                issues.append(
                    f"historical critical file hash mismatch:{historical_path}"
                )

        # Scientific/evaluation evidence remains immutable in the current
        # checkout even though implementation/test paths may evolve.
        for historical_path, expected_sha in sorted(critical_hashes.items()):
            if not historical_path.startswith("evaluation/"):
                continue

            current_path = ROOT / historical_path

            if not current_path.is_file():
                issues.append(
                    f"current frozen evidence missing:{historical_path}"
                )
            elif _sha256_file(current_path) != expected_sha:
                issues.append(
                    f"current frozen evidence hash mismatch:{historical_path}"
                )

        # Relocated implementation is a current replay/verification surface,
        # not part of the historical byte identity.
        for historical_path, current_relative in (
            CURRENT_REPLAY_CRITICAL_FILES.items()
        ):
            if historical_path not in critical_hashes:
                issues.append(
                    f"historical replay identity missing:{historical_path}"
                )
                continue

            current_path = ROOT / current_relative

            if not current_path.is_file():
                issues.append(
                    f"current replay critical file missing:{current_relative}"
                )

        for current_relative in CURRENT_REGRESSION_SURFACE:
            if not (ROOT / current_relative).is_file():
                issues.append(
                    f"current regression surface missing:{current_relative}"
                )

    if ready.get("ready") is not True:
        issues.append("FREEZE_READY ready flag false")
    if ready.get("freeze_id") != freeze_id:
        issues.append("FREEZE_READY freeze ID mismatch")
    if ready.get("manifest_sha256") != manifest_sha:
        issues.append("FREEZE_READY manifest SHA mismatch")
    if ready.get("source_adjudication_commit") != manifest.get("source_adjudication_commit"):
        issues.append("FREEZE_READY source commit mismatch")
    if ready.get("automatic_next_stage_authorized") is not False:
        issues.append("FREEZE_READY automatic next-stage flag changed")
    if ready.get("fresh_reserve_c_consumed") is not False:
        issues.append("FREEZE_READY Reserve C consumption changed")
    if ready.get("stop") is not True:
        issues.append("FREEZE_READY STOP flag missing")

    required_false = [
        "human_scientist_reviewer_present",
        "hypothesis_rewrite_called",
        "r1_authorized_for_any_hypothesis",
        "r2_started",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in required_false:
        if manifest.get(key) is not False:
            issues.append(f"freeze manifest guard must be false:{key}")
    if manifest.get("scientific_reviewer_llm_used") is not True:
        issues.append("scientific reviewer LLM usage must remain explicit")
    if manifest.get("deterministic_r0_router_llm_calls") != 0:
        issues.append("deterministic router LLM count changed")
    if manifest.get("r0_scientific_adjudication_complete") is not True:
        issues.append("R0 scientific adjudication completion flag changed")
    if manifest.get("network_calls_during_freeze_creation") != 0:
        issues.append("freeze creation must remain offline")
    if manifest.get("stop_after_freeze") is not True:
        issues.append("stop_after_freeze must remain true")

    source_commit = manifest.get("source_adjudication_commit")
    if not isinstance(source_commit, str):
        issues.append("source adjudication commit missing")
    else:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode != 0:
            issues.append("source adjudication commit is not an ancestor of HEAD")

    if issues:
        print("SERS R0 manual adjudication freeze verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("R1 authorized:", False)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("SERS R0 manual adjudication freeze verification: PASS")
    print("Freeze ID:", freeze_id)
    print("Manifest SHA256:", manifest_sha)
    print("Source adjudication commit:", source_commit)
    print("Scientific reviewer LLM used:", True)
    print("Human scientist reviewer present:", False)
    print("Deterministic R0 router LLM calls:", 0)
    print("Network calls during verification:", 0)
    print("R1 authorized:", False)
    print("R2 started:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
