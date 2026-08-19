from __future__ import annotations

import json
import subprocess
import sys

from campaigns.sers_alpha4_epoch.post_t1.cli.freeze_sers_i0_integrated_orchestration_v1 import (
    FREEZE_ROOT,
    MANIFEST_PATH,
    READY_PATH,
)
from campaigns.sers_alpha4_epoch.post_t1.cli.run_sers_i0_integrated_orchestration_v1 import (
    ROOT,
    canonical,
    git_bytes_at,
    is_ancestor,
    sha256_bytes,
    sha256_file,
    tracked_at,
)


CURRENT_REPLAY_CRITICAL_FILES = {
    "scripts/freeze_sers_i0_integrated_orchestration_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "freeze_sers_i0_integrated_orchestration_v1.py"
    ),
    "scripts/run_sers_i0_integrated_orchestration_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "run_sers_i0_integrated_orchestration_v1.py"
    ),
    "scripts/verify_sers_i0_integrated_orchestration_freeze_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "verify_sers_i0_integrated_orchestration_freeze_v1.py"
    ),
    "scripts/verify_sers_i0_integrated_orchestration_v1.py": (
        "campaigns/sers_alpha4_epoch/post_t1/cli/"
        "verify_sers_i0_integrated_orchestration_v1.py"
    ),
}

CURRENT_REGRESSION_SURFACE = (
    "tests/test_sers_i0_integrated_orchestration_v1.py",
)


def main() -> int:
    base = subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_i0_integrated_orchestration_v1"],
        cwd=ROOT,
        text=True,
    )
    if base.returncode != 0:
        return 2

    issues: list[str] = []
    if not MANIFEST_PATH.is_file():
        issues.append("I0 freeze manifest missing")
    if not READY_PATH.is_file():
        issues.append("I0 FREEZE_READY missing")
    if issues:
        print("SERS I0 integrated orchestration freeze verification: FAIL")
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
        issues.append("I0 freeze manifest SHA mismatch")
    if freeze_id != "sers_i0_integrated_orchestration_freeze_v1:" + recomputed[:20]:
        issues.append("I0 freeze ID mismatch")

    critical_hashes = manifest.get("critical_file_sha256", {})
    historical_source_commit = manifest.get("source_i0_handoff_commit")

    if not isinstance(critical_hashes, dict) or not critical_hashes:
        issues.append("I0 critical-file hash map missing")

    if (
        not isinstance(historical_source_commit, str)
        or not historical_source_commit
    ):
        issues.append("I0 historical source handoff commit missing")

    if (
        isinstance(critical_hashes, dict)
        and critical_hashes
        and isinstance(historical_source_commit, str)
        and historical_source_commit
    ):
        # Historical freeze identity belongs to the original source commit
        # and original scripts/* path vocabulary.
        for historical_path, expected_sha in sorted(
            critical_hashes.items()
        ):
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
                    f"I0 historical critical file missing:{historical_path}"
                )
                continue

            observed_sha = sha256_bytes(
                committed
            )

            if observed_sha != expected_sha:
                issues.append(
                    "I0 historical critical file hash mismatch:"
                    f"{historical_path}"
                )

        # Scientific spec and frozen evaluation artifacts remain
        # byte-immutable in the current checkout.
        for historical_path, expected_sha in sorted(
            critical_hashes.items()
        ):
            if not (
                historical_path.startswith("evaluation/")
                or historical_path
                == "dac_her/sers_i0_integrated_orchestration_spec_v1.json"
            ):
                continue

            current_path = ROOT / historical_path

            if not current_path.is_file():
                issues.append(
                    f"I0 current immutable file missing:{historical_path}"
                )
            elif sha256_file(current_path) != expected_sha:
                issues.append(
                    f"I0 current immutable file hash mismatch:{historical_path}"
                )

        # Relocated implementation is the current replay surface.  Its current
        # bytes may differ because relocation/provenance repairs are not part of
        # the historical scientific identity.
        for historical_path, current_relative in (
            CURRENT_REPLAY_CRITICAL_FILES.items()
        ):
            if historical_path not in critical_hashes:
                issues.append(
                    f"I0 historical replay identity missing:{historical_path}"
                )
                continue

            current_path = ROOT / current_relative

            if not current_path.is_file():
                issues.append(
                    f"I0 current replay critical file missing:{current_relative}"
                )
            elif not tracked_at("HEAD", current_relative):
                issues.append(
                    f"I0 current replay critical file not tracked:{current_relative}"
                )

        for current_relative in CURRENT_REGRESSION_SURFACE:
            if not (ROOT / current_relative).is_file():
                issues.append(
                    f"I0 current regression surface missing:{current_relative}"
                )

    if ready.get("ready") is not True or ready.get("stop") is not True:
        issues.append("I0 FREEZE_READY ready/STOP invalid")
    for key in ["freeze_id", "manifest_sha256", "source_i0_handoff_commit"]:
        if ready.get(key) != manifest.get(key):
            issues.append(f"I0 FREEZE_READY mismatch:{key}")
    if ready.get("i0_complete") is not True:
        issues.append("I0 FREEZE_READY completion flag false")
    for key in [
        "fresh_reserve_c_readiness_assessed",
        "fresh_reserve_c_authorized",
        "fresh_reserve_c_consumed",
        "automatic_next_stage_authorized",
    ]:
        if ready.get(key) is not False:
            issues.append(f"I0 FREEZE_READY guard changed:{key}")

    required_false = [
        "human_scientist_reviewer_present",
        "scientific_reassessment_performed",
        "new_scientific_judgment_performed",
        "new_retrieval_performed",
        "ranker_called",
        "claim_reviewer_called",
        "hypothesis_rewrite_called",
        "r1_executed",
        "fresh_reserve_c_readiness_assessed",
        "fresh_reserve_c_authorized",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_marker_write_allowed",
        "holdout_execution_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in required_false:
        if manifest.get(key) is not False:
            issues.append(f"I0 freeze guard must be false:{key}")
    if manifest.get("upstream_scientific_reviewer_llm_used") is not True:
        issues.append("upstream scientific reviewer LLM usage must remain explicit")
    if manifest.get("i0_runtime_llm_calls") != 0:
        issues.append("I0 runtime LLM calls changed")
    if manifest.get("i0_runtime_network_calls") != 0:
        issues.append("I0 runtime network calls changed")
    if manifest.get("network_calls_during_freeze_creation") != 0:
        issues.append("I0 freeze creation network count changed")
    if manifest.get("i0_complete") is not True:
        issues.append("I0 completion flag false")
    if manifest.get("stop_after_freeze") is not True:
        issues.append("I0 freeze STOP boundary missing")

    source_commit = manifest.get("source_i0_handoff_commit")
    if not isinstance(source_commit, str):
        issues.append("I0 source handoff commit missing")
    else:
        if not is_ancestor(source_commit, "HEAD"):
            issues.append("I0 source handoff commit not ancestor of HEAD")
        for rp in [
            "evaluation/sers_novelty_gap/i0_integrated_orchestration_run_v1/i0_handoff.json",
            "evaluation/sers_novelty_gap/i0_integrated_orchestration_run_v1/I0_COMPLETE.json",
        ]:
            if not tracked_at(source_commit, rp):
                issues.append(f"I0 source handoff commit missing tracked artifact:{rp}")
            else:
                expected_sha = manifest.get("critical_file_sha256", {}).get(rp)
                if not isinstance(expected_sha, str):
                    issues.append(f"I0 freeze missing critical hash:{rp}")
                elif sha256_bytes(git_bytes_at(source_commit, rp)) != expected_sha:
                    issues.append(f"I0 source handoff commit artifact hash mismatch:{rp}")

    r2_freeze_commit = manifest.get("source_r2_freeze_commit")
    if not isinstance(r2_freeze_commit, str):
        issues.append("I0 source R2 freeze commit missing")
    elif not is_ancestor(r2_freeze_commit, "HEAD"):
        issues.append("I0 source R2 freeze commit not ancestor of HEAD")

    if issues:
        print("SERS I0 integrated orchestration freeze verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        print("Fresh Reserve C authorized:", False)
        return 2

    print("SERS I0 integrated orchestration freeze verification: PASS")
    print("Freeze ID:", freeze_id)
    print("Manifest SHA256:", manifest_sha)
    print("Source I0 handoff commit:", source_commit)
    print("Source R2 freeze commit:", r2_freeze_commit)
    print(
        "Primary remaining candidate:",
        manifest["primary_remaining_candidate_hypothesis_id"],
    )
    print("Scientific reassessment performed:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("R1 executed:", False)
    print("Fresh Reserve C readiness assessed:", False)
    print("Fresh Reserve C authorized:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
