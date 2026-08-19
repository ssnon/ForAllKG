from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
FREEZE_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t1_live_targeted_retrieval_freeze_v2"
)
MANIFEST_PATH = FREEZE_ROOT / "freeze_manifest.json"

EXPECTED_SOURCE_EVIDENCE_COMMIT = "e11fa8b0d2d33c44e567214f3fd6f0063b6c4385"
EXPECTED_T0_FREEZE_ID = (
    "sers_targeted_retrieval_t0_freeze_v2:"
    "e6f582c672e77945060a"
)
EXPECTED_INPUT_BUNDLE_ID = (
    "sers_targeted_retrieval_t1_input_bundle_v1:"
    "836a5b53c31a668e38be"
)
EXPECTED_SPEC_ID = (
    "sers_targeted_retrieval_t1_live_spec:"
    "8ea007ccbea7cc1b9dea"
)
EXPECTED_V1_FAILURE_FREEZE_ID = (
    "sers_targeted_retrieval_t1_v1_failure_freeze:"
    "752e95b3d4719d2b75af"
)
EXPECTED_V2_RUN_ID = (
    "sers_targeted_retrieval_t1_live_v2:"
    "9a3c03bc59085c0af5fe"
)
EXPECTED_V2_OUTCOME = (
    "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_MECHANICAL_PASS"
)

CRITICAL_FILES = ['evaluation/sers_novelty_gap/t0_targeted_retrieval_canonicalization_freeze_v2/freeze_manifest.json', 'evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1/bundle_manifest.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/t1_spec.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/provider_plan.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_spec_v1/PREPARE_PASS.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_v1_failure_freeze_v1/failure_manifest.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/LIVE_ATTEMPT_CONSUMED.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/FATAL_ERROR.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/gap_01/augmented_plan.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/gap_01/delta_plan.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/gap_01/delta_packet.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v1/gap_01/merged_packet.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/LIVE_ATTEMPT_CONSUMED.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/MECHANICAL_PASS.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_01/RECOVERED_FROM_V1.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_01/gap_audit.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_02/SKIPPED.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_03/augmented_plan.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_03/delta_plan.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_03/delta_packet.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_03/merged_packet.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/gap_03/gap_audit.json', 'evaluation/sers_novelty_gap/t1_live_targeted_retrieval_run_v2/t1_live_report.json', 'dac_her/sers_targeted_retrieval_t1_live_guard.py', 'dac_her/sers_targeted_retrieval_t1_live_validation_v2.py', 'dac_her/sers_targeted_retrieval_t1_live_recovery_v2.py', 'campaigns/sers_novelty_gap/history/cli/verify_sers_targeted_retrieval_t0_freeze_v2.py', 'campaigns/sers_novelty_gap/history/cli/verify_sers_targeted_retrieval_t1_input_bundle_v1.py', 'scripts/verify_sers_targeted_retrieval_t1_v1_failure_evidence.py', 'scripts/preflight_sers_targeted_retrieval_t1_live_v2.py', 'scripts/run_sers_targeted_retrieval_t1_live_v2.py', 'scripts/verify_sers_targeted_retrieval_t1_live_v2.py', 'tests/test_sers_targeted_retrieval_t1_live_validation_v2.py']


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def _run_module(module: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    issues: list[str] = []

    if not MANIFEST_PATH.is_file():
        print("T1 final freeze v2 verification: FAIL")
        print(" - freeze manifest missing")
        return 2

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    body = dict(manifest)
    observed_id = body.pop("freeze_id", None)
    observed_sha = body.pop("manifest_sha256", None)
    expected_sha = _sha256_json(body)
    expected_id = (
        "sers_targeted_retrieval_t1_final_freeze_v2:"
        + expected_sha[:20]
    )
    if observed_sha != expected_sha:
        issues.append("freeze manifest SHA mismatch")
    if observed_id != expected_id:
        issues.append("freeze ID mismatch")

    if (
        manifest.get("source_evidence_commit")
        != EXPECTED_SOURCE_EVIDENCE_COMMIT
    ):
        issues.append("source evidence commit mismatch")

    # The freeze commit may be a child of the evidence commit. Require ancestry.
    current_head = _git("rev-parse", "HEAD")
    if current_head.returncode != 0:
        issues.append("cannot resolve current HEAD")
        current_head_text = "UNKNOWN"
    else:
        current_head_text = current_head.stdout.strip()
        anc = _git(
            "merge-base",
            "--is-ancestor",
            EXPECTED_SOURCE_EVIDENCE_COMMIT,
            current_head_text,
        )
        if anc.returncode != 0:
            issues.append(
                "source evidence commit is not an ancestor of current HEAD"
            )

    # Verify the final verifier's own bytes.
    expected_verifier_sha = manifest.get(
        "freeze_verifier_sha256"
    )
    observed_verifier_sha = _sha256(Path(__file__).resolve())
    if observed_verifier_sha != expected_verifier_sha:
        issues.append("freeze verifier SHA mismatch")

    # Verify current bytes and exact bytes at the evidence commit.
    file_hashes = manifest.get("critical_file_sha256", {})
    if set(file_hashes) != set(CRITICAL_FILES):
        issues.append("critical-file manifest set mismatch")
    for rel in CRITICAL_FILES:
        current = ROOT / rel
        expected = file_hashes.get(rel)
        if not current.is_file():
            issues.append(f"missing critical file: {rel}")
            continue
        if _sha256(current) != expected:
            issues.append(f"current critical-file SHA mismatch: {rel}")

        show = subprocess.run(
            [
                "git",
                "show",
                f"{EXPECTED_SOURCE_EVIDENCE_COMMIT}:{rel}",
            ],
            cwd=ROOT,
            capture_output=True,
        )
        if show.returncode != 0:
            issues.append(
                f"critical file absent at source evidence commit: {rel}"
            )
        else:
            observed_commit_sha = hashlib.sha256(
                show.stdout
            ).hexdigest()
            if observed_commit_sha != expected:
                issues.append(
                    f"source-commit critical-file SHA mismatch: {rel}"
                )

    # Re-check semantic identifiers and pass assertions from saved evidence.
    t0_manifest = json.loads(
        (
            ROOT / CRITICAL_FILES[0]
        ).read_text(encoding="utf-8")
    )
    bundle_manifest = json.loads(
        (
            ROOT / CRITICAL_FILES[1]
        ).read_text(encoding="utf-8")
    )
    spec = json.loads(
        (
            ROOT / CRITICAL_FILES[2]
        ).read_text(encoding="utf-8")
    )
    failure_manifest = json.loads(
        (
            ROOT
            / "evaluation/sers_novelty_gap/"
            "t1_live_targeted_retrieval_v1_failure_freeze_v1/"
            "failure_manifest.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads(
        (
            ROOT
            / "evaluation/sers_novelty_gap/"
            "t1_live_targeted_retrieval_run_v2/"
            "t1_live_report.json"
        ).read_text(encoding="utf-8")
    )
    v2_marker = json.loads(
        (
            ROOT
            / "evaluation/sers_novelty_gap/"
            "t1_live_targeted_retrieval_run_v2/"
            "LIVE_ATTEMPT_CONSUMED.json"
        ).read_text(encoding="utf-8")
    )

    checks = {
        "t0_freeze_id":
            t0_manifest.get("freeze_id") == EXPECTED_T0_FREEZE_ID,
        "input_bundle_id":
            bundle_manifest.get("bundle_id") == EXPECTED_INPUT_BUNDLE_ID,
        "spec_id":
            spec.get("spec_id") == EXPECTED_SPEC_ID,
        "v1_failure_freeze_id":
            failure_manifest.get("failure_freeze_id")
            == EXPECTED_V1_FAILURE_FREEZE_ID,
        "v1_rerun_forbidden":
            failure_manifest.get("v1_rerun_authorized") is False,
        "v1_gap1_network_replay_forbidden":
            failure_manifest.get(
                "v1_gap1_network_replay_authorized"
            ) is False,
        "v2_run_id":
            report.get("run_id") == EXPECTED_V2_RUN_ID,
        "v2_outcome":
            report.get("outcome") == EXPECTED_V2_OUTCOME,
        "provider_mode":
            report.get("provider_mode") == "STANDARD_2_PROVIDER",
        "providers":
            report.get("providers") == ["openalex", "crossref"],
        "provider_executions":
            report.get("successful_provider_execution_count") == 12
            and report.get("failed_provider_execution_count") == 0,
        "every_targeted_query_operational":
            report.get("every_targeted_query_operational") is True,
        "all_structural_checks_pass":
            report.get("all_structural_checks_pass") is True,
        "v1_gap1_not_replayed":
            report.get("v1_gap1_network_replayed") is False,
        "scientific_novelty_not_reassessed":
            report.get("scientific_novelty_reassessed") is False,
        "ranker_not_called":
            report.get("ranker_called") is False,
        "claim_reviewer_not_called":
            report.get("claim_reviewer_called") is False,
        "llm_calls_zero":
            report.get("llm_calls") == 0,
        "hypothesis_rewrite_not_called":
            report.get("hypothesis_rewrite_called") is False,
        "fresh_reserve_c_unconsumed":
            report.get("fresh_reserve_c_consumed") is False
            and v2_marker.get("fresh_reserve_c_consumed") is False,
        "automatic_next_stage_false":
            report.get("automatic_next_stage_authorized") is False,
        "v2_rerun_forbidden":
            v2_marker.get("rerun_authorized") is False,
    }
    for name, passed in checks.items():
        if not passed:
            issues.append(f"semantic check failed: {name}")

    # Re-run all offline lineage verifiers. These are network-free by contract.
    verifier_modules = [
        "scripts.verify_sers_targeted_retrieval_t0_freeze_v2",
        "scripts.verify_sers_targeted_retrieval_t1_input_bundle_v1",
        "scripts.verify_sers_targeted_retrieval_t1_v1_failure_evidence",
        "scripts.verify_sers_targeted_retrieval_t1_live_v2",
    ]
    verifier_outputs: list[tuple[str, bool, str]] = []
    for module in verifier_modules:
        ok, output = _run_module(module)
        verifier_outputs.append((module, ok, output))
        if not ok:
            issues.append(f"offline verifier failed: {module}")

    if issues:
        print("T1 final freeze v2 verification: FAIL")
        for issue in issues:
            print(" -", issue)
        for module, ok, output in verifier_outputs:
            if not ok:
                print()
                print(f"===== {module} =====")
                print(output)
        print("Network calls during freeze verification:", 0)
        print("LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T1 final freeze v2 verification: PASS")
    print("Freeze ID:", observed_id)
    print("Manifest SHA256:", observed_sha)
    print(
        "Source evidence commit:",
        EXPECTED_SOURCE_EVIDENCE_COMMIT,
    )
    print("Current HEAD:", current_head_text)
    print("T0 freeze:", EXPECTED_T0_FREEZE_ID)
    print("T1 input bundle:", EXPECTED_INPUT_BUNDLE_ID)
    print("T1 live spec:", EXPECTED_SPEC_ID)
    print("V1 failure freeze:", EXPECTED_V1_FAILURE_FREEZE_ID)
    print("V2 Run ID:", EXPECTED_V2_RUN_ID)
    print("V2 outcome:", EXPECTED_V2_OUTCOME)
    print("Providers: ['openalex', 'crossref']")
    print("Provider executions: 12 success / 0 failed")
    print("Every targeted query operational:", True)
    print("All structural checks pass:", True)
    print("V1 gap_01 network replayed:", False)
    print("Scientific novelty reassessed:", False)
    print("Ranker called:", False)
    print("Claim reviewer called:", False)
    print("LLM calls:", 0)
    print("Hypothesis rewrite called:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("Network calls during freeze verification:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
