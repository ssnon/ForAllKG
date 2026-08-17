from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_sers_r2_final_reassessment_v1 import (
    COMPLETE_PATH,
    REPORT_PATH,
    ROOT,
    build_report,
    canonical,
    sha256_bytes,
    validate_inputs,
)


def main() -> int:
    try:
        ctx = validate_inputs(require_output_absent=False)
    except Exception as exc:
        print("SERS R2 final reassessment verification: FAIL")
        print(" - input validation:", exc)
        return 2

    issues: list[str] = []
    if not REPORT_PATH.is_file():
        issues.append("R2 report missing")
    if not COMPLETE_PATH.is_file():
        issues.append("R2 complete marker missing")
    if issues:
        print("SERS R2 final reassessment verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    marker = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    expected = build_report(ctx)
    if canonical(report) != canonical(expected):
        issues.append("R2 report differs from deterministic reconstruction")

    payload = dict(report)
    report_id = payload.pop("report_id", None)
    report_sha = payload.pop("report_sha256", None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    if report_sha != recomputed:
        issues.append("R2 report SHA mismatch")
    if report_id != "sers_r2_final_reassessment_report_v1:" + recomputed[:20]:
        issues.append("R2 report ID mismatch")

    if marker.get("complete") is not True or marker.get("stop") is not True:
        issues.append("R2 complete/STOP marker invalid")
    if marker.get("report_id") != report_id or marker.get("report_sha256") != report_sha:
        issues.append("R2 marker report lineage mismatch")

    for key in [
        "r1_executed",
        "hypothesis_rewrite_called",
        "i0_started",
        "fresh_reserve_c_consumed",
        "automatic_next_stage_authorized",
    ]:
        if marker.get(key) is not False:
            issues.append(f"R2 marker guard changed:{key}")

    guards = report.get("epistemic_guards", {})
    expected_false = [
        "r1_executed",
        "hypothesis_rewrite_called",
        "external_prior_art_used_as_positive_premise",
        "literature_absence_claimed",
        "i0_started",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in expected_false:
        if guards.get(key) is not False:
            issues.append(f"R2 report guard changed:{key}")
    if guards.get("runtime_llm_calls") != 0:
        issues.append("R2 runtime LLM calls changed")
    if guards.get("runtime_network_calls") != 0:
        issues.append("R2 runtime network calls changed")
    if guards.get("stop_after_r2") is not True:
        issues.append("R2 report STOP flag missing")

    decisions = {row["hypothesis_id"]: row for row in report.get("hypothesis_decisions", [])}
    if len(decisions) != 3:
        issues.append("R2 must contain exactly three hypothesis decisions")
    dispositions = {row.get("candidate_disposition") for row in decisions.values()}
    if dispositions != {"KEEP_BOUNDED_EXTENSION", "REJECT_AS_FORMULATED", "KEEP_RELATIONAL_GAP_CANDIDATE"}:
        issues.append("R2 disposition set changed")
    if report.get("portfolio_decision", {}).get("hypothesis_rewrites") != 0:
        issues.append("portfolio records hypothesis rewrite")
    if report.get("portfolio_decision", {}).get("r1_executed") is not False:
        issues.append("portfolio records R1 execution")

    source_code_commit = report.get("source_lineage", {}).get("source_r2_code_commit")
    if not isinstance(source_code_commit, str):
        issues.append("R2 source code commit missing")
    else:
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_code_commit, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0:
            issues.append("R2 source code commit not ancestor of HEAD")

    if issues:
        print("SERS R2 final reassessment verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("SERS R2 final reassessment verification: PASS")
    print("Report ID:", report_id)
    print("Report SHA256:", report_sha)
    for row in report["hypothesis_decisions"]:
        print(row["hypothesis_id"], "=>", row["candidate_disposition"])
    print("Primary remaining candidate:", report["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"])
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
