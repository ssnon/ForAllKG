from __future__ import annotations

import json
from pathlib import Path

from dac_her.canonicalization_doi_conflict_recheck import (
    OUTPUT_ROOT,
    canonical_json,
    run_recheck,
)

ROOT = Path.cwd()


def main() -> int:
    root = ROOT / OUTPUT_ROOT
    canonical_path = root / "canonical_prior_art_v2.json"
    report_path = root / "recheck_report.json"
    if not canonical_path.is_file() or not report_path.is_file():
        print("canonicalization DOI-conflict verification: FAIL")
        print(" - required recheck artifacts missing")
        return 2

    stored_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    stored_canonical = json.loads(
        canonical_path.read_text(encoding="utf-8")
    )

    recomputed_canonical, recomputed_report = run_recheck(ROOT)

    issues = []
    if canonical_json(stored_report) != canonical_json(recomputed_report):
        issues.append("offline report recomputation mismatch")
    if canonical_json(stored_canonical) != canonical_json(
        recomputed_canonical.model_dump(mode="json")
    ):
        issues.append("offline canonical packet recomputation mismatch")
    if stored_report.get("outcome") != (
        "CANONICALIZATION_DOI_CONFLICT_HARDENING_PASS"
    ):
        issues.append("outcome is not PASS")
    if stored_report.get(
        "canonical_packet_eligible_for_dev_ranker_validation"
    ) is not True:
        issues.append("canonical packet not ranker-eligible")
    if stored_report.get("network_calls") != 0:
        issues.append("unexpected network calls recorded")
    if stored_report.get("llm_calls") != 0:
        issues.append("unexpected LLM calls")

    if issues:
        print("canonicalization DOI-conflict verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    print("canonicalization DOI-conflict verification: PASS")
    print("Run ID:", stored_report["run_id"])
    print("Outcome:", stored_report["outcome"])
    print("Checks:", stored_report["checks"])
    print("Counts:", stored_report["counts"])
    print(
        "Canonical packet eligible for DEV ranker validation:",
        True,
    )
    print("Network calls:", 0)
    print("Ranker used:", False)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
