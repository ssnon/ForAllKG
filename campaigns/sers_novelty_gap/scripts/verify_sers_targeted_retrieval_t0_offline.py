from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaigns.sers_novelty_gap.sers_targeted_retrieval_t0_offline_validation import (
    build_t0_offline_report,
)
from campaigns.sers_novelty_gap.scripts.run_sers_targeted_retrieval_t0_offline import DEFAULT_RUN_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    report_path = args.run_root / "t0_offline_report.json"
    if not report_path.is_file():
        print("T0 offline verification: FAIL")
        print(" - report missing:", report_path)
        return 2

    stored = json.loads(report_path.read_text(encoding="utf-8"))
    recomputed = build_t0_offline_report()
    issues = []
    if stored != recomputed:
        issues.append("stored report != offline recomputation")
    if stored.get("structural_outcome") != (
        "SERS_TARGETED_RETRIEVAL_T0_CANONICALIZATION_OFFLINE_PASS"
    ):
        issues.append("stored structural outcome is not PASS")

    if issues:
        print("T0 offline verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls during verification:", 0)
        print("LLM calls during verification:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T0 offline verification: PASS")
    print("Run ID:", stored["run_id"])
    print("Shared canonicalization checks:", len(stored["checks"]))
    print("Targeted retrieval called:", False)
    print("Provider calls:", 0)
    print("Network calls during verification:", 0)
    print("LLM calls during verification:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
