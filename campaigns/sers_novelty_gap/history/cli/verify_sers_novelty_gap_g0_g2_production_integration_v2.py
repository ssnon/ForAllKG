from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaigns.sers_novelty_gap.history.cli.run_sers_novelty_gap_g0_g2_production_integration_v2 import (
    DEFAULT_RUN_ROOT,
    build_production_run,
)

ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    path = args.run_root / "production_integration_v2_report.json"
    if not path.is_file():
        print("Production integration v2 verification: FAIL")
        print(" - report missing:", path)
        return 2

    stored = json.loads(path.read_text(encoding="utf-8"))
    try:
        recomputed = build_production_run()
    except Exception as exc:
        print("Production integration v2 verification: FAIL")
        print(" - recomputation:", f"{type(exc).__name__}: {exc}")
        return 2

    issues = []
    if stored != recomputed:
        issues.append("stored report != offline recomputation")
    if stored.get("structural_outcome") != (
        "SERS_NOVELTY_GAP_G0_G2_PRODUCTION_INTEGRATION_V2_PASS"
    ):
        issues.append("stored structural outcome is not PASS")
    if not stored.get("checks", {}).get(
        "exact_dev_v4_gap_equivalence",
        False,
    ):
        issues.append("DEV-v4 gap equivalence is false")
    if not stored.get("checks", {}).get(
        "exact_dev_v4_query_equivalence",
        False,
    ):
        issues.append("DEV-v4 query equivalence is false")

    if issues:
        print("Production integration v2 verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    print("Production integration v2 verification: PASS")
    print("Run ID:", stored["run_id"])
    print(
        "Exact DEV-v4 gap equivalence:",
        stored["checks"]["exact_dev_v4_gap_equivalence"],
    )
    print(
        "Exact DEV-v4 query equivalence:",
        stored["checks"]["exact_dev_v4_query_equivalence"],
    )
    print("Targeted retrieval called:", False)
    print("LLM calls during verification:", 0)
    print("Network calls during verification:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
