from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaigns.sers_novelty_gap.sers_targeted_retrieval_t0_offline_validation import (
    build_t0_offline_report,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t0_targeted_retrieval_canonicalization_v1_run"
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-t0-offline", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    if not args.run or not args.confirm_t0_offline:
        parser.error("--run and --confirm-t0-offline are required")
    if args.run_root.exists():
        print("T0 offline run: FAIL")
        print(" - run root already exists:", args.run_root)
        return 2

    report = build_t0_offline_report()
    args.run_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.run_root / "t0_offline_report.json", report)
    marker = (
        "STRUCTURAL_PASS.json"
        if report["structural_outcome"].endswith("_PASS")
        else "STRUCTURAL_FAIL.json"
    )
    _atomic_json(
        args.run_root / marker,
        {
            "run_id": report["run_id"],
            "structural_outcome": report["structural_outcome"],
        },
    )

    print("SERS Targeted Retrieval T0 Offline")
    print("Run ID:", report["run_id"])
    print("Structural outcome:", report["structural_outcome"])
    for key, value in report["checks"].items():
        print(f"{key}: {value}")
    print("Scenario counts:", report["scenario_counts"])
    print("Targeted retrieval called:", False)
    print("Provider calls:", 0)
    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0 if report["structural_outcome"].endswith("_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
