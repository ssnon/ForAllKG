from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.adaptive_yield_evaluation import (
    AdaptiveYieldCaseAudit,
    summarize_cohort,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", action="append", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    cases = [
        AdaptiveYieldCaseAudit.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.case
    ]
    payload = summarize_cohort(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Adaptive Scientific Yield cohort summary complete")
    print("cases:", payload["case_count"])
    print("scientific superiority established: false")
    print("overall winner selected: false")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
