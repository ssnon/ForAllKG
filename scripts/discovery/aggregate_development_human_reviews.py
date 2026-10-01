
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.development_human_review import (
    aggregate_review_responses,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate blinded development human-review JSON responses."
    )
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    responses = sorted(args.responses_dir.glob("*.json"))
    result = aggregate_review_responses(
        key_path=args.key,
        response_paths=responses,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Development human review aggregation complete")
    print("response files:", len(responses))
    print("rating rows:", result["rating_row_count"])
    for arm, dimensions in result["arm_summary"].items():
        pursue = dimensions["pursue_likelihood"]
        distinct = dimensions["conceptual_distinctiveness"]
        print(
            arm,
            "| pursue mean=", pursue["mean"],
            "| conceptual distinctiveness mean=", distinct["mean"],
        )
    print("output:", args.output)
    print("DEVELOPMENT_EXPLORATORY_ONLY=True")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
