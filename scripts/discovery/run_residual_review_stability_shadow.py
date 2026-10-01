
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.residual_review_stability_shadow import (
    compare_residual_runs,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fulltext-a", required=True, type=Path)
    p.add_argument("--fulltext-b", required=True, type=Path)
    p.add_argument("--aggregation-a", required=True, type=Path)
    p.add_argument("--aggregation-b", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = compare_residual_runs(
        fulltext_a=load(args.fulltext_a),
        fulltext_b=load(args.fulltext_b),
        aggregation_a=load(args.aggregation_a),
        aggregation_b=load(args.aggregation_b),
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("===== RESIDUAL REVIEW STABILITY SHADOW =====")
    print(
        "Authority-relevant stable:",
        report["authority_relevant_stable"],
    )
    print(
        "Relation instabilities:",
        report[
            "authority_relevant_relation_instability_count"
        ],
    )
    print(
        "Disposition instabilities:",
        report[
            "aggregation_disposition_instability_count"
        ],
    )
    for row in report["composites"]:
        print(
            row["hypothesis_id"],
            "|", row["claim_id"],
            "| stable=", row["stable"],
            "|", row["disposition_a"],
            "=>", row["disposition_b"],
        )
    print("SHADOW_ONLY=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
