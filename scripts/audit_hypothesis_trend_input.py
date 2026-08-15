from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.hypothesis_trend_grounding import (
    HypothesisTrendGroundingBundle,
)
from dac_her.hypothesis_trend_input import (
    audit_trend_grounding_for_input,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an alpha4c.5a grounding bundle for deterministic "
            "alpha4c.5b hypothesis-input role projection."
        )
    )
    parser.add_argument(
        "--trend-grounding",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--input-semantics-id",
        default=(
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    grounding = HypothesisTrendGroundingBundle.model_validate_json(
        args.trend_grounding.read_text(encoding="utf-8")
    )
    audit = audit_trend_grounding_for_input(
        grounding,
        input_semantics_id=args.input_semantics_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        audit.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Hypothesis Trend input projection audit")
    print("Grounding bundle:", audit.grounding_bundle_id)
    print("Grounding SHA256:", audit.grounding_bundle_sha256)
    print("Corpus ID:", audit.corpus_binding.corpus_id)
    print("Relations:", audit.relation_count)
    print("Views:", audit.view_count)
    print(
        "Cross-context statuses:",
        audit.cross_context_status_counts,
    )
    print("Lane counts:", audit.lane_counts)
    print("Issues:", audit.issues)
    print("Structural gate:", audit.structural_gate)
    print("Saved:", args.output)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
