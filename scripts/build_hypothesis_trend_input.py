from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_trend_grounding import (
    HypothesisTrendGroundingBundle,
)
from dac_her.hypothesis_trend_input import (
    build_trend_aware_hypothesis_input,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic Trend-aware hypothesis input envelope. "
            "This stage does not call or modify Hypothesis Maker."
        )
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
        help="Existing validated HypothesisContext JSON.",
    )
    parser.add_argument(
        "--trend-grounding",
        required=True,
        type=Path,
        help="alpha4c.5a HypothesisTrendGroundingBundle JSON.",
    )
    parser.add_argument(
        "--input-semantics-id",
        default=(
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    grounding = HypothesisTrendGroundingBundle.model_validate_json(
        args.trend_grounding.read_text(encoding="utf-8")
    )
    value = build_trend_aware_hypothesis_input(
        grounded_context=context,
        trend_grounding=grounding,
        input_semantics_id=args.input_semantics_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        value.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Trend-aware Hypothesis input built")
    print("Input ID:", value.input_id)
    print("Input SHA256:", value.input_sha256)
    print("Contract semantics:", value.contract_semantics_id)
    print("Input semantics:", value.input_semantics_id)
    print("Domain profile:", value.domain_profile_id)
    print("Corpus ID:", value.corpus_id)
    print("Grounded context:", value.grounded_context.context_id)
    print(
        "Trend grounding:",
        value.trend_grounding.bundle_id,
    )
    print("Trend views:", len(value.trend_views))
    print("Lane counts:", value.lane_counts)
    print("Hypothesis Maker consumption enabled:", False)
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
