from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.nonobviousness_dual_run_comparison import (
    build_nonobviousness_dual_run_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an observational v1/v2 non-obviousness "
            "comparison artifact. This runner does not create or "
            "modify Alpha6 production authority."
        )
    )

    parser.add_argument(
        "--query-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--intake-shadow",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--full-shadow",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--runtime-authority-policy",
        choices=[
            "v1_only",
            "v2_production",
        ],
        default="v1_only",
        help=(
            "Record which separately compiled production "
            "gate the surrounding runtime will consume. "
            "The comparison artifact itself remains "
            "non-authoritative."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(
            encoding="utf-8"
        )
    )

    intake = json.loads(
        args.intake_shadow.read_text(
            encoding="utf-8"
        )
    )

    full = json.loads(
        args.full_shadow.read_text(
            encoding="utf-8"
        )
    )

    result = (
        build_nonobviousness_dual_run_comparison(
            query_plan=plan,
            intake_shadow=intake,
            full_shadow=full,
            runtime_authority_policy=(
                args.runtime_authority_policy
            ),
        )
    )

    # Defensive runner-level authority assertions.
    if result.get("comparison_only") is not True:
        raise RuntimeError(
            "dual-run artifact is not comparison-only"
        )

    if result.get("production_authority") is not False:
        raise RuntimeError(
            "dual-run comparison unexpectedly has "
            "production authority"
        )

    if (
        result.get(
            "candidate_has_production_authority"
        )
        is not False
    ):
        raise RuntimeError(
            "v2 candidate unexpectedly has "
            "production authority"
        )

    if (
        result.get("authority_policy")
        != args.runtime_authority_policy
    ):
        raise RuntimeError(
            "dual-run comparison authority policy "
            "does not match requested runtime policy"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "N10 non-obviousness dual-run comparison built"
    )

    print(
        "Hypotheses:",
        result["hypothesis_count"],
    )

    print(
        "Transitions:",
        result[
            "selection_transition_counts"
        ],
    )

    print(
        "Positive-authority divergence:",
        result[
            "positive_authority_divergence_count"
        ],
    )

    print(
        "Authority policy:",
        result["authority_policy"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
