from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.nonobviousness_shadow import (
    build_nonobviousness_shadow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the N9 non-obviousness shadow intake: "
            "external novelty -> residue -> branch "
            "specification gate. This stage does not "
            "perform targeted closure or final adjudication."
        )
    )

    parser.add_argument(
        "--query-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--external-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--required-bridge-bindings",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic typed source-span required_bridge bindings. "
            "These do not carry novelty authority."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(
            encoding="utf-8"
        )
    )

    report = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(
            encoding="utf-8"
        )
    )

    source_portfolio = (
        HypothesisPortfolio.model_validate_json(
            args.portfolio.read_text(
                encoding="utf-8"
            )
        )
        if args.portfolio is not None
        else None
    )

    required_bridge_bindings = None
    if args.required_bridge_bindings is not None:
        payload = json.loads(
            args.required_bridge_bindings.read_text(
                encoding="utf-8"
            )
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != "typed-required-bridge-live-ablation-v1"
            or not isinstance(payload.get("bindings_by_claim"), dict)
        ):
            raise ValueError(
                "invalid typed required_bridge binding artifact"
            )
        required_bridge_bindings = payload["bindings_by_claim"]

    result = build_nonobviousness_shadow(
        plan=plan,
        report=report,
        source_portfolio=source_portfolio,
        required_bridge_bindings=required_bridge_bindings,
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

    print("N9 non-obviousness shadow built")
    print(
        "Hypotheses:",
        result["hypothesis_count"],
    )
    print(
        "Claims:",
        result["claim_count"],
    )
    print(
        "States:",
        result["shadow_state_counts"],
    )
    print(
        "Scientific selection changed:",
        result["scientific_selection_changed"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
