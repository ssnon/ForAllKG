from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.nonobviousness_post_generation import (
    filter_alpha6_portfolio_by_nonobviousness,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


def _gate_arg(
    value: str,
) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--gate must be CANDIDATE_ID=PATH"
        )

    candidate_id, raw_path = (
        value.split(
            "=",
            1,
        )
    )

    candidate_id = (
        candidate_id.strip()
    )

    if not candidate_id:
        raise argparse.ArgumentTypeError(
            "empty candidate ID in --gate"
        )

    return (
        candidate_id,
        Path(raw_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Alpha6 survivors using fresh candidate-specific "
            "N10 non-obviousness production gates."
        )
    )

    parser.add_argument(
        "--portfolio",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--refinement-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        type=_gate_arg,
    )

    parser.add_argument(
        "--output-portfolio",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    portfolio = (
        HypothesisPortfolio
        .model_validate_json(
            args.portfolio.read_text(
                encoding="utf-8"
            )
        )
    )

    refinement = (
        NoveltyRefinementReport
        .model_validate_json(
            args.refinement_report
            .read_text(
                encoding="utf-8"
            )
        )
    )

    gates = {}

    for candidate_id, path in args.gate:
        if candidate_id in gates:
            raise ValueError(
                "duplicate --gate candidate ID: "
                + candidate_id
            )

        gates[
            candidate_id
        ] = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    filtered, report = (
        filter_alpha6_portfolio_by_nonobviousness(
            portfolio=portfolio,
            refinement_report=refinement,
            gates_by_candidate_id=gates,
        )
    )

    args.output_portfolio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_portfolio.write_text(
        json.dumps(
            filtered.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    args.output_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Alpha6 post-generation N10 enforcement complete"
    )
    print(
        "Alpha6 survivors:",
        report["alpha6_survivor_count"],
    )
    print(
        "Final survivors:",
        report["final_survivor_count"],
    )
    print(
        "Removed by N10:",
        report[
            "removed_by_post_generation_n10_count"
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
