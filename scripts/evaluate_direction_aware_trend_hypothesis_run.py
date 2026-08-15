from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_directional_run_record import (
    DirectionAwareTrendHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_evaluation import (
    evaluate_run,
    load_protocol,
    load_reserve_manifest,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("seen_regression", "reserve"),
        required=True,
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--reserve-manifest", type=Path, default=None)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--final-draft", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load(path: Path, model):
    return model.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    reserve = (
        load_reserve_manifest(args.reserve_manifest)
        if args.reserve_manifest is not None
        else None
    )
    source = _load(args.input, TrendAwareHypothesisInput)
    run = _load(
        args.run,
        DirectionAwareTrendHypothesisMakerRunRecord,
    )
    draft = _load(
        args.final_draft,
        DirectionAwareTrendHypothesisPortfolioDraft,
    )
    portfolio = _load(
        args.portfolio,
        DirectionAwareTrendHypothesisPortfolio,
    )

    report = evaluate_run(
        root=Path.cwd(),
        protocol=protocol,
        source=source,
        final_draft=draft,
        run_record=run,
        portfolio=portfolio,
        evaluation_mode=args.mode,
        reserve_manifest=reserve,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("alpha4c.5e Trend/Hypothesis evaluation")
    print("Mode:", report.evaluation_mode)
    print("Evaluation ID:", report.evaluation_id)
    print("Protocol SHA256:", report.protocol_sha256)
    print("Fatal issues:", report.fatal_issue_count)
    print("Nonfatal observations:", report.observation_count)
    print("Hypotheses:", report.hypothesis_count)
    print("Abstained:", report.abstained)
    print("Generation attempts:", report.generation_attempts)
    print("Repair attempts:", report.repair_attempts)
    print(
        "Revalidation errors/warnings:",
        report.revalidation_errors,
        report.revalidation_warnings,
    )
    print(
        "Count thresholds used for acceptance:",
        report.count_thresholds_used_for_acceptance,
    )
    print("Accepted:", report.accepted)
    print("Reserve consumed:", report.reserve_consumed)
    print("Output:", args.output)

    if not report.accepted:
        print("Fatal rule codes:")
        for row in report.issues:
            if row.severity == "fatal":
                print(
                    f"  - {row.code}: {row.message}"
                )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
