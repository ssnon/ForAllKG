from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one HypothesisPortfolio with v2.6.2 deterministic hard gates."
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_path = Path(args.context)
    portfolio_path = Path(args.portfolio)

    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    report = HypothesisBenchmarkEvaluator().evaluate(context, portfolio)

    output = (
        Path(args.output)
        if args.output
        else portfolio_path.with_name(portfolio_path.stem + ".evaluation_v262.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Hypothesis evaluation complete")
    print("Evaluator:", report.evaluator_version)
    print("Hard gate:", "PASS" if report.hard_gate_passed else "FAIL")
    print("Hard errors/warnings:", report.hard_gate_errors, report.hard_gate_warnings)
    print("Diagnostics:", len(report.diagnostics))
    for issue in report.hard_gate_issues:
        print(f"{issue.severity.upper()} {issue.code} @ {issue.location}: {issue.message}")
    for issue in report.diagnostics:
        print(f"DIAGNOSTIC {issue.code} @ {issue.location}: {issue.message}")
    print("Saved:", output)

    if not report.hard_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
