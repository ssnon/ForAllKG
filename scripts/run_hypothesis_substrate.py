from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_compiler import HypothesisCompileError
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolioDraft
from dac_her.hypothesis_maker import HypothesisMakerSubstrate


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and validate a HypothesisPortfolioDraft without an LLM")
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    context = HypothesisContext.model_validate(_load(args.context))
    draft = HypothesisPortfolioDraft.model_validate(_load(args.draft))
    substrate = HypothesisMakerSubstrate()

    try:
        outcome = substrate.run(context, draft)
    except HypothesisCompileError as exc:
        print("Hypothesis compilation failed")
        for issue in exc.issues:
            print(f"ERROR {issue.code} @ {issue.location}: {issue.message}")
        return 3

    portfolio_path = Path(str(args.output_prefix) + ".portfolio.json")
    validation_path = Path(str(args.output_prefix) + ".validation.json")
    portfolio_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_path.write_text(outcome.portfolio.model_dump_json(indent=2), encoding="utf-8")
    validation_path.write_text(outcome.validation.model_dump_json(indent=2), encoding="utf-8")

    print("Hypothesis substrate:", "PASS" if outcome.accepted else "FAIL")
    print("Hypotheses:", len(outcome.portfolio.hypotheses))
    print("Validation errors/warnings:", outcome.validation.errors, outcome.validation.warnings)
    for issue in outcome.validation.issues:
        print(f"{issue.severity.upper()} {issue.code} @ {issue.location}: {issue.message}")
    print("Saved portfolio:", portfolio_path)
    print("Saved validation:", validation_path)
    return 0 if outcome.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
