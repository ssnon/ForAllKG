from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.hypothesis_trend_compiler import (
    TrendAwareHypothesisCompiler,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
)
from dac_her.hypothesis_trend_validator import (
    TrendAwareHypothesisValidator,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and validate an explicit trend-aware hypothesis "
            "draft. No LLM is called."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--draft",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--portfolio-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    source = TrendAwareHypothesisInput.model_validate_json(
        args.input.read_text(encoding="utf-8")
    )
    draft = (
        TrendAwareHypothesisPortfolioDraft.
        model_validate_json(
            args.draft.read_text(encoding="utf-8")
        )
    )

    compiler = TrendAwareHypothesisCompiler()
    portfolio = compiler.compile(source, draft)
    validation = TrendAwareHypothesisValidator().validate(
        source,
        portfolio,
    )

    args.portfolio_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.validation_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.portfolio_output.write_text(
        portfolio.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.validation_output.write_text(
        validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Trend-aware Hypothesis deterministic compile")
    print("Portfolio:", portfolio.portfolio_id)
    print("Hypotheses:", len(portfolio.hypotheses))
    print(
        "Compiler semantics:",
        compiler.semantics_id,
    )
    print(
        "Validator semantics:",
        validation.semantics_id,
    )
    print("Validation passes:", validation.passes)
    print("Errors:", validation.errors)
    print("Warnings:", validation.warnings)
    print(
        "Portfolio saved:",
        args.portfolio_output,
    )
    print(
        "Validation saved:",
        args.validation_output,
    )
    return 0 if validation.passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
