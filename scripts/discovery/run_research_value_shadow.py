from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.feasibility.experimental_contracts import (
    ExperimentalRealizabilityReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.research_value_shadow import (
    assess_research_value_portfolio,
)
from pipeline_core.runtime.validation_contracts import (
    ValidationSpecification,
)


def _load_many(
    directory: Path,
    model,
):
    return [
        model.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(
            directory.glob("*.json")
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only research-value assessment over final hypothesis "
            "portfolio + existing validation/experimental feasibility artifacts. "
            "Novelty signals are intentionally excluded."
        )
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--feasibility-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    portfolio = (
        HypothesisPortfolio.model_validate_json(
            args.portfolio.read_text(
                encoding="utf-8"
            )
        )
    )

    validation_dir = (
        args.feasibility_dir
        / "validation"
    )
    experimental_dir = (
        args.feasibility_dir
        / "experimental"
    )

    if not validation_dir.is_dir():
        raise RuntimeError(
            f"missing validation directory: {validation_dir}"
        )

    if not experimental_dir.is_dir():
        raise RuntimeError(
            f"missing experimental directory: {experimental_dir}"
        )

    specifications = _load_many(
        validation_dir,
        ValidationSpecification,
    )
    experimental = _load_many(
        experimental_dir,
        ExperimentalRealizabilityReport,
    )

    report = assess_research_value_portfolio(
        portfolio=portfolio,
        specifications=specifications,
        experimental_reports=experimental,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== RESEARCH VALUE SHADOW ==="
    )
    print(
        "portfolio:",
        portfolio.portfolio_id,
    )
    print(
        "assessed:",
        report.assessed_count,
        "/",
        report.hypothesis_count,
    )

    for card in report.cards:
        print(
            "\n",
            card.hypothesis_id,
            "|",
            card.value_argument_class,
        )
        print(
            " discrimination:",
            card.mechanistic_discrimination.signal,
        )
        print(
            " two-sided outcome:",
            card.two_sided_outcome_informativeness.signal,
        )
        print(
            " observable decisiveness:",
            card.observable_decisiveness.signal,
        )
        print(
            " information-gain proxy:",
            card.information_gain_proxy.signal,
        )
        print(
            " experimental resolvability:",
            card.experimental_resolvability.signal,
        )
        print(
            " cost/effort:",
            card.relative_cost_burden,
            "/",
            card.relative_effort_burden,
        )

    print(
        "\nNovelty signals consumed: False"
    )
    print(
        "Diagnostic only; selection unchanged."
    )
    print(
        "artifact:",
        args.output,
    )
    print(
        "RESEARCH_VALUE_SHADOW_COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
