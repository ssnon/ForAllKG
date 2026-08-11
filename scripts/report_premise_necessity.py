from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.premise_necessity_diagnostic import (
    PremiseNecessityDiagnosticAssessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PS1 deterministic premise-necessity / over-selection "
            "diagnostic for one Alpha4 portfolio."
        )
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--axis-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--axis-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def _write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.read_text(
            encoding="utf-8"
        )
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(
            encoding="utf-8"
        )
    )
    plan = DiscoveryAxisPlan.model_validate_json(
        args.axis_plan.read_text(
            encoding="utf-8"
        )
    )
    axis_report = (
        DiscoveryAxisSynthesisReport.model_validate_json(
            args.axis_report.read_text(
                encoding="utf-8"
            )
        )
    )

    index_dir = args.index_dir or (
        PROJECT_ROOT
        / "data_dac"
        / "corpus"
        / context.corpus_id
        / "mechanism"
        / "navigation"
        / "node_index"
    )

    assessor = PremiseNecessityDiagnosticAssessor(
        index_dir=index_dir,
        device=args.device,
    )
    report = assessor.assess(
        context,
        portfolio,
        plan,
        axis_report,
    )
    _write_json(
        args.output,
        report,
    )

    print("PS1 premise necessity diagnostic")
    print("Report:", report.report_id)
    print(
        "Hypotheses / eligible / used:",
        f"{report.hypothesis_count}/"
        f"{report.eligible_statement_count}/"
        f"{report.used_statement_count}",
    )
    print(
        "Exact same premise set across all:",
        report.exact_same_premise_set_across_all_hypotheses,
    )
    print(
        "Shared core:",
        report.shared_core_statement_count,
        report.shared_core_statement_ids,
    )
    print(
        "Selected Pareto-dominated incidences:",
        report.selected_pareto_dominated_incidence_count,
    )
    print(
        "Hypotheses with dominated selected premise:",
        report.hypotheses_with_pareto_dominated_selected_premise_count,
    )

    print()
    print("GLOBAL PREMISE USE")
    for row in sorted(
        report.global_premise_diagnostics,
        key=lambda x: (
            -x.usage_count,
            x.statement_id,
        ),
    ):
        print(
            f"- {row.statement_id}: "
            f"use={row.usage_count}/{row.hypothesis_count}; "
            f"core_mean={row.mean_core_score:.3f}; "
            f"core_range={row.core_score_range:.3f}; "
            f"axis_mean={row.mean_axis_score:.3f}; "
            f"axis_range={row.axis_score_range:.3f}"
        )

    print()
    print("PER HYPOTHESIS")
    for card in report.cards:
        print()
        print(card.title)
        print(
            " axis:",
            card.axis_label,
        )
        for row in card.selected_premises:
            s = row.scores
            p = row.provenance
            print(
                f"  {row.statement_id}: "
                f"core={s.core_score:.3f}(r{s.core_rank}); "
                f"axis={s.axis_score:.3f}(r{s.axis_rank}); "
                f"pred={s.prediction_score:.3f}(r{s.prediction_rank}); "
                f"uniq_edge={p.unique_scientific_edge_fraction:.2f}; "
                f"uniq_node={p.unique_scientific_node_fraction:.2f}"
            )
            if row.pareto_dominated_by_unselected_statement_ids:
                print(
                    "    PARETO-DOMINATED BY:",
                    row.pareto_dominated_by_unselected_statement_ids,
                )
            if row.diagnostic_flags:
                print(
                    "    flags:",
                    row.diagnostic_flags,
                )

        if card.best_unselected_by_core_score is not None:
            row = card.best_unselected_by_core_score
            print(
                "  best unselected core:",
                row.statement_id,
                f"{row.scores.core_score:.3f}",
                f"(rank {row.scores.core_rank})",
            )
        if card.best_unselected_by_axis_score is not None:
            row = card.best_unselected_by_axis_score
            print(
                "  best unselected axis:",
                row.statement_id,
                f"{row.scores.axis_score:.3f}",
                f"(rank {row.scores.axis_rank})",
            )

    print()
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
