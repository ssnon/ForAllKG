from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityAssessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SD1 diagnostic-only statement-level evidence diversity "
            "for an existing hypothesis portfolio."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    report = HypothesisEvidenceDiversityAssessor().assess(context, portfolio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Hypothesis evidence diversity report")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print(
        "Statement coverage:",
        f"{report.used_statement_count}/{report.eligible_statement_count}",
        f"({report.eligible_statement_coverage:.3f})",
    )
    print("Shared-core statements:", report.shared_core_statement_count)
    print("Distinct premise sets:", report.distinct_premise_set_count)
    print(
        "Exact duplicate groups:",
        report.exact_premise_set_duplicate_group_count,
    )
    print(
        "Mean/max pairwise statement Jaccard:",
        f"{report.mean_pairwise_statement_jaccard:.3f}/"
        f"{report.max_pairwise_statement_jaccard:.3f}",
    )
    print(
        "Multi-paper used statements:",
        report.multi_paper_used_statement_count,
    )
    print(
        "Mean papers per used statement:",
        f"{report.mean_papers_per_used_statement:.3f}",
    )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
