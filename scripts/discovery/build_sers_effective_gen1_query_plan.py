
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.sers_effective_gen1_certification_closeout import (
    build_effective_subset_query_plan,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-query-plan", required=True, type=Path)
    p.add_argument("--effective-portfolio", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    plan = LiteratureQueryPlan.model_validate_json(
        args.source_query_plan.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.effective_portfolio.read_text(encoding="utf-8")
    )
    subset = build_effective_subset_query_plan(
        source_plan=plan,
        effective_portfolio=portfolio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        subset.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Effective Gen1 query-plan subset built")
    print("Portfolio:", subset.source_portfolio_id)
    print("Hypotheses:", len(subset.claims))
    print("Queries:", len(subset.queries))
    print("Plan:", subset.plan_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
