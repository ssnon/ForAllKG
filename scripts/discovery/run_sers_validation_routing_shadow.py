#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.fdtd_applicability_contracts import SERSFDTDApplicabilityBundle
from domains.sers.validation_routing import SERSHypothesisValidationRouter


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shadow-only SERS validation-route plan. FDTD is routed only "
            "to classical-EM subclaims; whole SERS outcomes remain separate."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument(
        "--fdtd-applicability",
        help=(
            "Optional existing FDTD applicability bundle. If omitted, the router "
            "runs the deterministic FDTD applicability analyzer first."
        ),
    )
    parser.add_argument("--hypothesis-id", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).resolve()
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )

    applicability = None
    if args.fdtd_applicability:
        applicability = SERSFDTDApplicabilityBundle.model_validate_json(
            Path(args.fdtd_applicability).resolve().read_text(encoding="utf-8")
        )

    result = SERSHypothesisValidationRouter().plan_portfolio(
        portfolio,
        fdtd_applicability=applicability,
        hypothesis_ids=set(args.hypothesis_id) or None,
    )
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "sers.validation_plans.shadow.json"
    _write_json(result_path, result)

    route_counts = Counter(
        route.route_kind
        for plan in result.plans
        for route in plan.routes
    )
    print(f"validation_plans={result_path}")
    print(f"plan_count={len(result.plans)}")
    print("route_counts=" + json.dumps(dict(sorted(route_counts.items())), sort_keys=True))
    print(
        "integrated_experiment_required_count="
        + str(sum(plan.integrated_experiment_required for plan in result.plans))
    )
    print(
        "mechanism_claim_count="
        + str(sum(plan.mechanism_claim_present for plan in result.plans))
    )
    print(
        "context_note_count="
        + str(sum(len(plan.context_notes) for plan in result.plans))
    )
    print("evidence_authority_created=false")
    print("hypothesis_rejection_authority=false")
    print("feedback_generation_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
