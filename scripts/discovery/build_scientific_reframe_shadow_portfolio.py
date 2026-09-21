from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.critic_contracts import (
    ScientificReframeCriticReport,
)
from pipeline_core.discovery.reframing.portfolio import (
    build_scientific_reframe_shadow_portfolio,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic Pareto/slot shadow portfolio from scientific "
            "reframe candidates and their multidimensional critic vectors."
        )
    )
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--critic", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    shadow = ScientificReframingShadowReport.model_validate(_load(args.shadow))
    critic = ScientificReframeCriticReport.model_validate(_load(args.critic))
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=shadow,
        critic=critic,
    )

    output = args.output or (args.shadow.parent / "scientific_reframing_portfolio.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")

    print("Scientific reframe shadow portfolio built")
    print("Task:", portfolio.source_task_id)
    print("Candidates:", len(portfolio.assessments))
    print("Pareto non-dominated (not a ranking):", len(portfolio.non_dominated_candidate_ids))
    for candidate_id in portfolio.non_dominated_candidate_ids:
        print("  frontier:", candidate_id)
    for row in portfolio.assessments:
        dominators = ",".join(row.dominated_by_candidate_ids) or "-"
        print(
            f"  {row.operator_id}: {row.candidate_id}; "
            f"pareto={row.pareto_status}; route={row.route_kind}; "
            f"slot={row.assigned_slot_id or '-'}; dominated_by={dominators}"
        )
        if row.risk_flags:
            print("    risk flags:", ", ".join(row.risk_flags))
    print("Slots:")
    for slot in portfolio.slots:
        print(f"  {slot.slot_id}: candidates={len(slot.candidate_ids)}")
        for candidate_id in slot.candidate_ids:
            print("    ", candidate_id)
        if slot.empty_reason:
            print("     empty:", slot.empty_reason)
    print("No overall score, winner, or production ranking was computed.")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
