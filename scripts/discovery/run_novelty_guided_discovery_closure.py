from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityReport,
)
from pipeline_core.discovery.novelty_guided_discovery_closure import (
    bind_novelty_guided_regeneration_closure,
    build_novelty_guided_negative_space_plan,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
    NoveltyRefinementReport,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Materialize the novelty-guided discovery closure from existing "
            "Alpha4/Alpha6 artifacts: NoveltyGap -> NegativeSpace/evidence/"
            "operator allocation -> Alpha6 regeneration -> fresh external "
            "verification trace. Artifact-only; no new LLM or retrieval calls."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--gap-plan", required=True, type=Path)
    p.add_argument("--evidence-diversity", required=True, type=Path)
    p.add_argument("--refinement-report", required=True, type=Path)
    p.add_argument("--plan-output", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def _exclusive_write(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        value.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        args.gap_plan.read_text(encoding="utf-8")
    )
    evidence = HypothesisEvidenceDiversityReport.model_validate_json(
        args.evidence_diversity.read_text(encoding="utf-8")
    )
    refinement = NoveltyRefinementReport.model_validate_json(
        args.refinement_report.read_text(encoding="utf-8")
    )

    if gap_plan.source_portfolio_id != portfolio.portfolio_id:
        raise RuntimeError(
            "gap-plan/source-portfolio mismatch: "
            f"{gap_plan.source_portfolio_id} != {portfolio.portfolio_id}"
        )

    if evidence.source_portfolio_id != portfolio.portfolio_id:
        raise RuntimeError(
            "evidence-diversity/source-portfolio mismatch: "
            f"{evidence.source_portfolio_id} != {portfolio.portfolio_id}"
        )

    if refinement.source_gap_plan_id != gap_plan.plan_id:
        raise RuntimeError(
            "refinement-report/gap-plan mismatch: "
            f"{refinement.source_gap_plan_id} != {gap_plan.plan_id}"
        )

    plan = build_novelty_guided_negative_space_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        evidence_diversity=evidence,
    )

    closure = bind_novelty_guided_regeneration_closure(
        negative_space_plan=plan,
        refinement_report=refinement,
    )

    _exclusive_write(args.plan_output, plan)
    _exclusive_write(args.output, closure)

    print("=== NOVELTY-GUIDED DISCOVERY CLOSURE ===")
    print("artifact only: true")
    print("negative-space targets:", plan.target_count)
    print("regeneration eligible:", plan.regeneration_eligible_count)
    print(
        "evidence alternative capacity targets:",
        plan.evidence_alternative_capacity_target_count,
    )
    print("operator targets:", plan.operator_target_count)
    print("operator opportunities:", plan.operator_opportunity_count)
    print(
        "regeneration attempted:",
        closure.regeneration_attempted_count,
    )
    print(
        "regeneration generated:",
        closure.regeneration_generated_count,
    )
    print(
        "accepted regeneration:",
        closure.accepted_regeneration_count,
    )
    print(
        "fresh external verification:",
        closure.fresh_external_verification_count,
    )
    print(
        "closed loops observed:",
        closure.closed_loop_observed_count,
    )
    print("gap actions:", closure.gap_action_counts)
    print("generation modes:", closure.generation_mode_counts)
    print("decisions:", closure.decision_counts)
    print("ranking computed: false")
    print("novelty authority created: false")
    print("production selection changed: false")
    print("negative-space plan:", args.plan_output)
    print("closure report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
