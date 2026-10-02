
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import (
    build_feedback_plan,
    load_context_source,
    run_feedback_generation,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--context-source", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--closeout", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--materialization-report", type=Path, default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--critic-model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    context = load_context_source(args.context_source)
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    closeout = load(args.closeout)
    external = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )
    materialization = (
        load(args.materialization_report)
        if args.materialization_report is not None
        else None
    )

    plan = build_feedback_plan(
        context=context,
        portfolio=portfolio,
        closeout=closeout,
        external=external,
        materialization_report=materialization,
    )
    write(args.output_dir / "feedback.plan.json", plan)

    report, gen1 = run_feedback_generation(
        context=context,
        portfolio=portfolio,
        external=external,
        plan=plan,
        model=args.model,
        critic_model=args.critic_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        output_dir=args.output_dir,
    )
    write(args.output_dir / "generation.report.json", report)
    write(args.output_dir / "gen1.portfolio.json", gen1)

    print("Closed-loop feedback generation complete")
    print("Plan:", plan["plan_id"])
    print("Routes:", plan["route_counts"])
    print("Decisions:", report["decision_counts"])
    print("Gen1 hypotheses:", report["output_hypothesis_count"])
    for row in report["records"]:
        print(
            row["source_hypothesis_id"],
            "|", row["route"],
            "|", row["decision"],
            "| gen1=", row.get("generated_hypothesis_id"),
            "| task=", row.get("task_preservation"),
        )
    print("SHADOW_ONLY=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
