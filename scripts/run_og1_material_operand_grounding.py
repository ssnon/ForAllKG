from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dac_her.discovery_axis_contracts import DiscoveryAxisPlan, DiscoveryAxisSynthesisReport
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.ig11_endpoint_scope import IG11PlanningReport, IG11StructuredGenerator
from dac_her.og1_material_operand_grounding import run_og1


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OG1 material-operand grounding gate. Filters hypotheses where a "
            "scientific operand is not grounded by selected positive premises."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--axis-plan", required=True, type=Path)
    p.add_argument("--axis-report", required=True, type=Path)
    p.add_argument("--planning-report", required=True, type=Path)
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header, metavar="KEY=VALUE")
    p.add_argument("--max-audit-repairs", type=int, choices=(0, 1), default=1)
    p.add_argument("--output-prefix", required=True, type=Path)
    p.add_argument("--save-prompts", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model:
        raise SystemExit("--model is required unless a critic/agent model env var is set.")

    context = HypothesisContext.model_validate_json(args.context.read_text(encoding="utf-8"))
    portfolio = HypothesisPortfolio.model_validate_json(args.portfolio.read_text(encoding="utf-8"))
    axis_plan = DiscoveryAxisPlan.model_validate_json(args.axis_plan.read_text(encoding="utf-8"))
    axis_report = DiscoveryAxisSynthesisReport.model_validate_json(args.axis_report.read_text(encoding="utf-8"))
    planning_report = IG11PlanningReport.model_validate_json(args.planning_report.read_text(encoding="utf-8"))

    generator = IG11StructuredGenerator(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
        max_output_tokens=args.max_output_tokens,
    )

    candidate_portfolio, candidate_axis_report, report, prompts = run_og1(
        context=context,
        portfolio=portfolio,
        axis_plan=axis_plan,
        axis_report=axis_report,
        planning_report=planning_report,
        generator=generator,
        max_audit_repairs=args.max_audit_repairs,
    )

    portfolio_path = Path(str(args.output_prefix) + ".portfolio.json")
    lineage_path = Path(str(args.output_prefix) + ".lineage.json")
    report_path = Path(str(args.output_prefix) + ".report.json")
    _write_json(portfolio_path, candidate_portfolio)
    _write_json(lineage_path, candidate_axis_report)
    _write_json(report_path, report)

    if args.save_prompts:
        prompt_dir = Path(str(args.output_prefix) + ".prompts")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for hypothesis_id, stage, messages in prompts:
            name = _safe_name(hypothesis_id) + "__" + _safe_name(stage) + ".prompt.txt"
            text = []
            for msg in messages:
                text.append(msg["role"].upper() + "\n======\n" + msg["content"] + "\n")
            (prompt_dir / name).write_text("\n".join(text), encoding="utf-8")

    print("OG1 Material Operand Grounding Gate")
    print("Report:", report.report_id)
    print("Source portfolio:", report.source_portfolio_id)
    print("Model:", report.model)
    print(
        "source / passed / filtered / valid audits:",
        f"{report.source_hypothesis_count}/{report.passed_count}/"
        f"{report.filtered_count}/{report.valid_audit_count}",
    )
    print("Audit repairs:", report.audit_repair_count)
    print("Operand grounding status counts:", report.grounding_status_counts)

    for row in report.records:
        print()
        print(row.title)
        print(" axis:", row.axis_label)
        print(
            " audit valid / attempts / repairs:",
            row.valid, "/", row.generation_attempts, "/", row.repair_count,
        )
        for operand in row.audit.operand_reviews:
            print(" operand:", repr(operand.operand_text), "=>", operand.grounding_status)
            if operand.supports:
                print("   support:", [x.statement_id for x in operand.supports])
        if row.audit.unlisted_material_operand_texts:
            print(" unlisted material operands:", row.audit.unlisted_material_operand_texts)
        print(
            " coverage / all grounded:",
            row.audit.coverage_complete, "/", row.audit.all_material_operands_grounded,
        )
        print(" OG1:", "PASS" if row.passes_gate else "FILTER")
        print(" reason:", row.disposition_reason)

    print()
    print("Saved candidate portfolio:", portfolio_path)
    print("Saved candidate axis report:", lineage_path)
    print("Saved OG1 report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
