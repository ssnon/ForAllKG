from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dac_her.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_clause_coverage import (
    HypothesisClauseCoverageReport,
)
from dac_her.hypothesis_clause_coverage_v31 import (
    HypothesisClauseCoverageRuntimeV31,
)
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.premise_role_necessity import (
    InstructorPremiseCriticBackend,
)


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Header must be KEY=VALUE"
        )
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            "Header key may not be empty"
        )
    return key, item


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


def _safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )[:120]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PS3.1 refined epistemic taxonomy audit using the exact "
            "evidence-blind clause decomposition from an existing PS3 report."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--axis-plan", required=True, type=Path)
    parser.add_argument("--axis-report", required=True, type=Path)
    parser.add_argument("--source-ps3-report", required=True, type=Path)

    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--instructor-mode",
        default="JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
    )

    parser.add_argument(
        "--hypothesis-id",
        action="append",
        default=None,
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model:
        raise SystemExit(
            "--model is required unless "
            "GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL or "
            "OPENROUTER_AGENT_MODEL is set."
        )

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    axis_plan = DiscoveryAxisPlan.model_validate_json(
        args.axis_plan.read_text(encoding="utf-8")
    )
    axis_report = (
        DiscoveryAxisSynthesisReport.model_validate_json(
            args.axis_report.read_text(encoding="utf-8")
        )
    )
    source_ps3 = HypothesisClauseCoverageReport.model_validate_json(
        args.source_ps3_report.read_text(encoding="utf-8")
    )

    backend = InstructorPremiseCriticBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )

    report, prompts = HypothesisClauseCoverageRuntimeV31(
        backend
    ).run(
        context,
        portfolio,
        axis_plan,
        axis_report,
        source_ps3,
        hypothesis_ids=(
            set(args.hypothesis_id)
            if args.hypothesis_id
            else None
        ),
    )

    output_path = Path(
        str(args.output_prefix)
        + ".hypothesis_clause_coverage.ps31.json"
    )
    _write_json(output_path, report)

    if args.save_prompts:
        prompt_dir = Path(
            str(args.output_prefix)
            + ".ps31.prompts"
        )
        prompt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for hypothesis_id, prompt in prompts:
            filename = (
                _safe_name(hypothesis_id)
                + "__coverage.prompt.txt"
            )
            (prompt_dir / filename).write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

    print("PS3.1 refined epistemic taxonomy audit")
    print("Report:", report.report_id)
    print("Source PS3:", report.source_ps3_report_id)
    print("Model:", report.critic_model)
    print(
        "Hypotheses / LLM calls:",
        f"{report.hypothesis_count}/{report.llm_call_count}",
    )
    print(
        "Decomposition / coverage calls:",
        f"{report.decomposition_llm_call_count}/"
        f"{report.coverage_llm_call_count}",
    )
    print(
        "Invalid source quotes / support refs:",
        f"{report.invalid_source_clause_quote_count}/"
        f"{report.invalid_support_reference_count}",
    )
    print(
        "Overall verdicts:",
        report.overall_verdict_counts,
    )
    print(
        "Hypothesis clause statuses:",
        report.clause_status_counts,
    )
    print(
        "Bridge statuses:",
        report.bridge_status_counts,
    )
    print(
        "Overall transitions:",
        report.overall_verdict_transition_counts,
    )
    print(
        "Clause transitions:",
        report.clause_status_transition_counts,
    )
    print(
        "Bridge transitions:",
        report.bridge_status_transition_counts,
    )

    for card in report.cards:
        print()
        print(card.title)
        print(" axis:", card.axis_label)
        print(
            " overall:",
            card.source_ps3_overall_verdict,
            "->",
            card.overall_verdict,
        )
        print(" reason:", card.verdict_reason)

        review_by_id = {
            row.local_id: row
            for row in card.hypothesis_clause_reviews
        }
        print(" hypothesis clauses:")
        for clause in card.hypothesis_clauses:
            review = review_by_id[clause.local_id]
            print(
                "  -",
                clause.local_id,
                f"[{clause.materiality}/{clause.clause_type}]",
                f"{review.source_ps3_status} -> {review.status}",
            )
            print("      text:", clause.text)
            print(
                "      support:",
                review.supporting_premise_statement_ids,
            )
            if review.limiting_premise_statement_ids:
                print(
                    "      limiting:",
                    review.limiting_premise_statement_ids,
                )
            if review.missing_relation_or_scope:
                print(
                    "      missing:",
                    review.missing_relation_or_scope,
                )

        bridge_by_id = {
            row.local_id: row
            for row in card.bridge_unit_reviews
        }
        if card.bridge_units:
            print(" bridge units:")
        for bridge in card.bridge_units:
            review = bridge_by_id[bridge.local_id]
            print(
                "  -",
                bridge.local_id,
                f"[{bridge.materiality}]",
                f"{review.source_ps3_status} -> {review.status}",
            )
            print("      text:", bridge.text)
            print(
                "      support:",
                review.supporting_premise_statement_ids,
            )
            if review.limiting_premise_statement_ids:
                print(
                    "      limiting:",
                    review.limiting_premise_statement_ids,
                )
            if review.missing_relation_or_scope:
                print(
                    "      missing:",
                    review.missing_relation_or_scope,
                )

        if card.critical_missing_links:
            print(" critical missing links:")
            for item in card.critical_missing_links:
                print("   *", item)

    print()
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
