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
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from dac_her.premise_necessity_diagnostic import (
    PremiseNecessityDiagnosticReport,
)
from dac_her.premise_role_necessity import (
    InstructorPremiseCriticBackend,
    PremiseRoleNecessityRuntime,
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
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )
    return value[:120]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PS2 isolated premise-role + blinded leave-one-premise-out "
            "grounding critic."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--axis-plan", required=True, type=Path)
    parser.add_argument("--axis-report", required=True, type=Path)
    parser.add_argument("--ps1-report", type=Path, default=None)

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
        "--premise-id",
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

    ps1 = None
    if args.ps1_report is not None:
        ps1 = PremiseNecessityDiagnosticReport.model_validate_json(
            args.ps1_report.read_text(encoding="utf-8")
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

    runtime = PremiseRoleNecessityRuntime(backend)
    report, prompts = runtime.run(
        context,
        portfolio,
        axis_plan,
        axis_report,
        ps1_report=ps1,
        hypothesis_ids=(
            set(args.hypothesis_id)
            if args.hypothesis_id
            else None
        ),
        premise_ids=(
            set(args.premise_id)
            if args.premise_id
            else None
        ),
    )

    report_path = Path(
        str(args.output_prefix)
        + ".premise_role_necessity.ps2.json"
    )
    _write_json(report_path, report)

    if args.save_prompts:
        prompt_dir = Path(
            str(args.output_prefix)
            + ".ps2.prompts"
        )
        prompt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for hypothesis_id, label, prompt in prompts:
            filename = (
                _safe_name(hypothesis_id)
                + "__"
                + _safe_name(label)
                + ".prompt.txt"
            )
            (prompt_dir / filename).write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

    print("PS2 premise role / ablation diagnostic")
    print("Report:", report.report_id)
    print("Model:", report.critic_model)
    print(
        "Hypotheses / premise incidences:",
        f"{report.hypothesis_count}/"
        f"{report.selected_premise_incidence_count}",
    )
    print(
        "LLM calls:",
        f"role={report.role_call_count}, "
        f"ablation={report.ablation_call_count}, "
        f"total={report.role_call_count + report.ablation_call_count}",
    )
    print(
        "Invalid quoted clauses:",
        report.invalid_quoted_clause_count,
    )
    print("Role counts:", report.role_counts)
    print(
        "Ablation statuses:",
        report.ablation_status_counts,
    )
    print(
        "Necessity verdicts:",
        report.necessity_verdict_counts,
    )

    for summary in report.hypothesis_summaries:
        print()
        print(summary.title)
        print(" axis:", summary.axis_label)
        print(
            " critical:",
            summary.critical_premise_statement_ids,
        )
        print(
            " material:",
            summary.material_premise_statement_ids,
        )
        print(
            " replaceable:",
            summary.replaceable_premise_statement_ids,
        )
        print(
            " contextual/nonessential:",
            summary.contextual_or_nonessential_statement_ids,
        )
        print(
            " scope/counterevidence:",
            summary.scope_problem_statement_ids,
        )
        print(
            " uncertain:",
            summary.uncertain_statement_ids,
        )

        rows = [
            row
            for row in report.cards
            if row.hypothesis_id == summary.hypothesis_id
        ]
        for row in rows:
            print(
                "  -",
                row.premise_statement_id,
                "role=",
                row.role_review.role,
                "ablation=",
                row.ablation_review.remaining_grounding_status,
                "bridge_grounded=",
                row.ablation_review.inferential_bridge_grounded,
                "=>",
                row.necessity_verdict,
            )
            if row.ps1_snapshot is not None:
                print(
                    "      PS1:",
                    f"core={row.ps1_snapshot.core_score:.3f}"
                    if row.ps1_snapshot.core_score is not None
                    else "core=-",
                    f"r{row.ps1_snapshot.core_rank}"
                    if row.ps1_snapshot.core_rank is not None
                    else "",
                    f"axis={row.ps1_snapshot.axis_score:.3f}"
                    if row.ps1_snapshot.axis_score is not None
                    else "axis=-",
                    f"r{row.ps1_snapshot.axis_rank}"
                    if row.ps1_snapshot.axis_rank is not None
                    else "",
                )
            if row.role_review.supported_hypothesis_clause:
                audit = (
                    row.role_review
                    .supported_hypothesis_clause_audit
                )
                print(
                    "      supports hypothesis:",
                    row.role_review.supported_hypothesis_clause,
                    "exact=",
                    (
                        audit.exact_substring_match
                        if audit is not None
                        else None
                    ),
                )
            if row.role_review.supported_bridge_clause:
                audit = (
                    row.role_review
                    .supported_bridge_clause_audit
                )
                print(
                    "      supports bridge:",
                    row.role_review.supported_bridge_clause,
                    "exact=",
                    (
                        audit.exact_substring_match
                        if audit is not None
                        else None
                    ),
                )
            if (
                row.ablation_review
                .unsupported_or_weak_hypothesis_clauses
            ):
                print(
                    "      weak after ablation:",
                    row.ablation_review
                    .unsupported_or_weak_hypothesis_clauses,
                )
            if row.ablation_review.critical_missing_link:
                print(
                    "      missing link:",
                    row.ablation_review.critical_missing_link,
                )

    print()
    print("Saved:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
