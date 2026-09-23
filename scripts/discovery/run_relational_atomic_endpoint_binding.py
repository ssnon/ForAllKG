from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    InstructorLiteralEndpointBindingBackend,
    build_endpoint_binding_prompt,
    run_endpoint_binding,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind literal relation endpoint spans to the frozen, "
            "specification-complete relational claim population. "
            "This annotation layer cannot rewrite claims or make "
            "novelty/production judgments."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = RelationalAtomicBindingPlan.model_validate_json(
        args.plan.read_text(encoding="utf-8")
    )

    print("Relational atomic literal endpoint binding")
    print("Source plan:", plan.plan_id)
    print("Binding-ready hypotheses:", plan.ready_hypothesis_count)
    print("Binding-ready claims:", plan.binding_ready_claim_count)
    print(
        "Novelty-bearing binding-ready claims:",
        plan.novelty_bearing_binding_ready_claim_count,
    )

    prompt = build_endpoint_binding_prompt(plan)
    if args.dry_run:
        if args.prompt_output is not None:
            args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
            args.prompt_output.write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )
        print("LLM calls: 0")
        print("Would execute literal endpoint binding: true")
        print("Scientific content added: false")
        print("Production authority: false")
        return 0

    if not args.model:
        raise SystemExit("--model is required")

    backend = InstructorLiteralEndpointBindingBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        telemetry_path=args.telemetry,
        telemetry_context={
            "source_binding_plan_id": plan.plan_id,
        },
    )
    report, prompt = run_endpoint_binding(
        plan=plan,
        backend=backend,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.prompt_output is not None and prompt is not None:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )

    print("Literal endpoint binding complete")
    print("LLM calls:", report.llm_calls_performed)
    print(
        "Selected hypotheses/claims:",
        report.selected_hypothesis_count,
        "/",
        report.selected_claim_count,
    )
    print(
        "Bound/abstained:",
        report.bound_claim_count,
        "/",
        report.abstained_claim_count,
    )
    print(
        "Novelty-bearing bound:",
        report.novelty_bearing_bound_claim_count,
    )
    for row in report.bindings:
        print(
            " ",
            row.claim_id,
            row.novelty_selection_role,
            row.outcome,
            repr(row.relation_endpoint_anchors),
        )
        if row.abstention_reason:
            print("   abstention:", row.abstention_reason)
    print("Scientific content added: false")
    print("Verifier result observed: false")
    print("Production authority: false")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
