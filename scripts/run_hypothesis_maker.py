from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from dac_her.hypothesis_prompt import HypothesisPromptAssembler
from dac_her.hypothesis_runtime import HypothesisMakerAgentRuntime


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hypothesis Maker v2.6.1: HypothesisContext -> structured "
            "HypothesisPortfolioDraft -> deterministic compile -> validation -> at most one repair."
        )
    )
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
        help=(
            "OpenAI-compatible model name. May also be set with "
            "GRAPHAGENTS_HYPOTHESIS_MODEL (or OPENROUTER_AGENT_MODEL as fallback)."
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-hypotheses", type=int, default=3)
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_header,
        metavar="KEY=VALUE",
        help="Optional default HTTP header for compatible providers; repeatable.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix without suffix. Defaults beside context file.",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save deterministic system/user prompt text for audit.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Build and save/print prompt metadata without calling a model.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")  # type: ignore[attr-defined]
    else:
        payload = value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    context_path = Path(args.context)
    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )

    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else context_path.with_suffix("").with_name(
            context_path.stem.replace(".context", "") + ".hypothesis_v261"
        )
    )

    assembler = HypothesisPromptAssembler(max_hypotheses=args.max_hypotheses)
    prompt = assembler.build(context)
    if args.save_prompt or args.dry_run_prompt:
        prompt_path = Path(str(prefix) + ".prompt.txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )
        print("Prompt version:", prompt.prompt_version)
        print("Prompt SHA256:", prompt.prompt_sha256)
        print("Prompt saved:", prompt_path)
    if args.dry_run_prompt:
        return

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or OPENROUTER_AGENT_MODEL is set."
        )

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )
    runtime = HypothesisMakerAgentRuntime(
        backend,
        prompt_assembler=assembler,
        max_repairs=args.max_repairs,
    )
    outcome = runtime.run(context)

    for index, draft in enumerate(outcome.draft_history):
        suffix = ".draft.json" if index == 0 else f".repair{index}.draft.json"
        _write_json(Path(str(prefix) + suffix), draft)

    _write_json(Path(str(prefix) + ".run.json"), outcome.run_record)

    if outcome.validation is not None:
        _write_json(Path(str(prefix) + ".validation.json"), outcome.validation)
    else:
        _write_json(
            Path(str(prefix) + ".validation.json"),
            {
                "passes": False,
                "stage": "compile",
                "issues": [
                    issue.model_dump(mode="json")
                    for issue in outcome.compile_issues
                ],
            },
        )

    if outcome.accepted_portfolio is not None:
        _write_json(
            Path(str(prefix) + ".portfolio.json"),
            outcome.accepted_portfolio,
        )
    elif outcome.last_portfolio is not None:
        _write_json(
            Path(str(prefix) + ".rejected_portfolio.json"),
            outcome.last_portfolio,
        )

    print("Hypothesis Maker run complete")
    print("Run ID:", outcome.run_record.run_id)
    print("Context SHA256:", context.context_sha256)
    print("Prompt:", outcome.run_record.prompt_version, outcome.run_record.prompt_sha256)
    print("Backend/model:", outcome.run_record.backend, outcome.run_record.model)
    print("Generation attempts:", outcome.run_record.generation_attempts)
    print("Repair attempts:", outcome.run_record.repair_attempts)
    print("Accepted:", outcome.accepted)
    print("Failure stage:", outcome.run_record.failure_stage)
    print(
        "Validation errors/warnings:",
        outcome.run_record.validation_errors,
        outcome.run_record.validation_warnings,
    )
    if outcome.accepted_portfolio is not None:
        print("Portfolio ID:", outcome.accepted_portfolio.portfolio_id)
        print("Portfolio SHA256:", outcome.run_record.portfolio_sha256)
        print("Hypotheses:", len(outcome.accepted_portfolio.hypotheses))
        print("Abstained:", not bool(outcome.accepted_portfolio.hypotheses))
        print("Saved portfolio:", Path(str(prefix) + ".portfolio.json"))
    else:
        print("No hypothesis portfolio was accepted; downstream Evidence Auditor must not consume this run.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
