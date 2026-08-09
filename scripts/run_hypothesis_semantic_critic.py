from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_semantic_llm import (
    InstructorOpenAICompatibleSemanticCriticBackend,
)
from dac_her.hypothesis_semantic_runtime import HypothesisSemanticCriticRuntime


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
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the v2.6.2-b structured semantic critic on one accepted hypothesis portfolio."
    )
    parser.add_argument("--context", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--header", action="append", default=[], type=_header)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--save-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL "
            "or OPENROUTER_AGENT_MODEL is set."
        )

    context_path = Path(args.context)
    portfolio_path = Path(args.portfolio)
    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else portfolio_path.with_suffix("").with_name(
            portfolio_path.stem.replace(".portfolio", "") + ".semantic_v262"
        )
    )

    backend = InstructorOpenAICompatibleSemanticCriticBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
    )
    outcome = HypothesisSemanticCriticRuntime(backend).run(context, portfolio)

    _write_json(Path(str(prefix) + ".hard_evaluation.json"), outcome.evaluation)
    _write_json(Path(str(prefix) + ".run.json"), outcome.run_record)

    if args.save_prompt:
        prompt_path = Path(str(prefix) + ".prompt.txt")
        prompt_path.write_text(
            "SYSTEM\n======\n"
            + outcome.prompt.system_prompt
            + "\n\nUSER\n====\n"
            + outcome.prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )
        print("Prompt saved:", prompt_path)

    if outcome.generation is not None:
        _write_json(Path(str(prefix) + ".draft.json"), outcome.generation.draft)
    if outcome.review is not None:
        _write_json(Path(str(prefix) + ".review.json"), outcome.review)

    print("Hypothesis semantic critic complete")
    print("Hard gate:", "PASS" if outcome.evaluation.hard_gate_passed else "FAIL")
    print("Generated:", outcome.run_record.generated)
    print("Accepted review:", outcome.accepted)
    print("Failure stage:", outcome.run_record.failure_stage)
    print("Prompt:", outcome.run_record.critic_prompt_version, outcome.run_record.critic_prompt_sha256)
    if outcome.review is not None:
        print("Review ID:", outcome.review.review_id)
        for row in outcome.review.dimensions:
            print(f"- {row.dimension}: {row.verdict}")
    if outcome.review_validation_issues:
        for issue in outcome.review_validation_issues:
            print("REVIEW VALIDATION:", issue)
    if not outcome.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
