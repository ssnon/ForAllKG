from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_core.discovery.reframing.final_output_blind_evaluation import (
    BlindFinalOutputEvaluationPacket,
    InstructorOpenAICompatibleFinalOutputEvaluator,
    build_final_output_evaluation_prompt,
    run_blind_final_output_evaluation,
)


def _headers(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--header values must use KEY=VALUE")
        key, item = value.split("=", 1)
        if not key.strip():
            raise ValueError("--header key must be non-empty")
        result[key.strip()] = item
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blindly evaluate two anonymous final hypothesis portfolios dimension by dimension. "
            "The evaluator never loads the blind key and never selects an overall winner."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--telemetry", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    packet_path = args.packet.expanduser().resolve()
    packet = BlindFinalOutputEvaluationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    output = args.output or packet_path.with_name("scientific_final_output_blind_evaluation.json")
    prompt_output = args.prompt_output

    if args.dry_run:
        prompt = build_final_output_evaluation_prompt(packet)
        if prompt_output is not None:
            prompt_output.parent.mkdir(parents=True, exist_ok=True)
            prompt_output.write_text(
                "SYSTEM\n======\n" + prompt.system_prompt + "\n\nUSER\n====\n" + prompt.user_prompt + "\n",
                encoding="utf-8",
            )
        print("Final hypothesis output blind evaluation dry run")
        print("LLM calls: 0")
        print("Would execute: true")
        print("Blind key loaded by evaluator: false")
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or OPENROUTER_AGENT_MODEL is set"
        )
    backend = InstructorOpenAICompatibleFinalOutputEvaluator(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=_headers(args.header),
        telemetry_path=args.telemetry,
        telemetry_context={"packet_id": packet.packet_id},
    )
    report, prompt = run_blind_final_output_evaluation(packet=packet, backend=backend)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if prompt_output is not None:
        prompt_output.parent.mkdir(parents=True, exist_ok=True)
        prompt_output.write_text(
            "SYSTEM\n======\n" + prompt.system_prompt + "\n\nUSER\n====\n" + prompt.user_prompt + "\n",
            encoding="utf-8",
        )

    print("Final hypothesis output blind evaluation complete")
    print("LLM calls: 1")
    print("Judge model:", report.model_name)
    print("Preference counts:", report.preference_counts)
    for row in report.judgments:
        print(f"  {row.dimension_id}: {row.preference}")
    print("Blind key loaded by evaluator: false")
    print("Candidate count treated as quality signal: false")
    print("Overall score computed: false")
    print("Overall winner selected: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
