from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    InstructorOpenAICompatibleAblationJudgeBackend,
    build_ablation_blind_evaluation_prompt,
    run_blind_scientific_reasoning_evaluation,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    ScientificReasoningAblationPacket,
)


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Header key may not be empty")
    return key, item


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run blind pairwise scientific-reasoning ablation evaluation. "
            "This evaluator intentionally has no blind-key input and cannot unblind source conditions."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
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
    parser.add_argument("--header", action="append", default=[], type=_header, metavar="KEY=VALUE")
    parser.add_argument("--telemetry", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_path = args.packet.expanduser().resolve()
    packet = ScientificReasoningAblationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    if not args.model:
        raise RuntimeError(
            "No evaluator model configured. Pass --model or set OPENROUTER_CRITIC_MODEL, "
            "GRAPHAGENTS_HYPOTHESIS_MODEL, or OPENROUTER_AGENT_MODEL."
        )

    prompt_dir = (
        args.prompt_dir.expanduser().resolve()
        if args.prompt_dir is not None
        else packet_path.parent / "scientific_reasoning_ablation_blind_prompts"
    )
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for comparison in packet.comparisons:
        prompt = build_ablation_blind_evaluation_prompt(
            packet=packet,
            comparison=comparison,
        )
        (prompt_dir / f"{comparison.comparison_alias}.system.txt").write_text(
            prompt.system_prompt + "\n", encoding="utf-8"
        )
        (prompt_dir / f"{comparison.comparison_alias}.user.txt").write_text(
            prompt.user_prompt + "\n", encoding="utf-8"
        )

    backend = InstructorOpenAICompatibleAblationJudgeBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
        telemetry_path=args.telemetry,
        telemetry_context={"source_packet_id": packet.packet_id},
    )
    report = run_blind_scientific_reasoning_evaluation(
        packet=packet,
        backend=backend,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else packet_path.parent / "scientific_reasoning_ablation_blind_evaluation.json"
    )
    _write_json(output, report)

    print("Blind scientific reasoning ablation evaluation complete")
    print(f"LLM calls: {report.llm_calls_performed}")
    print(f"Comparisons: {len(report.comparison_evaluations)}")
    for comparison in report.comparison_evaluations:
        counts = {}
        for row in comparison.dimensions:
            counts[row.preference] = counts.get(row.preference, 0) + 1
        print(
            f"  {comparison.comparison_alias}: "
            + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        )
    print("Blind key loaded by evaluator: false")
    print("Candidate count treated as quality signal: false")
    print("Overall score computed: false")
    print("Overall winner selected: false")
    print("Scientific quality ranking performed: false")
    print("Production selection changed: false")
    print(f"Prompts: {prompt_dir}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
