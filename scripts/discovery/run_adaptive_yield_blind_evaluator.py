from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.adaptive_yield_blind_evaluator import (
    BlindYieldEvaluationDraft,
    SYSTEM_PROMPT,
    build_prompt,
    report_payload,
)
from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldBlindPacket
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--telemetry", type=Path, default=None)
    args = p.parse_args()

    if not args.model:
        raise RuntimeError("--model is required")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env: {args.api_key_env}")

    packet = AdaptiveYieldBlindPacket.model_validate_json(
        args.packet.read_text(encoding="utf-8")
    )
    import instructor
    from openai import OpenAI

    raw = OpenAI(api_key=api_key, base_url=args.base_url, timeout=180.0)
    client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    draft, _event = run_instructor_structured_call(
        client.chat.completions,
        model=args.model,
        response_model=BlindYieldEvaluationDraft,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(packet)},
        ],
        temperature=0.0,
        max_retries=args.parse_retries,
        telemetry_path=args.telemetry,
        telemetry_context={
            "pipeline": "adaptive_scientific_evaluation",
            "stage": "blind_quality",
            "packet_id": packet.packet_id,
        },
    )
    if not isinstance(draft, BlindYieldEvaluationDraft):
        draft = BlindYieldEvaluationDraft.model_validate(draft)

    report = report_payload(packet, draft, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Adaptive-yield blind evaluation complete")
    print("overall winner selected: false")
    print("quality ranking performed: false")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
