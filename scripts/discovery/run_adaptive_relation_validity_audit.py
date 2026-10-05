from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.adaptive_relation_validity import (
    RelationValidityAudit,
    RelationValidityBatchDraft,
    SYSTEM_PROMPT,
    build_prompt,
)
from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldCaseAudit
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-audit", required=True, type=Path)
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

    audit = AdaptiveYieldCaseAudit.model_validate_json(
        args.case_audit.read_text(encoding="utf-8")
    )
    unique_count = len({x.scientific_fingerprint for x in audit.candidates})
    if unique_count == 0:
        report = RelationValidityAudit(
            case_id=audit.case_id,
            model=args.model,
            candidates=[],
            verdict_counts={},
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print("Adaptive relation-validity audit complete: no candidates")
        return 0

    import instructor
    from openai import OpenAI

    raw = OpenAI(api_key=api_key, base_url=args.base_url, timeout=180.0)
    client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    draft, _event = run_instructor_structured_call(
        client.chat.completions,
        model=args.model,
        response_model=RelationValidityBatchDraft,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(audit)},
        ],
        temperature=0.0,
        max_retries=args.parse_retries,
        telemetry_path=args.telemetry,
        telemetry_context={
            "pipeline": "adaptive_scientific_evaluation",
            "stage": "relation_validity",
            "case_id": audit.case_id,
        },
    )
    if not isinstance(draft, RelationValidityBatchDraft):
        draft = RelationValidityBatchDraft.model_validate(draft)
    if len(draft.candidates) != unique_count:
        raise RuntimeError("relation-validity evaluator returned wrong candidate count")

    counts = Counter(
        dim.verdict for row in draft.candidates for dim in row.dimensions
    )
    report = RelationValidityAudit(
        case_id=audit.case_id,
        model=args.model,
        candidates=draft.candidates,
        verdict_counts=dict(counts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print("Adaptive relation-validity audit complete")
    print("verdicts:", dict(counts))
    print("rejection authority: false")
    print("production selection changed: false")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
