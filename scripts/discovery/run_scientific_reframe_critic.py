from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ScientificReframeCriticReport,
)
from pipeline_core.discovery.reframing.critic_llm import (
    InstructorOpenAICompatibleReframeCriticBackend,
)
from pipeline_core.discovery.reframing.critic_prompt import (
    ScientificReframeCriticPromptAssembler,
)
from pipeline_core.discovery.reframing.critic_runtime import (
    ScientificReframeCriticRuntime,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


def _read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--header values must use KEY=VALUE")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--header key must be non-empty")
        headers[key] = item
    return headers


def _save_prompt(path: Path, system_prompt: str, user_prompt: str) -> None:
    path.write_text(
        "SYSTEM\n======\n"
        + system_prompt
        + "\n\nUSER\n====\n"
        + user_prompt
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate scientific reframe candidates with a shadow-only multidimensional "
            "critic. No overall score, ranking, novelty authority, or production selection."
        )
    )
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
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
    parser.add_argument("--telemetry", default=None)
    args = parser.parse_args()

    shadow = ScientificReframingShadowReport.model_validate(_read_json(args.shadow))
    evidence = ScientificReframeEvidencePacket.model_validate(_read_json(args.evidence))
    if shadow.source_task_id != evidence.task_id:
        raise SystemExit("shadow/evidence task mismatch")

    output = (
        Path(args.output)
        if args.output
        else Path(args.shadow).parent / "scientific_reframing_critic.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    assembler = ScientificReframeCriticPromptAssembler()
    if args.dry_run:
        prompt_dir = Path(args.shadow).parent / "scientific_reframing_critic_prompts"
        if args.save_prompts:
            prompt_dir.mkdir(parents=True, exist_ok=True)
        print("Scientific reframe critic dry run")
        print("Task:", evidence.task_id)
        print("Candidates:", len(shadow.candidates))
        print("Dimensions:", len(CRITIC_DIMENSIONS))
        print("LLM calls: 0")
        for candidate in shadow.candidates:
            print(f"  {candidate.operator_id}: {candidate.reframe_id}")
            if args.save_prompts:
                prompt = assembler.build(candidate=candidate, evidence=evidence)
                safe = candidate.operator_id.lower() + "_" + candidate.reframe_id.rsplit(":", 1)[-1]
                _save_prompt(
                    prompt_dir / f"{safe}.prompt.txt",
                    prompt.system_prompt,
                    prompt.user_prompt,
                )
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    backend = InstructorOpenAICompatibleReframeCriticBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=_parse_headers(args.header),
        telemetry_path=args.telemetry,
        telemetry_context={
            "task_id": evidence.task_id,
            "source_context_id": evidence.source_context_id,
            "source_shadow_report_id": shadow.report_id,
        },
    )
    outcome = ScientificReframeCriticRuntime(backend).run(
        shadow=shadow,
        evidence=evidence,
    )
    output.write_text(
        outcome.report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    if args.save_prompts:
        prompt_dir = Path(args.shadow).parent / "scientific_reframing_critic_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for row in outcome.prompts:
            candidate = next(
                item for item in shadow.candidates
                if item.reframe_id == row.candidate_id
            )
            safe = candidate.operator_id.lower() + "_" + candidate.reframe_id.rsplit(":", 1)[-1]
            _save_prompt(
                prompt_dir / f"{safe}.prompt.txt",
                row.prompt.system_prompt,
                row.prompt.user_prompt,
            )

    print("Scientific reframe critic complete")
    print("Task:", outcome.report.source_task_id)
    print("Candidates reviewed:", len(outcome.report.reviews))
    print("LLM calls:", outcome.report.llm_calls_succeeded, "/", outcome.report.llm_calls_attempted)
    for review in outcome.report.reviews:
        print(f"  {review.operator_id}: {review.candidate_id}")
        print("    structural:", review.structural_audit.model_dump(mode="json"))
        for dimension in review.dimensions:
            rating = "NA" if dimension.rating is None else str(dimension.rating)
            print(
                f"    {dimension.dimension}: {rating}/3 "
                f"({dimension.review_status})"
            )
            if dimension.rationale:
                print("      ", dimension.rationale)
            for concern in dimension.concerns:
                print("       concern:", concern)
        for note in review.normalization_notes:
            print("    normalization:", note)
        if review.llm_error_type:
            print("    critic error:", review.llm_error_type)
    print("No overall score or ranking was computed.")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
