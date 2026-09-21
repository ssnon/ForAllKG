from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.reframing.proxy_challenge import (
    InstructorOpenAICompatibleProxyChallengeBackend,
    ProxyChallengeShadowRuntime,
    build_proxy_challenge_input,
    build_proxy_challenge_prompt,
    load_proxy_annotations,
)
from pipeline_core.discovery.reframing.proxy_enrichment import (
    ProxySemanticEnrichmentReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


def _read_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--header values must use KEY=VALUE")
        key, item = value.split("=", 1)
        if not key.strip():
            raise ValueError("--header key must be non-empty")
        headers[key.strip()] = item
    return headers


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shadow-only PROXY_CHALLENGE operator over grounded reframing "
            "evidence plus completed proxy-semantic enrichment sidecars."
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--enrichment-report", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--input-output", default=None)
    parser.add_argument("--prompt-output", default=None)
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
    parser.add_argument("--telemetry", default=None)
    args = parser.parse_args()

    evidence = ScientificReframeEvidencePacket.model_validate(_read_json(args.evidence))
    enrichment = ProxySemanticEnrichmentReport.model_validate(
        _read_json(args.enrichment_report)
    )
    annotations = load_proxy_annotations(args.annotations)
    proxy_input = build_proxy_challenge_input(
        evidence=evidence,
        enrichment_report=enrichment,
        annotations=annotations,
        annotation_path=args.annotations,
    )

    base = Path(args.evidence).parent
    input_output = (
        Path(args.input_output)
        if args.input_output
        else base / "scientific_proxy_challenge_input.json"
    )
    output = (
        Path(args.output)
        if args.output
        else base / "scientific_proxy_challenge_shadow.json"
    )
    input_output.write_text(proxy_input.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("PROXY_CHALLENGE shadow input")
    print("Task:", evidence.task_id)
    print("Proxy annotations:", len(proxy_input.seeds))
    print("Task-relevant semantic seeds:", len(proxy_input.task_relevant_seed_ids))
    print("Enrichment coverage complete: true")

    if args.dry_run:
        print("LLM calls: 0")
        if proxy_input.task_relevant_seed_ids:
            prompt = build_proxy_challenge_prompt(evidence=evidence, proxy_input=proxy_input)
            if args.prompt_output:
                Path(args.prompt_output).write_text(
                    "SYSTEM\n======\n"
                    + prompt.system_prompt
                    + "\n\nUSER\n====\n"
                    + prompt.user_prompt
                    + "\n",
                    encoding="utf-8",
                )
            print("Would execute: true")
        else:
            print("Would execute: false")
        print("Input:", input_output)
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    backend = InstructorOpenAICompatibleProxyChallengeBackend(
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
        },
    )
    report, prompt = ProxyChallengeShadowRuntime(backend).run(
        evidence=evidence,
        proxy_input=proxy_input,
    )
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if args.prompt_output and prompt is not None:
        Path(args.prompt_output).write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )

    print("PROXY_CHALLENGE shadow complete")
    print("Decision:", report.decision)
    print("LLM calls:", report.llm_calls_performed)
    print("Candidates:", len(report.candidates))
    print("Rejected candidates:", report.rejected_candidate_count)
    for row in report.candidates:
        print(f"  {row.candidate_id}: {row.title}")
    print("Canonical graph mutated: false")
    print("Production selection changed: false")
    print("Output:", output)
    print("Input:", input_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
