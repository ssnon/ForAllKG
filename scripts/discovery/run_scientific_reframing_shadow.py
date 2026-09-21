from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.reframing.operator_readiness import (
    ReframingOperatorReadinessReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    build_scientific_reframe_evidence_packet,
)
from pipeline_core.discovery.reframing.reframe_llm import (
    InstructorOpenAICompatibleReframeBackend,
)
from pipeline_core.discovery.reframing.reframe_prompt import (
    ScientificReframePromptAssembler,
)
from pipeline_core.discovery.reframing.reframe_runtime import (
    ScientificReframingShadowRuntime,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ScientificReframeTriggerReport,
)


_IMPLEMENTED: tuple[ImplementedReframeOperatorId, ...] = (
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
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
            "Run shadow-only LATENT_VARIABLE / REGIME_BOUNDARY scientific "
            "reframing. No canonical hypothesis portfolio is mutated."
        )
    )
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--readiness", default=None)
    parser.add_argument(
        "--trigger",
        default=None,
        help=(
            "Optional scientific_reframing_triggers.json. When supplied, only "
            "operators with decision=triggered may make an LLM call."
        ),
    )
    parser.add_argument(
        "--scope-mode",
        choices=("premise_and_gap", "premise_only"),
        default="premise_and_gap",
    )
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="error",
    )
    parser.add_argument(
        "--operator",
        action="append",
        choices=_IMPLEMENTED,
        default=None,
        help="Repeat to select operators. Default: both implemented operators.",
    )
    parser.add_argument("--max-condition-examples", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--evidence-output", default=None)

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

    grounded, bundles = resolve_grounded_semantic_task_scope(
        packet_path=args.packet,
        context_path=args.context,
        corpus_roots=args.root,
        scope_mode=args.scope_mode,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )
    context = HypothesisContext.model_validate(_read_json(args.context))
    readiness_path = Path(args.readiness) if args.readiness else (
        Path(args.context).parent / "reframing_operator_readiness.json"
    )
    readiness = ReframingOperatorReadinessReport.model_validate(
        _read_json(readiness_path)
    )
    trigger = (
        ScientificReframeTriggerReport.model_validate(_read_json(args.trigger))
        if args.trigger
        else None
    )
    evidence = build_scientific_reframe_evidence_packet(
        context=context,
        grounded=grounded,
        bundles=bundles,
        max_condition_examples=args.max_condition_examples,
    )
    operators: list[ImplementedReframeOperatorId] = (
        list(args.operator) if args.operator else list(_IMPLEMENTED)
    )

    output = (
        Path(args.output)
        if args.output
        else Path(args.context).parent / "scientific_reframing_shadow.json"
    )
    evidence_output = (
        Path(args.evidence_output)
        if args.evidence_output
        else Path(args.context).parent / "scientific_reframing_evidence.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        evidence.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    assembler = ScientificReframePromptAssembler()
    assessment_by_operator = {
        row.operator_id: row
        for row in readiness.assessments
    }

    if args.dry_run:
        prompt_dir = Path(args.context).parent / "scientific_reframing_prompts"
        if args.save_prompts:
            prompt_dir.mkdir(parents=True, exist_ok=True)
        print("Scientific reframing dry run")
        print("Task:", evidence.task_id)
        print("Premises:", len(evidence.premise_statements))
        print("Gaps:", len(evidence.gap_statements))
        print("Condition examples:", len(evidence.condition_examples))
        print("LLM calls: 0")
        for operator_id in operators:
            assessment = assessment_by_operator.get(operator_id)
            if assessment is None:
                raise SystemExit(f"Missing readiness assessment for {operator_id}")
            executable = assessment.status in {
                "ready_now",
                "ready_with_optional_enrichment",
            }
            trigger_assessment = (
                next(
                    (row for row in trigger.assessments if row.operator_id == operator_id),
                    None,
                )
                if trigger is not None
                else None
            )
            triggered = (
                trigger_assessment is None
                or trigger_assessment.decision == "triggered"
            )
            would_execute = executable and triggered
            trigger_text = (
                trigger_assessment.decision
                if trigger_assessment is not None
                else "legacy_not_evaluated"
            )
            print(
                f"{operator_id}: readiness={assessment.status}; "
                f"trigger={trigger_text}; "
                f"would_execute={str(would_execute).lower()}"
            )
            if would_execute:
                prompt = assembler.build(operator_id=operator_id, evidence=evidence)
                if args.save_prompts:
                    _save_prompt(
                        prompt_dir / f"{operator_id.lower()}.prompt.txt",
                        prompt.system_prompt,
                        prompt.user_prompt,
                    )
        print("Evidence:", evidence_output)
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required for generation unless GRAPHAGENTS_HYPOTHESIS_MODEL "
            "or OPENROUTER_AGENT_MODEL is set"
        )

    backend = InstructorOpenAICompatibleReframeBackend(
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
    outcome = ScientificReframingShadowRuntime(backend).run(
        evidence=evidence,
        readiness=readiness,
        operators=operators,
        trigger_report=trigger,
    )

    output.write_text(
        outcome.report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.save_prompts:
        prompt_dir = Path(args.context).parent / "scientific_reframing_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for row in outcome.prompts:
            _save_prompt(
                prompt_dir / f"{row.operator_id.lower()}.prompt.txt",
                row.prompt.system_prompt,
                row.prompt.user_prompt,
            )

    print("Scientific reframing shadow complete")
    print("Task:", outcome.report.source_task_id)
    print("LLM calls:", outcome.report.llm_calls_performed)
    print("Candidates:", len(outcome.report.candidates))
    for run in outcome.report.runs:
        print(
            f"  {run.operator_id}: readiness={run.readiness_status}; "
            f"decision={run.decision}; candidates={len(run.candidate_ids)}"
        )
        if run.abstention_reason:
            print("    abstention:", run.abstention_reason)
        if run.generation_error_type:
            print("    generation error:", run.generation_error_type)
        if run.compile_issues:
            for issue in run.compile_issues:
                print("    compile issue:", issue)
    for index, candidate in enumerate(outcome.report.candidates, start=1):
        print(f"[{index}] {candidate.operator_id}: {candidate.title}")
        print("    baseline:", candidate.baseline_model.summary)
        print("    alternative:", candidate.alternative_model.summary)
        for note in candidate.provenance_normalizations:
            print("    provenance normalization:", note)
        print("    differential predictions:", len(candidate.differential_predictions))
    print("Evidence:", evidence_output)
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
