from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.idea_evolution import (
    IdeaEvolutionReport,
    build_idea_evolution_plan,
)
from pipeline_core.discovery.idea_evolution_audit import build_idea_evolution_audit
from pipeline_core.discovery.idea_evolution_prompt import build_idea_evolution_prompt
from pipeline_core.discovery.idea_evolution_runtime import (
    IdeaEvolutionRuntime,
    InstructorOpenAICompatibleIdeaEvolutionBackend,
)
from pipeline_core.discovery.reframing.proxy_challenge import ProxyChallengeRunReport
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header values must use KEY=VALUE")
    key, item = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("header key may not be empty")
    return key, item


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the shadow-only Idea Evolution milestone over an audited Frontier. "
            "Native operators perform cross-source bridge, backbone mutation, and "
            "candidate interpretation. Existing scientific-reframing artifacts may "
            "be imported without rerunning those subsystems."
        )
    )
    p.add_argument("--population", required=True, type=Path)
    p.add_argument("--frontier-audit", required=True, type=Path)
    p.add_argument("--scientific-reframe-shadow", type=Path, default=None)
    p.add_argument("--proxy-challenge-shadow", type=Path, default=None)
    p.add_argument("--max-cross-source-outputs", type=int, default=6)
    p.add_argument("--max-backbone-mutation-outputs", type=int, default=6)
    p.add_argument("--max-candidate-interpretation-outputs", type=int, default=4)
    p.add_argument("--max-candidate-parent-pool", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--output-plan", required=True, type=Path)
    p.add_argument("--output-report", type=Path, default=None)
    p.add_argument("--output-audit", type=Path, default=None)
    p.add_argument("--save-prompts-dir", type=Path, default=None)
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    return p


def main() -> int:
    args = parser().parse_args()
    for name in (
        "max_cross_source_outputs",
        "max_backbone_mutation_outputs",
        "max_candidate_interpretation_outputs",
        "max_candidate_parent_pool",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")

    population_path = args.population.expanduser().resolve()
    audit_path = args.frontier_audit.expanduser().resolve()
    population = _load(population_path, FrontierIdeaPopulation)
    frontier_audit = _load(audit_path, FrontierExplorationAudit)

    plan = build_idea_evolution_plan(
        population=population,
        exploration_audit=frontier_audit,
        max_cross_source_outputs=args.max_cross_source_outputs,
        max_backbone_mutation_outputs=args.max_backbone_mutation_outputs,
        max_candidate_interpretation_outputs=args.max_candidate_interpretation_outputs,
        max_candidate_parent_pool=args.max_candidate_parent_pool,
    )
    _write(args.output_plan, plan)

    if args.prepare_only:
        print("Idea Evolution plan prepared")
        print("plan:", args.output_plan)
        for row in plan.operator_plans:
            print(
                f"{row.operator_id}: enabled={row.enabled}; "
                f"parent_pool={len(row.parent_pool_idea_ids)}; "
                f"max_outputs={row.max_output_count}"
            )
        print("LLM calls: 0")
        print("PRODUCTION_SELECTION_AUTHORITY=False")
        return 0

    if args.output_report is None or args.output_audit is None:
        raise SystemExit(
            "--output-report and --output-audit are required unless --prepare-only is used"
        )
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    reframe = None
    reframe_path = ""
    reframe_sha = ""
    if args.scientific_reframe_shadow is not None:
        p = args.scientific_reframe_shadow.expanduser().resolve()
        reframe = _load(p, ScientificReframingShadowReport)
        reframe_path = str(p)
        reframe_sha = _sha256(p)

    proxy = None
    proxy_path = ""
    proxy_sha = ""
    if args.proxy_challenge_shadow is not None:
        p = args.proxy_challenge_shadow.expanduser().resolve()
        proxy = _load(p, ProxyChallengeRunReport)
        proxy_path = str(p)
        proxy_sha = _sha256(p)

    backend = InstructorOpenAICompatibleIdeaEvolutionBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
        telemetry_path=args.output_report.with_suffix(".telemetry.jsonl"),
        telemetry_context={
            "source_population_id": population.population_id,
            "source_context_id": population.source_context_id,
        },
    )
    outcome = IdeaEvolutionRuntime(backend).run(
        population=population,
        exploration_audit=frontier_audit,
        plan=plan,
        population_artifact=str(population_path),
        population_artifact_sha256=_sha256(population_path),
        scientific_reframe_report=reframe,
        scientific_reframe_artifact=reframe_path,
        scientific_reframe_artifact_sha256=reframe_sha,
        proxy_challenge_report=proxy,
        proxy_challenge_artifact=proxy_path,
        proxy_challenge_artifact_sha256=proxy_sha,
    )
    _write(args.output_report, outcome.report)

    audit = build_idea_evolution_audit(
        exploration_audit=frontier_audit,
        evolution_report=outcome.report,
    )
    _write(args.output_audit, audit)

    if args.save_prompts_dir is not None:
        prompt_dir = args.save_prompts_dir.expanduser().resolve()
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for prompt in outcome.prompts:
            (prompt_dir / f"{prompt.operator_id.lower()}.prompt.txt").write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

    print("Idea Evolution shadow complete")
    print("ideas:", outcome.report.idea_count)
    print("by operator:", outcome.report.idea_count_by_operator)
    print("cross-source bridges:", audit.cross_source_bridge_count)
    print("open-world parent coverage:", audit.open_world_parent_coverage_count)
    print("new mutated backbone families:", audit.new_mutated_backbone_family_count)
    print("candidate interpretations:", audit.candidate_interpretation_count)
    print("reframes imported:", audit.reframe_idea_count)
    print("conceptual transition observed:", audit.conceptual_transition_observed)
    print(
        "native LLM calls attempted/succeeded:",
        outcome.report.native_llm_calls_attempted,
        "/",
        outcome.report.native_llm_calls_succeeded,
    )
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("report:", args.output_report)
    print("audit:", args.output_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
