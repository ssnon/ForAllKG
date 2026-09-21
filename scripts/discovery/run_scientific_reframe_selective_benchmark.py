from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.critic_contracts import ScientificReframeCriticReport
from pipeline_core.discovery.reframing.portfolio_contracts import ScientificReframeShadowPortfolio
from pipeline_core.discovery.reframing.reframe_contracts import ScientificReframingShadowReport
from pipeline_core.discovery.reframing.selective_execution import (
    BenchmarkTaskSelectiveExecution,
    blocked_stage,
    failed_stage,
    load_json_model,
    resumable_critic_report,
    resumable_portfolio_report,
    resumable_shadow_report,
    skipped_stage,
    stage_from_critic,
    stage_from_portfolio,
    stage_from_shadow,
    summarize_selective_execution,
    triggered_operator_ids,
)
from pipeline_core.discovery.reframing.trigger_contracts import ScientificReframeTriggerReport


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        if len(output) > 5000:
            output = output[-5000:]
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(command)
            + ("\n" + output if output else "")
        )


def _model_args(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.model:
        values += ["--model", args.model]
    values += ["--api-key-env", args.api_key_env]
    if args.base_url:
        values += ["--base-url", args.base_url]
    values += ["--instructor-mode", args.instructor_mode]
    values += ["--temperature", str(args.temperature)]
    values += ["--parse-retries", str(args.parse_retries)]
    values += ["--timeout", str(args.timeout)]
    for header in args.header:
        values += ["--header", header]
    if args.telemetry:
        values += ["--telemetry", args.telemetry]
    return values


def _generation_command(
    *,
    args: argparse.Namespace,
    canonical: Path,
    shadow_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.discovery.run_scientific_reframing_shadow",
    ]
    for root in args.root:
        command += ["--root", root]
    command += [
        "--packet", str(canonical / "explorer.packet.json"),
        "--context", str(canonical / "hypothesis.context.json"),
        "--readiness", str(canonical / "reframing_operator_readiness.json"),
        "--trigger", str(canonical / "scientific_reframing_triggers.json"),
        "--scope-mode", args.scope_mode,
        "--duplicate-paper-policy", args.duplicate_paper_policy,
        "--cross-root-duplicate-policy", args.cross_root_duplicate_policy,
        "--max-condition-examples", str(args.max_condition_examples),
        "--output", str(shadow_path),
        "--evidence-output", str(canonical / "scientific_reframing_evidence.json"),
    ]
    if args.save_prompts:
        command.append("--save-prompts")
    command += _model_args(args)
    return command


def _critic_command(
    *,
    args: argparse.Namespace,
    canonical: Path,
    shadow_path: Path,
    critic_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.discovery.run_scientific_reframe_critic",
        "--shadow", str(shadow_path),
        "--evidence", str(canonical / "scientific_reframing_evidence.json"),
        "--output", str(critic_path),
    ]
    if args.save_prompts:
        command.append("--save-prompts")
    command += _model_args(args)
    return command


def _portfolio_command(
    *,
    shadow_path: Path,
    critic_path: Path,
    portfolio_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.discovery.build_scientific_reframe_shadow_portfolio",
        "--shadow", str(shadow_path),
        "--critic", str(critic_path),
        "--output", str(portfolio_path),
    ]


def _skipped_task(row) -> BenchmarkTaskSelectiveExecution:
    return BenchmarkTaskSelectiveExecution(
        task_key=row.task_key,
        case_key=row.case_key,
        replicate_key=row.replicate_key,
        canonical_dir=row.canonical_dir,
        task_id=row.task_id,
        trigger_pattern=row.trigger_pattern,
        triggered_operator_ids=[],
        status="skipped_not_triggered",
        generation=skipped_stage("generation"),
        critic=skipped_stage("critic"),
        portfolio=skipped_stage("portfolio"),
        llm_calls_attempted=0,
        llm_calls_succeeded=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the scientific reframing shadow lane selectively across a "
            "benchmark: trigger-gated generation, multidimensional critic, and "
            "deterministic portfolio. Valid current-lineage artifacts are resumed."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--audit", type=Path, default=None)
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
    parser.add_argument("--max-condition-examples", type=int, default=30)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shadow-name", default="scientific_reframing_shadow_selective.json")
    parser.add_argument("--critic-name", default="scientific_reframing_critic_selective.json")
    parser.add_argument("--portfolio-name", default="scientific_reframing_portfolio_selective.json")

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

    benchmark_root = args.benchmark_root.resolve()
    audit_path = args.audit or benchmark_root / "scientific_reframing_trigger_audit.json"
    audit = load_json_model(audit_path, ScientificReframeBenchmarkAuditReport)
    if Path(audit.benchmark_root).resolve() != benchmark_root:
        raise SystemExit("benchmark root does not match source trigger audit")

    resume = not args.no_resume
    task_results: list[BenchmarkTaskSelectiveExecution] = []

    print("Scientific reframing selective benchmark execution")
    print("Tasks:", len(audit.task_audits))
    print("Resume:", str(resume).lower())

    for row in audit.task_audits:
        operators = triggered_operator_ids(row)
        if row.status != "complete":
            generation = failed_stage(
                "generation",
                exc=RuntimeError("source benchmark audit row is not complete"),
            )
            task_results.append(
                BenchmarkTaskSelectiveExecution(
                    task_key=row.task_key,
                    case_key=row.case_key,
                    replicate_key=row.replicate_key,
                    canonical_dir=row.canonical_dir,
                    task_id=row.task_id,
                    trigger_pattern=row.trigger_pattern,
                    triggered_operator_ids=operators,
                    status="error",
                    generation=generation,
                    critic=blocked_stage("critic"),
                    portfolio=blocked_stage("portfolio"),
                    llm_calls_attempted=0,
                    llm_calls_succeeded=0,
                )
            )
            print(f"{row.task_key}: ERROR source audit row incomplete")
            continue
        if not operators:
            task_results.append(_skipped_task(row))
            print(f"{row.task_key}: skipped_not_triggered")
            continue

        canonical = Path(row.canonical_dir)
        trigger_path = canonical / "scientific_reframing_triggers.json"
        shadow_path = canonical / args.shadow_name
        critic_path = canonical / args.critic_name
        portfolio_path = canonical / args.portfolio_name

        generation_stage = None
        critic_stage = None
        portfolio_stage = None
        shadow = None
        critic = None
        portfolio = None
        resume_used = False

        try:
            trigger = load_json_model(trigger_path, ScientificReframeTriggerReport)
            audit_triggered = set(operators)
            artifact_triggered = {
                item.operator_id
                for item in trigger.assessments
                if item.decision == "triggered"
            }
            if audit_triggered != artifact_triggered:
                raise ValueError(
                    "benchmark audit/task trigger artifact mismatch: "
                    f"audit={sorted(audit_triggered)} artifact={sorted(artifact_triggered)}"
                )
        except Exception as exc:
            generation_stage = failed_stage("generation", exc=exc, artifact_path=shadow_path)
            task_results.append(
                BenchmarkTaskSelectiveExecution(
                    task_key=row.task_key,
                    case_key=row.case_key,
                    replicate_key=row.replicate_key,
                    canonical_dir=row.canonical_dir,
                    task_id=row.task_id,
                    trigger_pattern=row.trigger_pattern,
                    triggered_operator_ids=operators,
                    status="error",
                    generation=generation_stage,
                    critic=blocked_stage("critic"),
                    portfolio=blocked_stage("portfolio"),
                    llm_calls_attempted=0,
                    llm_calls_succeeded=0,
                )
            )
            print(f"{row.task_key}: ERROR trigger lineage: {type(exc).__name__}")
            continue

        if resume:
            shadow, reason = resumable_shadow_report(
                path=shadow_path,
                trigger=trigger,
                expected_task_id=row.task_id,
            )
            if shadow is not None:
                resume_used = True
                generation_stage = stage_from_shadow(
                    report=shadow,
                    path=shadow_path,
                    status="resumed",
                    resume_reason=reason,
                )
        if shadow is None:
            try:
                _run(_generation_command(args=args, canonical=canonical, shadow_path=shadow_path))
                shadow = load_json_model(shadow_path, ScientificReframingShadowReport)
                generation_stage = stage_from_shadow(
                    report=shadow,
                    path=shadow_path,
                    status="executed",
                )
            except Exception as exc:
                generation_stage = failed_stage("generation", exc=exc, artifact_path=shadow_path)
                task_results.append(
                    BenchmarkTaskSelectiveExecution(
                        task_key=row.task_key,
                        case_key=row.case_key,
                        replicate_key=row.replicate_key,
                        canonical_dir=row.canonical_dir,
                        task_id=row.task_id,
                        trigger_pattern=row.trigger_pattern,
                        triggered_operator_ids=operators,
                        status="error",
                        generation=generation_stage,
                        critic=blocked_stage("critic"),
                        portfolio=blocked_stage("portfolio"),
                        llm_calls_attempted=0,
                        llm_calls_succeeded=0,
                    )
                )
                print(f"{row.task_key}: ERROR generation: {type(exc).__name__}")
                continue

        if resume:
            critic, reason = resumable_critic_report(path=critic_path, shadow=shadow)
            if critic is not None:
                resume_used = True
                critic_stage = stage_from_critic(
                    report=critic,
                    path=critic_path,
                    status="resumed",
                    resume_reason=reason,
                )
        if critic is None:
            try:
                _run(
                    _critic_command(
                        args=args,
                        canonical=canonical,
                        shadow_path=shadow_path,
                        critic_path=critic_path,
                    )
                )
                critic = load_json_model(critic_path, ScientificReframeCriticReport)
                critic_stage = stage_from_critic(
                    report=critic,
                    path=critic_path,
                    status="executed",
                )
            except Exception as exc:
                critic_stage = failed_stage("critic", exc=exc, artifact_path=critic_path)
                generation_stage = generation_stage or blocked_stage("generation")
                attempted = generation_stage.llm_calls_attempted
                succeeded = generation_stage.llm_calls_succeeded
                task_results.append(
                    BenchmarkTaskSelectiveExecution(
                        task_key=row.task_key,
                        case_key=row.case_key,
                        replicate_key=row.replicate_key,
                        canonical_dir=row.canonical_dir,
                        task_id=row.task_id,
                        trigger_pattern=row.trigger_pattern,
                        triggered_operator_ids=operators,
                        status="error",
                        generation=generation_stage,
                        critic=critic_stage,
                        portfolio=blocked_stage("portfolio"),
                        candidate_count=len(shadow.candidates),
                        llm_calls_attempted=attempted,
                        llm_calls_succeeded=succeeded,
                        resume_used=resume_used,
                    )
                )
                print(f"{row.task_key}: ERROR critic: {type(exc).__name__}")
                continue

        if resume:
            portfolio, reason = resumable_portfolio_report(
                path=portfolio_path,
                shadow=shadow,
                critic=critic,
            )
            if portfolio is not None:
                resume_used = True
                portfolio_stage = stage_from_portfolio(
                    path=portfolio_path,
                    status="resumed",
                    resume_reason=reason,
                )
        if portfolio is None:
            try:
                _run(
                    _portfolio_command(
                        shadow_path=shadow_path,
                        critic_path=critic_path,
                        portfolio_path=portfolio_path,
                    )
                )
                portfolio = load_json_model(portfolio_path, ScientificReframeShadowPortfolio)
                portfolio_stage = stage_from_portfolio(
                    path=portfolio_path,
                    status="executed",
                )
            except Exception as exc:
                portfolio_stage = failed_stage("portfolio", exc=exc, artifact_path=portfolio_path)
                attempted = generation_stage.llm_calls_attempted + critic_stage.llm_calls_attempted
                succeeded = generation_stage.llm_calls_succeeded + critic_stage.llm_calls_succeeded
                task_results.append(
                    BenchmarkTaskSelectiveExecution(
                        task_key=row.task_key,
                        case_key=row.case_key,
                        replicate_key=row.replicate_key,
                        canonical_dir=row.canonical_dir,
                        task_id=row.task_id,
                        trigger_pattern=row.trigger_pattern,
                        triggered_operator_ids=operators,
                        status="error",
                        generation=generation_stage,
                        critic=critic_stage,
                        portfolio=portfolio_stage,
                        candidate_count=len(shadow.candidates),
                        critic_review_count=len(critic.reviews),
                        llm_calls_attempted=attempted,
                        llm_calls_succeeded=succeeded,
                        resume_used=resume_used,
                    )
                )
                print(f"{row.task_key}: ERROR portfolio: {type(exc).__name__}")
                continue

        stage_errors = any(
            stage.status == "executed_with_errors"
            for stage in (generation_stage, critic_stage, portfolio_stage)
        )
        attempted = sum(
            stage.llm_calls_attempted
            for stage in (generation_stage, critic_stage, portfolio_stage)
        )
        succeeded = sum(
            stage.llm_calls_succeeded
            for stage in (generation_stage, critic_stage, portfolio_stage)
        )
        task = BenchmarkTaskSelectiveExecution(
            task_key=row.task_key,
            case_key=row.case_key,
            replicate_key=row.replicate_key,
            canonical_dir=row.canonical_dir,
            task_id=row.task_id,
            trigger_pattern=row.trigger_pattern,
            triggered_operator_ids=operators,
            status="completed_with_errors" if stage_errors else "completed",
            generation=generation_stage,
            critic=critic_stage,
            portfolio=portfolio_stage,
            candidate_count=len(shadow.candidates),
            critic_review_count=len(critic.reviews),
            portfolio_assessment_count=len(portfolio.assessments),
            resume_used=resume_used,
            llm_calls_attempted=attempted,
            llm_calls_succeeded=succeeded,
        )
        task_results.append(task)
        print(
            f"{row.task_key}: {task.status}; operators={','.join(operators)}; "
            f"candidates={task.candidate_count}; llm_calls={attempted}; "
            f"resume={str(resume_used).lower()}"
        )

    report = summarize_selective_execution(
        audit=audit,
        extraction_roots=list(args.root),
        resume_enabled=resume,
        task_executions=task_results,
    )
    output = args.output or benchmark_root / "scientific_reframing_selective_execution.json"
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("\nScientific reframing selective benchmark execution complete")
    print("Triggered tasks:", report.triggered_task_count)
    print("Skipped tasks:", report.skipped_task_count)
    print("Completed tasks:", report.completed_task_count)
    print("Completed with errors:", report.completed_with_errors_task_count)
    print("Error tasks:", report.error_task_count)
    print("Resumed tasks:", report.resumed_task_count)
    print("Generation LLM calls attempted:", report.generation_calls_attempted)
    print("Critic LLM calls attempted:", report.critic_calls_attempted)
    print("Total LLM calls attempted:", report.llm_calls_attempted)
    print("Total LLM calls succeeded:", report.llm_calls_succeeded)
    print("No novelty, N10, canonical graph mutation, winner, or production selection was performed.")
    print("Output:", output)

    if args.fail_on_error and report.error_task_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
