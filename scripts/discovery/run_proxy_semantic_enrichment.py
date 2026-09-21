from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.discovery.reframing.operator_readiness import (
    ReframingOperatorReadinessReport,
)
from pipeline_core.discovery.reframing.proxy_enrichment import (
    InstructorOpenAICompatibleProxyBackend,
    execute_proxy_semantic_enrichment,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute task-scoped PROXY_CHALLENGE semantic enrichment from the "
            "existing mandatory backfill plan. Canonical KG artifacts are never mutated."
        )
    )
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--readiness", default=None)
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
        "--max-calls",
        type=int,
        default=None,
        help=(
            "Maximum new LLM calls in this invocation. Completed chunk reviews are "
            "resumed without a call. Omit to execute all pending targets."
        ),
    )
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--review-path", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_ENRICHMENT_MODEL")
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
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--telemetry", default=None)
    args = parser.parse_args()

    if args.max_calls is not None and args.max_calls < 0:
        raise SystemExit("--max-calls must be >= 0")

    grounded, bundles = resolve_grounded_semantic_task_scope(
        packet_path=args.packet,
        context_path=args.context,
        corpus_roots=args.root,
        scope_mode=args.scope_mode,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )
    readiness_path = Path(args.readiness) if args.readiness else (
        Path(args.context).parent / "reframing_operator_readiness.json"
    )
    readiness = ReframingOperatorReadinessReport.model_validate(
        _read_json(readiness_path)
    )
    if readiness.scope_id != grounded.selection.task_id:
        raise SystemExit(
            "readiness scope_id does not match the grounded task: "
            f"{readiness.scope_id} != {grounded.selection.task_id}"
        )
    assessments = {
        row.operator_id: row
        for row in readiness.assessments
    }
    proxy = assessments.get("PROXY_CHALLENGE")
    if proxy is None:
        raise SystemExit("readiness report has no PROXY_CHALLENGE assessment")
    plan = proxy.mandatory_backfill_plan
    grounded_chunk_keys = {
        ref.identity_key() for ref in grounded.resolution.source_chunks
    }
    stale_targets = [
        target.source_chunk_ref.object_id
        for target in plan.targets
        if target.source_chunk_ref.identity_key() not in grounded_chunk_keys
    ]
    if stale_targets:
        raise SystemExit(
            "readiness backfill plan contains source chunks outside the current grounded scope: "
            + ", ".join(stale_targets)
        )
    if proxy.status == "blocked_insufficient_substrate":
        raise SystemExit(
            "PROXY_CHALLENGE is blocked by insufficient grounded substrate; enrichment is not allowed"
        )
    if plan.target_count == 0:
        print("PROXY_CHALLENGE has no mandatory proxy_semantics backfill targets.")
        print("LLM calls: 0")
        return 0
    if proxy.status != "targeted_enrichment_required":
        raise SystemExit(
            "mandatory proxy_semantics targets exist but readiness status is not targeted_enrichment_required"
        )

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_ENRICHMENT_MODEL, "
            "GRAPHAGENTS_HYPOTHESIS_MODEL, or OPENROUTER_AGENT_MODEL is set"
        )

    canonical_dir = Path(args.context).parent
    annotation_path = Path(args.annotation_path) if args.annotation_path else (
        canonical_dir / "annotations" / "proxy_semantics_v1_2.jsonl"
    )
    review_path = Path(args.review_path) if args.review_path else (
        canonical_dir / "annotations" / "proxy_semantics_v1_2_reviews.jsonl"
    )
    report_path = Path(args.report) if args.report else (
        canonical_dir / "proxy_semantic_enrichment_report_v1_2.json"
    )

    backend = InstructorOpenAICompatibleProxyBackend(
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
            "task_id": grounded.selection.task_id,
            "source_plan_id": plan.plan_id,
        },
    )
    report = execute_proxy_semantic_enrichment(
        scope_id=grounded.selection.task_id,
        plan=plan,
        bundles=bundles,
        backend=backend,
        annotation_path=annotation_path,
        review_path=review_path,
        max_new_calls=args.max_calls,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("PROXY_CHALLENGE targeted semantic enrichment complete")
    print("Task:", report.scope_id)
    print("Plan targets:", report.plan_target_count)
    print("Completed targets:", report.completed_target_count)
    print("Resumed targets:", report.resumed_target_count)
    print("New completed targets:", report.new_target_count)
    print("Pending targets:", report.pending_target_count)
    print("Failed targets:", report.failed_target_count)
    print("LLM calls:", report.llm_calls_performed)
    print("New annotations:", report.new_annotation_count)
    print("Total annotations:", report.total_annotation_count)
    print("New rejected annotation candidates:", report.new_rejected_candidate_count)
    print("Total rejected annotation candidates:", report.total_rejected_candidate_count)
    print("Coverage complete for plan:", str(report.coverage_complete_for_plan).lower())
    print("Canonical graph mutated: false")
    print("Annotation sidecar:", report.annotation_path)
    print("Review ledger:", report.review_path)
    print("Report:", report_path)
    for error in report.errors:
        print(
            "ERROR",
            error.source_chunk_ref.paper_id,
            error.source_chunk_ref.chunk_id,
            error.error_type,
            error.error_message,
        )
    return 2 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
