#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.higher_order_external_shadow_plan import build_higher_order_external_shadow_batch_plan
from pipeline_core.discovery.higher_order_hypothesis_batch import HigherOrderShadowBatchRuntime
from pipeline_core.discovery.higher_order_modifier_eligibility import screen_confirmed_known_modifiers
from pipeline_core.discovery.higher_order_synthesis_context import build_higher_order_synthesis_contexts
from pipeline_core.discovery.higher_order_topology_carrier import build_topology_native_synthesis_carriers
from pipeline_core.discovery.higher_order_topology_composition import compose_higher_order_topologies
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.relation_component_composition import (
    confirmed_known_component_from_mapping,
    compose_relation_component_topologies,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_components(path: Path):
    rows, seen = [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                component = confirmed_known_component_from_mapping(row)
            except Exception as exc:
                raise ValueError(f"accepted-pattern validation failed at row {row_number}") from exc
            if component.component_id in seen:
                continue
            seen.add(component.component_id)
            rows.append(component)
    return tuple(rows)


def _arm_payload(index: int, arm, arm_dir: Path) -> dict[str, Any]:
    ho = arm.higher_order_context
    projection = arm.projection
    authorization = arm.authorization
    run = arm.run_outcome
    canonical = run.canonical_outcome

    paths = {
        "higher_order_source_context": arm_dir / "higher_order.source_context.json",
        "derived_hypothesis_context": arm_dir / "hypothesis.context.json",
        "materialization": arm_dir / "higher_order.materialization.json",
        "authorization": arm_dir / "higher_order.authorization.json",
        "run_record": arm_dir / "hypothesis.run.json",
        "validation": arm_dir / "hypothesis.validation.json",
        "final_draft": arm_dir / "hypothesis.final_draft.json",
        "portfolio": arm_dir / "hypothesis.portfolio.json",
        "prompt": arm_dir / "hypothesis.prompt.txt",
    }
    _write(paths["higher_order_source_context"], ho)
    _write(paths["derived_hypothesis_context"], projection.context)
    _write(paths["materialization"], projection.materialization)
    _write(paths["authorization"], authorization)
    _write(paths["run_record"], canonical.run_record)
    paths["prompt"].write_text(
        "SYSTEM\n======\n" + canonical.prompt.system_prompt + "\n\nUSER\n====\n" + canonical.prompt.user_prompt + "\n",
        encoding="utf-8",
    )
    if canonical.final_draft is not None:
        _write(paths["final_draft"], canonical.final_draft)
    if canonical.validation is not None:
        _write(paths["validation"], canonical.validation)
    if canonical.accepted_portfolio is not None:
        _write(paths["portfolio"], canonical.accepted_portfolio)

    return {
        "pool_index": index,
        "modifier": ho.structural_opportunity.modifier_text,
        "modifier_component_id": ho.lineage.modifier_component_id,
        "higher_order_context_id": ho.context_id,
        "higher_order_topology_id": ho.higher_order_topology_id,
        "status": run.status,
        "shadow_contract_passed": run.shadow_contract_passed,
        "canonical_runtime_accepted": canonical.accepted,
        "final_validation_passed": canonical.run_record.final_validation_passed,
        "failure_stage": canonical.run_record.failure_stage,
        "generation_attempts": canonical.run_record.generation_attempts,
        "repair_attempts": canonical.run_record.repair_attempts,
        "shadow_failure_code": run.shadow_failure_code,
        "portfolio": str(paths["portfolio"]) if canonical.accepted_portfolio is not None else None,
        "hypothesis_statements": (
            [x.hypothesis_statement for x in canonical.accepted_portfolio.hypotheses]
            if canonical.accepted_portfolio is not None else []
        ),
        "artifacts": {
            k: (str(v) if (v.exists() or k in {"higher_order_source_context", "derived_hypothesis_context", "materialization", "authorization", "run_record", "prompt"}) else None)
            for k, v in paths.items() if k != "portfolio"
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Authority-neutral S24 higher-order shadow generation lane.")
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--accepted-patterns", type=Path, required=True)
    p.add_argument(
        "--task-axis-report",
        type=Path,
        default=None,
        help=(
            "Preferred canonical E2E endpoint source. Reads the resolved "
            "requested_source/requested_target produced by Stage 7.5."
        ),
    )
    p.add_argument(
        "--requested-source",
        default=None,
        help="Standalone compatibility only; do not use from canonical parent E2E.",
    )
    p.add_argument(
        "--requested-target",
        default=None,
        help="Standalone compatibility only; do not use from canonical parent E2E.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-contexts", type=int, default=12)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--max-repairs", type=int, choices=(0, 1), default=1)
    args = p.parse_args()

    if args.max_contexts < 1:
        raise ValueError("--max-contexts must be >= 1")

    task_axis_report_path = (
        args.task_axis_report.expanduser().resolve()
        if args.task_axis_report is not None
        else None
    )

    if task_axis_report_path is not None:
        if args.requested_source is not None or args.requested_target is not None:
            raise ValueError(
                "--task-axis-report is mutually exclusive with direct "
                "--requested-source/--requested-target"
            )
        if not task_axis_report_path.is_file():
            raise FileNotFoundError(
                f"task-axis report not found: {task_axis_report_path}"
            )

        task_axis_report = json.loads(
            task_axis_report_path.read_text(encoding="utf-8")
        )
        requested_source = task_axis_report.get("requested_source")
        requested_target = task_axis_report.get("requested_target")

        if not isinstance(requested_source, str) or not requested_source.strip():
            raise ValueError(
                "Stage-7.5 task-axis report lacks a non-empty requested_source"
            )
        if not isinstance(requested_target, str) or not requested_target.strip():
            raise ValueError(
                "Stage-7.5 task-axis report lacks a non-empty requested_target"
            )

        requested_source = requested_source.strip()
        requested_target = requested_target.strip()
        endpoint_resolution_source = "stage7_5_task_axis_report"
        endpoint_resolution_mode = task_axis_report.get(
            "endpoint_resolution_mode"
        )
    else:
        if not args.requested_source or not args.requested_target:
            raise ValueError(
                "Provide --task-axis-report, or both standalone "
                "--requested-source and --requested-target"
            )
        requested_source = str(args.requested_source).strip()
        requested_target = str(args.requested_target).strip()
        endpoint_resolution_source = "standalone_direct_endpoints"
        endpoint_resolution_mode = None

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "generation_report.json"

    source_context = HypothesisContext.model_validate_json(args.context.read_text(encoding="utf-8"))
    components = _load_components(args.accepted_patterns)
    backbones = compose_relation_component_topologies(
        components=components,
        requested_source=requested_source,
        requested_target=requested_target,
        require_endpoint_fidelity=True,
    )
    screen = screen_confirmed_known_modifiers(backbones=backbones, components=components)
    topologies = compose_higher_order_topologies(backbones=backbones, modifiers=screen.eligible)
    carriers = build_topology_native_synthesis_carriers(
        topologies=topologies,
        requested_source=requested_source,
        requested_target=requested_target,
    )
    contexts = build_higher_order_synthesis_contexts(carriers=carriers)

    _write(out / "modifier_eligibility_audit.json", screen.audit)
    _write(out / "backbones.json", [x.model_dump(mode="json") for x in backbones])
    _write(out / "topologies.json", [x.model_dump(mode="json") for x in topologies])
    _write(out / "carriers.json", [x.model_dump(mode="json") for x in carriers])
    _write(out / "contexts.json", [x.model_dump(mode="json") for x in contexts])

    base_report = {
        "schema_version": "higher-order-shadow-lane-generation-v1",
        "requested_source": requested_source,
        "requested_target": requested_target,
        "endpoint_resolution_source": endpoint_resolution_source,
        "endpoint_resolution_mode": endpoint_resolution_mode,
        "task_axis_report": (
            str(task_axis_report_path)
            if task_axis_report_path is not None
            else None
        ),
        "source_context_id": source_context.context_id,
        "source_context_sha256": source_context.context_sha256,
        "accepted_pattern_component_count": len(components),
        "strict_backbone_count": len(backbones),
        "eligible_modifier_count": len(screen.eligible),
        "higher_order_topology_count": len(topologies),
        "higher_order_context_count": len(contexts),
        "shadow_only": True,
        "scientific_quality_ranking_performed": False,
        "external_novelty_evaluated": False,
        "n10_run": False,
        "production_selection_changed": False,
        "legacy_portfolio_mutated": False,
        "novelty_authority": False,
    }
    if not contexts:
        _write(report_path, {**base_report, "selected_context_count": 0, "candidate_count": 0, "proposed_count": 0, "arms": [], "status": "NO_HIGHER_ORDER_CONTEXTS"})
        print("Higher-order shadow: no eligible contexts; legacy lane unchanged.")
        print("report:", report_path)
        return 0

    backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=args.parse_retries,
        telemetry_path=out / "telemetry.jsonl",
        telemetry_context={"pipeline": "higher_order_shadow_lane", "stage": "generation"},
    )
    outcome = HigherOrderShadowBatchRuntime(backend, max_repairs=args.max_repairs).run(
        source_context=source_context,
        contexts=contexts,
        max_contexts=min(args.max_contexts, len(contexts)),
    )
    arms = []
    for index, arm in enumerate(outcome.arms, start=1):
        arm_dir = out / f"{index:02d}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        arms.append(_arm_payload(index, arm, arm_dir))

    external_plan_path = None
    if outcome.proposed_arms:
        external_plan_path = out / "external_shadow_plan.json"
        _write(external_plan_path, build_higher_order_external_shadow_batch_plan(outcome))

    report = {
        **base_report,
        "selected_context_count": outcome.record.selected_context_count,
        "candidate_count": len(arms),
        "proposed_count": outcome.record.proposed_count,
        "abstained_count": outcome.record.abstained_count,
        "canonical_rejected_count": outcome.record.canonical_rejected_count,
        "shadow_contract_rejected_count": outcome.record.shadow_contract_rejected_count,
        "batch_record": outcome.record.model_dump(mode="json"),
        "external_shadow_plan": str(external_plan_path) if external_plan_path else None,
        "arms": arms,
        "status": "HIGHER_ORDER_SHADOW_GENERATION_COMPLETE",
    }
    _write(report_path, report)
    print("Higher-order shadow generation complete")
    print("strict backbones:", len(backbones))
    print("eligible modifiers:", len(screen.eligible))
    print("topologies:", len(topologies))
    print("selected contexts:", outcome.record.selected_context_count)
    print("proposed:", outcome.record.proposed_count)
    print("report:", report_path)
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
