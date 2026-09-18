#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline_core.discovery.discovery_bundle import DiscoveryPolicy
from pipeline_core.discovery.discovery_contracts import DiscoveryBundle
from pipeline_core.discovery.higher_order_external_shadow_plan import build_higher_order_external_shadow_batch_plan
from pipeline_core.discovery.higher_order_hypothesis_batch import HigherOrderShadowBatchRuntime
from pipeline_core.discovery.higher_order_modifier_eligibility import screen_confirmed_known_modifiers
from pipeline_core.discovery.higher_order_synthesis_context import build_higher_order_synthesis_contexts
from pipeline_core.discovery.higher_order_topology_carrier import build_topology_native_synthesis_carriers
from pipeline_core.discovery.higher_order_topology_composition import compose_higher_order_topologies
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.relation_component_composition import (
    candidate_inspiration_component,
    confirmed_known_component_from_mapping,
    compose_relation_component_topologies,
)
from pipeline_core.discovery.task_backbone_chain import (
    compose_three_component_task_backbones,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    candidate_relation_from_mapping,
)
from scripts.discovery.build_task_conditioned_axis_plan import (
    _candidate_mapping,
    _candidate_unit_id,
    _profile_mediator_equivalences,
    _quality_eligible,
    _replay_and_capture,
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


def _candidate_modifier_components_from_canonical_root(
    *,
    canonical_root: Path,
    domain_profile: str,
) -> tuple[tuple[object, ...], dict[str, Any]]:
    # Reconstruct reserve-quality candidate-unit components from the same
    # frozen pool used by the canonical DiscoveryBundle. Replay is accepted
    # only when its bundle SHA matches the canonical bundle exactly.
    preferred_bundle = (
        canonical_root
        / ".a17f.generic_bundle.replay.json"
    )
    if preferred_bundle.is_file():
        bundle_path = preferred_bundle
    else:
        bundle_paths = sorted(
            canonical_root.glob(
                ".*.generic_bundle.replay.json"
            )
        )
        if not bundle_paths:
            return (), {
                "status": "NO_GENERIC_BUNDLE",
                "candidate_modifier_component_count": 0,
                "generic_bundle_selection_changed": False,
            }
        bundle_path = bundle_paths[0]

    final_traversal = canonical_root / "traversal.json"
    candidate_traversal = (
        canonical_root
        / "candidate_unit.traversal.a3.json"
    )
    if (
        not final_traversal.is_file()
        or not candidate_traversal.is_file()
    ):
        return (), {
            "status": "NO_CANDIDATE_REPLAY_TRAVERSALS",
            "candidate_modifier_component_count": 0,
            "generic_bundle": str(bundle_path),
            "generic_bundle_selection_changed": False,
        }

    expected_bundle = DiscoveryBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )

    replay_threshold = 0.30
    manifest_path = canonical_root / "e2e_runner.manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        replay_threshold = float(
            (
                manifest.get("candidate_unit_policy")
                or {}
            ).get(
                "min_candidate_unit_score",
                replay_threshold,
            )
        )

    reserve_threshold = float(
        DiscoveryPolicy()
        .min_reserved_candidate_unit_score
    )

    with tempfile.TemporaryDirectory(
        prefix="forallkg_s27b_candidate_replay_"
    ) as tmp:
        builder, replay, enriched_rows = (
            _replay_and_capture(
                final_traversal=final_traversal,
                candidate_traversal=candidate_traversal,
                domain_profile=domain_profile,
                discovery_top_k=int(
                    expected_bundle.selected_count
                    or DiscoveryPolicy().top_k
                ),
                min_candidate_unit_score=replay_threshold,
                expected_bundle=expected_bundle,
                scratch_bundle=(
                    Path(tmp)
                    / "discovery_bundle.replay.json"
                ),
            )
        )

        by_unit: dict[
            str,
            tuple[
                float,
                float,
                object,
            ],
        ] = {}

        for enriched in enriched_rows:
            mapping = _candidate_mapping(enriched)
            unit_id = _candidate_unit_id(enriched)

            if mapping is None or not unit_id:
                continue

            try:
                inspiration = builder._materialize_inspiration(
                    corpus_id=expected_bundle.corpus_id,
                    rank=1,
                    row=enriched,
                    semantic_mode=(
                        replay.semantic_diversity_mode
                    ),
                )
            except Exception:
                continue

            if not _quality_eligible(
                inspiration,
                min_candidate_unit_score=(
                    reserve_threshold
                ),
            ):
                continue

            try:
                relation = (
                    candidate_relation_from_mapping(
                        mapping
                    )
                )
                component = (
                    candidate_inspiration_component(
                        relation=relation,
                        candidate_unit_score=float(
                            inspiration
                            .candidate_unit_score
                        ),
                        exploration_score=float(
                            inspiration
                            .exploration_score
                        ),
                        quality_eligible=True,
                        source_path_id=str(
                            inspiration.source_path_id
                        ),
                    )
                )
            except Exception:
                continue

            score = float(
                inspiration.candidate_unit_score
            )
            exploration = float(
                inspiration.exploration_score
            )

            previous = by_unit.get(unit_id)
            if (
                previous is None
                or score > previous[0]
                or (
                    score == previous[0]
                    and exploration > previous[1]
                )
            ):
                by_unit[unit_id] = (
                    score,
                    exploration,
                    component,
                )

    ordered = sorted(
        by_unit.items(),
        key=lambda item: (
            -item[1][0],
            -item[1][1],
            item[0],
        ),
    )
    components = tuple(
        payload[2]
        for _, payload in ordered
    )

    return components, {
        "status": "CANDIDATE_MODIFIER_REPLAY_COMPLETE",
        "generic_bundle": str(bundle_path),
        "generic_bundle_id": expected_bundle.bundle_id,
        "generic_bundle_sha256": (
            expected_bundle.bundle_sha256
        ),
        "replay_bundle_sha256": replay.bundle_sha256,
        "replay_sha_matches_canonical": (
            replay.bundle_sha256
            == expected_bundle.bundle_sha256
        ),
        "stage7_5_replay_candidate_threshold": (
            replay_threshold
        ),
        "reserve_candidate_threshold": (
            reserve_threshold
        ),
        "reserve_candidate_threshold_source": (
            "DiscoveryPolicy.min_reserved_candidate_unit_score"
        ),
        "candidate_modifier_component_count": (
            len(components)
        ),
        "generic_bundle_selection_changed": False,
        "global_axis_threshold_changed": False,
        "candidate_positive_premise_authority_created": False,
        "novelty_authority_created": False,
    }



def _modifier_candidate_component_id(
    context: object,
) -> str | None:
    premises = getattr(context, "premises", ())

    for premise in premises:
        if getattr(
            premise,
            "premise_role",
            None,
        ) != "modifier_relation":
            continue

        authority = getattr(
            premise,
            "authority",
            None,
        )
        authority_value = getattr(
            authority,
            "value",
            authority,
        )

        if str(authority_value) != "candidate_inspiration":
            return None

        component_id = str(
            getattr(
                premise,
                "component_id",
                "",
            )
        ).strip()
        return component_id or None

    return None


def _select_shadow_generation_contexts(
    *,
    contexts: tuple[object, ...] | list[object],
    max_contexts: int,
    candidate_reserve_fraction: float = 0.25,
    candidate_reserve_cap: int = 3,
) -> tuple[
    tuple[object, ...],
    dict[str, Any],
]:
    """
    Deterministic shadow-only coverage selector.

    Existing global context order remains authoritative. This helper only
    reserves bounded representation for distinct candidate-inspiration
    modifiers when such contexts would otherwise fall outside the generation
    prefix.

    It performs no scientific-quality ranking and creates no evidence,
    premise, gap, novelty, or production-selection authority.
    """
    rows = tuple(contexts)
    budget = min(
        max(int(max_contexts), 0),
        len(rows),
    )

    if budget == 0:
        return (), {
            "policy": "bounded_candidate_modifier_coverage_v1",
            "input_context_count": len(rows),
            "budget": 0,
            "candidate_context_count": 0,
            "unique_candidate_modifier_count": 0,
            "candidate_reserve_target": 0,
            "selected_candidate_context_count": 0,
            "selected_unique_candidate_modifier_count": 0,
            "selected_original_indices": [],
            "selection_changed_from_prefix": False,
            "scientific_quality_ranking_performed": False,
            "production_selection_changed": False,
        }

    candidate_first_index: dict[str, int] = {}
    candidate_context_count = 0

    for index, context in enumerate(rows, start=1):
        component_id = _modifier_candidate_component_id(context)
        if component_id is None:
            continue

        candidate_context_count += 1
        candidate_first_index.setdefault(
            component_id,
            index,
        )

    unique_candidate_count = len(candidate_first_index)

    if unique_candidate_count == 0:
        selected_indices = list(range(1, budget + 1))
        return (
            tuple(rows[index - 1] for index in selected_indices),
            {
                "policy": "bounded_candidate_modifier_coverage_v1",
                "input_context_count": len(rows),
                "budget": budget,
                "candidate_context_count": 0,
                "unique_candidate_modifier_count": 0,
                "candidate_reserve_target": 0,
                "selected_candidate_context_count": 0,
                "selected_unique_candidate_modifier_count": 0,
                "selected_original_indices": selected_indices,
                "selection_changed_from_prefix": False,
                "scientific_quality_ranking_performed": False,
                "production_selection_changed": False,
            },
        )

    reserve_from_fraction = max(
        1,
        int(
            budget
            * float(candidate_reserve_fraction)
        ),
    )
    reserve_target = min(
        unique_candidate_count,
        int(candidate_reserve_cap),
        reserve_from_fraction,
        budget,
    )

    required_indices = sorted(
        list(candidate_first_index.values())[
            :reserve_target
        ]
    )
    required_set = set(required_indices)

    selected_indices = list(required_indices)

    for index in range(1, len(rows) + 1):
        if len(selected_indices) >= budget:
            break
        if index in required_set:
            continue
        selected_indices.append(index)

    selected_indices = sorted(selected_indices[:budget])

    selected_candidate_ids: list[str] = []
    selected_candidate_context_count = 0

    for index in selected_indices:
        component_id = _modifier_candidate_component_id(
            rows[index - 1]
        )
        if component_id is None:
            continue

        selected_candidate_context_count += 1
        if component_id not in selected_candidate_ids:
            selected_candidate_ids.append(component_id)

    baseline_indices = list(range(1, budget + 1))

    return (
        tuple(rows[index - 1] for index in selected_indices),
        {
            "policy": "bounded_candidate_modifier_coverage_v1",
            "input_context_count": len(rows),
            "budget": budget,
            "candidate_context_count": candidate_context_count,
            "unique_candidate_modifier_count": unique_candidate_count,
            "candidate_reserve_fraction": float(
                candidate_reserve_fraction
            ),
            "candidate_reserve_cap": int(
                candidate_reserve_cap
            ),
            "candidate_reserve_target": reserve_target,
            "selected_candidate_context_count": (
                selected_candidate_context_count
            ),
            "selected_unique_candidate_modifier_count": len(
                selected_candidate_ids
            ),
            "selected_candidate_modifier_component_ids": (
                selected_candidate_ids
            ),
            "selected_original_indices": selected_indices,
            "baseline_prefix_indices": baseline_indices,
            "selection_changed_from_prefix": (
                selected_indices
                != baseline_indices
            ),
            "scientific_quality_ranking_performed": False,
            "production_selection_changed": False,
        },
    )


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
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Build and write higher-order backbone/modifier/topology/context "
            "artifacts, but do not invoke an LLM backend."
        ),
    )
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

    strict_two_component_backbones = (
        compose_relation_component_topologies(
            components=components,
            requested_source=requested_source,
            requested_target=requested_target,
            require_endpoint_fidelity=True,
        )
    )

    strict_three_component_backbones = (
        compose_three_component_task_backbones(
            components=components,
            requested_source=requested_source,
            requested_target=requested_target,
            mediator_equivalences=(
                _profile_mediator_equivalences(
                    source_context.domain_profile_id
                )
            ),
        )
    )

    backbones = tuple(
        [
            *strict_two_component_backbones,
            *strict_three_component_backbones,
        ]
    )

    candidate_modifier_components: tuple[object, ...] = ()
    candidate_modifier_replay = {
        "status": "STANDALONE_KNOWN_ONLY",
        "candidate_modifier_component_count": 0,
        "generic_bundle_selection_changed": False,
        "global_axis_threshold_changed": False,
    }

    if task_axis_report_path is not None:
        (
            candidate_modifier_components,
            candidate_modifier_replay,
        ) = _candidate_modifier_components_from_canonical_root(
            canonical_root=(
                task_axis_report_path.parent
            ),
            domain_profile=(
                source_context.domain_profile_id
            ),
        )

    modifier_components = tuple(
        [
            *components,
            *candidate_modifier_components,
        ]
    )

    screen = screen_confirmed_known_modifiers(
        backbones=backbones,
        components=modifier_components,
    )
    topologies = compose_higher_order_topologies(
        backbones=backbones,
        modifiers=screen.eligible,
    )
    carriers = build_topology_native_synthesis_carriers(
        topologies=topologies,
        requested_source=requested_source,
        requested_target=requested_target,
    )
    contexts = build_higher_order_synthesis_contexts(
        carriers=carriers
    )

    (
        generation_contexts,
        generation_context_selection,
    ) = _select_shadow_generation_contexts(
        contexts=contexts,
        max_contexts=args.max_contexts,
    )

    candidate_component_ids = {
        component.component_id
        for component in candidate_modifier_components
    }
    eligible_candidate_modifier_count = sum(
        1
        for row in screen.eligible
        if row.component.component_id
        in candidate_component_ids
    )

    _write(out / "modifier_eligibility_audit.json", screen.audit)
    _write(
        out / "candidate_modifier_replay_audit.json",
        candidate_modifier_replay,
    )
    _write(
        out / "generation_context_selection.json",
        generation_context_selection,
    )
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
        "strict_two_component_backbone_count": len(
            strict_two_component_backbones
        ),
        "strict_three_component_backbone_count": len(
            strict_three_component_backbones
        ),
        "candidate_modifier_component_count": len(
            candidate_modifier_components
        ),
        "candidate_modifier_replay": candidate_modifier_replay,
        "eligible_modifier_count": len(screen.eligible),
        "eligible_candidate_modifier_count": (
            eligible_candidate_modifier_count
        ),
        "higher_order_topology_count": len(topologies),
        "higher_order_context_count": len(contexts),
        "generation_context_selection": (
            generation_context_selection
        ),
        "generation_context_budget": len(
            generation_contexts
        ),
        "generation_candidate_modifier_context_count": (
            generation_context_selection.get(
                "selected_candidate_context_count",
                0,
            )
        ),
        "generation_unique_candidate_modifier_count": (
            generation_context_selection.get(
                "selected_unique_candidate_modifier_count",
                0,
            )
        ),
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

    if args.prepare_only:
        _write(
            report_path,
            {
                **base_report,
                "selected_context_count": 0,
                "candidate_count": 0,
                "proposed_count": 0,
                "arms": [],
                "generation_skipped": True,
                "status": "HIGHER_ORDER_SHADOW_PREPARED",
            },
        )
        print("Higher-order shadow preparation complete")
        print(
            "strict 2-component backbones:",
            len(strict_two_component_backbones),
        )
        print(
            "strict 3-component backbones:",
            len(strict_three_component_backbones),
        )
        print(
            "candidate modifier components:",
            len(candidate_modifier_components),
        )
        print(
            "eligible candidate modifiers:",
            eligible_candidate_modifier_count,
        )
        print("topologies:", len(topologies))
        print("contexts:", len(contexts))
        print(
            "generation contexts:",
            len(generation_contexts),
        )
        print(
            "generation candidate modifier contexts:",
            generation_context_selection.get(
                "selected_candidate_context_count",
                0,
            ),
        )
        print(
            "generation unique candidate modifiers:",
            generation_context_selection.get(
                "selected_unique_candidate_modifier_count",
                0,
            ),
        )
        print(
            "generation original indices:",
            generation_context_selection.get(
                "selected_original_indices",
                [],
            ),
        )
        print("report:", report_path)
        print("PRODUCTION_SELECTION_CHANGED=False")
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
        contexts=generation_contexts,
        max_contexts=len(generation_contexts),
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
    print(
        "strict 2-component backbones:",
        len(strict_two_component_backbones),
    )
    print(
        "strict 3-component backbones:",
        len(strict_three_component_backbones),
    )
    print(
        "candidate modifier components:",
        len(candidate_modifier_components),
    )
    print(
        "eligible candidate modifiers:",
        eligible_candidate_modifier_count,
    )
    print("eligible modifiers:", len(screen.eligible))
    print("topologies:", len(topologies))
    print(
        "generation candidate modifier contexts:",
        generation_context_selection.get(
            "selected_candidate_context_count",
            0,
        ),
    )
    print(
        "generation unique candidate modifiers:",
        generation_context_selection.get(
            "selected_unique_candidate_modifier_count",
            0,
        ),
    )
    print("selected contexts:", outcome.record.selected_context_count)
    print("proposed:", outcome.record.proposed_count)
    print("report:", report_path)
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
