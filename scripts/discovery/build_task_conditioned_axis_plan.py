from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.discovery_axis_planner import (
    DiscoveryAxisPlanner,
    _sha256_json,
    _stable_id,
)
from pipeline_core.discovery.discovery_bundle import (
    DiscoveryBundleBuilder,
)
from pipeline_core.discovery.discovery_contracts import (
    DiscoveryBundle,
    DiscoveryInspiration,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    candidate_relation_from_mapping,
    compose_task_bridge_candidates,
)
from pipeline_core.discovery.task_bridge_composite_axis import (
    materialize_task_bridge_composite_axis,
)

import scripts.discovery.build_discovery_bundle as bundle_cli


GRAMMAR = re.compile(
    r"^How does (.+?) relate to (.+?)\?$",
    flags=re.DOTALL,
)

MAX_TASK_AXES = 2
MAX_TOTAL_AXES = 3

MIN_EXPLORATION_SCORE = 0.05
MAX_GROUNDING_SEMANTIC_OVERLAP = 0.95
MAX_CONTEXT_SWITCH_PENALTY = 0.50


def _json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write_model(
    path: Path,
    model: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        model.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def _lane_id(
    *,
    question: str,
    unit_id: str,
    path_id: str,
) -> str:
    payload = (
        question
        + "|"
        + unit_id
        + "|"
        + path_id
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]
    return (
        "task_lane_source:"
        + digest
    )


def _parse_question(
    question: str,
) -> tuple[str, str] | None:
    match = GRAMMAR.fullmatch(
        question.strip()
    )
    if match is None:
        return None

    source = match.group(1).strip()
    target = match.group(2).strip()

    if not source or not target:
        return None

    return source, target


def _candidate_mapping(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    value = row.get(
        "candidate_unit"
    )
    if isinstance(value, dict):
        return value

    value = row.get(
        "candidate_unit_selection"
    )
    if isinstance(value, dict):
        candidate = value.get(
            "candidate_unit"
        )
        if isinstance(candidate, dict):
            return candidate

    return None


def _candidate_unit_id(
    row: dict[str, Any],
) -> str:
    unit = _candidate_mapping(row)

    if unit is not None:
        return str(
            unit.get("unit_id", "")
            or
            unit.get("candidate_unit_id", "")
            or
            ""
        )

    return ""


def _sanitize_lane_inspiration(
    *,
    inspiration: DiscoveryInspiration,
    inspiration_id: str,
) -> DiscoveryInspiration:
    payload = inspiration.model_dump(
        mode="json"
    )

    payload["inspiration_id"] = (
        inspiration_id
    )

    payload["source_mode"] = (
        "task_composite_source_lane_full_pool"
    )

    payload[
        "max_semantic_similarity_to_selected"
    ] = 0.0

    payload[
        "semantic_diversity_mode"
    ] = "disabled"

    reasons = [
        str(value)
        for value in payload.get(
            "reason_codes",
            [],
        )
        if (
            not str(value).startswith(
                "bundle_rank:"
            )
            and
            str(value)
            !=
            "semantic_diversity_relaxed"
        )
    ]

    reasons.extend(
        [
            "task_composite_source_lane",
            "full_pool_representative",
            "not_generic_bundle_selected",
        ]
    )

    payload["reason_codes"] = sorted(
        set(reasons)
    )

    return DiscoveryInspiration.model_validate(
        payload
    )


def _quality_eligible(
    inspiration: DiscoveryInspiration,
    *,
    min_candidate_unit_score: float,
) -> bool:
    return bool(
        inspiration.path_type
        == "CANDIDATE_EXPLORATION"
        and
        inspiration.candidate_unit_id
        and
        float(
            inspiration.candidate_unit_score
        )
        >= float(
            min_candidate_unit_score
        )
        and
        float(
            inspiration.exploration_score
        )
        >= MIN_EXPLORATION_SCORE
        and
        float(
            inspiration.semantic_similarity_to_grounding
        )
        <= MAX_GROUNDING_SEMANTIC_OVERLAP
        and
        float(
            inspiration.context_switch_penalty
        )
        <= MAX_CONTEXT_SWITCH_PENALTY
    )


def _replay_and_capture(
    *,
    final_traversal: Path,
    candidate_traversal: Path,
    domain_profile: str,
    discovery_top_k: int,
    min_candidate_unit_score: float,
    expected_bundle: DiscoveryBundle,
    scratch_bundle: Path,
) -> tuple[
    DiscoveryBundleBuilder,
    DiscoveryBundle,
    list[dict[str, Any]],
]:
    captured: list[
        dict[str, Any]
    ] = []

    builders: list[
        DiscoveryBundleBuilder
    ] = []

    original = (
        DiscoveryBundleBuilder
        ._enrich_candidate_path
    )

    def wrapped(
        self: DiscoveryBundleBuilder,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enriched = original(
            self,
            **kwargs,
        )

        if not builders:
            builders.append(self)

        captured.append(
            enriched
        )

        return enriched

    old_argv = list(sys.argv)

    DiscoveryBundleBuilder._enrich_candidate_path = (
        wrapped
    )

    try:
        sys.argv = [
            "build_discovery_bundle",
            "--traversal",
            str(final_traversal),
            "--traversal",
            str(candidate_traversal),
            "--domain-profile",
            domain_profile,
            "--top-k",
            str(discovery_top_k),
            "--min-reserved-candidate-unit-score",
            str(min_candidate_unit_score),
            "--output",
            str(scratch_bundle),
        ]

        rc = bundle_cli.main()

    finally:
        sys.argv = old_argv
        (
            DiscoveryBundleBuilder
            ._enrich_candidate_path
        ) = original

    if rc not in (
        0,
        None,
    ):
        raise RuntimeError(
            "DiscoveryBundle replay failed: "
            f"rc={rc}"
        )

    if not builders:
        raise RuntimeError(
            "DiscoveryBundle replay captured no builder"
        )

    replay = DiscoveryBundle.model_validate_json(
        scratch_bundle.read_text(
            encoding="utf-8"
        )
    )

    if (
        replay.bundle_sha256
        !=
        expected_bundle.bundle_sha256
    ):
        raise RuntimeError(
            "A17F replay changed the generic "
            "DiscoveryBundle SHA: "
            f"{replay.bundle_sha256} != "
            f"{expected_bundle.bundle_sha256}"
        )

    return (
        builders[0],
        replay,
        captured,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen A17F task-conditioned "
            "2-task + 1-generic discovery-axis plan "
            "without changing DiscoveryBundle selection."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
    )
    parser.add_argument(
        "--final-traversal",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--candidate-traversal",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--discovery-bundle",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--discovery-top-k",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--min-candidate-unit-score",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--max-axes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--output-dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-axis-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    bundle = (
        DiscoveryBundle
        .model_validate_json(
            args.discovery_bundle.read_text(
                encoding="utf-8"
            )
        )
    )

    old_dual = (
        DualHypothesisContext
        .model_validate_json(
            args.dual_context.read_text(
                encoding="utf-8"
            )
        )
    )

    planner = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(
            max_axes=args.max_axes,
        )
    )

    generic_plan = planner.build(
        old_dual
    )

    parsed = _parse_question(
        args.question
    )

    # --------------------------------------------------------------
    # Exact grammar does not apply:
    # preserve old production behavior exactly.
    # --------------------------------------------------------------
    if parsed is None:
        _write_model(
            args.output_dual_context,
            old_dual,
        )
        _write_model(
            args.output_axis_plan,
            generic_plan,
        )

        args.output_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version":
                        "task-conditioned-axis-plan-report-v1",
                    "status":
                        "GRAMMAR_NOT_APPLICABLE",
                    "grammar":
                        "HOW_DOES_RELATE_TO_GRAMMAR_V1",
                    "generic_bundle_changed":
                        False,
                    "task_axis_count":
                        0,
                    "generic_axis_count":
                        len(generic_plan.axes),
                    "architecture_tuning":
                        False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "A17F_STATUS=GRAMMAR_NOT_APPLICABLE"
        )
        print(
            "TASK_AXIS_COUNT=0"
        )
        print(
            "GENERIC_BUNDLE_CHANGED=False"
        )
        return 0

    requested_source, requested_target = (
        parsed
    )

    scratch_bundle = (
        args.output_report.parent
        / ".a17f.generic_bundle.replay.json"
    )

    (
        builder,
        replay_bundle,
        enriched_rows,
    ) = _replay_and_capture(
        final_traversal=(
            args.final_traversal
        ),
        candidate_traversal=(
            args.candidate_traversal
        ),
        domain_profile=(
            args.domain_profile
        ),
        discovery_top_k=(
            args.discovery_top_k
        ),
        min_candidate_unit_score=(
            args.min_candidate_unit_score
        ),
        expected_bundle=bundle,
        scratch_bundle=scratch_bundle,
    )

    # --------------------------------------------------------------
    # Materialize every captured candidate path with neutral
    # non-selection semantics, then apply the frozen quality gates.
    # --------------------------------------------------------------
    rows_by_unit: dict[
        str,
        list[
            tuple[
                DiscoveryInspiration,
                dict[str, Any],
            ]
        ],
    ] = {}

    relation_by_unit: dict[
        str,
        CandidateRelationView,
    ] = {}

    for enriched in enriched_rows:
        unit = _candidate_mapping(
            enriched
        )

        if unit is None:
            continue

        unit_id = _candidate_unit_id(
            enriched
        )

        if not unit_id:
            continue

        relation = (
            candidate_relation_from_mapping(
                unit
            )
        )

        relation_by_unit.setdefault(
            unit_id,
            relation,
        )

        materialized = (
            builder._materialize_inspiration(
                corpus_id=(
                    bundle.corpus_id
                ),
                rank=1,
                row=enriched,
                semantic_mode=(
                    replay_bundle
                    .semantic_diversity_mode
                ),
            )
        )

        rows_by_unit.setdefault(
            unit_id,
            [],
        ).append(
            (
                materialized,
                enriched,
            )
        )

    representatives: dict[
        str,
        tuple[
            DiscoveryInspiration,
            dict[str, Any],
        ],
    ] = {}

    for unit_id, rows in (
        rows_by_unit.items()
    ):
        eligible = [
            pair
            for pair in rows
            if _quality_eligible(
                pair[0],
                min_candidate_unit_score=(
                    args
                    .min_candidate_unit_score
                ),
            )
        ]

        if not eligible:
            continue

        eligible.sort(
            key=lambda pair: (
                -float(
                    pair[0]
                    .exploration_score
                ),
                str(
                    pair[0]
                    .source_path_id
                ),
            )
        )

        representatives[
            unit_id
        ] = eligible[0]

    all_relations = [
        relation_by_unit[unit_id]
        for unit_id in sorted(
            relation_by_unit
        )
    ]

    composites = (
        compose_task_bridge_candidates(
            candidates=all_relations,
            requested_source=(
                requested_source
            ),
            requested_target=(
                requested_target
            ),
            max_composites=12,
        )
    )

    # Frozen A17F7D choice:
    # earliest GLOBAL A10 rank per quality-eligible source.
    selected_composites = []

    seen_sources: set[str] = set()

    final_cap = min(
        max(
            int(args.max_axes),
            1,
        ),
        MAX_TOTAL_AXES,
    )

    task_cap = min(
        MAX_TASK_AXES,
        max(
            final_cap - 1,
            0,
        ),
    )

    for composite in composites:
        source_id = (
            composite.source_unit_id
        )

        if source_id in seen_sources:
            continue

        if (
            source_id
            not in representatives
        ):
            continue

        # Require actual lexical task-source support.
        if not composite.source_overlap_tokens:
            continue

        selected_composites.append(
            composite
        )
        seen_sources.add(
            source_id
        )

        if (
            len(selected_composites)
            >=
            task_cap
        ):
            break

    # --------------------------------------------------------------
    # If no task source survives, preserve generic behavior.
    # --------------------------------------------------------------
    if not selected_composites:
        _write_model(
            args.output_dual_context,
            old_dual,
        )
        _write_model(
            args.output_axis_plan,
            generic_plan,
        )

        args.output_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version":
                        "task-conditioned-axis-plan-report-v1",
                    "status":
                        "NO_TASK_COMPOSITE",
                    "grammar":
                        "HOW_DOES_RELATE_TO_GRAMMAR_V1",
                    "generic_bundle_changed":
                        False,
                    "quality_eligible_source_count":
                        len(representatives),
                    "global_composite_count":
                        len(composites),
                    "task_axis_count":
                        0,
                    "generic_axis_count":
                        len(generic_plan.axes),
                    "architecture_tuning":
                        False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "A17F_STATUS=NO_TASK_COMPOSITE"
        )
        print(
            "TASK_AXIS_COUNT=0"
        )
        print(
            "GENERIC_BUNDLE_CHANGED=False"
        )
        return 0

    # --------------------------------------------------------------
    # Build task-lane inspirations.
    # --------------------------------------------------------------
    lane_inspirations = []

    for composite in (
        selected_composites
    ):
        source_id = (
            composite.source_unit_id
        )

        inspiration, _ = (
            representatives[
                source_id
            ]
        )

        lane = (
            _sanitize_lane_inspiration(
                inspiration=inspiration,
                inspiration_id=_lane_id(
                    question=args.question,
                    unit_id=source_id,
                    path_id=(
                        inspiration
                        .source_path_id
                    ),
                ),
            )
        )

        if (
            lane
            .eligible_as_positive_premise
            is not False
        ):
            raise RuntimeError(
                "Task-lane inspiration became "
                "positive-premise eligible"
            )

        lane_inspirations.append(
            lane
        )

    new_dual = (
        DualHypothesisContext.build(
            old_dual.grounded_context,
            old_dual.discovery_bundle,
            task_lane_inspirations=(
                lane_inspirations
            ),
        )
    )

    if (
        new_dual
        .discovery_bundle
        .bundle_sha256
        !=
        old_dual
        .discovery_bundle
        .bundle_sha256
    ):
        raise RuntimeError(
            "Generic DiscoveryBundle changed "
            "while adding task lane"
        )

    # --------------------------------------------------------------
    # Materialize task composite axes.
    # --------------------------------------------------------------
    final_axes = []

    for rank, (
        composite,
        lane,
    ) in enumerate(
        zip(
            selected_composites,
            lane_inspirations,
        ),
        start=1,
    ):
        planner_score = (
            planner._score(
                lane
            )
        )

        source_axis = (
            planner._materialize_axis(
                dual_context_sha256=(
                    new_dual
                    .dual_context_sha256
                ),
                item=lane,
                planner_score=(
                    planner_score
                ),
                axis_rank=rank,
            )
        )

        composite_axis = (
            materialize_task_bridge_composite_axis(
                composite=composite,
                source_axis=source_axis,
                requested_source=(
                    requested_source
                ),
                requested_target=(
                    requested_target
                ),
                axis_rank=rank,
            )
        )

        final_axes.append(
            composite_axis
        )

    # --------------------------------------------------------------
    # Frozen generic reserve:
    # preserve original generic-plan order.
    # Rematerialize against the NEW dual SHA.
    # --------------------------------------------------------------
    inspiration_by_id = {
        item.inspiration_id:
        item
        for item in (
            old_dual
            .discovery_bundle
            .inspirations
        )
    }

    for old_axis in generic_plan.axes:
        if len(final_axes) >= final_cap:
            break

        inspiration = (
            inspiration_by_id.get(
                old_axis.inspiration_id
            )
        )

        if inspiration is None:
            raise RuntimeError(
                "Generic axis inspiration missing "
                "from unchanged DiscoveryBundle: "
                + old_axis.inspiration_id
            )

        rank = (
            len(final_axes)
            + 1
        )

        generic_axis = (
            planner._materialize_axis(
                dual_context_sha256=(
                    new_dual
                    .dual_context_sha256
                ),
                item=inspiration,
                planner_score=(
                    planner._score(
                        inspiration
                    )
                ),
                axis_rank=rank,
            )
        )

        final_axes.append(
            generic_axis
        )

    if not final_axes:
        raise RuntimeError(
            "A17F produced zero final axes"
        )

    if (
        len(final_axes)
        > final_cap
    ):
        raise RuntimeError(
            "A17F exceeded frozen axis cap"
        )

    if (
        final_cap >= 2
        and
        not any(
            axis.source_mode
            !=
            "task_conditioned_composite_bridge_projection"
            for axis in final_axes
        )
    ):
        raise RuntimeError(
            "A17F violated generic-reserve invariant"
        )

    plan_id = _stable_id(
        "discovery_axis_plan",
        new_dual.dual_context_sha256,
        bundle.bundle_sha256,
        *[
            axis.axis_id
            for axis in final_axes
        ],
    )

    payload = {
        "schema_version":
            generic_plan.schema_version,
        "plan_id":
            plan_id,
        "source_dual_context_id":
            new_dual.dual_context_id,
        "source_dual_context_sha256":
            new_dual.dual_context_sha256,
        "source_bundle_id":
            bundle.bundle_id,
        "source_bundle_sha256":
            bundle.bundle_sha256,
        "corpus_id":
            bundle.corpus_id,
        "axes":
            [
                axis.model_dump(
                    mode="json"
                )
                for axis in final_axes
            ],
        "excluded_inspiration_ids":
            generic_plan
            .excluded_inspiration_ids,
        "policy":
            generic_plan
            .policy
            .model_dump(
                mode="json"
            ),
    }

    final_plan = DiscoveryAxisPlan(
        **payload,
        plan_sha256=_sha256_json(
            payload
        ),
    )

    _write_model(
        args.output_dual_context,
        new_dual,
    )

    _write_model(
        args.output_axis_plan,
        final_plan,
    )

    report = {
        "schema_version":
            "task-conditioned-axis-plan-report-v1",
        "status":
            "TASK_CONDITIONED",
        "grammar":
            "HOW_DOES_RELATE_TO_GRAMMAR_V1",
        "requested_source":
            requested_source,
        "requested_target":
            requested_target,
        "generic_bundle_changed":
            False,
        "generic_bundle_sha256":
            bundle.bundle_sha256,
        "replay_bundle_sha256":
            replay_bundle.bundle_sha256,
        "quality_eligible_source_count":
            len(representatives),
        "global_composite_count":
            len(composites),
        "selected_source_unit_ids":
            [
                row.source_unit_id
                for row in (
                    selected_composites
                )
            ],
        "selected_composite_ids":
            [
                row.composite_id
                for row in (
                    selected_composites
                )
            ],
        "task_axis_count":
            sum(
                axis.source_mode
                ==
                "task_conditioned_composite_bridge_projection"
                for axis in final_axes
            ),
        "generic_axis_count":
            sum(
                axis.source_mode
                !=
                "task_conditioned_composite_bridge_projection"
                for axis in final_axes
            ),
        "axis_modes":
            [
                axis.source_mode
                for axis in final_axes
            ],
        "axis_ids":
            [
                axis.axis_id
                for axis in final_axes
            ],
        "dual_context_id":
            new_dual.dual_context_id,
        "dual_context_sha256":
            new_dual.dual_context_sha256,
        "axis_plan_id":
            final_plan.plan_id,
        "axis_plan_sha256":
            final_plan.plan_sha256,
        "architecture_tuning":
            False,
    }

    args.output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "A17F_STATUS=TASK_CONDITIONED"
    )
    print(
        "TASK_AXIS_COUNT=",
        report["task_axis_count"],
    )
    print(
        "GENERIC_AXIS_COUNT=",
        report["generic_axis_count"],
    )
    print(
        "AXIS_MODES=",
        report["axis_modes"],
    )
    print(
        "SELECTED_SOURCE_UNIT_IDS=",
        report[
            "selected_source_unit_ids"
        ],
    )
    print(
        "GENERIC_BUNDLE_CHANGED=False"
    )
    print(
        "DUAL_CONTEXT_SHA256=",
        new_dual.dual_context_sha256,
    )
    print(
        "AXIS_PLAN_SHA256=",
        final_plan.plan_sha256,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
