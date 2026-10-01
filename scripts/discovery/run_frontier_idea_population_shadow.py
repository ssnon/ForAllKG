#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
)
from pipeline_core.discovery.frontier_idea_population import (
    build_frontier_idea_population,
)
from pipeline_core.discovery.higher_order_competing_explanations import (
    CompetingExplanationSet,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidateSet,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
)
from pipeline_core.discovery.open_world_discovery_axis import (
    OpenWorldExternalAxisBundle,
)


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(
    path: Path,
    value: object,
) -> None:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_higher_order_topologies(
    path: Path | None,
) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()

    payload = _load_json(
        path
    )

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "higher-order topology artifact "
            "must be a JSON list"
        )

    rows = []
    for index, row in enumerate(
        payload,
        start=1,
    ):
        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "higher-order topology row "
                f"{index} is not an object"
            )
        rows.append(row)

    return tuple(rows)


def _load_direct_higher_order_topologies(
    path: Path | None,
) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()

    payload = _load_json(
        path
    )

    if (
        not isinstance(
            payload,
            dict,
        )
        or payload.get(
            "schema_version"
        )
        != "direct-higher-order-topology-shadow-report-v1"
    ):
        raise ValueError(
            "unexpected direct-HO topology "
            "artifact schema"
        )

    rows = payload.get(
        "topologies",
        [],
    )

    if not isinstance(
        rows,
        list,
    ):
        raise ValueError(
            "direct-HO topology artifact "
            "topologies must be a list"
        )

    if int(
        payload.get(
            "topology_count",
            len(rows),
        )
    ) != len(rows):
        raise ValueError(
            "direct-HO topology count mismatch"
        )

    return tuple(
        dict(row)
        for row in rows
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Frontier Idea Population v1 as a "
            "shadow-only union of existing exploration "
            "artifacts. This stage performs no LLM calls, "
            "retrieval, scientific ranking, generation, "
            "or production selection."
        )
    )
    parser.add_argument(
        "--source-context",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--question",
        required=True,
    )
    parser.add_argument(
        "--task-source",
        required=True,
    )
    parser.add_argument(
        "--task-target",
        required=True,
    )
    parser.add_argument(
        "--kg-axis-plan",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--open-world-axis-plan",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--open-world-axis-bundle",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--higher-order-topologies",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--direct-higher-order-topologies",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--tension-candidates",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--competing-explanations",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    return build_parser().parse_args(
        argv
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(
        argv
    )

    source_context = (
        HypothesisContext.model_validate_json(
            args.source_context.read_text(
                encoding="utf-8"
            )
        )
    )

    if (
        source_context.question
        != args.question
    ):
        raise ValueError(
            "frontier population question "
            "does not match source context"
        )

    kg_plan = (
        DiscoveryAxisPlan.model_validate_json(
            args.kg_axis_plan.read_text(
                encoding="utf-8"
            )
        )
    )

    open_world_plan = None
    open_world_bundle = None

    if bool(
        args.open_world_axis_plan
    ) != bool(
        args.open_world_axis_bundle
    ):
        raise ValueError(
            "open-world plan and bundle "
            "must be supplied together"
        )

    if args.open_world_axis_plan:
        open_world_plan = (
            DiscoveryAxisPlan.model_validate_json(
                args.open_world_axis_plan.read_text(
                    encoding="utf-8"
                )
            )
        )
        open_world_bundle = (
            OpenWorldExternalAxisBundle.model_validate_json(
                args.open_world_axis_bundle.read_text(
                    encoding="utf-8"
                )
            )
        )

    higher_order_topologies = (
        _load_higher_order_topologies(
            args.higher_order_topologies
        )
    )

    direct_topologies = (
        _load_direct_higher_order_topologies(
            args.direct_higher_order_topologies
        )
    )

    tensions = (
        ScientificTensionCandidateSet
        .model_validate_json(
            args.tension_candidates.read_text(
                encoding="utf-8"
            )
        )
        if args.tension_candidates
        else None
    )

    explanations = (
        CompetingExplanationSet.model_validate_json(
            args.competing_explanations.read_text(
                encoding="utf-8"
            )
        )
        if args.competing_explanations
        else None
    )

    population = (
        build_frontier_idea_population(
            source_context_id=(
                source_context.context_id
            ),
            source_context_sha256=(
                source_context.context_sha256
            ),
            research_question=(
                args.question
            ),
            task_source=(
                args.task_source
            ),
            task_target=(
                args.task_target
            ),
            kg_axis_plan=kg_plan,
            kg_axis_artifact=str(
                args.kg_axis_plan.resolve()
            ),
            kg_axis_artifact_sha256=(
                _sha256_file(
                    args.kg_axis_plan
                )
            ),
            open_world_axis_plan=(
                open_world_plan
            ),
            open_world_axis_artifact=(
                str(
                    args
                    .open_world_axis_plan
                    .resolve()
                )
                if args.open_world_axis_plan
                else None
            ),
            open_world_axis_artifact_sha256=(
                _sha256_file(
                    args.open_world_axis_plan
                )
                if args.open_world_axis_plan
                else None
            ),
            open_world_axis_bundle=(
                open_world_bundle
            ),
            higher_order_topologies=(
                higher_order_topologies
            ),
            higher_order_topology_artifact=(
                str(
                    args
                    .higher_order_topologies
                    .resolve()
                )
                if args.higher_order_topologies
                else None
            ),
            higher_order_topology_artifact_sha256=(
                _sha256_file(
                    args.higher_order_topologies
                )
                if args.higher_order_topologies
                else None
            ),
            direct_higher_order_topologies=(
                direct_topologies
            ),
            direct_higher_order_topology_artifact=(
                str(
                    args
                    .direct_higher_order_topologies
                    .resolve()
                )
                if args.direct_higher_order_topologies
                else None
            ),
            direct_higher_order_topology_artifact_sha256=(
                _sha256_file(
                    args.direct_higher_order_topologies
                )
                if args.direct_higher_order_topologies
                else None
            ),
            tension_candidates=tensions,
            tension_artifact=(
                str(
                    args
                    .tension_candidates
                    .resolve()
                )
                if args.tension_candidates
                else None
            ),
            tension_artifact_sha256=(
                _sha256_file(
                    args.tension_candidates
                )
                if args.tension_candidates
                else None
            ),
            competing_explanations=(
                explanations
            ),
            competing_explanations_artifact=(
                str(
                    args
                    .competing_explanations
                    .resolve()
                )
                if args.competing_explanations
                else None
            ),
            competing_explanations_artifact_sha256=(
                _sha256_file(
                    args.competing_explanations
                )
                if args.competing_explanations
                else None
            ),
        )
    )

    _write_json(
        args.output,
        population,
    )

    print(
        "Frontier Idea Population v1 complete"
    )
    print(
        "total ideas:",
        population.total_idea_count,
    )
    print(
        "by source:",
        population.idea_count_by_source_kind,
    )
    print(
        "by form:",
        population.idea_count_by_idea_form,
    )
    print(
        "cross-source exact duplicate groups:",
        population
        .cross_source_exact_duplicate_group_count,
    )
    print(
        "cross-source overlap diagnostics:",
        population
        .cross_source_overlap_pair_count,
    )
    print(
        "cross-source structural-overlap diagnostics:",
        population
        .cross_source_structural_overlap_pair_count,
    )
    print(
        "STAGE8_INPUT_CHANGED=False"
    )
    print(
        "PRODUCTION_SELECTION_AUTHORITY=False"
    )
    print(
        "artifact:",
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
