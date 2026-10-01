#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from pipeline_core.discovery.frontier_exploration_audit import (
    build_frontier_exploration_audit,
)
from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.higher_order_topology_composition import (
    compose_higher_order_topologies,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _runtime_topology_cap() -> int | None:
    parameter = inspect.signature(
        compose_higher_order_topologies
    ).parameters.get("max_topologies")
    if parameter is None:
        return None
    value = parameter.default
    if value is inspect.Parameter.empty:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Factorize one Frontier Idea Population into primitive, topology, "
            "interpretive, cross-source, and capacity layers. Diagnostic only; "
            "no LLM/retrieval/ranking/selection authority."
        )
    )
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--higher-order-contexts", type=Path, default=None)
    parser.add_argument(
        "--higher-order-generation-report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--higher-order-topology-cap",
        type=int,
        default=None,
        help=(
            "Explicit cap used when the topology artifact was generated. "
            "When omitted, the current runtime default is recorded as such."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    population = FrontierIdeaPopulation.model_validate_json(
        args.population.read_text(encoding="utf-8")
    )

    contexts = ()
    if args.higher_order_contexts is not None:
        payload = _load_json(args.higher_order_contexts)
        if not isinstance(payload, list):
            raise ValueError("higher-order contexts artifact must be a JSON list")
        contexts = tuple(payload)

    report = None
    if args.higher_order_generation_report is not None:
        payload = _load_json(args.higher_order_generation_report)
        if not isinstance(payload, dict):
            raise ValueError("higher-order generation report must be a JSON object")
        report = payload

    if args.higher_order_topology_cap is not None:
        cap = args.higher_order_topology_cap
        cap_source = "EXPLICIT_CLI"
    else:
        cap = _runtime_topology_cap()
        cap_source = (
            "CURRENT_RUNTIME_DEFAULT"
            if cap is not None
            else "UNRESOLVED"
        )

    audit = build_frontier_exploration_audit(
        population=population,
        higher_order_contexts=contexts,
        higher_order_generation_report=report,
        higher_order_topology_cap=cap,
        higher_order_topology_cap_source=cap_source,
    )
    _write_json(args.output, audit)

    print("Exploration Frontier audit complete")
    print("raw ideas:", audit.raw_total_idea_count)
    print(
        "primitive families:",
        audit.primitive_layer.unique_primitive_family_count,
    )
    print(
        "backbone families:",
        audit.topology_layer.unique_backbone_family_count,
    )
    print(
        "modifier families:",
        audit.topology_layer.unique_modifier_family_count,
    )
    print(
        "largest backbone share:",
        round(audit.topology_layer.largest_backbone_family_share, 4),
    )
    print(
        "open-world primitive reuse in topology:",
        audit.primitive_layer.open_world_axis_primitive_reused_in_topology_count,
    )
    print(
        "tensions / explanation pairs:",
        audit.interpretive_layer.tension_seed_count,
        "/",
        audit.interpretive_layer.competing_explanation_pair_count,
    )
    print(
        "verified derivation depth:",
        audit.interpretive_layer.max_verified_structural_derivation_depth,
    )
    print(
        "HO cap reached:",
        audit.capacity.higher_order_topology_cap_reached,
        "cap=",
        audit.capacity.higher_order_topology_cap,
        "source=",
        audit.capacity.higher_order_topology_cap_source,
    )
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
