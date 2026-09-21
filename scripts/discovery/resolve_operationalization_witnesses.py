#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx

from pipeline_core.discovery.higher_order_operationalization_resolver import (
    resolve_operationalization_witnesses,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirementSet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve S27j operationalization witness requirements against "
            "one canonical corpus graph without mutating the graph."
        )
    )
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--corpus-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.50)
    parser.add_argument(
        "--max-candidates-per-observable",
        type=int,
        default=20,
    )
    parser.add_argument("--max-pairs", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    requirement_path = Path(args.requirements).expanduser().resolve()
    graph_path = Path(args.corpus_graph).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not requirement_path.is_file():
        raise FileNotFoundError(requirement_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    requirements = OperationalizationWitnessRequirementSet.model_validate_json(
        requirement_path.read_text(encoding="utf-8")
    )
    graph = nx.read_graphml(graph_path, force_multigraph=True)

    result = resolve_operationalization_witnesses(
        graph=graph,
        requirements=requirements,
        minimum_coverage=args.minimum_coverage,
        max_candidates_per_observable=(
            args.max_candidates_per_observable
        ),
        max_pairs=args.max_pairs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("S27K_OPERATIONALIZATION_RESOLUTION=PASS")
    print("requirements:", requirement_path)
    print("requirements_sha256:", _sha256(requirement_path))
    print("corpus_graph:", graph_path)
    print("corpus_graph_sha256:", _sha256(graph_path))
    print("graph_nodes:", result.graph_node_count)
    print("graph_edges:", result.graph_edge_count)
    print("requirement_count:", result.requirement_count)
    print("resolution_count:", result.resolution_count)
    print("status_counts:", result.status_counts)
    print(
        "measurement_candidate_count:",
        result.measurement_candidate_count,
    )
    print("pair_candidate_count:", result.pair_candidate_count)
    print(
        "distinct_operationalization_resolution_count:",
        result.distinct_operationalization_resolution_count,
    )
    print(
        "measurement_independence_verified_count:",
        result.measurement_independence_verified_count,
    )
    print(
        "independence_certification_authority:",
        result.independence_certification_authority,
    )
    print("output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
