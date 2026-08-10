from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from dac_her.candidate_unit_selection import (
    CandidateUnitSelectionPolicy,
    CandidateUnitSelector,
    endpoint_pair_payload,
)
from dac_her.domains import get_domain_profile
from dac_her.domains.extraction_registry import (
    get_extraction_adapter,
)
from dac_her.candidate_units import (
    CandidateUnitBuilder,
    candidate_unit_inventory,
    confirmed_navigation_graph,
    edge_is_alignment,
    edge_is_candidate,
    edge_is_reverse,
    node_label,
    paper_ids_from_node,
)
from dac_her.node_mapping import NodeMapper, QueryConcept, load_node_embedding_index
from dac_her.path_quality import PathQualityScorer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _types(raw: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (raw or "").split(",") if item.strip())


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Candidate-unit-aware discovery traversal. Maps grounded endpoints, "
            "recovers multi-anchor semantic candidate units, and ranks "
            "source -> exactly one candidate unit -> target routes."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", default="dac_her")
    parser.add_argument("--data-root", default=None)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", default=None)
    source.add_argument("--source-node-id", default=None)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", default=None)
    target.add_argument("--target-node-id", default=None)

    parser.add_argument("--source-types", default=None)
    parser.add_argument("--target-types", default=None)
    parser.add_argument("--node-map-k", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-routes-per-unit", type=int, default=1)
    parser.add_argument("--max-unit-semantic-similarity", type=float, default=0.90)
    parser.add_argument("--min-unit-relevance", type=float, default=0.0)
    parser.add_argument("--min-selection-score", type=float, default=0.0)
    parser.add_argument("--navigation-graphml", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-candidate-paths",
        action="store_true",
        help="Persist the full candidate-unit route pool for DiscoveryBundle ranking.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _map(
    mapper: NodeMapper,
    text: str,
    *,
    allowed_types: tuple[str, ...],
    top_k: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    return [
        row.to_dict()
        for row in mapper.map(
            QueryConcept(
                text=text,
                allowed_node_types=allowed_types,
                allow_candidates=False,
                allow_alignment_hubs=False,
                top_k=top_k,
                min_similarity=min_similarity,
            )
        )
    ]


def _index_vectors(index: Any) -> dict[str, np.ndarray]:
    records = list(index.records)
    embeddings = np.asarray(index.embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError("Invalid node embedding index")
    return {
        str(record.get("node_id", "")): embeddings[i]
        for i, record in enumerate(records)
        if str(record.get("node_id", "")).strip()
    }


def _step(graph: nx.DiGraph, source: str, target: str, *, unit: dict[str, Any]) -> dict[str, Any]:
    attrs = dict(graph.edges[source, target])
    candidate_id = str(unit["candidate_node_id"])
    entry_id = str(unit["entry_anchor_id"])
    exit_id = str(unit["exit_anchor_id"])
    if source == entry_id and target == candidate_id:
        candidate_role = "candidate_unit_entry"
    elif source == candidate_id and target == exit_id:
        candidate_role = "candidate_unit_exit"
    else:
        candidate_role = ""
    return {
        "source": source,
        "target": target,
        "navigation_edge_id": str(attrs.get("edge_id", "")),
        "relation": str(attrs.get("relation", "RELATED_TO")),
        "edge_class": str(attrs.get("edge_class", "")),
        "exploration_cost": float(attrs.get("exploration_cost", 1.0)),
        "requires_verification": bool(edge_is_candidate(attrs)),
        "traversal_direction": str(attrs.get("traversal_direction", "forward")),
        "scientific_direction": str(attrs.get("scientific_direction", "")),
        "selected_original_edge_id": str(attrs.get("selected_original_edge_id", "")),
        "source_paper_ids": [
            str(item)
            for item in _json_list(attrs.get("source_paper_ids_json", "[]"))
            if str(item).strip()
        ],
        "alternative_count": int(attrs.get("alternative_count", 1)),
        "alternatives": [
            item
            for item in _json_list(attrs.get("edge_alternatives_json", "[]"))
            if isinstance(item, dict)
        ],
        "candidate_unit_role": candidate_role,
        "candidate_unit_id": str(unit["unit_id"]) if candidate_role else "",
    }


def _materialize_route(graph: nx.DiGraph, route: Any, quality_scorer: PathQualityScorer) -> dict[str, Any]:
    base = route.to_dict()
    unit = dict(base["candidate_unit"])
    nodes = [str(node) for node in base["nodes"]]
    steps = [
        _step(graph, left, right, unit=unit)
        for left, right in zip(nodes, nodes[1:], strict=False)
    ]
    alignment_count = sum(edge_is_alignment(dict(graph.edges[left, right])) for left, right in zip(nodes, nodes[1:], strict=False))
    candidate_count = sum(edge_is_candidate(dict(graph.edges[left, right])) for left, right in zip(nodes, nodes[1:], strict=False))
    reverse_count = sum(edge_is_reverse(dict(graph.edges[left, right])) for left, right in zip(nodes, nodes[1:], strict=False))

    visited_papers: set[str] = set(base.get("visited_paper_ids", []))
    supporting_papers: set[str] = set()
    hub_scope_papers: set[str] = set()
    for node in nodes:
        visited_papers.update(paper_ids_from_node(graph, node))
    for step in steps:
        papers = {str(item) for item in step.get("source_paper_ids", []) if str(item).strip()}
        if str(step.get("edge_class", "")) in {"registry_alignment", "pattern_alignment"}:
            hub_scope_papers.update(papers)
        else:
            supporting_papers.update(papers)
    supporting_papers.update(visited_papers)

    endpoint_pair = endpoint_pair_payload(base["source_match"], base["target_match"])
    path_id = _stable_id(
        "path",
        "candidate_unit_top_n",
        base["route_id"],
        *nodes,
        *[step["navigation_edge_id"] for step in steps],
    )
    row: dict[str, Any] = {
        "path_id": path_id,
        "algorithm": "candidate_unit_top_n",
        "mode": "exploratory",
        "source": nodes[0],
        "target": nodes[-1],
        "semantic_stop": None,
        "nodes": nodes,
        "steps": steps,
        "total_cost": float(base["total_cost"]),
        "hop_count": len(steps),
        "scientific_edge_count": max(0, len(steps) - alignment_count),
        "alignment_edge_count": alignment_count,
        "candidate_edge_count": candidate_count,
        "candidate_unit_count": 1,
        "reverse_edge_count": reverse_count,
        "source_paper_ids": sorted(visited_papers),
        "cross_paper_count": len(visited_papers),
        "visited_paper_ids": sorted(visited_papers),
        "visited_paper_count": len(visited_papers),
        "supporting_paper_ids": sorted(supporting_papers),
        "hub_scope_paper_ids": sorted(hub_scope_papers),
        "requires_verification": True,
        "source_match": base["source_match"],
        "target_match": base["target_match"],
        "stop_match": None,
        "waypoint": None,
        "endpoint_pair": endpoint_pair,
        "candidate_unit": unit,
        "candidate_unit_selection": base["candidate_unit_selection"],
        "candidate_unit_core_node_ids": [
            unit["entry_anchor_id"],
            unit["candidate_node_id"],
            unit["exit_anchor_id"],
        ],
        "candidate_unit_semantic_text": " | ".join(
            part
            for part in [
                str(unit.get("label", "")),
                str(unit.get("proposed_subject", "")),
                str(unit.get("proposed_relation", "")),
                str(unit.get("proposed_object", "")),
                str(unit.get("entry_anchor_label", "")),
                str(unit.get("exit_anchor_label", "")),
            ]
            if part.strip()
        ),
        "context_node_labels": list(
            base.get("context_node_labels", base.get("reaction_node_labels", []))
        ),
        "reaction_node_labels": list(
            base.get("context_node_labels", base.get("reaction_node_labels", []))
        ),
    }
    row["path_quality"] = quality_scorer.score(row).to_dict()
    return row


def main() -> None:
    args = parse_args()
    domain_profile = get_domain_profile(args.domain_profile)
    extraction_adapter = get_extraction_adapter(
        domain_profile.profile_id
    )
    data_root = Path(
        args.data_root
        or extraction_adapter.default_data_root
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    if args.max_depth < 2:
        raise ValueError("--max-depth must be >= 2 for a candidate unit traversal")
    if not 0.0 <= args.max_unit_semantic_similarity <= 1.0:
        raise ValueError("--max-unit-semantic-similarity must be between 0 and 1")

    mode_root = data_root / "corpus" / args.corpus_id / "exploratory"
    navigation_root = mode_root / "navigation"
    graph_path = Path(args.navigation_graphml) if args.navigation_graphml else navigation_root / "graph.graphml"
    index_dir = Path(args.index_dir) if args.index_dir else navigation_root / "node_index"

    graph = nx.read_graphml(graph_path)
    index = load_node_embedding_index(index_dir)
    mapper = NodeMapper(index, device=args.device)

    source_matches = (
        [{"node_id": args.source_node_id, "direct_node_id": True, "exact_label_match": True, "semantic_similarity": 1.0}]
        if args.source_node_id
        else _map(
            mapper,
            args.source,
            allowed_types=_types(args.source_types),
            top_k=args.node_map_k,
            min_similarity=args.min_similarity,
        )
    )
    target_matches = (
        [{"node_id": args.target_node_id, "direct_node_id": True, "exact_label_match": True, "semantic_similarity": 1.0}]
        if args.target_node_id
        else _map(
            mapper,
            args.target,
            allowed_types=_types(args.target_types),
            top_k=args.node_map_k,
            min_similarity=args.min_similarity,
        )
    )
    if not source_matches:
        raise RuntimeError("No grounded source nodes matched")
    if not target_matches:
        raise RuntimeError("No grounded target nodes matched")

    units = CandidateUnitBuilder(graph).build(bridge_capable_only=True)
    confirmed = confirmed_navigation_graph(graph)
    vectors = _index_vectors(index)

    combined_query = f"{args.source or args.source_node_id} ; candidate scientific bridge ; {args.target or args.target_node_id}"
    query_vector = np.asarray(mapper.encoder.encode_query(combined_query), dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 0:
        raise RuntimeError("Candidate-unit query embedding has zero norm")
    query_vector = query_vector / query_norm

    unit_relevance: dict[str, float] = {}
    unit_vectors: dict[str, np.ndarray] = {}
    for unit in units:
        vector = vectors.get(unit.candidate_node_id)
        if vector is None:
            continue
        normalized = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(normalized))
        if norm <= 0:
            continue
        normalized = normalized / norm
        unit_vectors[unit.candidate_node_id] = normalized
        unit_relevance[unit.candidate_node_id] = max(0.0, min(1.0, float(np.dot(normalized, query_vector))))

    policy = CandidateUnitSelectionPolicy(
        max_depth=args.max_depth,
        top_k=args.top_k,
        max_routes_per_unit=args.max_routes_per_unit,
        max_unit_semantic_similarity=args.max_unit_semantic_similarity,
        min_unit_relevance=args.min_unit_relevance,
        min_selection_score=args.min_selection_score,
    )
    selector = CandidateUnitSelector(
        graph,
        confirmed,
        policy=policy,
        unit_relevance=unit_relevance,
        unit_vectors=unit_vectors,
        domain_profile=domain_profile,
    )
    routes = selector.enumerate_routes(units, source_matches, target_matches)
    selected_routes = selector.select(routes)
    quality_scorer = PathQualityScorer(
        graph,
        discovery_semantics=domain_profile.discovery,
    )
    all_paths = [_materialize_route(graph, route, quality_scorer) for route in routes]
    paths = [_materialize_route(graph, route, quality_scorer) for route in selected_routes]

    candidate_groups = {"CANDIDATE_EXPLORATION": [row["path_id"] for row in all_paths]}
    returned_groups = {"CANDIDATE_EXPLORATION": [row["path_id"] for row in paths]}
    inventory = candidate_unit_inventory(units)
    payload = {
        "schema_version": "candidate-unit-traversal-v1",
        "corpus_id": args.corpus_id,
        "domain_profile_id": domain_profile.profile_id,
        "data_root": str(data_root),
        "mode": "exploratory",
        "algorithm": "candidate_unit_top_n",
        "source_query": args.source,
        "target_query": args.target,
        "semantic_stop_query": None,
        "constraints": {
            "mode": "exploratory",
            "max_depth": args.max_depth,
            "candidate_unit_count": 1,
            "candidate_edge_count_per_unit": 2,
            "distinct_entry_exit_required": True,
            "confirmed_prefix_suffix_only": True,
        },
        "effective_max_depth": args.max_depth,
        "source_matches": source_matches,
        "target_matches": target_matches,
        "stop_matches": [],
        "candidate_unit_inventory": inventory,
        "candidate_unit_route_count": len(routes),
        "selected_candidate_unit_count": len({row["candidate_unit"]["unit_id"] for row in paths}),
        "candidate_path_count": len(all_paths),
        "returned_path_count": len(paths),
        "candidate_path_type_counts": {"CANDIDATE_EXPLORATION": len(all_paths)},
        "returned_path_type_counts": {"CANDIDATE_EXPLORATION": len(paths)},
        "candidate_path_groups": candidate_groups,
        "returned_path_groups": returned_groups,
        "path_count": len(paths),
        "path_type_counts": {"CANDIDATE_EXPLORATION": len(paths)},
        "path_groups": returned_groups,
        "selection_policy": policy.to_dict(),
        "candidate_paths_included": bool(args.include_candidate_paths),
        "candidate_paths": all_paths if args.include_candidate_paths else [],
        "paths": paths,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Candidate-unit traversal complete")
    print("Corpus:", args.corpus_id)
    print("Bridge-capable candidate units:", inventory["candidate_unit_count"])
    print("Valid source→unit→target routes:", len(all_paths))
    print("Returned paths:", len(paths))
    print("Full candidate pool included:", bool(args.include_candidate_paths))
    for rank, row in enumerate(paths, start=1):
        unit = row["candidate_unit"]
        score = row["candidate_unit_selection"]
        print(
            f"[{rank}] score={score['total']:.3f} unit_rel={score['unit_relevance']:.3f} "
            f"mech={score['mechanistic_continuity']:.2f} context_penalty={score['context_switch_penalty']:.2f} "
            f"hops={row['hop_count']} cost={row['total_cost']:.2f}"
        )
        print("     unit:", unit["label"])
        print("     entry:", unit["entry_anchor_label"])
        print("     exit :", unit["exit_anchor_label"])
        print("     core :", " -> ".join(node_label(graph, node) for node in row["candidate_unit_core_node_ids"]))
    print("Saved:", output)


if __name__ == "__main__":
    main()
