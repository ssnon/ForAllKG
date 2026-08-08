from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from dac_her.node_mapping import NodeMapper, QueryConcept
from dac_her.traversal_engine import TraversalConstraints, TraversalEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _types(raw: str | None) -> tuple[str, ...]:
    return tuple(x.strip() for x in (raw or "").split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map scientific concepts to nodes and run policy-constrained graph traversal.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--mode", choices=("evidence", "mechanism", "exploratory"), default="mechanism")
    parser.add_argument("--algorithm", choices=("shortest", "top_n", "bounded_dfs", "semantic_stop"), default="top_n")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", default=None)
    source.add_argument("--source-node-id", default=None)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", default=None)
    target.add_argument("--target-node-id", default=None)

    parser.add_argument("--stop", default=None)
    parser.add_argument("--stop-node-id", default=None)
    parser.add_argument("--source-types", default=None)
    parser.add_argument("--target-types", default=None)
    parser.add_argument("--stop-types", default=None)
    parser.add_argument("--node-map-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--allow-candidate-endpoints", action="store_true")
    parser.add_argument("--allow-alignment-hub-endpoints", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-alignment-edges", type=int, default=2)
    parser.add_argument("--min-scientific-edges", type=int, default=1)
    parser.add_argument("--max-expansions", type=int, default=20_000)
    parser.add_argument("--navigation-graphml", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _map(mapper: NodeMapper, text: str, allowed_types: tuple[str, ...], args: argparse.Namespace) -> list[dict]:
    return [x.to_dict() for x in mapper.map(QueryConcept(
        text=text,
        allowed_node_types=allowed_types,
        allow_candidates=args.allow_candidate_endpoints,
        allow_alignment_hubs=args.allow_alignment_hub_endpoints,
        top_k=args.node_map_k,
        min_similarity=args.min_similarity,
    ))]


def main() -> None:
    args = parse_args()
    if args.algorithm == "semantic_stop" and not (args.stop or args.stop_node_id):
        raise ValueError("semantic_stop requires --stop or --stop-node-id.")

    mode_root = PROJECT_ROOT / "data_dac" / "corpus" / args.corpus_id / args.mode
    navigation_root = mode_root / "navigation"
    graph_path = Path(args.navigation_graphml) if args.navigation_graphml else navigation_root / "graph.graphml"
    index_dir = Path(args.index_dir) if args.index_dir else navigation_root / "node_index"

    graph = nx.read_graphml(graph_path)
    engine = TraversalEngine(graph)
    mapper: NodeMapper | None = None

    def get_mapper() -> NodeMapper:
        nonlocal mapper
        if mapper is None:
            mapper = NodeMapper.from_directory(index_dir, device=args.device)
        return mapper

    source_matches = [{"node_id": args.source_node_id, "direct_node_id": True}] if args.source_node_id else _map(get_mapper(), args.source, _types(args.source_types), args)
    target_matches = [{"node_id": args.target_node_id, "direct_node_id": True}] if args.target_node_id else _map(get_mapper(), args.target, _types(args.target_types), args)
    if args.stop_node_id:
        stop_matches: list[dict | None] = [{"node_id": args.stop_node_id, "direct_node_id": True}]
    elif args.stop:
        stop_matches = _map(get_mapper(), args.stop, _types(args.stop_types), args)
    else:
        stop_matches = [None]

    if not source_matches:
        raise RuntimeError("No source nodes matched.")
    if not target_matches:
        raise RuntimeError("No target nodes matched.")
    if args.algorithm == "semantic_stop" and not stop_matches:
        raise RuntimeError("No semantic-stop nodes matched.")

    constraints = TraversalConstraints(
        mode=args.mode,
        top_k=args.top_k,
        max_depth=args.max_depth,
        max_alignment_edges=args.max_alignment_edges,
        min_scientific_edges=args.min_scientific_edges,
        max_expansions=args.max_expansions,
    )

    collected: dict[str, dict] = {}
    for source_match in source_matches:
        for target_match in target_matches:
            for stop_match in stop_matches:
                stop_node = str(stop_match["node_id"]) if stop_match else None
                for path in engine.traverse(
                    str(source_match["node_id"]),
                    str(target_match["node_id"]),
                    algorithm=args.algorithm,
                    constraints=constraints,
                    semantic_stop=stop_node,
                ):
                    row = path.to_dict()
                    row["source_match"] = source_match
                    row["target_match"] = target_match
                    row["stop_match"] = stop_match
                    collected[path.path_id] = row

    paths = sorted(collected.values(), key=lambda row: (float(row["total_cost"]), int(row["hop_count"]), str(row["path_id"])))[: args.top_k]
    payload = {
        "corpus_id": args.corpus_id,
        "mode": args.mode,
        "algorithm": args.algorithm,
        "source_query": args.source,
        "target_query": args.target,
        "semantic_stop_query": args.stop,
        "constraints": constraints.to_dict(),
        "source_matches": source_matches,
        "target_matches": target_matches,
        "stop_matches": [] if stop_matches == [None] else stop_matches,
        "path_count": len(paths),
        "paths": paths,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "source_matches": [{"node_id": row["node_id"], "label": row.get("label"), "similarity": row.get("semantic_similarity")} for row in source_matches],
        "target_matches": [{"node_id": row["node_id"], "label": row.get("label"), "similarity": row.get("semantic_similarity")} for row in target_matches],
        "path_count": len(paths),
    }, ensure_ascii=False, indent=2))

    for index, path in enumerate(paths, start=1):
        print(f"\n[{index}] cost={path['total_cost']:.3f} hops={path['hop_count']} papers={path['cross_paper_count']} candidate={path['candidate_edge_count']} reverse={path['reverse_edge_count']}")
        for step in path["steps"]:
            marker = " <-reverse" if step["traversal_direction"] == "reverse" else ""
            print("  ", step["source"], "--", step["relation"], "-->", step["target"], marker)

    if args.output:
        print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
