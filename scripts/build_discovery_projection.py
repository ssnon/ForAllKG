from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from dac_her.bridge_graph import build_discovery_projection
from dac_her.graph_io import save_graphml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GraphAgents-friendly DiGraph projection with either the "
            "strict canonical core alone or the core plus bridge concepts."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("core-only", "core-plus-bridge"),
        default="core-plus-bridge",
    )
    parser.add_argument("--canonical-graphml", default=None)
    parser.add_argument("--bridge-graphml", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_root = PROJECT_ROOT / "data_dac" / "extracted" / args.paper_id
    canonical_path = (
        Path(args.canonical_graphml)
        if args.canonical_graphml
        else paper_root / f"{args.paper_id}.graphml"
    )
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical graph not found: {canonical_path}")

    bridge_path = (
        Path(args.bridge_graphml)
        if args.bridge_graphml
        else paper_root / f"{args.paper_id}.bridge.graphml"
    )
    include_bridge = args.mode == "core-plus-bridge"
    if include_bridge and not bridge_path.exists():
        raise FileNotFoundError(f"Bridge graph not found: {bridge_path}")

    canonical_graph = nx.read_graphml(canonical_path, force_multigraph=True)
    bridge_graph = (
        nx.read_graphml(bridge_path, force_multigraph=True)
        if include_bridge
        else None
    )
    projection = build_discovery_projection(
        canonical_graph,
        bridge_graph=bridge_graph,
        include_bridge=include_bridge,
    )

    output_path = (
        Path(args.output)
        if args.output
        else paper_root / "discovery" / f"{args.mode}.graphml"
    )
    save_graphml(projection, output_path)

    print("Discovery projection built")
    print("Mode:", args.mode)
    print("Graph class:", type(projection).__name__)
    print("Nodes/edges:", projection.number_of_nodes(), projection.number_of_edges())
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
