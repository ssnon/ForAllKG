from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from dac_her.graph_io import save_graphml
from dac_her.graphagents_adapter import (
    build_graphagents_projection,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build GraphAgents-ready evidence, mechanism, or exploratory "
            "projections from a canonical DAC-HER graph and Bridge v2 graph."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="mechanism",
    )
    parser.add_argument("--canonical-graphml", default=None)
    parser.add_argument("--bridge-graphml", default=None)
    parser.add_argument("--output-dir", default=None)
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
    bridge_required = args.mode in {"mechanism", "exploratory"}
    if bridge_required and not bridge_path.exists():
        raise FileNotFoundError(f"Bridge graph not found: {bridge_path}")

    canonical_graph = nx.read_graphml(canonical_path, force_multigraph=True)
    bridge_graph = (
        nx.read_graphml(bridge_path, force_multigraph=True)
        if bridge_required
        else None
    )
    projection, node_rows, evidence_rows = build_graphagents_projection(
        canonical_graph,
        bridge_graph=bridge_graph,
        mode=args.mode,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else paper_root / "graphagents" / args.mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = save_graphml(projection, output_dir / "graph.graphml")
    node_text_path = write_jsonl(output_dir / "node_text.jsonl", node_rows)
    evidence_path = write_jsonl(output_dir / "edge_evidence.jsonl", evidence_rows)

    summary = {
        "paper_id": args.paper_id,
        "mode": args.mode,
        "canonical_graphml": str(canonical_path),
        "bridge_graphml": str(bridge_path) if bridge_graph is not None else "",
        "nodes": projection.number_of_nodes(),
        "edges": projection.number_of_edges(),
        "node_text_rows": len(node_rows),
        "edge_evidence_rows": len(evidence_rows),
        "graphml": str(graph_path),
        "node_text": str(node_text_path),
        "edge_evidence": str(evidence_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("GraphAgents projection built")
    print("Mode:", args.mode)
    print("Nodes/edges:", projection.number_of_nodes(), projection.number_of_edges())
    print("Saved:", graph_path)
    print("Node text:", node_text_path)
    print("Edge evidence:", evidence_path)


if __name__ == "__main__":
    main()
