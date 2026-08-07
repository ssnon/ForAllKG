from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from dac_her.corpus_graph import audit_corpus_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a cross-paper GraphAgents corpus graph."
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--graphml", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (
        PROJECT_ROOT
        / "data_dac"
        / "corpus"
        / args.corpus_id
        / args.mode
    )
    graph_path = Path(args.graphml) if args.graphml else root / "graph.graphml"
    manifest_path = Path(args.manifest) if args.manifest else root / "manifest.json"
    output_path = Path(args.output) if args.output else root / "audit.json"

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    audit = audit_corpus_graph(
        graph,
        expected_papers=manifest.get("paper_ids", []),
        expected_source_nodes=int(manifest.get("source_projection_nodes", 0)),
        expected_source_edges=int(manifest.get("source_projection_edges", 0)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Corpus audit finished")
    print("Graph:", graph_path)
    print("Papers:", len(audit["seen_papers"]))
    print("Nodes/edges:", audit["nodes"], audit["edges"])
    print("Paper-local nodes:", audit["paper_local_nodes"])
    print("Alignment hubs:", audit["alignment_hubs"])
    print("Direct cross-paper source edges:", audit["direct_cross_paper_source_edges"])
    print("Issues:", audit["issue_count"])
    print("Structural gate:", audit["passes_structural_gate"])
    print("Saved:", output_path)

    if not audit["passes_structural_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
