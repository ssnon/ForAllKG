from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx

from pipeline_core.corpus.graph.graph_io import save_graphml
from pipeline_core.corpus.navigation_graph import NavigationPolicy, build_navigation_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provenance-preserving traversal DiGraph from a corpus graph.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--data-root", default="data_dac")
    parser.add_argument("--mode", choices=("evidence", "mechanism", "exploratory"), default="exploratory")
    parser.add_argument("--corpus-graphml", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reverse-penalty", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    mode_root = data_root / "corpus" / args.corpus_id / args.mode
    source_path = Path(args.corpus_graphml) if args.corpus_graphml else mode_root / "graph.graphml"
    output_dir = Path(args.output_dir) if args.output_dir else mode_root / "navigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = nx.read_graphml(source_path, force_multigraph=True)
    policy = NavigationPolicy(reverse_penalty=args.reverse_penalty)
    navigation, sidecar_rows, summary = build_navigation_graph(corpus, policy=policy)

    graph_path = save_graphml(navigation, output_dir / "graph.graphml")
    sidecar_path = output_dir / "edge_alternatives.jsonl"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        for row in sidecar_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary.update({
        "corpus_id": args.corpus_id,
        "mode": args.mode,
        "source_graphml": str(source_path),
        "source_graph_sha256": _sha256_file(source_path),
        "graphml": str(graph_path),
        "graphml_sha256": _sha256_file(graph_path),
        "edge_alternatives": str(sidecar_path),
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Navigation graph built")
    print("Corpus:", args.corpus_id)
    print("Mode:", args.mode)
    print("Source nodes/edges:", summary["source_nodes"], summary["source_edges"])
    print("Navigation nodes/edges:", summary["navigation_nodes"], summary["navigation_edges"])
    print("Reverse edges:", summary["reverse_navigation_edges"])
    print("Candidate edges:", summary["candidate_navigation_edges"])
    print("Edge classes:", summary["edge_class_counts"])
    print("Saved:", graph_path)


if __name__ == "__main__":
    main()
