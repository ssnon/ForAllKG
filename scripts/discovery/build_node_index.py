from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.node_mapping import DEFAULT_EMBED_MODEL, build_node_embedding_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exact cosine-search embedding index for corpus navigation nodes.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--data-root", default="data_dac")
    parser.add_argument("--mode", choices=("evidence", "mechanism", "exploratory"), default="exploratory")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--navigation-graphml", default=None)
    parser.add_argument("--node-text", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-alignment-hubs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    mode_root = data_root / "corpus" / args.corpus_id / args.mode
    navigation_root = mode_root / "navigation"
    graph_path = Path(args.navigation_graphml) if args.navigation_graphml else navigation_root / "graph.graphml"
    node_text_path = Path(args.node_text) if args.node_text else mode_root / "node_text.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else navigation_root / "node_index"

    manifest = build_node_embedding_index(
        navigation_graph_path=graph_path,
        node_text_path=node_text_path,
        output_dir=output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        include_alignment_hubs=args.include_alignment_hubs,
    )
    print("Node index built")
    print("Model:", manifest["model_name"])
    print("Nodes:", manifest["node_count"])
    print("Dimension:", manifest["embedding_dimension"])
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
