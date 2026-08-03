from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from dac_her.claim_overlap import write_claim_overlap_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-destructive paper-level claim-overlap candidates.")
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--minimum-score", type=float, default=0.68)
    args = parser.parse_args()

    graph_path = Path(args.graphml).resolve()
    graph = nx.read_graphml(graph_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else graph_path.parent / "claim_audit"
    summary = write_claim_overlap_audit(graph, output_dir, minimum_score=args.minimum_score)
    print("Claim-overlap audit successful")
    print("Output:", output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
