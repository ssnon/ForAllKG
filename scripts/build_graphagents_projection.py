from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
from collections import Counter

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
    parser.add_argument(
        "--candidate-bridge-graphml",
        default=None,
        help=(
            "Semantic-candidate Bridge GraphML. "
            "Required for exploratory mode. "
            "It must belong to the same extraction "
            "and policy run as --bridge-graphml."
        ),
    )
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
    candidate_bridge_path = (
        Path(
            args.candidate_bridge_graphml
        )
        if args.candidate_bridge_graphml
        else (
            paper_root
            / (
                f"{args.paper_id}"
                ".bridge.candidates.graphml"
            )
        )
    )

    candidate_required = (
        args.mode == "exploratory"
    )

    if (
        candidate_required
        and not candidate_bridge_path.exists()
    ):
        raise FileNotFoundError(
            "Candidate Bridge graph not found: "
            f"{candidate_bridge_path}"
        )

    candidate_bridge_graph = (
        nx.read_graphml(
            candidate_bridge_path,
            force_multigraph=True,
        )
        if candidate_required
        else None
    )

    if (
        bridge_graph is not None
        and candidate_bridge_graph
        is not None
    ):
        metadata_keys = (
            "bridge_extraction_id",
            "bridge_policy_run_id",
            "bridge_policy_version",
        )

        mismatches: list[str] = []

        for key in metadata_keys:
            confirmed_value = str(
                bridge_graph.graph.get(
                    key,
                    "",
                )
            )
            candidate_value = str(
                candidate_bridge_graph
                .graph.get(
                    key,
                    "",
                )
            )

            if (
                not confirmed_value
                or not candidate_value
            ):
                mismatches.append(
                    f"{key}: missing metadata "
                    f"({confirmed_value!r}, "
                    f"{candidate_value!r})"
                )
                continue

            if (
                confirmed_value
                != candidate_value
            ):
                mismatches.append(
                    f"{key}: "
                    f"{confirmed_value!r} != "
                    f"{candidate_value!r}"
                )

        if mismatches:
            raise RuntimeError(
                "Confirmed and candidate Bridge "
                "graphs are not from the same "
                "policy materialization:\n- "
                + "\n- ".join(mismatches)
            )
    projection, node_rows, evidence_rows = (
        build_graphagents_projection(
            canonical_graph,
            bridge_graph=bridge_graph,
            candidate_bridge_graph=(
                candidate_bridge_graph
            ),
            mode=args.mode,
        )
    )
    evidence_status_counts = Counter(
        str(
            row.get(
                "evidence_status",
                "",
            )
        )
        for row in evidence_rows
    )

    multi_evidence_edges = 0
    max_support_count = 0

    for _, _, attrs in projection.edges(
        data=True
    ):
        try:
            edge_ids = json.loads(
                str(
                    attrs.get(
                        "projection_edge_ids_json",
                        "[]",
                    )
                )
            )
        except json.JSONDecodeError:
            edge_ids = []

        support_count = max(
            1,
            len(edge_ids),
        )

        max_support_count = max(
            max_support_count,
            support_count,
        )

        if support_count > 1:
            multi_evidence_edges += 1

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
        "candidate_bridge_graphml": (
            str(candidate_bridge_path)
            if candidate_bridge_graph
            is not None
            else ""
        ),

        "bridge_extraction_id": (
            str(
                bridge_graph.graph.get(
                    "bridge_extraction_id",
                    "",
                )
            )
            if bridge_graph is not None
            else ""
        ),

        "bridge_policy_run_id": (
            str(
                bridge_graph.graph.get(
                    "bridge_policy_run_id",
                    "",
                )
            )
            if bridge_graph is not None
            else ""
        ),

        "candidate_bridge_policy_run_id": (
            str(
                candidate_bridge_graph
                .graph.get(
                    "bridge_policy_run_id",
                    "",
                )
            )
            if candidate_bridge_graph
            is not None
            else ""
        ),
        "nodes": projection.number_of_nodes(),
        "edges": projection.number_of_edges(),
        "node_text_rows": len(node_rows),
        "edge_evidence_rows": len(evidence_rows),
        "graphml": str(graph_path),
        "node_text": str(node_text_path),
        "edge_evidence": str(evidence_path),
        "source_asserted_evidence_rows": (
            evidence_status_counts[
                "source_asserted"
            ]
        ),
        "derived_evidence_rows": (
            evidence_status_counts[
                "derived_projection"
            ]
        ),
        "multi_evidence_graph_edges": (
            multi_evidence_edges
        ),
        "max_support_count": (
            max_support_count
        ),
        "evidence_rows_per_graph_edge": (
            len(evidence_rows)
            / max(
                1,
                projection.number_of_edges(),
            )
        ),
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
