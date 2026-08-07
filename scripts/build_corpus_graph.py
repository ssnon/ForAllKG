from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.corpus_graph import (
    audit_corpus_graph,
    build_corpus_graph,
    load_projection_bundle,
    write_jsonl,
)
from dac_her.graph_io import save_graphml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive cross-paper GraphAgents corpus graph "
            "from frozen per-paper projections."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--paper-ids",
        nargs="+",
        required=True,
        help="Paper IDs to include, e.g. Kiwook_1 ... Kiwook_10.",
    )
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-registry-alignment",
        action="store_true",
        help="Disable deterministic Metal/Reaction alignment hubs.",
    )
    parser.add_argument(
        "--no-pattern-alignment",
        action="store_true",
        help="Disable exact confirmed Bridge pattern alignment hubs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.paper_ids)) != len(args.paper_ids):
        raise ValueError("--paper-ids contains duplicates.")

    bundles = [
        load_projection_bundle(
            project_root=PROJECT_ROOT,
            paper_id=paper_id,
            mode=args.mode,
        )
        for paper_id in args.paper_ids
    ]

    (
        graph,
        node_rows,
        evidence_rows,
        registry_rows,
        pattern_rows,
        candidate_rows,
        manifest,
    ) = build_corpus_graph(
        bundles,
        corpus_id=args.corpus_id,
        mode=args.mode,
        add_registry_alignment=(not args.no_registry_alignment),
        add_pattern_alignment=(not args.no_pattern_alignment),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            PROJECT_ROOT
            / "data_dac"
            / "corpus"
            / args.corpus_id
            / args.mode
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = save_graphml(graph, output_dir / "graph.graphml")
    node_text_path = write_jsonl(output_dir / "node_text.jsonl", node_rows)
    evidence_path = write_jsonl(
        output_dir / "edge_evidence.jsonl",
        evidence_rows,
    )
    registry_path = write_jsonl(
        output_dir / "registry_alignments.jsonl",
        registry_rows,
    )
    pattern_path = write_jsonl(
        output_dir / "pattern_alignments.jsonl",
        pattern_rows,
    )
    candidates_path = write_jsonl(
        output_dir / "cross_paper_resolution_candidates.jsonl",
        candidate_rows,
    )

    audit = audit_corpus_graph(
        graph,
        expected_papers=args.paper_ids,
        expected_source_nodes=manifest["source_projection_nodes"],
        expected_source_edges=manifest["source_projection_edges"],
    )

    manifest.update({
        "graphml": str(graph_path),
        "node_text": str(node_text_path),
        "edge_evidence": str(evidence_path),
        "registry_alignments": str(registry_path),
        "pattern_alignments": str(pattern_path),
        "cross_paper_resolution_candidates_path": str(candidates_path),
        "audit": str(output_dir / "audit.json"),
        "passes_structural_gate": audit["passes_structural_gate"],
    })

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Corpus graph built")
    print("Corpus ID:", args.corpus_id)
    print("Mode:", args.mode)
    print("Papers:", len(args.paper_ids))
    print(
        "Source projection nodes/edges:",
        manifest["source_projection_nodes"],
        manifest["source_projection_edges"],
    )
    print("Corpus nodes/edges:", graph.number_of_nodes(), graph.number_of_edges())
    print("Registry alignment hubs:", len(registry_rows))
    print("Pattern alignment hubs:", len(pattern_rows))
    print("Cross-paper review candidates:", len(candidate_rows))
    print("Structural gate:", audit["passes_structural_gate"])
    print("Saved:", graph_path)

    if not audit["passes_structural_gate"]:
        raise RuntimeError(
            "Corpus structural audit failed. See: "
            f"{output_dir / 'audit.json'}"
        )


if __name__ == "__main__":
    main()
