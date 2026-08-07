from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.corpus_visualization import (
    build_corpus_visualization,
    load_corpus_visualization_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static and interactive visualizations for a merged "
            "GraphAgents DAC/HER corpus graph."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help=(
            "Optional corpus directory. Default: "
            "data_dac/corpus/<corpus-id>/<mode>."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-overview-hubs",
        type=int,
        default=30,
        help="Maximum cross-paper hubs shown in the static overview SVG.",
    )
    parser.add_argument(
        "--max-evidence-per-edge",
        type=int,
        default=4,
        help="Compact evidence excerpts embedded per collapsed viewer edge.",
    )
    parser.add_argument(
        "--allow-failed-audit",
        action="store_true",
        help=(
            "Generate a visualization even when the source corpus structural "
            "audit did not pass. The viewer will display the failed status."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = (
        Path(args.corpus_dir)
        if args.corpus_dir
        else (
            PROJECT_ROOT
            / "data_dac"
            / "corpus"
            / args.corpus_id
            / args.mode
        )
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else corpus_dir / "visualization"
    )

    bundle = load_corpus_visualization_bundle(corpus_dir)

    manifest_corpus_id = str(bundle.manifest.get("corpus_id", ""))
    manifest_mode = str(bundle.manifest.get("mode", ""))
    if manifest_corpus_id and manifest_corpus_id != args.corpus_id:
        raise ValueError(
            f"Corpus ID mismatch: {manifest_corpus_id!r} != {args.corpus_id!r}."
        )
    if manifest_mode and manifest_mode != args.mode:
        raise ValueError(
            f"Projection mode mismatch: {manifest_mode!r} != {args.mode!r}."
        )

    passes = bool(bundle.audit.get("passes_structural_gate", False))
    if not passes and not args.allow_failed_audit:
        raise RuntimeError(
            "Corpus structural audit did not pass. Refusing visualization by "
            "default. Re-run with --allow-failed-audit only for diagnosis."
        )

    summary = build_corpus_visualization(
        bundle,
        output_dir=output_dir,
        max_overview_hubs=args.max_overview_hubs,
        max_evidence_per_edge=args.max_evidence_per_edge,
    )

    print("Corpus visualization built")
    print("Corpus ID:", summary["corpus_id"])
    print("Mode:", summary["mode"])
    print("Papers:", summary["papers"])
    print("Nodes/raw edges:", summary["nodes"], summary["raw_edges"])
    print("Alignment hubs:", summary["alignment_hubs"])
    print("Viewer:", summary["viewer"])
    print("Overview:", summary["overview_svg"])
    print("Similarity:", summary["paper_similarity_svg"])


if __name__ == "__main__":
    main()
