from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.corpus_publication import (
    CorpusPublicationOptions,
    StrictBridgeCorpusPublisher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a completed Strict/Bridge corpus into a traversal-ready "
            "production layout with complete selected-work lifecycle accounting "
            "and corpus -> navigation -> node-index fingerprint binding."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="mechanism",
    )
    parser.add_argument("--selected-works", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--m4-dir", required=True, type=Path)
    parser.add_argument("--m4-5-dir", required=True, type=Path)
    parser.add_argument("--outcomes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument(
        "--target-status",
        choices=("STRICT_USABLE", "BRIDGE_USEFUL", "CORPUS_ELIGIBLE"),
        default="CORPUS_ELIGIBLE",
    )
    parser.add_argument("--skip-node-index", action="store_true")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--include-alignment-hubs-in-index", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publisher = StrictBridgeCorpusPublisher(
        project_root=Path(args.project_root),
        corpus_id=args.corpus_id,
        domain_profile=args.domain_profile,
        data_root=args.data_root,
        selected_works_path=args.selected_works,
        m3_dir=args.m3_dir,
        m4_dir=args.m4_dir,
        m4_5_dir=args.m4_5_dir,
        outcomes_path=args.outcomes,
        output_dir=args.output_dir,
        options=CorpusPublicationOptions(
            mode=args.mode,
            target_count=args.target_count,
            target_status=args.target_status,
            build_node_index=not args.skip_node_index,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            include_alignment_hubs_in_index=args.include_alignment_hubs_in_index,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        ),
    )
    result = publisher.run()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
