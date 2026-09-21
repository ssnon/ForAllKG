from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir.corpus_audit import (
    audit_existing_extraction_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover paper extraction attempts below a corpus root and "
            "aggregate their reusable semantic capabilities. No LLM calls "
            "are performed."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Corpus directory containing paper-local active_chunks.json files, "
            "one attempt directory, or an active_chunks.json file."
        ),
    )
    parser.add_argument(
        "--corpus-id",
        default=None,
        help="Optional stable corpus identifier. Defaults to the root basename.",
    )
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
        help=(
            "How to handle multiple active_chunks.json files for the same "
            "paper under the selected root. Default: fail closed."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path. Defaults to "
            "<root>/semantic_corpus_capability_audit.json for a directory root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_existing_extraction_corpus(
        args.root,
        corpus_id=args.corpus_id,
        duplicate_paper_policy=args.duplicate_paper_policy,
    )

    root_path = Path(args.root)
    default_parent = root_path if root_path.is_dir() else root_path.parent
    output_path = (
        Path(args.output)
        if args.output
        else default_parent / "semantic_corpus_capability_audit.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            audit.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Corpus:", audit.corpus_id)
    print("Root:", audit.root)
    print("Discovered active_chunks:", audit.discovery.discovered_active_chunks_count)
    print("Selected attempts:", audit.discovery.selected_attempt_count)
    print("Papers:", audit.discovery.paper_count)
    print("Duplicate papers:", audit.discovery.duplicate_paper_count)
    print("Manifest:", audit.manifest.manifest_id)
    print("Output:", output_path)
    print("Capabilities:")
    for name, record in audit.manifest.capabilities.items():
        metric = audit.metrics[name]
        coverage = (
            "n/a"
            if metric.coverage_fraction is None
            else f"{metric.coverage_fraction:.1%}"
        )
        print(
            f"  {name}: {record.state} "
            f"({metric.supported_count}/{metric.applicable_count}, "
            f"coverage={coverage}, mode={metric.assessment_mode}, "
            f"papers={metric.paper_assessed_count})"
        )


if __name__ == "__main__":
    main()
