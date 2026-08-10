from __future__ import annotations

import argparse

from dac_her.ingestion.corpus_manifest import build_corpus_manifest
from dac_her.ingestion.registry import PaperRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild an ingestion corpus manifest from the registry.")
    parser.add_argument("--registry", default="data_dac/ingestion/registry/papers.json")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-warnings", action="store_true")
    args = parser.parse_args()
    payload = build_corpus_manifest(
        PaperRegistry(args.registry),
        args.output,
        args.corpus_id,
        include_warnings=not args.exclude_warnings,
    )
    print(f"documents: {payload['document_count']}")
    print(args.output)


if __name__ == "__main__":
    main()
