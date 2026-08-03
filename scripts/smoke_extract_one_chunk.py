from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from dac_her.chunking import create_chunks
from dac_her.config import get_paper_config
from dac_her.document_package import load_document_package, select_document_sources
from dac_her.extraction import extract_one_chunk
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.figure_extraction import format_asset_context


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one configured document chunk through the extractor."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--document-index", type=int, default=0)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or os.getenv("OPENROUTER_EXTRACT_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_EXTRACT_MODEL is not defined.")

    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    try:
        document = paper.documents[args.document_index]
    except IndexError as error:
        raise IndexError("Invalid --document-index.") from error

    package = load_document_package(paper_id=paper.paper_id, config=document)
    sources = select_document_sources(package=package, config=document)
    try:
        source = sources[args.source_index]
    except IndexError as error:
        raise IndexError("Invalid --source-index.") from error

    policy = replace(ExtractionPolicy(), concurrency=1)
    chunks = create_chunks(
        paper_id=paper.paper_id,
        document_id=document.document_id,
        document_role=document.role,
        section=source.section,
        section_text=source.text,
        policy=policy,
        assets=package.assets,
    )
    try:
        chunk = chunks[args.chunk_index]
    except IndexError as error:
        raise IndexError("Invalid --chunk-index.") from error

    by_id = {asset.asset_id: asset for asset in package.assets}
    chunk = replace(
        chunk,
        asset_context=format_asset_context(
            [by_id[asset_id] for asset_id in chunk.asset_ids],
            {},
        ),
    )

    output_root = PROJECT_ROOT / "data_dac" / "smoke_output" / paper.paper_id
    record = extract_one_chunk(
        chunk=chunk,
        model=model,
        provider=args.provider,
        policy=policy,
        chunk_output_dir=output_root / "chunks",
        debug_dir=output_root / "debug",
        force=args.force,
    )
    print("Smoke extraction finished")
    print("Status:", record["status"])
    print("Chunk:", record["chunk_id"])
    print("Document:", record["document_id"])
    print("Assets:", record.get("asset_ids", []))
    print("Output:", record.get("output_path"))


if __name__ == "__main__":
    main()
