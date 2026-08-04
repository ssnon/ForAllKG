from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.config import get_paper_config
from dac_her.chunking import create_chunks
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.document_package import (
    extract_supplementary_references,
    load_document_package,
    select_document_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview main/SI source selection without calling an LLM."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    packages = {
        document.document_id: load_document_package(
            paper_id=paper.paper_id,
            config=document,
        )
        for document in paper.documents
    }

    selected_main = []
    for document in paper.documents:
        if document.role != "main":
            continue
        selected_main.extend(
            select_document_sources(
                package=packages[document.document_id],
                config=document,
            )
        )

    si_documents = [
        document
        for document in paper.documents
        if document.role == "supporting_information"
        and document.selection.mode == "referenced_blocks"
    ]
    use_whole_main = any(
        document.selection.reference_scope == "whole_main"
        for document in si_documents
    )
    reference_texts = (
        [
            packages[document.document_id].markdown
            for document in paper.documents
            if document.role == "main"
        ]
        if use_whole_main
        else [source.text for source in selected_main]
    )
    references = extract_supplementary_references(reference_texts)

    documents = []
    for document in paper.documents:
        if document.role == "supporting_information" and document.selection.mode == "referenced_blocks":
            sources = select_document_sources(
                package=packages[document.document_id],
                config=document,
                supplementary_references=references,
            )
        elif document.role == "main":
            sources = [
                source for source in selected_main
                if source.document_id == document.document_id
            ]
        else:
            sources = select_document_sources(
                package=packages[document.document_id],
                config=document,
            )
        package = packages[document.document_id]
        source_details = []
        linked_asset_ids: set[str] = set()
        for source in sources:
            chunks = create_chunks(
                paper_id=paper.paper_id,
                document_id=source.document_id,
                document_role=source.document_role,
                section=source.section,
                section_text=source.text,
                policy=ExtractionPolicy(),
                assets=package.assets,
            )
            source_assets = sorted({
                asset_id for chunk in chunks for asset_id in chunk.asset_ids
            })
            linked_asset_ids.update(source_assets)
            source_details.append({
                "section": source.section,
                "chunk_count": len(chunks),
                "linked_asset_count": len(source_assets),
                "linked_asset_ids": source_assets,
            })
        documents.append({
            "document_id": document.document_id,
            "role": document.role,
            "selection_mode": document.selection.mode,
            "reference_scope": document.selection.reference_scope,
            "fallback": document.selection.fallback,
            "source_count": len(sources),
            "sections": [source.section for source in sources],
            "indexed_asset_count": len(package.assets),
            "loose_asset_count": sum(
                not asset.referenced_in_markdown for asset in package.assets
            ),
            "linked_asset_count": len(linked_asset_ids),
            "linked_asset_ids": sorted(linked_asset_ids),
            "source_details": source_details,
        })

    payload = {
        "paper_id": paper.paper_id,
        "reference_scope": "whole_main" if use_whole_main else "selected_main",
        "supplementary_references": list(references),
        "documents": documents,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Paper:", paper.paper_id)
    print("Reference scope:", payload["reference_scope"])
    print("Supplementary references:", len(references))
    for reference in references:
        print("  -", reference)
    print("Documents:")
    for document in documents:
        print(
            f"  {document['document_id']} ({document['role']}): "
            f"{document['source_count']} selected source(s); "
            f"assets indexed={document['indexed_asset_count']} "
            f"loose={document['loose_asset_count']} "
            f"linked={document['linked_asset_count']}"
        )
        for detail in document["source_details"]:
            print(
                "    -", detail["section"],
                f"chunks={detail['chunk_count']}",
                f"assets={detail['linked_asset_count']}",
            )


if __name__ == "__main__":
    main()
