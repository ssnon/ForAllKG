from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.asset_index import write_assets_jsonl
from dac_her.config import get_paper_config
from dac_her.document_package import load_document_package
from dac_her.locator_index import (
    build_locator_index,
    write_locator_index_csv,
    write_locator_index_json,
)
from dac_her.run_state import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Marker assets and Figure/Table locators without an LLM."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    output_root = PROJECT_ROOT / "data_dac" / "document_index" / paper.paper_id
    override_path = (
        PROJECT_ROOT
        / "configs"
        / "locator_overrides"
        / f"{paper.paper_id}.yaml"
    )

    total = 0
    missing = 0
    all_locators = []
    for document in paper.documents:
        package = load_document_package(paper_id=paper.paper_id, config=document)
        locators = build_locator_index(
            document_id=document.document_id,
            document_role=document.role,
            markdown=package.markdown,
            assets=package.assets,
            override_path=override_path if override_path.exists() else None,
        )
        document_dir = output_root / document.document_id
        write_json(document_dir / "document.json", {
            "paper_id": paper.paper_id,
            "document_id": document.document_id,
            "role": document.role,
            "package_dir": str(document.package_dir),
            "markdown_path": str(document.markdown_path),
            "metadata_path": str(document.metadata_path) if document.metadata_path else None,
            "asset_count": len(package.assets),
            "loose_asset_count": sum(
                not asset.referenced_in_markdown for asset in package.assets
            ),
            "locator_count": len(locators),
            "visual_locator_count": sum(
                item.kind in {"figure", "scheme"} for item in locators
            ),
            "visual_locator_with_asset_count": sum(
                item.kind in {"figure", "scheme"} and bool(item.asset_ids)
                for item in locators
            ),
            "table_locator_count": sum(item.kind == "table" for item in locators),
        })
        write_assets_jsonl(document_dir / "assets.jsonl", package.assets)
        write_locator_index_json(document_dir / "locator_index.json", locators)
        write_locator_index_csv(document_dir / "locator_index.csv", locators)
        all_locators.extend(locators)
        total += len(package.assets)
        missing += sum(not asset.exists for asset in package.assets)
        print(
            f"[{document.document_id}] assets={len(package.assets)} "
            f"loose={sum(not asset.referenced_in_markdown for asset in package.assets)} "
            f"missing={sum(not asset.exists for asset in package.assets)} "
            f"locators={len(locators)} "
            f"visual-linked={sum(item.kind in {'figure', 'scheme'} and bool(item.asset_ids) for item in locators)}"
        )

    write_locator_index_json(output_root / "locator_index.json", all_locators)
    write_locator_index_csv(output_root / "locator_index.csv", all_locators)
    print("Indexed assets:", total)
    print("Missing assets:", missing)
    print("Locator records:", len(all_locators))
    print("Output:", output_root)


if __name__ == "__main__":
    main()
