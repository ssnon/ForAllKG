from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.asset_index import write_assets_jsonl
from dac_her.config import get_paper_config
from dac_her.document_package import load_document_package
from dac_her.run_state import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Marker Markdown packages without calling an LLM."
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

    total = 0
    missing = 0
    for document in paper.documents:
        package = load_document_package(paper_id=paper.paper_id, config=document)
        document_dir = output_root / document.document_id
        write_json(document_dir / "document.json", {
            "paper_id": paper.paper_id,
            "document_id": document.document_id,
            "role": document.role,
            "package_dir": str(document.package_dir),
            "markdown_path": str(document.markdown_path),
            "metadata_path": str(document.metadata_path) if document.metadata_path else None,
            "asset_count": len(package.assets),
        })
        write_assets_jsonl(document_dir / "assets.jsonl", package.assets)
        total += len(package.assets)
        missing += sum(not asset.exists for asset in package.assets)
        print(
            f"[{document.document_id}] assets={len(package.assets)} "
            f"missing={sum(not asset.exists for asset in package.assets)}"
        )

    print("Indexed assets:", total)
    print("Missing assets:", missing)
    print("Output:", output_root)


if __name__ == "__main__":
    main()
