from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from dac_her.config import get_paper_config
from dac_her.document_package import load_document_package
from dac_her.figure_extraction import analyze_figure, resolve_vision_model


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run on-demand vision analysis for selected Marker assets."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=os.getenv("OPENROUTER_PROVIDER") or None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    try:
        document = next(
            item for item in paper.documents if item.document_id == args.document_id
        )
    except StopIteration as error:
        raise KeyError(f"Unknown document_id: {args.document_id}") from error

    package = load_document_package(paper_id=paper.paper_id, config=document)
    requested = set(args.asset)
    selected = [
        asset
        for asset in package.assets
        if args.all
        or asset.asset_id in requested
        or asset.relative_path in requested
        or Path(asset.relative_path).name in requested
    ]
    if not selected:
        raise ValueError("No assets selected. Use --asset ... or --all.")

    model = args.model or resolve_vision_model(
        document.figure_processing,
        os.getenv("OPENROUTER_EXTRACT_MODEL"),
    )
    output_dir = (
        PROJECT_ROOT
        / "data_dac"
        / "document_index"
        / paper.paper_id
        / document.document_id
        / "vision"
    )
    for asset in selected:
        result = analyze_figure(
            asset=asset,
            model=model,
            provider=args.provider,
            output_dir=output_dir,
            force=args.force,
        )
        print("[VISION]", result.asset_id, "->", output_dir)


if __name__ == "__main__":
    main()
