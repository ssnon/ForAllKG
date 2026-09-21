from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir import (
    audit_semantic_ir_capabilities,
    load_existing_extraction_semantic_ir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit semantic capabilities already present in an existing "
            "paper extraction attempt. No LLM calls are performed."
        )
    )
    parser.add_argument(
        "--attempt",
        required=True,
        help=(
            "Extraction attempt directory or its active_chunks.json path."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output path. Defaults to "
            "<attempt>/semantic_capability_audit.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported = load_existing_extraction_semantic_ir(args.attempt)
    audit = audit_semantic_ir_capabilities(imported.bundle)

    active_path = Path(imported.active_chunks_path)
    output_path = (
        Path(args.output)
        if args.output
        else active_path.parent / "semantic_capability_audit.json"
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

    print("Paper:", audit.paper_id)
    print("Bundle:", audit.bundle_id)
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
            f"coverage={coverage}, mode={metric.assessment_mode})"
        )


if __name__ == "__main__":
    main()
