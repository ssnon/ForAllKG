
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.sers_certification_context_resolution import (
    write_grounded_hypothesis_context,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--metadata-output", type=Path, default=None)
    args = p.parse_args()

    meta = write_grounded_hypothesis_context(
        source_path=args.source,
        portfolio_path=args.portfolio,
        output_path=args.output,
    )

    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.metadata_output.write_text(
            json.dumps(
                {
                    "schema_version":
                        "sers-certification-context-resolution-v1",
                    **meta,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

    print("Grounded HypothesisContext resolved")
    for key, value in meta.items():
        print(key + ":", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
