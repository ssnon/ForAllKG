from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_novelty_production_gate import (
    build_scientific_novelty_fallback_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action-batch",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = json.loads(
        args.action_batch.read_text(
            encoding="utf-8"
        )
    )

    result = build_scientific_novelty_fallback_gate(
        payload
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Scientific novelty production fallback gate built"
    )
    print(
        "Gate count:",
        result["gate_count"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
