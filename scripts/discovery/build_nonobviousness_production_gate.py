from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.nonobviousness_production_gate import (
    build_nonobviousness_fallback_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile N10 non-obviousness shadow results into "
            "Alpha6 original-fallback production authority."
        )
    )

    parser.add_argument(
        "--intake-shadow",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--full-shadow",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    intake = json.loads(
        args.intake_shadow.read_text(
            encoding="utf-8"
        )
    )

    full = json.loads(
        args.full_shadow.read_text(
            encoding="utf-8"
        )
    )

    result = (
        build_nonobviousness_fallback_gate(
            intake_shadow=intake,
            full_shadow=full,
        )
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
        "N10 non-obviousness production gate built"
    )

    print(
        "Gates:",
        result["gate_count"],
    )

    print(
        "Fallback allowed:",
        sum(
            bool(row["fallback_allowed"])
            for row in result["gates"]
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
