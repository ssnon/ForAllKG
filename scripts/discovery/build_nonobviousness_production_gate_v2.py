from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.nonobviousness_production_gate_v2 import (
    build_nonobviousness_production_gate_v2,
)


def _load_json(
    path: Path,
) -> dict:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"expected JSON object: {path}"
        )

    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a frozen candidate-only "
            "N10 v2 gate into authoritative "
            "Alpha6 original-fallback authority."
        )
    )

    parser.add_argument(
        "--candidate-gate",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    artifact = (
        build_nonobviousness_production_gate_v2(
            candidate_gate=_load_json(
                args.candidate_gate
            )
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "N10 role-aware production gate v2 built"
    )

    print(
        "Hypotheses:",
        artifact["gate_count"],
    )

    print(
        "Selections:",
        artifact["selection_counts"],
    )

    print(
        "Fallback allowed:",
        artifact[
            "fallback_allowed_count"
        ],
    )

    print(
        "Production authority:",
        artifact[
            "production_authority"
        ],
    )


if __name__ == "__main__":
    main()
