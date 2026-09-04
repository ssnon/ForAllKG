from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.nonobviousness_post_generation_production_gate_v2 import (
    build_nonobviousness_post_generation_production_gate_v2,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Promote a role-aware N10 v2 candidate gate "
            "to Alpha6 post-generation production authority."
        )
    )

    result.add_argument(
        "--candidate-gate",
        required=True,
    )

    result.add_argument(
        "--output",
        required=True,
    )

    return result


def main() -> None:
    args = parser().parse_args()

    source = Path(
        args.candidate_gate
    )

    output = Path(
        args.output
    )

    candidate = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    result = (
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    counts = result.get(
        "selection_counts",
        {}
    )

    print(
        "N10 post-generation role-aware production gate built"
    )

    print(
        "Selections:",
        counts,
    )

    print(
        "Production authority:",
        result.get(
            "production_authority"
        ),
    )

    print(
        "Authority scope:",
        result.get(
            "authority_scope"
        ),
    )


if __name__ == "__main__":
    main()
