from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.nonobviousness_production_gate_v2_candidate import (
    build_nonobviousness_production_gate_v2_candidate,
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
            "Build the candidate-only role-aware "
            "N10 v2 fallback gate."
        )
    )

    parser.add_argument(
        "--query-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--intake-shadow",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--full-shadow",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    query_plan = (
        LiteratureQueryPlan.model_validate_json(
            args.query_plan.read_text(
                encoding="utf-8"
            )
        )
    )

    artifact = (
        build_nonobviousness_production_gate_v2_candidate(
            query_plan=query_plan,
            intake_shadow=_load_json(
                args.intake_shadow
            ),
            full_shadow=_load_json(
                args.full_shadow
            ),
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
        "N10 role-aware production candidate gate built"
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
        "Candidate fallback allowed:",
        artifact[
            "candidate_fallback_allowed_count"
        ],
    )

    print(
        "Production authority:",
        artifact["production_authority"],
    )


if __name__ == "__main__":
    main()
