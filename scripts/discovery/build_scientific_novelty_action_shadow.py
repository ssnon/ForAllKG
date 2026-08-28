from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_novelty_action_batch import (
    build_scientific_novelty_action_batch,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessReview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile external novelty plus two-pass semantic "
            "distinctiveness into deterministic scientific-novelty "
            "action decisions. Shadow only; production selection is "
            "not mutated."
        )
    )

    parser.add_argument(
        "--external-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--semantic-review",
        action="append",
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

    external_payload = json.loads(
        args.external_report.read_text(
            encoding="utf-8"
        )
    )

    semantic_reviews = [
        SemanticDistinctivenessReview.model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )
        for path in args.semantic_review
    ]

    result = build_scientific_novelty_action_batch(
        external_payload=external_payload,
        semantic_reviews=semantic_reviews,
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
        "Scientific novelty action shadow batch built"
    )
    print(
        "Decision count:",
        result["decision_count"],
    )
    print(
        "Production selection changed:",
        result["scientific_selection_changed"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
