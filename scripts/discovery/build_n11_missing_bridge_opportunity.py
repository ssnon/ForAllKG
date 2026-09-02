from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.nonobviousness_missing_bridge import (
    compile_missing_bridge_opportunity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile an N10 missing lower-order bridge "
            "into a provenance-safe N11 upstream search opportunity."
        )
    )

    parser.add_argument(
        "--detail-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> int:
    args = parse_args()

    detail = args.detail_dir

    execution = load_json(
        detail / "execution_plan.json"
    )

    reviews = load_json(
        detail / "slot_reviews.json"
    )

    relationships = load_json(
        detail
        / "closure_relationships.json"
    )

    result = (
        compile_missing_bridge_opportunity(
            execution_plan=execution,
            slot_reviews=reviews,
            closure_relationships=(
                relationships
            ),
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        result.model_dump_json(
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "N11 missing-bridge compilation built"
    )
    print("Status:", result.status)
    print(
        "Reason codes:",
        result.reason_codes,
    )

    if result.opportunity is not None:
        print(
            "Opportunity:",
            result.opportunity.opportunity_id,
        )
        print(
            "Factor anchors:",
            result.opportunity.factor_identity_terms,
        )
        print(
            "Base context:",
            result.opportunity.base_relation_terms,
        )
        print(
            "Allowed paths:",
            result.opportunity
            .search_requirement
            .allowed_path_classes,
        )
        print(
            "Blocked paths:",
            result.opportunity
            .search_requirement
            .blocked_path_classes,
        )

    print(
        "Production authority:",
        result.production_authority,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
