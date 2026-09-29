from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.direct_higher_order_alpha6_trigger_shadow import (
    recommend_direct_higher_order_alpha6_triggers,
)
from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderShadowBundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build diagnostic-only Alpha6 trigger recommendations from "
            "structural external novelty + conceptual first-gap level."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    bundle = DirectHigherOrderShadowBundle.model_validate_json(
        args.bundle.read_text(encoding="utf-8")
    )

    rows = recommend_direct_higher_order_alpha6_triggers(
        bundle
    )

    payload = {
        "schema_version":
            "direct-higher-order-alpha6-trigger-shadow-report-v1",
        "source_bundle_id":
            bundle.bundle_id,
        "recommendations": [
            row.model_dump(mode="json")
            for row in rows
        ],
        "counts": {
            key: sum(
                row.recommendation == key
                for row in rows
            )
            for key in (
                "KEEP",
                "REAXIS_CANDIDATE",
                "HOLD_UNRESOLVED",
                "NO_RECOMMENDATION",
            )
        },
        "authority": {
            "diagnostic_only":
                True,
            "alpha6_trigger_authority":
                False,
            "candidate_survival_authority":
                False,
            "production_selection_authority":
                False,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== DIRECT-HO ALPHA6 TRIGGER SHADOW ==="
    )
    print(
        "bundle:",
        bundle.bundle_id,
    )
    print(
        "counts:",
        payload["counts"],
    )
    print(
        "diagnostic only; Alpha6 behavior unchanged."
    )
    print(
        "artifact:",
        args.output,
    )
    print(
        "DIRECT_HO_ALPHA6_TRIGGER_SHADOW_COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
