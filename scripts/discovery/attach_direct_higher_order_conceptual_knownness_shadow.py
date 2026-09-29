from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.direct_higher_order_conceptual_knownness_contracts import (
    DirectHigherOrderConceptualKnownnessProfile,
)
from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderConceptualKnownnessSlot,
    DirectHigherOrderShadowBundle,
)


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Attach an already-computed normalized conceptual-knownness "
            "profile to a DirectHigherOrderShadowBundle. Diagnostic only."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--profile",
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
    profile = (
        DirectHigherOrderConceptualKnownnessProfile.model_validate_json(
            args.profile.read_text(encoding="utf-8")
        )
    )

    if profile.source_bundle_id != bundle.bundle_id:
        raise ValueError(
            "conceptual-knownness profile / direct-HO bundle mismatch"
        )

    records = {
        row.hypothesis_id: row
        for row in profile.records
    }

    if len(records) != len(profile.records):
        raise ValueError(
            "duplicate conceptual-knownness hypothesis_id"
        )

    updated_arms = []
    attached = 0

    for arm in bundle.arms:
        hypothesis_ids = {
            row.hypothesis_id
            for row in arm.structural_views
        }

        matched = [
            records[hypothesis_id]
            for hypothesis_id in sorted(hypothesis_ids)
            if hypothesis_id in records
        ]

        if not matched:
            updated_arms.append(arm)
            continue

        # Current direct-HO generation normally yields one accepted hypothesis
        # per arm. Fail closed if a future portfolio contains multiple
        # conceptual profiles with conflicting first-gap levels.
        first_gap_levels = {
            row.first_gap_level
            for row in matched
            if row.first_gap_level is not None
        }

        if len(first_gap_levels) > 1:
            raise ValueError(
                "conflicting first_gap_level values within one direct-HO arm"
            )

        first_gap = (
            next(iter(first_gap_levels))
            if first_gap_levels
            else None
        )

        updated_arms.append(
            arm.model_copy(
                update={
                    "conceptual_knownness":
                        DirectHigherOrderConceptualKnownnessSlot(
                            status="AVAILABLE",
                            profile_path=str(args.profile),
                            first_gap_level=first_gap,
                        )
                }
            )
        )
        attached += 1

    updated = bundle.model_copy(
        update={
            "arms":
                updated_arms,
            "conceptual_knownness_integrated":
                True,
        }
    )

    # Revalidate count/authority invariants after model_copy.
    updated = DirectHigherOrderShadowBundle.model_validate(
        updated.model_dump(mode="json")
    )

    _write(
        args.output,
        updated,
    )

    print(
        "=== DIRECT-HO CONCEPTUAL KNOWNNESS ATTACH ==="
    )
    print(
        "bundle:",
        bundle.bundle_id,
    )
    print(
        "profile:",
        profile.profile_id,
    )
    print(
        "arms attached:",
        attached,
        "/",
        len(bundle.arms),
    )
    print(
        "diagnostic only; production selection unchanged."
    )
    print(
        "artifact:",
        args.output,
    )
    print(
        "DIRECT_HO_CONCEPTUAL_KNOWNNESS_ATTACHED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
