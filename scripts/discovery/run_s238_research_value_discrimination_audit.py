from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "mechanistic_discrimination",
    "two_sided_outcome_informativeness",
    "observable_decisiveness",
    "information_gain_proxy",
    "experimental_resolvability",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
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


def dimension_signature(card: dict[str, Any]) -> tuple[str, ...]:
    dims = card.get("dimensions") or {}
    return tuple(str(dims.get(name) or "MISSING") for name in DIMENSIONS)


def full_signature(card: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(card.get("value_argument_class") or "MISSING"),
        *dimension_signature(card),
    )


def discrimination_status(
    *,
    card_count: int,
    unique_value_classes: int,
    unique_dimension_signatures: int,
    dominant_signature_fraction: float,
) -> str:
    if card_count < 2:
        return "INSUFFICIENT_SAMPLE"

    if (
        unique_value_classes == 1
        and unique_dimension_signatures == 1
    ):
        return "NON_DISCRIMINATIVE"

    if (
        dominant_signature_fraction >= 0.80
        or unique_dimension_signatures <= 2
    ):
        return "LOW_DISCRIMINATION"

    return "DISCRIMINATION_OBSERVED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S238 artifact-only Research Value discrimination audit. "
            "Measures whether the existing S222 shadow actually separates "
            "prospective-cohort hypotheses. No novelty or selection authority."
        )
    )
    p.add_argument(
        "--s226-matrix",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    matrix_path = args.s226_matrix.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not matrix_path.is_file():
        raise RuntimeError(
            "missing S226 matrix: " + str(matrix_path)
        )
    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(output_path)
        )

    matrix = load_json(matrix_path)

    cards: list[dict[str, Any]] = []
    unavailable_cases: list[dict[str, Any]] = []

    for case in matrix.get("cases", []):
        case_id = str(case.get("source_case_id") or "UNKNOWN")
        role = str(case.get("case_role") or "")
        rv = case.get("research_value") or {}

        if rv.get("measurement_status") != "AVAILABLE":
            unavailable_cases.append(
                {
                    "source_case_id": case_id,
                    "case_role": role,
                    "measurement_status": rv.get("measurement_status"),
                }
            )
            continue

        for card in rv.get("cards", []):
            if not isinstance(card, dict):
                continue
            cards.append(
                {
                    "source_case_id": case_id,
                    "case_role": role,
                    **card,
                }
            )

    value_class_counts = Counter(
        str(card.get("value_argument_class") or "MISSING")
        for card in cards
    )

    per_dimension_counts: dict[str, dict[str, int]] = {}
    for name in DIMENSIONS:
        counts = Counter(
            str((card.get("dimensions") or {}).get(name) or "MISSING")
            for card in cards
        )
        per_dimension_counts[name] = dict(sorted(counts.items()))

    dim_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    full_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for card in cards:
        dim_groups[dimension_signature(card)].append(card)
        full_groups[full_signature(card)].append(card)

    dominant_dim_count = max(
        (len(rows) for rows in dim_groups.values()),
        default=0,
    )
    dominant_dim_fraction = (
        dominant_dim_count / len(cards)
        if cards
        else 0.0
    )

    status = discrimination_status(
        card_count=len(cards),
        unique_value_classes=len(value_class_counts),
        unique_dimension_signatures=len(dim_groups),
        dominant_signature_fraction=dominant_dim_fraction,
    )

    constant_dimensions = [
        name
        for name, counts in per_dimension_counts.items()
        if len(counts) <= 1
    ]
    varying_dimensions = [
        name
        for name, counts in per_dimension_counts.items()
        if len(counts) > 1
    ]

    hypothesis_types = sorted(
        {
            str(card.get("hypothesis_type") or "MISSING")
            for card in cards
        }
    )
    burden_pairs = sorted(
        {
            (
                str(card.get("relative_cost_burden") or "MISSING"),
                str(card.get("relative_effort_burden") or "MISSING"),
            )
            for card in cards
        }
    )

    duplicate_signature_groups = []
    for signature, rows in sorted(
        dim_groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        duplicate_signature_groups.append(
            {
                "dimension_signature": {
                    name: value
                    for name, value in zip(DIMENSIONS, signature)
                },
                "card_count": len(rows),
                "cards": [
                    {
                        "source_case_id": row["source_case_id"],
                        "hypothesis_id": row.get("hypothesis_id"),
                        "hypothesis_type": row.get("hypothesis_type"),
                        "value_argument_class": row.get("value_argument_class"),
                        "experimental_disposition":
                            row.get("experimental_disposition"),
                        "relative_cost_burden":
                            row.get("relative_cost_burden"),
                        "relative_effort_burden":
                            row.get("relative_effort_burden"),
                    }
                    for row in rows
                ],
            }
        )

    class_masking = (
        len(cards) >= 2
        and len(value_class_counts) == 1
        and (
            len(hypothesis_types) > 1
            or len(burden_pairs) > 1
            or len(dim_groups) > 1
        )
    )

    summary = {
        "schema_version":
            "research-value-discrimination-audit-s238-v1",
        "source_s226_matrix": str(matrix_path),
        "card_count": len(cards),
        "available_case_count": len(
            {
                card["source_case_id"]
                for card in cards
            }
        ),
        "unavailable_cases": unavailable_cases,
        "value_class_counts":
            dict(sorted(value_class_counts.items())),
        "unique_value_class_count":
            len(value_class_counts),
        "per_dimension_signal_counts":
            per_dimension_counts,
        "constant_dimensions":
            constant_dimensions,
        "varying_dimensions":
            varying_dimensions,
        "unique_dimension_signature_count":
            len(dim_groups),
        "unique_full_signature_count":
            len(full_groups),
        "dominant_dimension_signature_count":
            dominant_dim_count,
        "dominant_dimension_signature_fraction":
            dominant_dim_fraction,
        "hypothesis_type_count":
            len(hypothesis_types),
        "hypothesis_types":
            hypothesis_types,
        "resource_burden_pair_count":
            len(burden_pairs),
        "resource_burden_pairs": [
            {
                "cost": cost,
                "effort": effort,
            }
            for cost, effort in burden_pairs
        ],
        "value_class_masks_input_variation":
            class_masking,
        "discrimination_status":
            status,
        "duplicate_signature_groups":
            duplicate_signature_groups,
        "interpretation_policy": {
            "NON_DISCRIMINATIVE": (
                "All assessed cards have one value class and one "
                "five-dimension signal signature."
            ),
            "LOW_DISCRIMINATION": (
                "The shadow produces some variation, but one signature "
                "dominates >=80% or there are at most two signatures."
            ),
            "DISCRIMINATION_OBSERVED": (
                "The existing shadow separates the cohort into multiple "
                "non-dominant dimension signatures. This does not establish "
                "scientific validity or ranking authority."
            ),
            "INSUFFICIENT_SAMPLE": (
                "Fewer than two Research Value cards were available."
            ),
            "novelty_signal_consumed": False,
            "ranking_computed": False,
            "research_value_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, summary)

    print("=== S238 RESEARCH VALUE DISCRIMINATION AUDIT ===")
    print("artifact only: true")
    print("cards:", len(cards))
    print("available cases:", summary["available_case_count"])
    print("value classes:", summary["value_class_counts"])
    print(
        "unique dimension signatures:",
        summary["unique_dimension_signature_count"],
    )
    print(
        "dominant signature fraction:",
        f"{dominant_dim_fraction:.3f}",
    )
    print("constant dimensions:", constant_dimensions or "-")
    print("varying dimensions:", varying_dimensions or "-")
    print("hypothesis types:", len(hypothesis_types))
    print("resource burden pairs:", len(burden_pairs))
    print("class masks input variation:", class_masking)
    print("discrimination:", status)
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
