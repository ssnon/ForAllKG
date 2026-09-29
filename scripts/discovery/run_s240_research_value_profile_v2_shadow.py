from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def comparison_structure(features: dict[str, Any]) -> str:
    distinct = int(features.get("distinct_comparison_count") or 0)
    contrasts = int(features.get("explicit_contrast_comparison_count") or 0)

    if distinct <= 0:
        return "NO_EXPLICIT_COMPARISON"
    if contrasts <= 0:
        return "COMPARISON_WITHOUT_EXPLICIT_CONTRAST"
    if distinct == 1:
        return "SINGLE_EXPLICIT_CONTRAST"
    return "MULTIPLE_EXPLICIT_CONTRASTS"


def outcome_structure(features: dict[str, Any]) -> str:
    predicted = int(features.get("predicted_observation_count") or 0)
    falsifiers = int(features.get("falsification_criterion_count") or 0)
    success = int(features.get("success_pattern_count") or 0)
    failure = int(features.get("falsification_pattern_count") or 0)

    if min(predicted, falsifiers, success, failure) <= 0:
        return "ONE_SIDED_OR_INCOMPLETE"

    if predicted == 1 and falsifiers == 1 and success == 1 and failure == 1:
        return "SINGLE_TWO_SIDED_OUTCOME"

    balance = features.get("two_sided_pattern_balance_ratio")
    if balance == 1.0:
        return "MULTI_OUTCOME_BALANCED"

    return "MULTI_OUTCOME_ASYMMETRIC"


def observable_structure(features: dict[str, Any]) -> str:
    shared = int(features.get("shared_observable_count") or 0)
    union = int(features.get("observable_union_count") or 0)

    if union <= 0:
        return "NO_OBSERVABLE_STRUCTURE"
    if shared <= 0:
        return "DISJOINT_PREDICTION_FALSIFIER_OBSERVABLES"
    if union == 1 and shared == 1:
        return "SINGLE_SHARED_OBSERVABLE"
    if shared == union:
        return "MULTIPLE_SHARED_OBSERVABLES"
    return "PARTIAL_OBSERVABLE_OVERLAP"


def primary_alignment(features: dict[str, Any]) -> str:
    ratios = [
        features.get("shared_primary_coverage_ratio"),
        features.get("prediction_primary_coverage_ratio"),
        features.get("falsifier_primary_coverage_ratio"),
    ]
    available = [
        float(value)
        for value in ratios
        if value is not None
    ]

    if not available:
        return "PRIMARY_ALIGNMENT_UNRESOLVED"

    if all(value >= 0.999999 for value in available):
        return "FULL_PRIMARY_ALIGNMENT"

    if any(value > 0 for value in available):
        return "PARTIAL_PRIMARY_ALIGNMENT"

    return "NO_PRIMARY_ALIGNMENT"


def experimental_profile(features: dict[str, Any]) -> str:
    disposition = str(
        features.get("experimental_disposition") or "MISSING"
    )
    requires = bool(
        features.get("requires_candidate_concretization", False)
    )

    suffix = (
        "REQUIRES_CONCRETIZATION"
        if requires
        else "NO_CONCRETIZATION_REQUIRED"
    )
    return disposition.upper() + "__" + suffix


def resource_profile(features: dict[str, Any]) -> str:
    cost = str(features.get("relative_cost_burden") or "MISSING").upper()
    effort = str(features.get("relative_effort_burden") or "MISSING").upper()
    return f"COST_{cost}__EFFORT_{effort}"


def build_profile(features: dict[str, Any]) -> dict[str, str]:
    return {
        "comparison_structure":
            comparison_structure(features),
        "outcome_structure":
            outcome_structure(features),
        "observable_structure":
            observable_structure(features),
        "primary_alignment":
            primary_alignment(features),
        "experimental_profile":
            experimental_profile(features),
        "resource_profile":
            resource_profile(features),
    }


def profile_signature(profile: dict[str, str]) -> str:
    return json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S240 Research Value profile-v2 shadow. Converts S239 raw "
            "validation/feasibility features into interpretable, non-scalar "
            "candidate profiles. No overall score, ranking, or authority."
        )
    )
    p.add_argument(
        "--s239-summary",
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

    source = args.s239_summary.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise RuntimeError("missing S239 summary: " + str(source))
    if output.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(output)
        )

    s239 = load_json(source)

    rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    dimension_counts: dict[str, Counter[str]] = {
        "comparison_structure": Counter(),
        "outcome_structure": Counter(),
        "observable_structure": Counter(),
        "primary_alignment": Counter(),
        "experimental_profile": Counter(),
        "resource_profile": Counter(),
    }

    for row in s239.get("cards", []):
        if not isinstance(row, dict):
            continue

        features = row.get("features") or {}
        profile = build_profile(features)
        signature = profile_signature(profile)

        item = {
            "source_case_id": row.get("source_case_id"),
            "case_role": row.get("case_role"),
            "hypothesis_id": row.get("hypothesis_id"),
            "hypothesis_type": features.get("hypothesis_type"),
            "validation_strategy": features.get("validation_strategy"),
            "profile": profile,
        }
        rows.append(item)
        groups[signature].append(item)

        for name, value in profile.items():
            dimension_counts[name][value] += 1

    dominant_count = max(
        (len(group) for group in groups.values()),
        default=0,
    )
    dominant_fraction = (
        dominant_count / len(rows)
        if rows
        else 0.0
    )

    varying_dimensions = [
        name
        for name, counts in dimension_counts.items()
        if len(counts) > 1
    ]
    constant_dimensions = [
        name
        for name, counts in dimension_counts.items()
        if len(counts) <= 1
    ]

    if len(rows) < 2:
        status = "INSUFFICIENT_SAMPLE"
    elif len(groups) == 1:
        status = "PROFILE_V2_NON_DISCRIMINATIVE"
    elif dominant_fraction >= 0.80:
        status = "PROFILE_V2_LOW_DISCRIMINATION"
    else:
        status = "PROFILE_V2_DISCRIMINATION_OBSERVED"

    group_rows = []
    for _, group in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        group_rows.append(
            {
                "profile": group[0]["profile"],
                "card_count": len(group),
                "cards": [
                    {
                        "source_case_id": item["source_case_id"],
                        "hypothesis_id": item["hypothesis_id"],
                        "hypothesis_type": item["hypothesis_type"],
                        "validation_strategy": item["validation_strategy"],
                    }
                    for item in group
                ],
            }
        )

    payload = {
        "schema_version":
            "research-value-profile-v2-shadow-s240-v1",
        "source_s239_summary": str(source),
        "card_count": len(rows),
        "unique_profile_count": len(groups),
        "dominant_profile_count": dominant_count,
        "dominant_profile_fraction": dominant_fraction,
        "varying_profile_dimensions": varying_dimensions,
        "constant_profile_dimensions": constant_dimensions,
        "profile_dimension_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in dimension_counts.items()
        },
        "profile_discrimination_status": status,
        "profile_groups": group_rows,
        "cards": rows,
        "interpretation_policy": {
            "profile_is_overall_value_score": False,
            "profile_is_ranking": False,
            "higher_profile_count_is_better": False,
            "hypothesis_type_is_value_evidence": False,
            "validation_strategy_is_value_evidence": False,
            "novelty_signal_consumed": False,
            "research_value_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    write_json(output, payload)

    print("=== S240 RESEARCH VALUE PROFILE V2 SHADOW ===")
    print("artifact only: true")
    print("cards:", len(rows))
    print("unique profiles:", len(groups))
    print(
        "dominant profile fraction:",
        f"{dominant_fraction:.3f}",
    )
    print("varying profile dimensions:")
    for name in varying_dimensions:
        print(" ", name, "=>", dict(sorted(dimension_counts[name].items())))
    print("constant profile dimensions:")
    for name in constant_dimensions:
        print(" ", name, "=>", dict(sorted(dimension_counts[name].items())))
    print("profile discrimination:", status)
    print("overall value score computed: false")
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
