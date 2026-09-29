from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTRAST_MARKERS = (
    "alternative",
    "compare",
    "versus",
    " vs ",
    "across",
    "matched",
    "while varying",
    "control",
)


def load_json(path: Path) -> Any:
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


def norm(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def load_hypothesis_cards(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("hypotheses", []) if isinstance(payload, dict) else []
    return {
        str(row["hypothesis_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("hypothesis_id")
    }


def load_hypothesis_index(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return result

    for path in sorted(directory.glob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        hypothesis_id = payload.get("hypothesis_id")
        if not hypothesis_id:
            continue

        key = str(hypothesis_id)
        if key in result and result[key] != payload:
            raise RuntimeError(
                f"conflicting artifacts for hypothesis {key} in {directory}"
            )
        result[key] = payload

    return result


def observable_set(rows: list[dict[str, Any]]) -> set[str]:
    return {
        norm(row.get("observable"))
        for row in rows
        if isinstance(row, dict) and norm(row.get("observable"))
    }


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def feature_vector(
    *,
    card: dict[str, Any],
    spec: dict[str, Any],
    experimental: dict[str, Any],
) -> dict[str, Any]:
    predicted_rows = [
        row
        for row in card.get("predicted_observations", [])
        if isinstance(row, dict)
    ]
    falsifier_rows = [
        row
        for row in card.get("falsification_criteria", [])
        if isinstance(row, dict)
    ]

    predicted = observable_set(predicted_rows)
    falsifier = observable_set(falsifier_rows)
    shared = predicted & falsifier
    union = predicted | falsifier

    primary = {
        norm(value)
        for value in spec.get("primary_observables", [])
        if norm(value)
    }

    required_comparisons = [
        norm(value)
        for value in spec.get("required_comparisons", [])
        if norm(value)
    ]
    distinct_comparisons = set(required_comparisons)

    contrast_comparisons = {
        value
        for value in distinct_comparisons
        if any(marker in f" {value} " for marker in CONTRAST_MARKERS)
    }

    success_patterns = [
        norm(value)
        for value in spec.get("success_patterns", [])
        if norm(value)
    ]
    falsification_patterns = [
        norm(value)
        for value in spec.get("falsification_patterns", [])
        if norm(value)
    ]

    pattern_max = max(
        len(success_patterns),
        len(falsification_patterns),
        0,
    )
    pattern_min = min(
        len(success_patterns),
        len(falsification_patterns),
    ) if pattern_max else 0

    return {
        "hypothesis_type":
            str(card.get("hypothesis_type") or "MISSING"),
        "validation_strategy":
            str(spec.get("validation_strategy") or "MISSING"),
        "requires_candidate_concretization":
            bool(spec.get("requires_candidate_concretization", False)),

        "predicted_observation_count":
            len(predicted_rows),
        "falsification_criterion_count":
            len(falsifier_rows),
        "prediction_observable_count":
            len(predicted),
        "falsifier_observable_count":
            len(falsifier),
        "shared_observable_count":
            len(shared),
        "observable_union_count":
            len(union),
        "observable_overlap_ratio":
            ratio(len(shared), len(union)),

        "primary_observable_count":
            len(primary),
        "shared_primary_observable_count":
            len(shared & primary),
        "shared_primary_coverage_ratio":
            ratio(len(shared & primary), len(shared)),
        "prediction_primary_coverage_ratio":
            ratio(len(predicted & primary), len(predicted)),
        "falsifier_primary_coverage_ratio":
            ratio(len(falsifier & primary), len(falsifier)),

        "required_comparison_count":
            len(required_comparisons),
        "distinct_comparison_count":
            len(distinct_comparisons),
        "explicit_contrast_comparison_count":
            len(contrast_comparisons),
        "explicit_contrast_fraction":
            ratio(len(contrast_comparisons), len(distinct_comparisons)),

        "success_pattern_count":
            len(success_patterns),
        "falsification_pattern_count":
            len(falsification_patterns),
        "two_sided_pattern_balance_ratio":
            ratio(pattern_min, pattern_max),

        "experimental_disposition":
            str(experimental.get("disposition") or "MISSING"),
        "relative_cost_burden":
            str(experimental.get("relative_cost_burden") or "MISSING"),
        "relative_effort_burden":
            str(experimental.get("relative_effort_burden") or "MISSING"),
    }


def signature(vector: dict[str, Any]) -> str:
    return json.dumps(
        vector,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S239 Research Value raw-feature shadow. Exposes deterministic "
            "candidate-level validation/feasibility features to determine "
            "whether S238 non-discrimination is caused by lossy RV-v1 "
            "compression or homogeneous upstream artifacts."
        )
    )
    p.add_argument(
        "--collector",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--s238-summary",
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

    collector_path = args.collector.expanduser().resolve()
    s238_path = args.s238_summary.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for path in (collector_path, s238_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))

    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(output_path)
        )

    collector = load_json(collector_path)
    s238 = load_json(s238_path)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case in collector.get("cases", []):
        if not isinstance(case, dict):
            continue

        rv = case.get("research_value") or {}
        if rv.get("measurement_status") != "AVAILABLE":
            skipped.append(
                {
                    "source_case_id": case.get("source_case_id"),
                    "reason": rv.get("measurement_status"),
                }
            )
            continue

        artifact_raw = rv.get("artifact")
        if not artifact_raw:
            skipped.append(
                {
                    "source_case_id": case.get("source_case_id"),
                    "reason": "MISSING_RV_ARTIFACT_PATH",
                }
            )
            continue

        run_dir = Path(artifact_raw).expanduser().resolve().parent
        portfolio_path = run_dir / "novelty_refinement_a6.portfolio.json"
        validation_dir = run_dir / "feasibility_final" / "validation"
        experimental_dir = run_dir / "feasibility_final" / "experimental"

        if not portfolio_path.is_file():
            raise RuntimeError(
                "missing refined portfolio: " + str(portfolio_path)
            )

        cards = load_hypothesis_cards(portfolio_path)
        specs = load_hypothesis_index(validation_dir)
        experiments = load_hypothesis_index(experimental_dir)

        common_ids = sorted(set(cards) & set(specs) & set(experiments))

        for hypothesis_id in common_ids:
            vector = feature_vector(
                card=cards[hypothesis_id],
                spec=specs[hypothesis_id],
                experimental=experiments[hypothesis_id],
            )
            rows.append(
                {
                    "source_case_id":
                        str(case.get("source_case_id") or "UNKNOWN"),
                    "case_role":
                        str(case.get("case_role") or ""),
                    "hypothesis_id":
                        hypothesis_id,
                    "features":
                        vector,
                }
            )

    sig_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sig_groups[signature(row["features"])].append(row)

    dominant_count = max(
        (len(group) for group in sig_groups.values()),
        default=0,
    )
    dominant_fraction = (
        dominant_count / len(rows)
        if rows
        else 0.0
    )

    feature_names = sorted(
        {
            key
            for row in rows
            for key in row["features"]
        }
    )

    feature_value_counts: dict[str, dict[str, int]] = {}
    constant_features = []
    varying_features = []

    for name in feature_names:
        counts = Counter(
            json.dumps(
                row["features"].get(name),
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        )
        feature_value_counts[name] = dict(sorted(counts.items()))
        if len(counts) <= 1:
            constant_features.append(name)
        else:
            varying_features.append(name)

    unique_count = len(sig_groups)

    if len(rows) < 2:
        diagnosis = "INSUFFICIENT_SAMPLE"
    elif unique_count == 1:
        diagnosis = "UPSTREAM_FEATURE_HOMOGENEITY"
    elif (
        str(s238.get("discrimination_status"))
        == "NON_DISCRIMINATIVE"
    ):
        diagnosis = "RV_V1_COMPRESSION_LOSS"
    else:
        diagnosis = "RAW_FEATURE_VARIATION_PRESENT"

    signature_groups = []
    for _, group in sorted(
        sig_groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        signature_groups.append(
            {
                "card_count": len(group),
                "features": group[0]["features"],
                "cards": [
                    {
                        "source_case_id": row["source_case_id"],
                        "case_role": row["case_role"],
                        "hypothesis_id": row["hypothesis_id"],
                    }
                    for row in group
                ],
            }
        )

    summary = {
        "schema_version":
            "research-value-raw-feature-shadow-s239-v1",
        "source_collector": str(collector_path),
        "source_s238_summary": str(s238_path),
        "card_count": len(rows),
        "unique_raw_feature_signature_count": unique_count,
        "dominant_raw_feature_signature_count": dominant_count,
        "dominant_raw_feature_signature_fraction": dominant_fraction,
        "constant_feature_count": len(constant_features),
        "varying_feature_count": len(varying_features),
        "constant_features": constant_features,
        "varying_features": varying_features,
        "feature_value_counts": feature_value_counts,
        "diagnosis": diagnosis,
        "signature_groups": signature_groups,
        "cards": rows,
        "skipped_cases": skipped,
        "interpretation_policy": {
            "RV_V1_COMPRESSION_LOSS": (
                "Raw validation/feasibility features vary across hypotheses "
                "but RV-v1 collapsed them to a non-discriminative signature. "
                "A richer RV representation is justified."
            ),
            "UPSTREAM_FEATURE_HOMOGENEITY": (
                "Raw available features are themselves identical. Refining "
                "RV-v1 thresholds cannot create meaningful discrimination; "
                "new scientific/experimental information is required."
            ),
            "RAW_FEATURE_VARIATION_PRESENT": (
                "Raw candidate-level variation exists; no ranking validity "
                "claim is made."
            ),
            "INSUFFICIENT_SAMPLE": (
                "Fewer than two complete candidate records are available."
            ),
            "numeric_value_score_computed": False,
            "ranking_computed": False,
            "novelty_signal_consumed": False,
            "research_value_selection_authority": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, summary)

    print("=== S239 RESEARCH VALUE RAW-FEATURE SHADOW ===")
    print("artifact only: true")
    print("cards:", len(rows))
    print("unique raw feature signatures:", unique_count)
    print(
        "dominant raw feature fraction:",
        f"{dominant_fraction:.3f}",
    )
    print("varying features:", len(varying_features))
    for name in varying_features:
        print(" ", name, "=>", feature_value_counts[name])
    print("constant features:", len(constant_features))
    print("diagnosis:", diagnosis)
    print("numeric value score computed: false")
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
