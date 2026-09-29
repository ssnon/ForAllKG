from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evidence_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("eligible_statement_count"),
        row.get("used_statement_count"),
        row.get("unused_eligible_statement_count"),
        row.get("exact_premise_set_duplicate_group_count"),
        row.get("max_pairwise_statement_jaccard"),
        row.get("evidence_recommendation"),
    )


def classify_evidence_discrimination(
    *,
    available_count: int,
    recommendation_count: int,
    signature_count: int,
    dominant_signature_fraction: float,
) -> str:
    if available_count < 2:
        return "INSUFFICIENT_SAMPLE"
    if recommendation_count == 1 and signature_count == 1:
        return "NON_DISCRIMINATIVE"
    if signature_count <= 2 or dominant_signature_fraction >= 0.8:
        return "LOW_DISCRIMINATION"
    return "DISCRIMINATION_OBSERVED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Artifact-only prospective cohort audit for portfolio evidence "
            "and operator diversity shadows. Measures discrimination and "
            "coverage only; does not rank or change selection."
        )
    )
    p.add_argument("--collector", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = args.collector.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise RuntimeError("missing collector: " + str(source))
    if output.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(output)
        )

    collector = load_json(source)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case in collector.get("cases", []):
        if not isinstance(case, dict):
            continue

        diversity = case.get("portfolio_diversity") or {}
        if diversity.get("measurement_status") != "AVAILABLE":
            skipped.append(
                {
                    "source_case_id": case.get("source_case_id"),
                    "measurement_status": diversity.get("measurement_status"),
                }
            )
            continue

        rows.append(
            {
                "source_case_id": case.get("source_case_id"),
                "case_role": case.get("case_role"),
                "eligible_statement_count":
                    diversity.get("eligible_statement_count"),
                "used_statement_count":
                    diversity.get("used_statement_count"),
                "unused_eligible_statement_count":
                    diversity.get("unused_eligible_statement_count"),
                "exact_premise_set_duplicate_group_count":
                    diversity.get("exact_premise_set_duplicate_group_count"),
                "max_pairwise_statement_jaccard":
                    diversity.get("max_pairwise_statement_jaccard"),
                "evidence_recommendation":
                    diversity.get("evidence_recommendation"),
                "operator_opportunity_count":
                    diversity.get("operator_opportunity_count"),
                "operator_recommendation":
                    diversity.get("operator_recommendation"),
                "operator_measurement_scope":
                    diversity.get("operator_measurement_scope"),
                "artifact":
                    diversity.get("artifact"),
            }
        )

    recommendation_counts = Counter(
        str(row.get("evidence_recommendation") or "MISSING")
        for row in rows
    )
    operator_recommendation_counts = Counter(
        str(row.get("operator_recommendation") or "MISSING")
        for row in rows
    )
    operator_scope_counts = Counter(
        str(row.get("operator_measurement_scope") or "MISSING")
        for row in rows
    )

    signatures = Counter(evidence_signature(row) for row in rows)
    dominant = max(signatures.values(), default=0)
    dominant_fraction = dominant / len(rows) if rows else 0.0

    evidence_status = classify_evidence_discrimination(
        available_count=len(rows),
        recommendation_count=len(recommendation_counts),
        signature_count=len(signatures),
        dominant_signature_fraction=dominant_fraction,
    )

    operator_measured_rows = [
        row
        for row in rows
        if str(row.get("operator_measurement_scope") or "")
        not in {
            "",
            "MISSING",
            "NO_NOVELTY_GAP_PLAN_SUPPLIED",
        }
    ]
    operator_nonzero_rows = [
        row
        for row in rows
        if int(row.get("operator_opportunity_count") or 0) > 0
    ]

    if not rows:
        operator_status = "INSUFFICIENT_SAMPLE"
    elif not operator_measured_rows:
        operator_status = "NOT_MEASURED_WITH_GAP_CONTEXT"
    elif not operator_nonzero_rows:
        operator_status = "NO_OPERATOR_OPPORTUNITY_OBSERVED"
    elif len(operator_recommendation_counts) == 1:
        operator_status = "LOW_DISCRIMINATION"
    else:
        operator_status = "DISCRIMINATION_OBSERVED"

    metric_counts = {
        "eligible_statement_count": Counter(
            str(row.get("eligible_statement_count")) for row in rows
        ),
        "used_statement_count": Counter(
            str(row.get("used_statement_count")) for row in rows
        ),
        "unused_eligible_statement_count": Counter(
            str(row.get("unused_eligible_statement_count")) for row in rows
        ),
        "exact_premise_set_duplicate_group_count": Counter(
            str(row.get("exact_premise_set_duplicate_group_count")) for row in rows
        ),
        "max_pairwise_statement_jaccard": Counter(
            str(row.get("max_pairwise_statement_jaccard")) for row in rows
        ),
    }

    varying_metrics = [
        name for name, counts in metric_counts.items()
        if len(counts) > 1
    ]
    constant_metrics = [
        name for name, counts in metric_counts.items()
        if len(counts) <= 1
    ]

    payload = {
        "schema_version":
            "portfolio-diversity-prospective-audit-v1",
        "source_collector": str(source),
        "available_case_count": len(rows),
        "skipped_cases": skipped,
        "evidence": {
            "recommendation_counts":
                dict(sorted(recommendation_counts.items())),
            "unique_signature_count": len(signatures),
            "dominant_signature_fraction": dominant_fraction,
            "varying_metrics": varying_metrics,
            "constant_metrics": constant_metrics,
            "metric_value_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in metric_counts.items()
            },
            "discrimination_status": evidence_status,
        },
        "operator": {
            "recommendation_counts":
                dict(sorted(operator_recommendation_counts.items())),
            "measurement_scope_counts":
                dict(sorted(operator_scope_counts.items())),
            "measured_with_gap_context_count":
                len(operator_measured_rows),
            "nonzero_opportunity_case_count":
                len(operator_nonzero_rows),
            "validation_status": operator_status,
        },
        "cases": rows,
        "interpretation_policy": {
            "unused_evidence_implies_relevance": False,
            "high_jaccard_implies_bad_science": False,
            "operator_absence_when_unmeasured_is_negative": False,
            "ranking_computed": False,
            "portfolio_selection_changed": False,
            "production_selection_changed": False,
        },
    }

    write_json(output, payload)

    print("=== PORTFOLIO DIVERSITY PROSPECTIVE AUDIT ===")
    print("artifact only: true")
    print("available cases:", len(rows))
    print("evidence recommendations:", dict(sorted(recommendation_counts.items())))
    print("unique evidence signatures:", len(signatures))
    print("dominant evidence signature fraction:", f"{dominant_fraction:.3f}")
    print("varying evidence metrics:", varying_metrics or "-")
    print("evidence discrimination:", evidence_status)
    print("operator scopes:", dict(sorted(operator_scope_counts.items())))
    print("operator nonzero cases:", len(operator_nonzero_rows))
    print("operator validation:", operator_status)
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
