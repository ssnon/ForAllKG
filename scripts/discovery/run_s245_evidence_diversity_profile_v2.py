from __future__ import annotations

import argparse
import json
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


def ratio(num: int | float, den: int | float) -> float | None:
    if not den:
        return None
    return round(float(num) / float(den), 6)


def build_profile(evidence: dict[str, Any]) -> dict[str, Any]:
    cards = [
        row
        for row in evidence.get("cards", [])
        if isinstance(row, dict)
    ]
    groups = [
        row
        for row in evidence.get("exact_premise_set_groups", [])
        if isinstance(row, dict)
    ]

    hypothesis_count = int(evidence.get("hypothesis_count") or len(cards) or 0)
    used = int(evidence.get("used_statement_count") or 0)
    eligible = int(evidence.get("eligible_statement_count") or 0)
    shared_core = int(evidence.get("shared_core_statement_count") or 0)

    zero_unique = sum(
        int(row.get("portfolio_unique_premise_count") or 0) == 0
        for row in cards
    )
    unique_counts = [
        int(row.get("portfolio_unique_premise_count") or 0)
        for row in cards
    ]

    duplicate_hypothesis_ids = set()
    for group in groups:
        for hid in group.get("hypothesis_ids", []):
            duplicate_hypothesis_ids.add(str(hid))

    distinct_sets = int(evidence.get("distinct_premise_set_count") or 0)
    unused = len(evidence.get("unused_eligible_statement_ids") or [])

    return {
        "hypothesis_count": hypothesis_count,
        "eligible_statement_count": eligible,
        "used_statement_count": used,
        "unused_eligible_statement_count": unused,
        "eligible_statement_coverage":
            evidence.get("eligible_statement_coverage"),

        "shared_core_statement_count": shared_core,
        "shared_core_fraction_of_used":
            ratio(shared_core, used),

        "zero_unique_support_hypothesis_count": zero_unique,
        "zero_unique_support_hypothesis_fraction":
            ratio(zero_unique, hypothesis_count),

        "mean_unique_premise_count_per_hypothesis": (
            round(sum(unique_counts) / len(unique_counts), 6)
            if unique_counts
            else None
        ),

        "distinct_premise_set_count": distinct_sets,
        "distinct_premise_set_fraction":
            ratio(distinct_sets, hypothesis_count),

        "exact_premise_set_duplicate_group_count":
            int(evidence.get("exact_premise_set_duplicate_group_count") or 0),
        "exact_duplicate_hypothesis_count":
            len(duplicate_hypothesis_ids),
        "exact_duplicate_hypothesis_fraction":
            ratio(len(duplicate_hypothesis_ids), hypothesis_count),

        "mean_pairwise_statement_jaccard":
            evidence.get("mean_pairwise_statement_jaccard"),
        "max_pairwise_statement_jaccard":
            evidence.get("max_pairwise_statement_jaccard"),

        "multi_paper_used_statement_count":
            evidence.get("multi_paper_used_statement_count"),
        "mean_papers_per_used_statement":
            evidence.get("mean_papers_per_used_statement"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "S245 descriptive Evidence Diversity profile-v2. "
            "Replaces unvalidated action labels with raw portfolio evidence "
            "structure. No recommendation, ranking, or selection authority."
        )
    )
    p.add_argument("--collector", required=True, type=Path)
    p.add_argument("--s244-summary", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    collector_path = args.collector.expanduser().resolve()
    s244_path = args.s244_summary.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for path in (collector_path, s244_path):
        if not path.is_file():
            raise RuntimeError("missing input: " + str(path))
    if output_path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(output_path)
        )

    collector = load_json(collector_path)
    s244 = load_json(s244_path)

    blind_by_case = {
        str(row.get("source_case_id")): row
        for row in s244.get("rows", [])
        if isinstance(row, dict)
    }

    rows: list[dict[str, Any]] = []

    for case in collector.get("cases", []):
        if not isinstance(case, dict):
            continue

        diversity = case.get("portfolio_diversity") or {}
        if diversity.get("measurement_status") != "AVAILABLE":
            continue

        artifact = diversity.get("artifact")
        if not artifact:
            continue

        run_dir = Path(str(artifact)).expanduser().resolve().parent
        evidence_path = run_dir / "hypothesis_axis_a4.evidence_diversity.json"
        if not evidence_path.is_file():
            raise RuntimeError(
                "missing evidence diversity report: " + str(evidence_path)
            )

        evidence = load_json(evidence_path)
        case_id = str(case.get("source_case_id") or "UNKNOWN")

        blind = blind_by_case.get(case_id) or {}
        blind_review = blind.get("blind_review") or {}

        rows.append(
            {
                "source_case_id": case_id,
                "case_role": case.get("case_role"),
                "legacy_shadow_recommendation":
                    diversity.get("evidence_recommendation"),
                "profile_v2": build_profile(evidence),
                "blind_validity_reference": {
                    "disposition": blind_review.get("disposition"),
                    "reasons": blind_review.get("reasons", []),
                    "legacy_recommendation_agreement":
                        blind.get("agreement"),
                },
            }
        )

    metric_names = sorted(
        {
            key
            for row in rows
            for key in row["profile_v2"].keys()
        }
    )

    varying = []
    constant = []
    for name in metric_names:
        values = {
            json.dumps(
                row["profile_v2"].get(name),
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        }
        if len(values) > 1:
            varying.append(name)
        else:
            constant.append(name)

    payload = {
        "schema_version":
            "portfolio-evidence-diversity-profile-v2-s245-v1",
        "source_collector": str(collector_path),
        "source_s244_summary": str(s244_path),
        "case_count": len(rows),
        "varying_profile_metrics": varying,
        "constant_profile_metrics": constant,
        "cases": rows,
        "interpretation_policy": {
            "legacy_v1_recommendation_validated": False,
            "legacy_NO_ACTION_is_authoritative": False,
            "legacy_REDUNDANCY_REVIEW_is_authoritative": False,
            "profile_v2_is_descriptive_only": True,
            "unused_evidence_implies_relevance": False,
            "high_overlap_implies_bad_science": False,
            "overall_diversity_score_computed": False,
            "ranking_computed": False,
            "portfolio_selection_changed": False,
            "production_selection_changed": False,
        },
    }

    write_json(output_path, payload)

    print("=== S245 EVIDENCE DIVERSITY PROFILE V2 ===")
    print("artifact only: true")
    print("cases:", len(rows))
    print("varying profile metrics:", len(varying))
    for name in varying:
        print(" ", name)
    print("constant profile metrics:", len(constant))
    print("legacy action recommendation validated: false")
    print("profile v2 descriptive only: true")
    print("ranking computed: false")
    print("production selection changed: false")
    print("artifact:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
