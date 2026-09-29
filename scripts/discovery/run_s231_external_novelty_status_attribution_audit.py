from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    ExternalNoveltyReport,
)


DECISIVE_NOVELTY_RANK = {
    "WELL_ESTABLISHED": 0,
    "LITERATURE_SUPPORTED_EXTENSION": 1,
    "NEW_COMBINATION_OF_KNOWN_EFFECTS": 2,
    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 3,
    "PLAUSIBLY_NOVEL": 4,
}

RELATION_BACKED = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}

GAP_LIKE = {
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
}

UNRESOLVED = {
    "TITLE_ONLY_NEIGHBORS",
    "INSUFFICIENT_METADATA",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_report(path: str | Path) -> ExternalNoveltyReport:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RuntimeError("missing novelty report: " + str(p))
    return ExternalNoveltyReport.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def card_map(report: ExternalNoveltyReport) -> dict[str, ExternalNoveltyCard]:
    return {
        row.hypothesis_id: row
        for row in report.cards
    }


def core_statuses(card: ExternalNoveltyCard) -> list[str]:
    core = [
        row.status
        for row in card.claim_reviews
        if str(row.importance) == "core"
    ]
    if core:
        return core
    return [row.status for row in card.claim_reviews]


def evidence_signature(card: ExternalNoveltyCard) -> dict[str, Any]:
    statuses = core_statuses(card)
    return {
        "card_status": card.status,
        "core_claim_count": len(statuses),
        "relation_backed_core_count": sum(
            status in RELATION_BACKED
            for status in statuses
        ),
        "gap_like_core_count": sum(
            status in GAP_LIKE
            for status in statuses
        ),
        "unresolved_core_count": sum(
            status in UNRESOLVED
            for status in statuses
        ),
        "conflicting_core_count": sum(
            status == "CONFLICTING_PRIOR_ART"
            for status in statuses
        ),
        "core_claim_statuses": statuses,
        "coverage_sufficient": bool(
            card.coverage.sufficient_for_absence_based_novelty
        ),
        "abstract_work_count": int(
            card.coverage.abstract_work_count
        ),
        "unique_work_count": int(
            card.coverage.unique_work_count
        ),
    }


def comparable_rank(status: str | None) -> int | None:
    if status is None:
        return None
    return DECISIVE_NOVELTY_RANK.get(status)


def transition_direction(
    before: str | None,
    after: str | None,
) -> str:
    if before == after:
        return "STABLE"
    br = comparable_rank(before)
    ar = comparable_rank(after)
    if br is None or ar is None:
        return "INCOMPARABLE"
    if ar > br:
        return "MORE_NOVEL"
    if ar < br:
        return "LESS_NOVEL"
    return "STABLE"


def attribution_class(
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    direction = transition_direction(
        before["card_status"],
        after["card_status"],
    )

    if direction == "STABLE":
        return "STATUS_STABLE"

    if direction == "INCOMPARABLE":
        return "STATUS_INCOMPARABLE"

    relation_delta = (
        after["relation_backed_core_count"]
        - before["relation_backed_core_count"]
    )
    unresolved_delta = (
        after["unresolved_core_count"]
        - before["unresolved_core_count"]
    )
    abstracts_delta = (
        after["abstract_work_count"]
        - before["abstract_work_count"]
    )

    if (
        direction == "MORE_NOVEL"
        and relation_delta > 0
    ):
        return "UPWARD_NOVELTY_DESPITE_MORE_RELATION_BACKED_CORE"

    if (
        direction == "MORE_NOVEL"
        and unresolved_delta < 0
        and abstracts_delta >= 0
    ):
        return "UPWARD_NOVELTY_AFTER_RESOLUTION"

    if direction == "MORE_NOVEL":
        return "UPWARD_NOVELTY_OTHER"

    if direction == "LESS_NOVEL":
        return "DOWNWARD_NOVELTY_WITH_STRONGER_PRIOR_ART"

    return "UNCLASSIFIED"


def compare_reports(
    *,
    case_id: str,
    scope: str,
    before_label: str,
    after_label: str,
    before_path: str,
    after_path: str,
) -> list[dict[str, Any]]:
    before_report = load_report(before_path)
    after_report = load_report(after_path)

    if (
        before_report.source_portfolio_id
        != after_report.source_portfolio_id
    ):
        raise RuntimeError(
            "source portfolio changed across comparison: "
            f"{case_id} | {scope}"
        )

    before_cards = card_map(before_report)
    after_cards = card_map(after_report)

    hypothesis_ids = sorted(
        set(before_cards) | set(after_cards)
    )

    rows: list[dict[str, Any]] = []
    for hypothesis_id in hypothesis_ids:
        b = before_cards.get(hypothesis_id)
        a = after_cards.get(hypothesis_id)

        if b is None or a is None:
            rows.append(
                {
                    "source_case_id": case_id,
                    "scope": scope,
                    "hypothesis_id": hypothesis_id,
                    "before_label": before_label,
                    "after_label": after_label,
                    "classification": "HYPOTHESIS_SET_CHANGED",
                    "before_present": b is not None,
                    "after_present": a is not None,
                }
            )
            continue

        b_sig = evidence_signature(b)
        a_sig = evidence_signature(a)

        rows.append(
            {
                "source_case_id": case_id,
                "scope": scope,
                "hypothesis_id": hypothesis_id,
                "before_label": before_label,
                "after_label": after_label,
                "before": b_sig,
                "after": a_sig,
                "status_direction": transition_direction(
                    b.status,
                    a.status,
                ),
                "classification": attribution_class(
                    b_sig,
                    a_sig,
                ),
                "relation_backed_core_delta": (
                    a_sig["relation_backed_core_count"]
                    - b_sig["relation_backed_core_count"]
                ),
                "gap_like_core_delta": (
                    a_sig["gap_like_core_count"]
                    - b_sig["gap_like_core_count"]
                ),
                "unresolved_core_delta": (
                    a_sig["unresolved_core_count"]
                    - b_sig["unresolved_core_count"]
                ),
                "abstract_work_delta": (
                    a_sig["abstract_work_count"]
                    - b_sig["abstract_work_count"]
                ),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S231 artifact-only external novelty status-attribution audit. "
            "No provider, retrieval, or LLM calls."
        )
    )
    parser.add_argument(
        "--s227-3-summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--s229-summary",
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

    s227_path = args.s227_3_summary.expanduser().resolve()
    s229_path = args.s229_summary.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    for path in (s227_path, s229_path):
        if not path.is_file():
            raise RuntimeError(
                "missing input: " + str(path)
            )

    s227 = load_json(s227_path)
    s229 = load_json(s229_path)

    s227_rows = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s227.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    }

    s229_rows = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s229.get("rows", [])
        if row.get("measurement_status") == "AVAILABLE"
    }

    keys = sorted(
        set(s227_rows) & set(s229_rows)
    )

    comparisons: list[dict[str, Any]] = []

    print("=== S231 EXTERNAL NOVELTY STATUS ATTRIBUTION AUDIT ===")
    print("artifact only: true")
    print("provider calls: false")
    print("literature retrieval: false")
    print("LLM calls: false")
    print("production selection changed: false")

    for case_id, scope in keys:
        c227 = s227_rows[(case_id, scope)]
        c229 = s229_rows[(case_id, scope)]

        print("\n" + "=" * 96)
        print(case_id, "|", scope)
        print("=" * 96)

        control_vs_treatment = compare_reports(
            case_id=case_id,
            scope=scope,
            before_label="S227_CONTROL_ORIGINAL_PACKET_REPLAY",
            after_label="S227_TREATMENT_RESOLVED_PACKET_REPLAY",
            before_path=c227["control_report_path"],
            after_path=c227["treatment_report_path"],
        )
        treatment_vs_integrated = compare_reports(
            case_id=case_id,
            scope=scope,
            before_label="S227_POSTHOC_TREATMENT",
            after_label="S229_INTEGRATED_PRE_REVIEW",
            before_path=c229["s227_treatment_report_path"],
            after_path=c229["integrated_report_path"],
        )

        rows = control_vs_treatment + treatment_vs_integrated
        comparisons.extend(rows)

        counts = Counter(
            row["classification"]
            for row in rows
        )
        for key, count in sorted(counts.items()):
            print(f"  {count:>3} {key}")

        flagged = [
            row
            for row in rows
            if row["classification"]
            in {
                "UPWARD_NOVELTY_DESPITE_MORE_RELATION_BACKED_CORE",
                "UPWARD_NOVELTY_AFTER_RESOLUTION",
                "UPWARD_NOVELTY_OTHER",
            }
        ]
        for row in flagged:
            before = row.get("before", {})
            after = row.get("after", {})
            print(
                " FLAG",
                row["hypothesis_id"],
                "|",
                before.get("card_status"),
                "->",
                after.get("card_status"),
                "| relation-backed",
                before.get("relation_backed_core_count"),
                "->",
                after.get("relation_backed_core_count"),
                "| unresolved",
                before.get("unresolved_core_count"),
                "->",
                after.get("unresolved_core_count"),
                "| abstracts",
                before.get("abstract_work_count"),
                "->",
                after.get("abstract_work_count"),
                "|",
                row["classification"],
            )

    counts = Counter(
        row["classification"]
        for row in comparisons
    )

    flagged_rows = [
        row
        for row in comparisons
        if row["classification"]
        in {
            "UPWARD_NOVELTY_DESPITE_MORE_RELATION_BACKED_CORE",
            "UPWARD_NOVELTY_AFTER_RESOLUTION",
            "UPWARD_NOVELTY_OTHER",
        }
    ]

    strong_incoherence = [
        row
        for row in comparisons
        if row["classification"]
        == "UPWARD_NOVELTY_DESPITE_MORE_RELATION_BACKED_CORE"
    ]

    summary = {
        "schema_version":
            "external-novelty-status-attribution-audit-s231-v1",
        "source_s227_3_summary": str(s227_path),
        "source_s229_summary": str(s229_path),
        "comparison_count": len(comparisons),
        "classification_counts":
            dict(sorted(counts.items())),
        "flagged_upward_novelty_count":
            len(flagged_rows),
        "strong_directional_incoherence_count":
            len(strong_incoherence),
        "comparisons": comparisons,
        "interpretation_policy": {
            "status_order_scope":
                "Only decisive non-conflict, non-insufficient statuses are ordinally compared.",
            "hard_monotonicity_enforced":
                False,
            "strong_directional_incoherence":
                (
                    "Hypothesis-level novelty became more novel while "
                    "the number of relation-backed core claims increased."
                ),
            "upward_after_resolution":
                (
                    "Hypothesis-level novelty became more novel while "
                    "unresolved core claims decreased and abstract coverage "
                    "did not decrease. This is audit-worthy, not automatically wrong."
                ),
            "provider_calls": False,
            "literature_retrieval_performed": False,
            "llm_calls": False,
            "production_selection_changed": False,
        },
    }

    write_json(out_path, summary)

    print("\nS231 audit complete")
    print("comparisons:", len(comparisons))
    print("classification counts:")
    for key, count in sorted(counts.items()):
        print(" ", count, key)
    print(
        "flagged upward novelty:",
        len(flagged_rows),
    )
    print(
        "strong directional incoherence:",
        len(strong_incoherence),
    )
    print("provider calls: false")
    print("LLM calls: false")
    print("production selection changed: false")
    print("artifact:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
