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


RELATION_BACKED_RELATIONSHIPS = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}

RELATION_BACKED_STATUSES = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}


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


def load_report(path: str | Path) -> ExternalNoveltyReport:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RuntimeError(
            "missing external novelty report: " + str(p)
        )
    return ExternalNoveltyReport.model_validate_json(
        p.read_text(encoding="utf-8")
    )


def card_for(
    report: ExternalNoveltyReport,
    hypothesis_id: str,
) -> ExternalNoveltyCard:
    matches = [
        card
        for card in report.cards
        if card.hypothesis_id == hypothesis_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one card for {hypothesis_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def review_map(card: ExternalNoveltyCard) -> dict[str, Any]:
    return {
        review.claim_id: review
        for review in card.claim_reviews
    }


def match_map(review) -> dict[str, Any]:
    return {
        match.work_id: match
        for match in review.matches
    }


def is_relation_backed_match(match, policy) -> bool:
    if match.relationship == "DIRECT_PRIOR_ART":
        return (
            match.abstract_available
            and match.confidence
            >= policy.direct_match_confidence
        )
    if match.relationship == "PARTIAL_PRIOR_ART":
        return (
            match.abstract_available
            and match.confidence
            >= policy.min_match_confidence
        )
    return False


def relation_backed_work_ids(review, policy) -> set[str]:
    return {
        match.work_id
        for match in review.matches
        if is_relation_backed_match(match, policy)
    }


def coverage_payload(review) -> dict[str, int]:
    cov = review.coverage
    return {
        "query_count": int(cov.query_count),
        "successful_query_count":
            int(cov.successful_query_count),
        "unique_work_count":
            int(cov.unique_work_count),
        "abstract_work_count":
            int(cov.abstract_work_count),
        "reviewed_work_count":
            int(cov.reviewed_work_count),
    }


def coverage_non_decreasing(
    before: dict[str, int],
    after: dict[str, int],
) -> bool:
    keys = (
        "successful_query_count",
        "unique_work_count",
        "abstract_work_count",
    )
    return all(
        after[key] >= before[key]
        for key in keys
    )


def classify_claim_drift(
    *,
    before_review,
    after_review,
    before_policy,
    after_policy,
) -> tuple[str, dict[str, Any]]:
    before_status = before_review.status
    after_status = after_review.status

    before_cov = coverage_payload(before_review)
    after_cov = coverage_payload(after_review)

    before_matches = match_map(before_review)
    after_matches = match_map(after_review)

    before_rb = relation_backed_work_ids(
        before_review,
        before_policy,
    )
    after_rb = relation_backed_work_ids(
        after_review,
        after_policy,
    )

    lost_rb = sorted(before_rb - after_rb)
    gained_rb = sorted(after_rb - before_rb)

    relabeled = []
    disappeared = []

    for work_id in lost_rb:
        b = before_matches[work_id]
        a = after_matches.get(work_id)

        if a is None:
            disappeared.append(
                {
                    "work_id": work_id,
                    "before_relationship":
                        b.relationship,
                    "before_confidence":
                        float(b.confidence),
                    "before_abstract_available":
                        bool(b.abstract_available),
                }
            )
            continue

        relabeled.append(
            {
                "work_id": work_id,
                "before_relationship":
                    b.relationship,
                "after_relationship":
                    a.relationship,
                "before_confidence":
                    float(b.confidence),
                "after_confidence":
                    float(a.confidence),
                "before_abstract_available":
                    bool(b.abstract_available),
                "after_abstract_available":
                    bool(a.abstract_available),
                "same_title":
                    b.title == a.title,
                "same_doi":
                    (b.doi or "") == (a.doi or ""),
            }
        )

    same_work_relationship_changes = []
    for work_id in sorted(
        set(before_matches) & set(after_matches)
    ):
        b = before_matches[work_id]
        a = after_matches[work_id]
        if (
            b.relationship != a.relationship
            or abs(
                float(b.confidence)
                - float(a.confidence)
            )
            > 1e-12
        ):
            same_work_relationship_changes.append(
                {
                    "work_id": work_id,
                    "before_relationship":
                        b.relationship,
                    "after_relationship":
                        a.relationship,
                    "before_confidence":
                        float(b.confidence),
                    "after_confidence":
                        float(a.confidence),
                    "abstract_before":
                        bool(b.abstract_available),
                    "abstract_after":
                        bool(a.abstract_available),
                    "same_title":
                        b.title == a.title,
                    "same_doi":
                        (b.doi or "") == (a.doi or ""),
                }
            )

    detail = {
        "before_status": before_status,
        "after_status": after_status,
        "claim_text_same":
            before_review.claim_text
            == after_review.claim_text,
        "importance_same":
            before_review.importance
            == after_review.importance,
        "before_coverage": before_cov,
        "after_coverage": after_cov,
        "coverage_non_decreasing":
            coverage_non_decreasing(
                before_cov,
                after_cov,
            ),
        "before_relation_backed_work_ids":
            sorted(before_rb),
        "after_relation_backed_work_ids":
            sorted(after_rb),
        "lost_relation_backed_work_ids":
            lost_rb,
        "gained_relation_backed_work_ids":
            gained_rb,
        "lost_relation_backed_same_work_relabeled":
            relabeled,
        "lost_relation_backed_match_disappeared":
            disappeared,
        "same_work_relationship_changes":
            same_work_relationship_changes,
        "before_reason_codes":
            list(before_review.reason_codes),
        "after_reason_codes":
            list(after_review.reason_codes),
        "before_reviewer_unknown_work_ids":
            list(
                before_review.reviewer_unknown_work_ids
            ),
        "after_reviewer_unknown_work_ids":
            list(
                after_review.reviewer_unknown_work_ids
            ),
    }

    if before_status == after_status:
        return "CLAIM_STATUS_STABLE", detail

    if (
        before_status in RELATION_BACKED_STATUSES
        and after_status
        not in RELATION_BACKED_STATUSES
    ):
        if relabeled:
            return (
                "RELATION_BACKED_LOST_SAME_WORK_RELABELED",
                detail,
            )

        if disappeared:
            if detail["coverage_non_decreasing"]:
                return (
                    "RELATION_BACKED_LOST_MATCH_DISAPPEARED_WITH_NONDECREASING_COVERAGE",
                    detail,
                )
            return (
                "RELATION_BACKED_LOST_MATCH_DISAPPEARED_WITH_COVERAGE_CHANGE",
                detail,
            )

        if detail["coverage_non_decreasing"]:
            return (
                "RELATION_BACKED_LOST_WITH_NONDECREASING_COVERAGE",
                detail,
            )

        return (
            "RELATION_BACKED_LOST_WITH_COVERAGE_DECREASE",
            detail,
        )

    if (
        before_status
        not in RELATION_BACKED_STATUSES
        and after_status
        in RELATION_BACKED_STATUSES
    ):
        return "RELATION_BACKED_GAINED", detail

    return "OTHER_CLAIM_STATUS_DRIFT", detail


def packet_root_cause(
    claim_rows: list[dict[str, Any]],
) -> str:
    classes = {
        row["classification"]
        for row in claim_rows
    }

    if (
        "RELATION_BACKED_LOST_SAME_WORK_RELABELED"
        in classes
    ):
        return "REVIEWER_RELATIONSHIP_DRIFT"

    if any(
        cls.startswith(
            "RELATION_BACKED_LOST_MATCH_DISAPPEARED"
        )
        for cls in classes
    ):
        return "MATCH_SET_OR_REVIEWER_OMISSION_DRIFT"

    if (
        "RELATION_BACKED_LOST_WITH_NONDECREASING_COVERAGE"
        in classes
    ):
        return "REVIEWER_OR_CANDIDATE_COMPOSITION_DRIFT"

    if (
        "RELATION_BACKED_LOST_WITH_COVERAGE_DECREASE"
        in classes
    ):
        return "COVERAGE_OR_RANKING_DRIFT"

    return "OTHER_OR_NO_CLAIM_LEVEL_DRIFT"


def report_paths_for_comparison(
    *,
    comparison: dict[str, Any],
    s227_rows: dict[tuple[str, str], dict[str, Any]],
    s229_rows: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str]:
    key = (
        str(comparison["source_case_id"]),
        str(comparison["scope"]),
    )

    before_label = comparison["before_label"]
    after_label = comparison["after_label"]

    if (
        before_label
        == "S227_CONTROL_ORIGINAL_PACKET_REPLAY"
        and after_label
        == "S227_TREATMENT_RESOLVED_PACKET_REPLAY"
    ):
        row = s227_rows[key]
        return (
            row["control_report_path"],
            row["treatment_report_path"],
        )

    if (
        before_label
        == "S227_POSTHOC_TREATMENT"
        and after_label
        == "S229_INTEGRATED_PRE_REVIEW"
    ):
        row = s229_rows[key]
        return (
            row["s227_treatment_report_path"],
            row["integrated_report_path"],
        )

    raise RuntimeError(
        "unsupported comparison labels: "
        f"{before_label} -> {after_label}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S232 claim-level attribution audit for the upward "
            "novelty transitions found by S231. Artifact only."
        )
    )
    parser.add_argument(
        "--s231-summary",
        required=True,
        type=Path,
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

    s231_path = (
        args.s231_summary.expanduser().resolve()
    )
    s227_path = (
        args.s227_3_summary.expanduser().resolve()
    )
    s229_path = (
        args.s229_summary.expanduser().resolve()
    )
    out_path = args.output.expanduser().resolve()

    for path in (
        s231_path,
        s227_path,
        s229_path,
    ):
        if not path.is_file():
            raise RuntimeError(
                "missing input: " + str(path)
            )

    s231 = load_json(s231_path)
    s227 = load_json(s227_path)
    s229 = load_json(s229_path)

    s227_rows = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s227.get("rows", [])
        if row.get("measurement_status")
        == "AVAILABLE"
    }

    s229_rows = {
        (
            str(row["source_case_id"]),
            str(row["scope"]),
        ): row
        for row in s229.get("rows", [])
        if row.get("measurement_status")
        == "AVAILABLE"
    }

    upward = [
        row
        for row in s231.get(
            "comparisons",
            [],
        )
        if str(
            row.get("classification", "")
        ).startswith("UPWARD_NOVELTY")
    ]

    outputs: list[dict[str, Any]] = []
    claim_class_counts: Counter[str] = Counter()
    root_counts: Counter[str] = Counter()

    print(
        "=== S232 CLAIM-LEVEL UPWARD-NOVELTY ATTRIBUTION AUDIT ==="
    )
    print("upward comparisons:", len(upward))
    print("provider calls: false")
    print("literature retrieval: false")
    print("LLM calls: false")
    print("production selection changed: false")

    for index, comparison in enumerate(
        upward,
        start=1,
    ):
        case_id = str(
            comparison["source_case_id"]
        )
        scope = str(comparison["scope"])
        hypothesis_id = str(
            comparison["hypothesis_id"]
        )

        before_path, after_path = (
            report_paths_for_comparison(
                comparison=comparison,
                s227_rows=s227_rows,
                s229_rows=s229_rows,
            )
        )

        before_report = load_report(
            before_path
        )
        after_report = load_report(
            after_path
        )

        before_card = card_for(
            before_report,
            hypothesis_id,
        )
        after_card = card_for(
            after_report,
            hypothesis_id,
        )

        before_reviews = review_map(
            before_card
        )
        after_reviews = review_map(
            after_card
        )

        claim_ids = sorted(
            set(before_reviews)
            | set(after_reviews)
        )

        claim_rows: list[
            dict[str, Any]
        ] = []

        for claim_id in claim_ids:
            before_review = (
                before_reviews.get(claim_id)
            )
            after_review = (
                after_reviews.get(claim_id)
            )

            if (
                before_review is None
                or after_review is None
            ):
                row = {
                    "claim_id": claim_id,
                    "classification":
                        "CLAIM_SET_CHANGED",
                    "before_present":
                        before_review is not None,
                    "after_present":
                        after_review is not None,
                }
                claim_rows.append(row)
                claim_class_counts[
                    row["classification"]
                ] += 1
                continue

            classification, detail = (
                classify_claim_drift(
                    before_review=before_review,
                    after_review=after_review,
                    before_policy=
                        before_report.policy,
                    after_policy=
                        after_report.policy,
                )
            )

            row = {
                "claim_id": claim_id,
                "claim_text":
                    before_review.claim_text,
                "importance":
                    before_review.importance,
                "classification":
                    classification,
                **detail,
            }
            claim_rows.append(row)
            claim_class_counts[
                classification
            ] += 1

        root = packet_root_cause(
            claim_rows
        )
        root_counts[root] += 1

        output = {
            "source_case_id": case_id,
            "scope": scope,
            "hypothesis_id":
                hypothesis_id,
            "before_label":
                comparison[
                    "before_label"
                ],
            "after_label":
                comparison[
                    "after_label"
                ],
            "before_report_path":
                str(
                    Path(
                        before_path
                    ).expanduser().resolve()
                ),
            "after_report_path":
                str(
                    Path(
                        after_path
                    ).expanduser().resolve()
                ),
            "before_hypothesis_status":
                before_card.status,
            "after_hypothesis_status":
                after_card.status,
            "s231_classification":
                comparison[
                    "classification"
                ],
            "root_cause_class":
                root,
            "claim_rows":
                claim_rows,
        }
        outputs.append(output)

        print(
            "\n"
            f"[{index}/{len(upward)}] "
            f"{case_id} | {scope}"
        )
        print(
            " hypothesis:",
            hypothesis_id,
        )
        print(
            " status:",
            before_card.status,
            "->",
            after_card.status,
        )
        print(
            " root cause:",
            root,
        )

        for claim in claim_rows:
            if (
                claim[
                    "classification"
                ]
                == "CLAIM_STATUS_STABLE"
            ):
                continue
            print(
                "  CLAIM",
                claim["claim_id"],
                "|",
                claim.get(
                    "before_status"
                ),
                "->",
                claim.get(
                    "after_status"
                ),
                "|",
                claim[
                    "classification"
                ],
            )

            for change in claim.get(
                "lost_relation_backed_same_work_relabeled",
                [],
            ):
                print(
                    "    SAME WORK",
                    change["work_id"],
                    change[
                        "before_relationship"
                    ],
                    "->",
                    change[
                        "after_relationship"
                    ],
                    "| abstract",
                    change[
                        "before_abstract_available"
                    ],
                    "->",
                    change[
                        "after_abstract_available"
                    ],
                )

            for lost in claim.get(
                "lost_relation_backed_match_disappeared",
                [],
            ):
                print(
                    "    DISAPPEARED",
                    lost["work_id"],
                    "|",
                    lost[
                        "before_relationship"
                    ],
                )

    summary = {
        "schema_version":
            "claim-level-upward-novelty-attribution-s232-v1",
        "source_s231_summary":
            str(s231_path),
        "source_s227_3_summary":
            str(s227_path),
        "source_s229_summary":
            str(s229_path),
        "upward_comparison_count":
            len(outputs),
        "root_cause_counts":
            dict(
                sorted(
                    root_counts.items()
                )
            ),
        "claim_classification_counts":
            dict(
                sorted(
                    claim_class_counts.items()
                )
            ),
        "rows": outputs,
        "interpretation_policy": {
            "REVIEWER_RELATIONSHIP_DRIFT":
                (
                    "A previously relation-backed work remains in the "
                    "later report but is relabeled to a non-relation-backed "
                    "relationship."
                ),
            "MATCH_SET_OR_REVIEWER_OMISSION_DRIFT":
                (
                    "A previously relation-backed work disappears from the "
                    "later review matches; report-only artifacts cannot "
                    "distinguish rank-set replacement from reviewer omission."
                ),
            "REVIEWER_OR_CANDIDATE_COMPOSITION_DRIFT":
                (
                    "Relation-backed claim support was lost despite "
                    "non-decreasing aggregate coverage, without a directly "
                    "observable same-work relabel."
                ),
            "COVERAGE_OR_RANKING_DRIFT":
                (
                    "Relation-backed support was lost while aggregate "
                    "coverage decreased."
                ),
            "provider_calls": False,
            "literature_retrieval_performed": False,
            "llm_calls": False,
            "production_selection_changed": False,
        },
    }

    write_json(
        out_path,
        summary,
    )

    print("\nS232 audit complete")
    print(
        "upward comparisons:",
        len(outputs),
    )
    print("root cause counts:")
    for key, count in sorted(
        root_counts.items()
    ):
        print(" ", count, key)

    print(
        "claim classification counts:"
    )
    for key, count in sorted(
        claim_class_counts.items()
    ):
        print(" ", count, key)

    print("provider calls: false")
    print("LLM calls: false")
    print(
        "production selection changed: false"
    )
    print("artifact:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
