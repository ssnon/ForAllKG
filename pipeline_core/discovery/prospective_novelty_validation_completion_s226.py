from __future__ import annotations

from typing import Any


GAP_STATUSES = {
    "SEARCH_BOUNDED_GAP",
    "SEARCH_BOUNDED_RELATIONAL_GAP",
}
UNCERTAIN_STATUSES = {
    "INSUFFICIENT_COVERAGE",
    "CONFLICTING_RELATION",
}


def first_gap_level(
    levels: list[tuple[str, str]],
) -> str | None:
    """Return earliest search-bounded gap; uncertainty blocks later inference."""
    for level, status in levels:
        if status in GAP_STATUSES:
            return level
        if status in UNCERTAIN_STATUSES:
            return None
    return None


def material_title_only_matches(
    card: dict[str, Any],
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for review in card.get("claim_reviews", []):
        if not isinstance(review, dict):
            continue
        if review.get("importance") != "core":
            continue
        for match in review.get("matches", []):
            if not isinstance(match, dict):
                continue
            if (
                match.get("relationship") == "TITLE_ONLY_NEIGHBOR"
                and float(match.get("confidence", 0.0) or 0.0)
                >= min_confidence
            ):
                rows.append(
                    {
                        "claim_id": review.get("claim_id"),
                        "work_id": match.get("work_id"),
                        "title": match.get("title"),
                        "doi": match.get("doi"),
                        "confidence": match.get("confidence"),
                        "relevance_score": match.get("relevance_score"),
                        "abstract_available": match.get(
                            "abstract_available",
                            False,
                        ),
                    }
                )
    return rows


def l3_status_from_external_card(
    card: dict[str, Any],
    *,
    min_confidence: float,
) -> dict[str, Any]:
    """
    Translate existing exact external novelty into conceptual L3 knownness.

    A gap-like exact status is withheld when high-confidence core title-only
    neighbors remain unresolved. This changes only S226 validation measurement,
    never the source external-novelty report.
    """
    source_status = str(card.get("status") or "")
    coverage = card.get("coverage") or {}
    sufficient = bool(
        coverage.get("sufficient_for_absence_based_novelty", False)
    )
    title_only = material_title_only_matches(
        card,
        min_confidence=min_confidence,
    )

    gap_like = source_status in {
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        "PLAUSIBLY_NOVEL",
    }

    if gap_like and title_only:
        status = "INSUFFICIENT_COVERAGE"
        reason = "MATERIAL_CORE_TITLE_ONLY_PRIOR_ART_UNRESOLVED"
        sufficient = False
    elif source_status == "WELL_ESTABLISHED":
        status = "RELATION_BACKED"
        reason = "EXACT_EXTERNAL_WELL_ESTABLISHED"
    elif source_status == "LITERATURE_SUPPORTED_EXTENSION":
        status = "PARTIAL_RELATION_BACKED"
        reason = "EXACT_EXTERNAL_SUPPORTED_EXTENSION"
    elif source_status == "CONFLICTING_PRIOR_ART":
        status = "CONFLICTING_RELATION"
        reason = "EXACT_EXTERNAL_CONFLICT"
    elif source_status == "INSUFFICIENT_SEARCH_EVIDENCE":
        status = "INSUFFICIENT_COVERAGE"
        reason = "EXACT_EXTERNAL_INSUFFICIENT_SEARCH"
    elif gap_like and not sufficient:
        status = "INSUFFICIENT_COVERAGE"
        reason = "EXACT_EXTERNAL_ABSENCE_COVERAGE_INCOMPLETE"
    elif source_status == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP":
        status = (
            "SEARCH_BOUNDED_RELATIONAL_GAP"
            if card.get("relational_gap_kind")
            == "HIGHER_ORDER_RELATIONAL_GAP"
            else "SEARCH_BOUNDED_GAP"
        )
        reason = "EXACT_EXTERNAL_COMPONENTS_WITH_GAP"
    elif source_status in {
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        "PLAUSIBLY_NOVEL",
    }:
        status = "SEARCH_BOUNDED_GAP"
        reason = "EXACT_EXTERNAL_GAP_LIKE"
    else:
        status = "INSUFFICIENT_COVERAGE"
        reason = "EXACT_EXTERNAL_STATUS_UNMAPPED"

    return {
        "status": status,
        "source_external_status": source_status,
        "sufficient_coverage": sufficient,
        "material_title_only_matches": title_only,
        "reason_code": reason,
    }


def audit_external_report_payload(
    report: dict[str, Any],
    *,
    packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = report.get("policy") or {}
    min_conf = float(
        policy.get("min_match_confidence", 0.65)
        or 0.65
    )

    cards = [
        row
        for row in report.get("cards", [])
        if isinstance(row, dict)
    ]

    material: list[dict[str, Any]] = []
    insufficient_metadata_core = 0
    insufficient_coverage_cards = 0

    for card in cards:
        material.extend(
            [
                {
                    "hypothesis_id": card.get("hypothesis_id"),
                    **row,
                }
                for row in material_title_only_matches(
                    card,
                    min_confidence=min_conf,
                )
            ]
        )

        coverage = card.get("coverage") or {}
        if not bool(
            coverage.get(
                "sufficient_for_absence_based_novelty",
                False,
            )
        ):
            insufficient_coverage_cards += 1

        for review in card.get("claim_reviews", []):
            if not isinstance(review, dict):
                continue
            if (
                review.get("importance") == "core"
                and review.get("status") == "INSUFFICIENT_METADATA"
            ):
                insufficient_metadata_core += 1

    packet_work_count = None
    packet_abstract_count = None
    packet_missing_abstract_count = None
    provider_execution_failures = None

    if isinstance(packet, dict):
        works = [
            row
            for row in packet.get("works", [])
            if isinstance(row, dict)
        ]
        packet_work_count = len(works)
        packet_abstract_count = sum(
            bool(str(row.get("abstract") or "").strip())
            for row in works
        )
        packet_missing_abstract_count = (
            packet_work_count - packet_abstract_count
        )
        provider_execution_failures = sum(
            not bool(row.get("success"))
            for row in packet.get("executions", [])
            if isinstance(row, dict)
        )

    if material or insufficient_metadata_core:
        disposition = "RESOLUTION_RISK"
    elif insufficient_coverage_cards:
        disposition = "COVERAGE_INCOMPLETE"
    else:
        disposition = "NO_OBSERVED_RESOLUTION_RISK"

    return {
        "disposition": disposition,
        "card_count": len(cards),
        "status_counts": dict(report.get("status_counts") or {}),
        "material_core_title_only_count": len(material),
        "material_core_title_only_matches": material,
        "core_insufficient_metadata_count": insufficient_metadata_core,
        "insufficient_absence_coverage_card_count":
            insufficient_coverage_cards,
        "packet_work_count": packet_work_count,
        "packet_abstract_count": packet_abstract_count,
        "packet_missing_abstract_count": packet_missing_abstract_count,
        "provider_execution_failure_count": provider_execution_failures,
        "retrieval_recall_completeness":
            "UNOBSERVABLE_WITHOUT_EXTERNAL_REFERENCE_SET",
        "search_bounded_only": True,
        "novelty_authority_created": False,
        "production_selection_authority": False,
    }


def research_value_detail_payload(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "measurement_status": "NOT_RUN",
            "cards": [],
        }

    cards = []
    dimensions = (
        "mechanistic_discrimination",
        "two_sided_outcome_informativeness",
        "observable_decisiveness",
        "information_gain_proxy",
        "experimental_resolvability",
    )

    for row in report.get("cards", []):
        if not isinstance(row, dict):
            continue
        cards.append(
            {
                "hypothesis_id": row.get("hypothesis_id"),
                "hypothesis_type": row.get("hypothesis_type"),
                "value_argument_class":
                    row.get("value_argument_class"),
                "dimensions": {
                    name: (
                        (row.get(name) or {}).get("signal")
                        if isinstance(row.get(name), dict)
                        else None
                    )
                    for name in dimensions
                },
                "experimental_disposition":
                    row.get("experimental_disposition"),
                "relative_cost_burden":
                    row.get("relative_cost_burden"),
                "relative_effort_burden":
                    row.get("relative_effort_burden"),
                "reason_codes":
                    list(row.get("reason_codes") or []),
            }
        )

    return {
        "measurement_status": "AVAILABLE",
        "hypothesis_count": report.get("hypothesis_count"),
        "assessed_count": report.get("assessed_count"),
        "cards": cards,
        "novelty_signal_consumed":
            bool(report.get("novelty_signal_consumed", False)),
        "research_value_selection_authority":
            bool(
                report.get(
                    "research_value_selection_authority",
                    False,
                )
            ),
        "production_selection_authority":
            bool(
                report.get(
                    "production_selection_authority",
                    False,
                )
            ),
    }


__all__ = [
    "GAP_STATUSES",
    "UNCERTAIN_STATUSES",
    "first_gap_level",
    "material_title_only_matches",
    "l3_status_from_external_card",
    "audit_external_report_payload",
    "research_value_detail_payload",
]
