from __future__ import annotations

from collections import Counter
from typing import Any


_RELATION_AUTHORITY = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "CONFLICTING_PRIOR_ART",
}


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise TypeError("expected mapping-like scientific artifact")
    return value


def _document(work: dict[str, Any]) -> str:
    return "\n".join(
        str(value or "")
        for value in (
            work.get("title"),
            work.get("abstract"),
            work.get("venue"),
        )
        if value
    )


def build_prior_art_domain_authority_audit(
    *,
    domain_profile: Any,
    packet: Any,
    report: Any,
) -> dict[str, Any]:
    """Audit raw retrieval vs compiled prior-art authority by scientific domain.

    Cross-domain records are allowed to remain in the retrieval packet as
    analogical/diagnostic neighbors. The invariant checked here is narrower:
    a document that is incompatible with the domain profile must not retain a
    compiled relation-level prior-art authority.

    This sidecar is diagnostic only and cannot change novelty or production
    selection.
    """
    packet_payload = _dump(packet)
    report_payload = _dump(report)

    works = {
        str(row.get("work_id")): row
        for row in packet_payload.get("works", [])
        if row.get("work_id")
    }

    compatibility: dict[str, bool] = {}
    for work_id, work in works.items():
        compatibility[work_id] = bool(
            domain_profile.novelty.document_is_compatible_for_positive_prior_art(
                _document(work)
            )
        )

    matches: list[dict[str, Any]] = []
    for card in report_payload.get("cards", []):
        hypothesis_id = str(card.get("hypothesis_id") or "")
        for review in card.get("claim_reviews", []):
            claim_id = str(review.get("claim_id") or "")
            claim_status = str(review.get("status") or "")
            claim_reason_codes = list(review.get("reason_codes") or [])
            for match in review.get("matches", []):
                work_id = str(match.get("work_id") or "")
                relationship = str(match.get("relationship") or "")
                domain_ok = compatibility.get(work_id)
                has_relation_authority = relationship in _RELATION_AUTHORITY
                violation = bool(
                    domain_ok is False and has_relation_authority
                )
                work = works.get(work_id, {})
                matches.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "claim_id": claim_id,
                        "claim_status": claim_status,
                        "claim_reason_codes": claim_reason_codes,
                        "work_id": work_id,
                        "title": work.get("title"),
                        "doi": work.get("doi"),
                        "relationship": relationship,
                        "domain_compatible": domain_ok,
                        "relation_authority": has_relation_authority,
                        "domain_authority_violation": violation,
                    }
                )

    incompatible_ids = sorted(
        work_id
        for work_id, ok in compatibility.items()
        if not ok
    )
    referenced_ids = {
        row["work_id"]
        for row in matches
        if row["work_id"]
    }
    incompatible_reviewed_ids = sorted(
        set(incompatible_ids) & referenced_ids
    )
    violations = [
        row for row in matches
        if row["domain_authority_violation"]
    ]

    relationship_counts = Counter(
        row["relationship"]
        for row in matches
        if row["relationship"]
    )
    incompatible_relationship_counts = Counter(
        row["relationship"]
        for row in matches
        if row["domain_compatible"] is False and row["relationship"]
    )

    return {
        "schema_version": "prior-art-domain-authority-audit-v1",
        "source_packet_id": packet_payload.get("packet_id"),
        "source_report_id": report_payload.get("report_id"),
        "domain_profile_id": getattr(domain_profile, "profile_id", None),
        "retrieved_work_count": len(works),
        "domain_incompatible_retrieved_work_count": len(incompatible_ids),
        "domain_incompatible_retrieved_work_ids": incompatible_ids,
        "compiled_match_count": len(matches),
        "domain_incompatible_compiled_match_work_count": len(
            incompatible_reviewed_ids
        ),
        "domain_incompatible_compiled_match_work_ids": incompatible_reviewed_ids,
        "compiled_relationship_counts": dict(sorted(relationship_counts.items())),
        "domain_incompatible_compiled_relationship_counts": dict(
            sorted(incompatible_relationship_counts.items())
        ),
        "domain_authority_violation_count": len(violations),
        "domain_authority_violations": violations,
        "pass": not violations,
        "matches": matches,
        "raw_cross_domain_retrieval_allowed": True,
        "diagnostic_only": True,
        "positive_premise_authority_created": False,
        "novelty_authority_created": False,
        "production_selection_authority": False,
    }


__all__ = ["build_prior_art_domain_authority_audit"]
