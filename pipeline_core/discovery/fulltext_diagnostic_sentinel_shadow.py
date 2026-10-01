
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pipeline_core.discovery.fulltext_lower_order_escalation_shadow import (
    _catalog_work,
    _claim_map,
    _excerpt,
    _extract_fulltext,
    _review_candidate,
)


BACKED = {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}


def _doi_key(value: object) -> str:
    text = str(value or "").strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.rstrip(" .")


def run_diagnostic_sentinel(
    *,
    query_plan: dict[str, Any],
    packet: dict[str, Any],
    claim_id: str,
    doi: str,
    output_root: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    max_excerpt_chars: int = 24000,
    parse_retries: int = 3,
) -> dict[str, Any]:
    claims = _claim_map(query_plan)
    claim = claims.get(claim_id)
    if claim is None:
        return {
            "schema_version": (
                "fulltext-diagnostic-sentinel-shadow-v1"
            ),
            "shadow_only": True,
            "diagnostic_only": True,
            "aggregation_eligible": False,
            "status": "CLAIM_NOT_FOUND",
            "claim_id": claim_id,
            "doi": doi,
            "novelty_authority_created": False,
        }

    target_doi = _doi_key(doi)
    work = next(
        (
            row
            for row in packet.get("works", [])
            if _doi_key(row.get("doi")) == target_doi
        ),
        None,
    )
    if work is None:
        return {
            "schema_version": (
                "fulltext-diagnostic-sentinel-shadow-v1"
            ),
            "shadow_only": True,
            "diagnostic_only": True,
            "aggregation_eligible": False,
            "status": "WORK_NOT_FOUND",
            "claim_id": claim_id,
            "doi": doi,
            "novelty_authority_created": False,
        }

    catalog = _catalog_work(work)
    acquired = _extract_fulltext(catalog, output_root)
    fulltext = str(acquired.get("text") or "")

    result = {
        "schema_version": (
            "fulltext-diagnostic-sentinel-shadow-v1"
        ),
        "shadow_only": True,
        "diagnostic_only": True,
        "aggregation_eligible": False,
        "claim_id": claim_id,
        "claim_text": claim.get("text"),
        "work_id": catalog.work_id,
        "title": catalog.title,
        "doi": catalog.doi,
        "artifact_status": (
            acquired.get("artifact", {}).get("status")
        ),
        "text_source": acquired.get("text_source"),
        "relationship": "INSUFFICIENT_FULLTEXT_EVIDENCE",
        "confidence": 0.0,
        "evidence_spans": [],
        "rationale": (
            "full text was not available or extractable"
        ),
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }

    if not fulltext:
        result["status"] = "FULLTEXT_UNAVAILABLE"
        return result

    excerpt = _excerpt(
        fulltext,
        claim,
        max_excerpt_chars,
    )
    draft = _review_candidate(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        claim=claim,
        work=work,
        excerpt=excerpt,
        parse_retries=parse_retries,
    )
    valid_spans = [
        span
        for span in draft.evidence_spans
        if span and span in excerpt
    ]
    relationship = draft.relationship
    if relationship in BACKED and not valid_spans:
        relationship = "INSUFFICIENT_FULLTEXT_EVIDENCE"

    result.update(
        {
            "status": "REVIEWED",
            "relationship": relationship,
            "confidence": draft.confidence,
            "evidence_spans": valid_spans,
            "rationale": draft.rationale,
            "excerpt_sha256": hashlib.sha256(
                excerpt.encode("utf-8")
            ).hexdigest(),
        }
    )
    return result
