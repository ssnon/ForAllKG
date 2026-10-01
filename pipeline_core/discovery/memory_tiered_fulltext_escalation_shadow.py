
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


def run_tiered_escalation(
    *,
    query_plan: dict[str, Any],
    packet: dict[str, Any],
    selection_plan: dict[str, Any],
    output_root: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    max_excerpt_chars: int = 24000,
    parse_retries: int = 3,
) -> dict[str, Any]:
    claims = _claim_map(query_plan)
    works = {
        str(work.get("work_id")): work
        for work in packet.get("works", [])
        if work.get("work_id")
    }

    targets: list[dict[str, Any]] = []

    for planned in selection_plan.get("targets", []):
        claim_id = str(planned.get("claim_id") or "")
        claim = claims.get(claim_id)

        if claim is None:
            targets.append(
                {
                    "claim_id": claim_id,
                    "status": "CLAIM_NOT_FOUND",
                    "candidate_count": 0,
                    "reviews": [],
                    "fulltext_relation_backed_work_ids": [],
                }
            )
            continue

        rows: list[dict[str, Any]] = []
        backed: list[str] = []

        for selected in planned.get("selected", []):
            wid = str(selected.get("work_id") or "")
            work = works.get(wid)
            if work is None:
                rows.append(
                    {
                        "work_id": wid,
                        "title": None,
                        "doi": None,
                        "selection_lane": selected.get(
                            "selection_lane"
                        ),
                        "memory_tier": selected.get("memory_tier"),
                        "memory_is_evidence": False,
                        "relationship": (
                            "INSUFFICIENT_FULLTEXT_EVIDENCE"
                        ),
                        "confidence": 0.0,
                        "evidence_spans": [],
                        "rationale": "selected work missing from packet",
                    }
                )
                continue

            catalog = _catalog_work(work)
            acquired = _extract_fulltext(catalog, output_root)
            fulltext = str(acquired.get("text") or "")

            row = {
                "work_id": catalog.work_id,
                "title": catalog.title,
                "doi": catalog.doi,
                "selection_lane": selected.get("selection_lane"),
                "memory_tier": selected.get("memory_tier"),
                "memory_is_evidence": False,
                "access_status": (
                    acquired.get("resolution", {}).get("status")
                ),
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
            }

            if fulltext:
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
                if (
                    relationship in BACKED
                    and not valid_spans
                ):
                    relationship = (
                        "INSUFFICIENT_FULLTEXT_EVIDENCE"
                    )

                row.update(
                    {
                        "relationship": relationship,
                        "confidence": draft.confidence,
                        "evidence_spans": valid_spans,
                        "rationale": draft.rationale,
                        "excerpt_sha256": hashlib.sha256(
                            excerpt.encode("utf-8")
                        ).hexdigest(),
                    }
                )

                if relationship in BACKED:
                    backed.append(catalog.work_id)

            rows.append(row)

        if backed:
            status = "FULLTEXT_RELATION_BACKED"
        elif any(
            row.get("artifact_status") == "downloaded"
            and row.get("text_source")
            for row in rows
        ):
            status = "FULLTEXT_REVIEWED_NOT_BACKED"
        else:
            status = "FULLTEXT_UNAVAILABLE"

        targets.append(
            {
                "claim_id": claim_id,
                "claim_text": claim.get("text"),
                "status": status,
                "candidate_count": len(rows),
                "reviews": rows,
                "fulltext_relation_backed_work_ids": sorted(
                    set(backed)
                ),
            }
        )

    return {
        "schema_version": (
            "memory-tiered-fulltext-escalation-shadow-v1"
        ),
        "shadow_only": True,
        "new_search_performed": False,
        "memory_is_evidence": False,
        "positive_premise_authority": False,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "target_claim_count": len(targets),
        "fulltext_relation_backed_target_count": sum(
            row["status"] == "FULLTEXT_RELATION_BACKED"
            for row in targets
        ),
        "targets": targets,
    }
