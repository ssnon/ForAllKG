
from __future__ import annotations

import re
from typing import Any

from pipeline_core.discovery.fulltext_lower_order_escalation_shadow import (
    _claim_map,
    _review_map,
    select_existing_candidates,
)
from pipeline_core.discovery.prior_art_candidate_memory_shadow import (
    bindings_compatible,
)


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−_/]+", " ", text)
    chars: list[str] = []
    for i, char in enumerate(text):
        if char != ".":
            chars.append(char)
            continue
        previous_is_digit = i > 0 and text[i - 1].isdigit()
        next_is_digit = i + 1 < len(text) and text[i + 1].isdigit()
        chars.append("." if previous_is_digit and next_is_digit else " ")
    text = "".join(chars)
    text = re.sub(r"[^\w\s+*]", " ", text)
    return " ".join(text.split())


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


def _work_key(work: dict[str, Any]) -> tuple[str, str]:
    doi = _doi_key(work.get("doi"))
    if doi:
        return ("doi", doi)
    work_id = str(work.get("work_id") or "").strip()
    if work_id:
        return ("work_id", work_id)
    return (
        "title_year",
        _norm(work.get("title")) + "|" + str(work.get("year") or ""),
    )


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9+*.]{1,}",
            _norm(value),
        )
        if len(token) >= 3
    }


def _review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        rows = payload.get("claim_reviews")
    return rows if isinstance(rows, list) else []


def _source_binding_records(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = payload.get("records")
    return rows if isinstance(rows, list) else []


def _historical_claim_ids(
    *,
    current_record: dict[str, Any],
    historical_source_binding: dict[str, Any],
) -> list[str]:
    return sorted(
        {
            str(row.get("claim_id"))
            for row in _source_binding_records(
                historical_source_binding
            )
            if row.get("claim_id")
            and bindings_compatible(row, current_record)
        }
    )


def _historical_fulltext_work_ids(
    historical_fulltext: dict[str, Any],
    claim_ids: set[str],
) -> set[str]:
    result: set[str] = set()
    for target in historical_fulltext.get("targets", []):
        if str(target.get("claim_id") or "") not in claim_ids:
            continue
        for row in target.get("reviews", []):
            wid = str(row.get("work_id") or "")
            if wid:
                result.add(wid)
    return result


def _historical_metadata_work_ids(
    historical_reviews: dict[str, Any],
    claim_ids: set[str],
) -> set[str]:
    result: set[str] = set()
    for review in _review_rows(historical_reviews):
        if str(review.get("claim_id") or "") not in claim_ids:
            continue
        for match in review.get("matches", []):
            wid = str(match.get("work_id") or "")
            if wid:
                result.add(wid)
    return result


def _historical_retrieval_work_ids(
    historical_packet: dict[str, Any],
    claim_ids: set[str],
) -> set[str]:
    result: set[str] = set()
    for work in historical_packet.get("works", []):
        retrieval = {
            str(x)
            for x in work.get("retrieval_claim_ids", [])
        }
        if retrieval & claim_ids:
            wid = str(work.get("work_id") or "")
            if wid:
                result.add(wid)
    return result


def _memory_score(
    *,
    claim: dict[str, Any],
    binding: dict[str, Any],
    work: dict[str, Any],
) -> tuple[int, int, int, int, int, str]:
    work_text = _norm(
        f"{work.get('title', '')} {work.get('abstract', '')}"
    )
    work_tokens = _tokens(work_text)

    anchor_hits = 0
    for anchor in binding.get("relation_endpoint_anchors", []):
        normalized = _norm(anchor)
        if normalized and normalized in work_text:
            anchor_hits += 1

    scope_hits = 0
    for span in binding.get("scope_qualifier_spans", []):
        normalized = _norm(span)
        if normalized and normalized in work_text:
            scope_hits += 1

    identity_tokens = _tokens(
        " ".join(
            [
                *map(str, claim.get("prior_art_identity_terms", [])),
                *map(
                    str,
                    binding.get("relation_endpoint_anchors", []),
                ),
            ]
        )
    )
    proposition_tokens = _tokens(
        binding.get("proposition_basis")
        or claim.get("text")
        or ""
    )

    identity_overlap = len(identity_tokens & work_tokens)
    proposition_overlap = len(proposition_tokens & work_tokens)
    access_hint = int(
        bool(work.get("open_access_url"))
        or bool(work.get("doi"))
        or bool((work.get("provider_ids") or {}).get("openalex"))
    )

    return (
        anchor_hits,
        identity_overlap,
        scope_hits,
        proposition_overlap,
        access_hint,
        str(work.get("work_id") or ""),
    )


def build_tiered_selection_plan(
    *,
    query_plan: dict[str, Any],
    topology_report: dict[str, Any],
    current_source_binding: dict[str, Any],
    current_packet: dict[str, Any],
    current_reviews: dict[str, Any],
    memory_packet: dict[str, Any],
    historical_source_binding: dict[str, Any],
    historical_packet: dict[str, Any],
    historical_reviews: dict[str, Any],
    historical_fulltext: dict[str, Any],
    current_slots: int = 4,
    tier_a_slots: int = 2,
    tier_b_slots: int = 2,
    tier_c_slots: int = 4,
) -> dict[str, Any]:
    from pipeline_core.discovery.fulltext_lower_order_escalation_shadow import (
        escalation_target_claim_ids,
    )

    claims = _claim_map(query_plan)
    current_review_map = _review_map(current_reviews)
    targets = escalation_target_claim_ids(topology_report)

    current_binding_by_claim = {
        str(row.get("claim_id")): row
        for row in _source_binding_records(current_source_binding)
        if row.get("claim_id")
    }

    memory_works = {
        str(work.get("work_id")): work
        for work in memory_packet.get("works", [])
        if work.get("work_id")
    }
    current_work_key_to_id = {
        _work_key(work): str(work.get("work_id"))
        for work in memory_packet.get("works", [])
        if work.get("work_id")
    }
    historical_works = {
        str(work.get("work_id")): work
        for work in historical_packet.get("works", [])
        if work.get("work_id")
    }

    rows: list[dict[str, Any]] = []
    total_cap = (
        max(0, current_slots)
        + max(0, tier_a_slots)
        + max(0, tier_b_slots)
        + max(0, tier_c_slots)
    )

    for claim_id in targets:
        claim = claims.get(claim_id)
        binding = current_binding_by_claim.get(claim_id)

        if claim is None or binding is None:
            rows.append(
                {
                    "claim_id": claim_id,
                    "status": (
                        "CLAIM_NOT_FOUND"
                        if claim is None
                        else "SOURCE_BINDING_NOT_FOUND"
                    ),
                    "selected": [],
                    "historical_claim_ids": [],
                }
            )
            continue

        historical_claim_ids = _historical_claim_ids(
            current_record=binding,
            historical_source_binding=historical_source_binding,
        )
        historical_claim_id_set = set(historical_claim_ids)

        a_hist = _historical_fulltext_work_ids(
            historical_fulltext,
            historical_claim_id_set,
        )
        b_hist = _historical_metadata_work_ids(
            historical_reviews,
            historical_claim_id_set,
        ) - a_hist
        c_hist = _historical_retrieval_work_ids(
            historical_packet,
            historical_claim_id_set,
        ) - a_hist - b_hist

        def to_current_ids(
            historical_ids: set[str],
        ) -> list[str]:
            result: list[str] = []
            for historical_id in historical_ids:
                work = historical_works.get(historical_id)
                if work is None:
                    continue
                current_id = current_work_key_to_id.get(
                    _work_key(work)
                )
                if current_id:
                    result.append(current_id)
            return sorted(set(result))

        tier_ids = {
            "HISTORICAL_FULLTEXT_REVIEWED": to_current_ids(a_hist),
            "HISTORICAL_METADATA_REVIEWED": to_current_ids(b_hist),
            "HISTORICAL_RETRIEVED_ONLY": to_current_ids(c_hist),
        }

        current_candidates = select_existing_candidates(
            claim_id=claim_id,
            claim=claim,
            review=current_review_map.get(claim_id),
            packet=current_packet,
            max_candidates=max(1, current_slots),
        )

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        for work in current_candidates[: max(0, current_slots)]:
            wid = str(work.get("work_id") or "")
            if not wid or wid in selected_ids:
                continue
            selected_ids.add(wid)
            selected.append(
                {
                    "work_id": wid,
                    "selection_lane": "CURRENT_RUN",
                    "memory_tier": None,
                    "memory_is_evidence": False,
                }
            )

        tier_slots = {
            "HISTORICAL_FULLTEXT_REVIEWED": max(0, tier_a_slots),
            "HISTORICAL_METADATA_REVIEWED": max(0, tier_b_slots),
            "HISTORICAL_RETRIEVED_ONLY": max(0, tier_c_slots),
        }

        tier_available_counts: dict[str, int] = {}
        tier_selected_counts: dict[str, int] = {}

        for tier in (
            "HISTORICAL_FULLTEXT_REVIEWED",
            "HISTORICAL_METADATA_REVIEWED",
            "HISTORICAL_RETRIEVED_ONLY",
        ):
            candidates = [
                memory_works[wid]
                for wid in tier_ids[tier]
                if wid in memory_works
                and wid not in selected_ids
            ]
            tier_available_counts[tier] = len(candidates)
            candidates.sort(
                key=lambda work: _memory_score(
                    claim=claim,
                    binding=binding,
                    work=work,
                ),
                reverse=True,
            )
            chosen = candidates[: tier_slots[tier]]
            tier_selected_counts[tier] = len(chosen)

            for work in chosen:
                wid = str(work.get("work_id") or "")
                selected_ids.add(wid)
                selected.append(
                    {
                        "work_id": wid,
                        "selection_lane": "HISTORICAL_MEMORY",
                        "memory_tier": tier,
                        "memory_is_evidence": False,
                    }
                )

        rows.append(
            {
                "claim_id": claim_id,
                "hypothesis_id": binding.get("hypothesis_id"),
                "claim_text": claim.get("text"),
                "proposition_basis": binding.get("proposition_basis"),
                "status": "SELECTED",
                "historical_claim_ids": historical_claim_ids,
                "selected": selected[:total_cap],
                "selected_count": min(len(selected), total_cap),
                "total_cap": total_cap,
                "tier_available_counts": tier_available_counts,
                "tier_selected_counts": tier_selected_counts,
            }
        )

    return {
        "schema_version": "memory-tiered-fulltext-selection-shadow-v1",
        "shadow_only": True,
        "new_search_performed": False,
        "memory_is_evidence": False,
        "selection_policy": {
            "current_slots": current_slots,
            "historical_fulltext_reviewed_slots": tier_a_slots,
            "historical_metadata_reviewed_slots": tier_b_slots,
            "historical_retrieved_only_slots": tier_c_slots,
            "total_cap": total_cap,
            "memory_ranking": (
                "endpoint-anchor coverage, identity-token overlap, "
                "scope coverage, proposition-token overlap, access hint"
            ),
        },
        "target_count": len(rows),
        "targets": rows,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
    }
