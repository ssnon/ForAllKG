
from __future__ import annotations

import copy
import re
from typing import Any


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−_/]+", " ", text)

    # Ignore sentence/surface punctuation '.', but preserve a decimal point
    # only when it is directly surrounded by digits (e.g. 0.5).
    chars: list[str] = []
    for i, char in enumerate(text):
        if char != ".":
            chars.append(char)
            continue

        previous_is_digit = (
            i > 0 and text[i - 1].isdigit()
        )
        next_is_digit = (
            i + 1 < len(text)
            and text[i + 1].isdigit()
        )
        chars.append(
            "."
            if previous_is_digit and next_is_digit
            else " "
        )

    text = "".join(chars)
    text = re.sub(r"[^\w\s+*.]", " ", text)
    return " ".join(text.split())



def _norm_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({_norm(x) for x in value if _norm(x)}))


def _binding_fingerprint(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("hypothesis_id") or ""),
        _norm(record.get("proposition_basis")),
    )


def _compatible_optional_set(
    left: object,
    right: object,
) -> bool:
    a = set(_norm_list(left))
    b = set(_norm_list(right))
    if not a or not b:
        return True
    return a == b


def bindings_compatible(
    historical: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    if _binding_fingerprint(historical) != _binding_fingerprint(current):
        return False
    if not _compatible_optional_set(
        historical.get("relation_endpoint_anchors"),
        current.get("relation_endpoint_anchors"),
    ):
        return False
    if not _compatible_optional_set(
        historical.get("scope_qualifier_spans"),
        current.get("scope_qualifier_spans"),
    ):
        return False
    if not _compatible_optional_set(
        historical.get("directional_qualifier_spans"),
        current.get("directional_qualifier_spans"),
    ):
        return False
    return True


def _review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        rows = payload.get("claim_reviews")
    if not isinstance(rows, list):
        rows = []
    return rows


def _work_key(work: dict[str, Any]) -> tuple[str, str]:
    doi = _norm(work.get("doi"))
    if doi:
        return ("doi", doi)
    work_id = str(work.get("work_id") or "").strip()
    if work_id:
        return ("work_id", work_id)
    return (
        "title_year",
        _norm(work.get("title")) + "|" + str(work.get("year") or ""),
    )


def _historical_candidate_ids_for_claim(
    *,
    claim_id: str,
    packet: dict[str, Any],
    reviews: dict[str, Any],
) -> set[str]:
    result: set[str] = set()

    for work in packet.get("works", []):
        if claim_id in {
            str(x)
            for x in work.get("retrieval_claim_ids", [])
        }:
            wid = str(work.get("work_id") or "")
            if wid:
                result.add(wid)

    for review in _review_rows(reviews):
        if str(review.get("claim_id") or "") != claim_id:
            continue
        for match in review.get("matches", []):
            wid = str(match.get("work_id") or "")
            if wid:
                result.add(wid)

    return result


def build_candidate_memory(
    *,
    historical_source_binding: dict[str, Any],
    historical_packet: dict[str, Any],
    historical_reviews: dict[str, Any],
    current_source_binding: dict[str, Any],
    current_packet: dict[str, Any],
    current_reviews: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    historical_records = list(
        historical_source_binding.get("records", [])
    )
    current_records = list(
        current_source_binding.get("records", [])
    )

    historical_works = {
        str(work.get("work_id")): work
        for work in historical_packet.get("works", [])
        if work.get("work_id")
    }

    augmented_packet = copy.deepcopy(current_packet)
    augmented_reviews = copy.deepcopy(current_reviews)

    current_works = {
        _work_key(work): work
        for work in augmented_packet.get("works", [])
    }
    current_by_id = {
        str(work.get("work_id")): work
        for work in augmented_packet.get("works", [])
        if work.get("work_id")
    }

    review_rows = _review_rows(augmented_reviews)
    reviews_by_claim = {
        str(row.get("claim_id")): row
        for row in review_rows
        if row.get("claim_id")
    }

    mappings: list[dict[str, Any]] = []
    reexposed_pairs: list[tuple[str, str]] = []
    new_work_count = 0

    for current in current_records:
        current_claim_id = str(current.get("claim_id") or "")
        if not current_claim_id:
            continue

        matches = [
            historical
            for historical in historical_records
            if bindings_compatible(historical, current)
        ]
        historical_claim_ids = sorted(
            {
                str(row.get("claim_id"))
                for row in matches
                if row.get("claim_id")
            }
        )

        candidate_hist_work_ids: set[str] = set()
        for historical_claim_id in historical_claim_ids:
            candidate_hist_work_ids.update(
                _historical_candidate_ids_for_claim(
                    claim_id=historical_claim_id,
                    packet=historical_packet,
                    reviews=historical_reviews,
                )
            )

        reexposed_current_work_ids: list[str] = []

        for hist_work_id in sorted(candidate_hist_work_ids):
            hist_work = historical_works.get(hist_work_id)
            if hist_work is None:
                continue

            key = _work_key(hist_work)
            current_work = current_works.get(key)

            if current_work is None:
                current_work = copy.deepcopy(hist_work)
                augmented_packet.setdefault("works", []).append(
                    current_work
                )
                current_works[key] = current_work
                wid = str(current_work.get("work_id") or "")
                if wid:
                    current_by_id[wid] = current_work
                new_work_count += 1

            current_work_id = str(
                current_work.get("work_id") or ""
            )
            if not current_work_id:
                continue

            tags = current_work.setdefault(
                "candidate_memory_reexposed_for_claim_ids",
                [],
            )
            if current_claim_id not in tags:
                tags.append(current_claim_id)

            reexposed_current_work_ids.append(current_work_id)
            reexposed_pairs.append(
                (current_claim_id, current_work_id)
            )

            review = reviews_by_claim.get(current_claim_id)
            if review is None:
                review = {
                    "claim_id": current_claim_id,
                    "matches": [],
                }
                review_rows.append(review)
                reviews_by_claim[current_claim_id] = review

            existing_ids = {
                str(match.get("work_id") or "")
                for match in review.get("matches", [])
            }
            if current_work_id not in existing_ids:
                review.setdefault("matches", []).append(
                    {
                        "work_id": current_work_id,
                        "relationship": "COMPONENT_ONLY",
                        "confidence": 1.0,
                        "relevance_score": 1.0,
                        "reaction_score": 0.5,
                        "scope_score": 1.0,
                        "candidate_memory_reexposed": True,
                        "candidate_memory_is_evidence": False,
                    }
                )

        mappings.append(
            {
                "current_claim_id": current_claim_id,
                "hypothesis_id": current.get("hypothesis_id"),
                "current_claim_local_id": current.get(
                    "claim_local_id"
                ),
                "proposition_basis": current.get(
                    "proposition_basis"
                ),
                "historical_claim_ids": historical_claim_ids,
                "historical_binding_match_count": len(matches),
                "reexposed_work_ids": sorted(
                    set(reexposed_current_work_ids)
                ),
            }
        )

    if "reviews" in augmented_reviews:
        augmented_reviews["reviews"] = review_rows
    elif "claim_reviews" in augmented_reviews:
        augmented_reviews["claim_reviews"] = review_rows
    else:
        augmented_reviews["reviews"] = review_rows

    report = {
        "schema_version": "prior-art-candidate-memory-shadow-v1",
        "shadow_only": True,
        "memory_is_evidence": False,
        "new_search_performed": False,
        "positive_premise_authority": False,
        "novelty_authority_created": False,
        "n9_authority_created": False,
        "n10_authority_created": False,
        "production_selection_changed": False,
        "mapping_policy": (
            "same hypothesis + exact normalized proposition_basis + "
            "compatible endpoint/scope/direction qualifiers"
        ),
        "current_claim_count": len(current_records),
        "mapped_current_claim_count": sum(
            bool(row["historical_claim_ids"])
            for row in mappings
        ),
        "reexposed_current_claim_count": sum(
            bool(row["reexposed_work_ids"])
            for row in mappings
        ),
        "reexposed_claim_work_pair_count": len(
            set(reexposed_pairs)
        ),
        "new_work_added_to_current_packet_count": (
            new_work_count
        ),
        "mappings": mappings,
    }

    return report, augmented_packet, augmented_reviews
