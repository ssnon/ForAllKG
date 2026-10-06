from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from pipeline_core.discovery.exploratory_portfolio_search_v1 import (
    ExploratoryCandidateRecord,
)


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in str(value or "").casefold().replace("/", " ").split()
        if token
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _diversity(
    row: ExploratoryCandidateRecord,
    selected: Sequence[ExploratoryCandidateRecord],
) -> float:
    if not selected:
        return 1.0
    premise = set(row.premise_statement_ids)
    text = _tokens(row.hypothesis_statement)
    worst = 0.0
    for other in selected:
        premise_similarity = _jaccard(
            premise,
            set(other.premise_statement_ids),
        )
        text_similarity = _jaccard(
            text,
            _tokens(other.hypothesis_statement),
        )
        worst = max(
            worst,
            0.55 * premise_similarity + 0.45 * text_similarity,
        )
    return max(0.0, 1.0 - worst)


def _score(
    row: ExploratoryCandidateRecord,
    slot: str,
    selected: Sequence[ExploratoryCandidateRecord],
) -> float:
    diversity = _diversity(row, selected)

    if slot == "EXPLOIT":
        return (
            0.60 * row.quality_prior
            + 0.20 * diversity
            + 0.12 * row.ucb_exploration_bonus
            - 0.08 * row.verification_cost_prior
        )

    if slot == "WILDCARD":
        return (
            0.40 * row.novelty_of_search_path
            + 0.28 * diversity
            + 0.18 * row.ucb_exploration_bonus
            + 0.14 * row.uncertainty
        )

    if slot == "UNCERTAINTY":
        return (
            0.46 * row.uncertainty
            + 0.24 * row.ucb_exploration_bonus
            + 0.20 * diversity
            + 0.10 * row.quality_prior
        )

    if slot == "PROSPECTIVE":
        prospect = (
            1.0
            if row.prospective_identifiability
            == "PROSPECTIVELY_IDENTIFIABLE"
            else 0.35
        )
        return (
            0.34 * row.quality_prior
            + 0.28 * prospect
            + 0.23 * diversity
            + 0.15 * row.novelty_of_search_path
        )

    raise ValueError(f"unknown slot: {slot}")


def reserve_diverse_search_portfolio(
    records: Sequence[ExploratoryCandidateRecord],
    *,
    max_retained_candidates: int = 8,
) -> dict[str, Any]:
    """
    Small-population-safe selection.

    Reserve distinct search roles before using remaining capacity for exploit.
    This avoids the EPS-v1 failure where an EXPLOIT quota of three consumed an
    entire 1-3 candidate population before UNCERTAINTY/WILDCARD could compete.
    """
    if max_retained_candidates < 1:
        raise ValueError("max_retained_candidates must be >= 1")

    working = [
        row.model_copy(deep=True)
        for row in records
        if row.reproductive_eligible
    ]

    selected: list[ExploratoryCandidateRecord] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()

    def eligible(
        row: ExploratoryCandidateRecord,
        slot: str,
    ) -> bool:
        if row.hypothesis_id in selected_ids:
            return False
        if row.family_signature in selected_families:
            return False
        if slot == "WILDCARD":
            return row.route in {
                "AXIS_MUTATION",
                "EVIDENCE_REAXIS",
                "REQUEST_GRAPH_RETRAVERSAL",
            }
        if slot == "PROSPECTIVE":
            return (
                row.prospective_identifiability
                == "PROSPECTIVELY_IDENTIFIABLE"
            )
        return True

    def take(slot: str) -> bool:
        if len(selected) >= max_retained_candidates:
            return False
        pool = [row for row in working if eligible(row, slot)]
        if not pool:
            return False
        pool.sort(
            key=lambda row: (
                -_score(row, slot, selected),
                -row.base_search_value,
                row.hypothesis_id,
            )
        )
        chosen = pool[0].model_copy(
            update={
                "selected_slot": slot,
                "selected_for_next_verification": True,
            }
        )
        selected.append(chosen)
        selected_ids.add(chosen.hypothesis_id)
        selected_families.add(chosen.family_signature)
        return True

    # Reservation order is deliberate:
    # 1) preserve one strong incumbent,
    # 2) force a genuinely exploratory branch when available,
    # 3) preserve uncertainty/information gain,
    # 4) preserve a prospectively testable branch,
    # 5) use remaining capacity for quality exploitation.
    take("EXPLOIT")
    take("WILDCARD")
    take("UNCERTAINTY")
    take("PROSPECTIVE")

    while len(selected) < max_retained_candidates:
        if not take("EXPLOIT"):
            break

    selected_by_id = {
        row.hypothesis_id: row
        for row in selected
    }
    annotated = [
        selected_by_id.get(row.hypothesis_id, row)
        for row in records
    ]
    counts = Counter(
        row.selected_slot
        for row in selected
        if row.selected_slot
    )

    return {
        "schema_version":
            "exploratory-portfolio-search-v1-1-selection-v1",
        "selection_policy":
            "RESERVED_EXPLORE_EXPLOIT_DIVERSITY_V1_1",
        "candidate_count": len(records),
        "eligible_candidate_count": len(working),
        "retained_count": len(selected),
        "retained_candidate_ids": [
            row.hypothesis_id
            for row in selected
        ],
        "retained_count_by_slot": dict(sorted(counts.items())),
        "records": [
            row.model_dump(mode="json")
            for row in annotated
        ],
        "candidate_archival_preserved": True,
        "search_parent_selection_authority": True,
        "compute_allocation_authority": True,
        "scientific_truth_authority": False,
        "literature_wide_novelty_authority": False,
        "production_selection_authority": False,
        "stage8_input_changed": False,
        "canonical_graph_mutated": False,
    }
