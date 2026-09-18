from __future__ import annotations

from types import SimpleNamespace

from scripts.discovery.run_higher_order_shadow_lane import (
    _select_shadow_generation_contexts,
)


def _context(index: int, candidate_id: str | None = None):
    if candidate_id is None:
        authority = "confirmed_known"
        component_id = f"known:{index}"
    else:
        authority = "candidate_inspiration"
        component_id = candidate_id

    premise = SimpleNamespace(
        premise_role="modifier_relation",
        authority=SimpleNamespace(value=authority),
        component_id=component_id,
    )
    return SimpleNamespace(
        context_id=f"context:{index}",
        premises=[premise],
    )


def test_selection_is_identity_when_candidate_coverage_is_already_in_prefix():
    rows = (
        _context(1),
        _context(2, "candidate:a"),
        _context(3),
        _context(4),
    )

    selected, audit = _select_shadow_generation_contexts(
        contexts=rows,
        max_contexts=3,
    )

    assert [row.context_id for row in selected] == [
        "context:1",
        "context:2",
        "context:3",
    ]
    assert audit["selection_changed_from_prefix"] is False
    assert audit["selected_unique_candidate_modifier_count"] == 1


def test_selection_reserves_distinct_candidate_modifiers_without_ranking():
    rows = tuple(
        [
            *[_context(i) for i in range(1, 32)],
            _context(32, "candidate:a"),
            *[_context(i) for i in range(33, 71)],
            _context(71, "candidate:b"),
            _context(72),
            _context(73, "candidate:c"),
            *[_context(i) for i in range(74, 90)],
        ]
    )

    selected, audit = _select_shadow_generation_contexts(
        contexts=rows,
        max_contexts=12,
    )

    assert audit["candidate_reserve_target"] == 3
    assert audit["selected_original_indices"] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 71, 73
    ]
    assert audit["selection_changed_from_prefix"] is True
    assert audit["selected_candidate_context_count"] == 3
    assert audit["selected_unique_candidate_modifier_count"] == 3
    assert audit["scientific_quality_ranking_performed"] is False
    assert audit["production_selection_changed"] is False
    assert [row.context_id for row in selected][-3:] == [
        "context:32",
        "context:71",
        "context:73",
    ]


def test_selection_deduplicates_candidate_modifier_reservation_by_component():
    rows = tuple(
        [
            *[_context(i) for i in range(1, 20)],
            _context(20, "candidate:a"),
            _context(21, "candidate:a"),
            _context(22, "candidate:b"),
            _context(23, "candidate:c"),
        ]
    )

    _, audit = _select_shadow_generation_contexts(
        contexts=rows,
        max_contexts=8,
    )

    assert audit["candidate_reserve_target"] == 2
    assert audit["selected_unique_candidate_modifier_count"] == 2
    assert audit["selected_candidate_modifier_component_ids"] == [
        "candidate:a",
        "candidate:b",
    ]
