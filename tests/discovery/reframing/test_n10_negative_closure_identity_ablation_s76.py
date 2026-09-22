from __future__ import annotations

from pipeline_core.discovery.reframing.n10_negative_closure_identity_ablation import (
    _global_constituent_identity_match,
    analyze_slot,
)


def test_dispersed_constituent_match_removes_only_local_window_requirement():
    anchor = ("architecture conditioned hotspot accessibility",)
    abstract = (
        "The architecture controls morphology across the substrate. "
        "Hotspot density is then quantified under repeated measurements. "
        "Accessibility of analytes is assessed independently."
    )
    assert _global_constituent_identity_match(
        abstract=abstract,
        anchors=anchor,
    )


def test_dispersed_constituent_match_still_requires_75_percent_identity_tokens():
    anchor = ("architecture conditioned hotspot accessibility",)
    abstract = "Architecture alone is discussed in this abstract."
    assert not _global_constituent_identity_match(
        abstract=abstract,
        anchors=anchor,
    )


def test_ablation_never_changes_positive_established_slot():
    target = {
        "slot": "FULL_RELATION",
        "identity_anchor_terms": [
            "architecture conditioned hotspot accessibility"
        ],
    }
    review = {
        "slot": "FULL_RELATION",
        "evidence_state": "ESTABLISHED",
        "successful_query_count": 1,
        "positive_work_ids": ["w1"],
        "matches": [
            {
                "work_id": "w1",
                "abstract_available": True,
                "relationship": "ESTABLISHES_SLOT",
            }
        ],
    }
    works = {
        "w1": {
            "work_id": "w1",
            "abstract": "Completely different wording.",
        }
    }
    result = analyze_slot(
        target=target,
        slot_review=review,
        works_by_id=works,
    )
    assert result.strict_shadow_state == "ESTABLISHED"
    assert result.dispersed_constituent_shadow_state == "ESTABLISHED"
    assert result.search_coverage_only_shadow_state == "ESTABLISHED"


def test_search_coverage_control_can_close_negative_without_identity():
    target = {
        "slot": "FULL_RELATION",
        "identity_anchor_terms": [
            "architecture conditioned hotspot accessibility"
        ],
    }
    review = {
        "slot": "FULL_RELATION",
        "evidence_state": "UNASSESSED",
        "successful_query_count": 3,
        "positive_work_ids": [],
        "matches": [
            {
                "work_id": f"w{i}",
                "abstract_available": True,
                "relationship": "COMPONENT_ONLY",
            }
            for i in range(3)
        ],
    }
    works = {
        f"w{i}": {
            "work_id": f"w{i}",
            "abstract": "SERS substrate component evidence.",
        }
        for i in range(3)
    }
    result = analyze_slot(
        target=target,
        slot_review=review,
        works_by_id=works,
    )
    assert result.strict_shadow_state == "UNASSESSED"
    assert result.dispersed_constituent_shadow_state == "UNASSESSED"
    assert result.search_coverage_only_shadow_state == "NOT_FOUND"
