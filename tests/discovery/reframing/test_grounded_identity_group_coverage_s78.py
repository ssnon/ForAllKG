from __future__ import annotations

from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotation,
    GroundedIdentityConstituentGroup,
    GroundedIdentityConstituentSpan,
)
from pipeline_core.discovery.reframing.grounded_identity_group_coverage import (
    analyze_slot_group_coverage,
)


def _group(label: str, text: str) -> GroundedIdentityConstituentGroup:
    return GroundedIdentityConstituentGroup(
        label=label,
        spans=[
            GroundedIdentityConstituentSpan(
                candidate_id=label,
                candidate_ref="CANDIDATE_01",
                exact_source_text=text,
                matched_source_paths=["scientific_proposal"],
            )
        ],
    )


def _annotation() -> GroundedIdentityAnnotation:
    return GroundedIdentityAnnotation(
        claim_id="c1",
        identity_term="architecture-conditioned hotspot accessibility",
        groups=[
            _group("architecture", "architecture"),
            _group("hotspot", "hotspot"),
            _group("accessibility", "accessibility"),
        ],
        rationale="test",
    )


def test_group_coverage_distinguishes_individual_from_joint_matches():
    review = {
        "slot": "FULL_RELATION",
        "matches": [
            {
                "work_id": "w1",
                "abstract_available": True,
                "relationship": "COMPONENT_ONLY",
            },
            {
                "work_id": "w2",
                "abstract_available": True,
                "relationship": "COMPONENT_ONLY",
            },
            {
                "work_id": "w3",
                "abstract_available": True,
                "relationship": "PARTIAL_SLOT_RELATION",
            },
        ],
    }
    works = {
        "w1": {
            "abstract": "Architecture and hotspot design are discussed."
        },
        "w2": {
            "abstract": "Hotspot accessibility is experimentally measured."
        },
        "w3": {
            "abstract": (
                "Architecture controls hotspot accessibility in the substrate."
            )
        },
    }

    result = analyze_slot_group_coverage(
        slot_review=review,
        annotation=_annotation(),
        works_by_id=works,
    )

    counts = {
        row.group_label: row.matching_material_work_count
        for row in result.groups
    }
    assert counts == {
        "architecture": 2,
        "hotspot": 3,
        "accessibility": 2,
    }
    assert result.all_groups_matching_work_ids == ["w3"]
    assert result.all_groups_matching_work_count == 1


def test_group_coverage_reports_zero_for_missing_constituent():
    review = {
        "slot": "BRIDGE_RELATION",
        "matches": [
            {
                "work_id": "w1",
                "abstract_available": True,
                "relationship": "COMPONENT_ONLY",
            }
        ],
    }
    works = {
        "w1": {
            "abstract": "Architecture and hotspot density are discussed."
        }
    }

    result = analyze_slot_group_coverage(
        slot_review=review,
        annotation=_annotation(),
        works_by_id=works,
    )

    counts = {
        row.group_label: row.matching_material_work_count
        for row in result.groups
    }
    assert counts["accessibility"] == 0
    assert result.all_groups_matching_work_count == 0
