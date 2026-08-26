from __future__ import annotations

from pipeline_core.discovery.explorer_packet import (
    _step_relation_matches_evidence,
)


def test_exact_navigation_relation_requires_exact_evidence_relation() -> None:
    assert _step_relation_matches_evidence(
        step_relation="HAS_ARCHITECTURE",
        evidence_relation="HAS_ARCHITECTURE",
    )

    assert not _step_relation_matches_evidence(
        step_relation="HAS_ARCHITECTURE",
        evidence_relation="HAS_COMPONENT",
    )


def test_multi_relation_is_aggregate_marker_not_exact_scientific_relation() -> None:
    assert _step_relation_matches_evidence(
        step_relation="MULTI_RELATION",
        evidence_relation="HAS_ARCHITECTURE",
    )

    assert _step_relation_matches_evidence(
        step_relation="MULTI_RELATION",
        evidence_relation="HAS_COMPONENT",
    )
