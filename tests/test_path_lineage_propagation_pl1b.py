from __future__ import annotations

from dac_her.path_lineage_propagation import (
    MinimalPathLineagePropagator,
    _minimum_full_cover,
)
from pipeline_core.discovery.path_lineage_diagnostics import (
    StatementPathLineageDiagnostic,
    StatementPathOverlap,
)


def _diag() -> StatementPathLineageDiagnostic:
    return StatementPathLineageDiagnostic(
        statement_id="S",
        text="test",
        epistemic_role="reported",
        claim_kind="mechanism",
        eligible_as_premise=True,
        scientific_support_node_ids=["n1", "n2", "n3"],
        scientific_support_edge_ids=["e1", "e2"],
        deterministic_attribution_candidate_path_ids=["P1", "P2", "P3"],
        deterministic_attribution_candidate_count=3,
        path_overlaps=[
            StatementPathOverlap(
                path_id="P1",
                bundle_rank=1,
                path_type="CROSS_PAPER_MECHANISTIC",
                mechanistic_content="high",
                mechanism_bearing=True,
                navigation_edge_fraction=0.8,
                overlapping_scientific_node_ids=["n1", "n2"],
                overlapping_scientific_edge_ids=["e1"],
                node_overlap_count=2,
                edge_overlap_count=1,
                statement_node_coverage=2/3,
                statement_edge_coverage=0.5,
                relationship="edge_supported_partial_route",
                attribution_candidate=True,
                mechanistic_attribution_candidate=True,
            ),
            StatementPathOverlap(
                path_id="P2",
                bundle_rank=2,
                path_type="CROSS_PAPER_MECHANISTIC",
                mechanistic_content="high",
                mechanism_bearing=True,
                navigation_edge_fraction=0.7,
                overlapping_scientific_node_ids=["n2", "n3"],
                overlapping_scientific_edge_ids=["e2"],
                node_overlap_count=2,
                edge_overlap_count=1,
                statement_node_coverage=2/3,
                statement_edge_coverage=0.5,
                relationship="edge_supported_partial_route",
                attribution_candidate=True,
                mechanistic_attribution_candidate=True,
            ),
            StatementPathOverlap(
                path_id="P3",
                bundle_rank=3,
                path_type="CROSS_PAPER_MECHANISTIC",
                mechanistic_content="high",
                mechanism_bearing=True,
                navigation_edge_fraction=0.75,
                overlapping_scientific_node_ids=["n1", "n2", "n3"],
                overlapping_scientific_edge_ids=["e1", "e2"],
                node_overlap_count=3,
                edge_overlap_count=2,
                statement_node_coverage=1.0,
                statement_edge_coverage=1.0,
                relationship="exact_support_route",
                attribution_candidate=True,
                mechanistic_attribution_candidate=True,
            ),
        ],
    )


def test_exact_minimum_cover_prefers_single_full_route():
    basis, universe, cover, tie_count = _minimum_full_cover(_diag())
    assert basis == "scientific_edges"
    assert universe == {"e1", "e2"}
    assert cover is not None
    assert cover.path_ids == ["P3"]
    assert cover.cardinality == 1
    assert cover.support_coverage == 1.0
    assert tie_count == 1


def test_tie_break_prefers_lower_navigation_after_scientific_equivalence():
    diag = _diag().model_copy(
        update={
            "deterministic_attribution_candidate_path_ids": ["P3", "P4"],
            "deterministic_attribution_candidate_count": 2,
            "path_overlaps": [
                _diag().path_overlaps[2],
                _diag().path_overlaps[2].model_copy(
                    update={
                        "path_id": "P4",
                        "bundle_rank": 4,
                        "navigation_edge_fraction": 0.6,
                    }
                ),
            ],
        }
    )
    _, _, cover, tie_count = _minimum_full_cover(diag)
    assert cover is not None
    assert cover.path_ids == ["P4"]
    assert tie_count == 2


def test_policy_keeps_propagation_as_provenance_only():
    from dac_her.path_lineage_propagation import PathLineagePropagationPolicy
    policy = PathLineagePropagationPolicy()
    assert policy.scientific_support_content_changed is False
    assert policy.premise_eligibility_changed is False
    assert policy.preserve_existing_explicit_paths is True
