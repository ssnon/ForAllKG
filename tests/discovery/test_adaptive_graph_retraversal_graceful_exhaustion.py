from scripts.discovery.run_adaptive_graph_retraversal_shadow import (
    build_structural_exhaustion_summary,
)


def test_structural_exhaustion_is_bounded_non_authoritative_result():
    summary = build_structural_exhaustion_summary(
        request_id="req",
        source_context_id="ctx_old",
        output_context_id="ctx_new",
        selected_path_count=6,
        new_eligible_premise_ids=[],
        selected_paper_expansion_count=2,
        paper_expansion_candidate_count=4,
        selected_traversal="/tmp/selected.json",
        output_context="/tmp/context.json",
        context_delta_audit="/tmp/delta.json",
        context_delta_audit_id="audit",
    )

    assert (
        summary["status"]
        == "NO_STRUCTURALLY_NEW_ELIGIBLE_POSITIVE_PREMISE"
    )
    assert summary["structurally_new_eligible_premise_count"] == 0
    assert summary["new_eligible_premise_count"] == 0
    assert summary["external_prior_art_as_positive_premise"] is False
    assert summary["external_boundary_used_for_positive_path_selection"] is False
    assert summary["canonical_graph_mutated"] is False


def test_structural_exhaustion_preserves_new_ids_without_promoting_them():
    summary = build_structural_exhaustion_summary(
        request_id=None,
        source_context_id="ctx_old",
        output_context_id="ctx_new",
        selected_path_count=3,
        new_eligible_premise_ids=["s_new"],
        selected_paper_expansion_count=0,
        paper_expansion_candidate_count=1,
        selected_traversal="selected.json",
        output_context="context.json",
        context_delta_audit="delta.json",
        context_delta_audit_id=None,
    )

    assert summary["new_eligible_premise_ids"] == ["s_new"]
    assert summary["new_eligible_premise_count"] == 1
    assert summary["structurally_new_eligible_premise_ids"] == []
    assert summary["structurally_new_eligible_premise_count"] == 0
