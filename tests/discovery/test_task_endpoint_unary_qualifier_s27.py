from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentProvenance,
    RelationComponentView,
)
from pipeline_core.discovery.task_endpoint_unary_qualifier import (
    UnaryEndpointQualifierStatus,
    build_task_endpoint_unary_qualifier_obligations,
    build_unary_endpoint_qualifier_obligation_view,
)


def _component(
    component_id: str,
    subject: str,
    object_value: str,
) -> RelationComponentView:
    accepted_pattern_id = f"accepted::{component_id}"

    return RelationComponentView(
        component_id=component_id,
        label=component_id,
        subject=subject,
        relation="VARIES_WITH",
        object=object_value,
        authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        provenance=RelationComponentProvenance(
            source_kind="accepted_pattern",
            source_id=accepted_pattern_id,
            paper_id="paper-1",
            chunk_id="chunk-1",
            document_id="doc-1",
        ),
        accepted_pattern_id=accepted_pattern_id,
    )


def test_q3_style_unary_qualifier_obstruction_is_diagnostic_only():
    components = (
        _component(
            "c1",
            "plasmonic response",
            "excitation wavelength",
        ),
        _component(
            "c2",
            "plasmonic optical response",
            "nanogap size",
        ),
    )

    view = build_unary_endpoint_qualifier_obligation_view(
        components=components,
        endpoint="broad plasmonic response",
    )

    assert (
        view.status
        == UnaryEndpointQualifierStatus
        .SINGLE_TOKEN_OBSTRUCTION_CANDIDATE
    )
    assert view.candidate_qualifier_surface == "broad"
    assert view.candidate_qualifier_tokens == ["broad"]
    assert view.candidate_core_endpoint == "plasmonic response"
    assert view.candidate_core_binding_count == 2

    assert view.diagnostic_only is True
    assert view.blocking is False
    assert view.eligibility_authority is False
    assert view.selection_authority is False
    assert view.rejection_authority is False
    assert view.endpoint_substitution_authority is False
    assert view.task_filter_relaxed is False
    assert view.positive_premise_authority_created is False
    assert view.novelty_authority_created is False


def test_full_noncoordinated_endpoint_does_not_create_candidate():
    components = (
        _component(
            "c1",
            "nanostructure shape",
            "local field enhancement",
        ),
    )

    view = build_unary_endpoint_qualifier_obligation_view(
        components=components,
        endpoint="nanostructure shape",
    )

    assert (
        view.status
        == UnaryEndpointQualifierStatus.FULL_ENDPOINT_BINDS
    )
    assert view.full_binding_count == 1
    assert view.candidate_qualifier_surface is None
    assert view.candidate_core_endpoint is None


def test_coordinated_endpoint_is_left_to_facet_coverage_lane():
    components = (
        _component(
            "c1",
            "SERS hotspot intensity",
            "SERS response",
        ),
    )

    view = build_unary_endpoint_qualifier_obligation_view(
        components=components,
        endpoint="electromagnetic hotspot location and intensity",
    )

    assert (
        view.status
        == UnaryEndpointQualifierStatus
        .NOT_APPLICABLE_COORDINATED_ENDPOINT
    )
    assert view.coordinated_endpoint is True
    assert view.candidate_qualifier_surface is None


def test_task_pair_preserves_all_authority_invariants():
    components = (
        _component(
            "c1",
            "plasmonic response",
            "excitation wavelength",
        ),
    )

    result = build_task_endpoint_unary_qualifier_obligations(
        components=components,
        requested_source="broad plasmonic response",
        requested_target="excitation wavelength",
    )

    assert (
        result.source.status
        == UnaryEndpointQualifierStatus
        .SINGLE_TOKEN_OBSTRUCTION_CANDIDATE
    )
    assert (
        result.target.status
        == UnaryEndpointQualifierStatus.FULL_ENDPOINT_BINDS
    )

    assert result.diagnostic_only is True
    assert result.blocking is False
    assert result.eligibility_changed is False
    assert result.threshold_changed is False
    assert result.task_filter_relaxed is False
    assert result.endpoint_substitution_performed is False
    assert result.positive_premise_authority_created is False
    assert result.novelty_authority_created is False
    assert result.selection_authority_created is False
    assert result.rejection_authority_created is False
