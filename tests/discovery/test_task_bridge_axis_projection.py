from __future__ import annotations

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.task_bridge_axis_projection import (
    build_task_bridge_axes,
    project_axis_to_task_bridge,
    select_diverse_bridge_basis_axes,
)


def _axis(
    *,
    axis_id: str,
    rank: int,
    subject: str,
    relation: str,
    object_: str,
    score: float,
) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=rank,
        inspiration_id=f"inspiration:{axis_id}",
        source_path_id=f"path:{axis_id}",
        candidate_unit_id=f"unit:{axis_id}",
        label=(
            f"{subject} {relation} {object_}"
        ),
        entry_anchor_id="entry",
        entry_anchor_label="old entry",
        exit_anchor_id="exit",
        exit_anchor_label="old exit",
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=object_,
        rendered_path=(
            f"{subject} -> {relation} -> {object_}"
        ),
        source_mode="exploratory",
        exploration_score=0.5,
        candidate_unit_score=0.5,
        planner_score=score,
        mechanistic_continuity_band="high",
        generic_entity_fraction=0.0,
        registry_hop_fraction=0.0,
        grounding_semantic_overlap=0.0,
        reaction_domain_switch_penalty=0.0,
        requires_verification=True,
        reason_codes=[
            "candidate_unit_traversal"
        ],
    )


def test_projection_preserves_requested_relation_nucleus():
    source_axis = _axis(
        axis_id="a",
        rank=1,
        subject="laser power",
        relation="VARIES_WITH",
        object_="SERS spectra",
        score=0.8,
    )

    projected, record = (
        project_axis_to_task_bridge(
            source_axis=source_axis,
            requested_source="HaB binding sites",
            requested_target="SERS behavior",
            projected_rank=1,
        )
    )

    assert (
        projected.proposed_subject
        == "HaB binding sites"
    )

    assert (
        projected.proposed_object
        == "SERS behavior"
    )

    assert (
        projected.proposed_relation
        == "MAY_RELATE_TO_VIA_CANDIDATE_BRIDGE"
    )

    assert (
        "laser power"
        in projected.rendered_path
    )

    assert (
        "VARIES_WITH"
        in projected.rendered_path
    )

    assert (
        "SERS spectra"
        in projected.rendered_path
    )

    assert projected.requires_verification

    assert (
        "candidate_bridge_inspiration_only"
        in projected.reason_codes
    )

    assert (
        record.bridge_subject
        == "laser power"
    )


def test_projection_is_deterministic():
    axis = _axis(
        axis_id="a",
        rank=1,
        subject="x",
        relation="R",
        object_="y",
        score=0.8,
    )

    p1, _ = project_axis_to_task_bridge(
        source_axis=axis,
        requested_source="source",
        requested_target="target",
        projected_rank=1,
    )

    p2, _ = project_axis_to_task_bridge(
        source_axis=axis,
        requested_source="source",
        requested_target="target",
        projected_rank=1,
    )

    assert p1.axis_id == p2.axis_id


def test_first_basis_preserves_best_planner_score():
    a = _axis(
        axis_id="a",
        rank=1,
        subject="a",
        relation="R",
        object_="b",
        score=0.95,
    )

    b = _axis(
        axis_id="b",
        rank=2,
        subject="c",
        relation="R",
        object_="d",
        score=0.70,
    )

    selected = (
        select_diverse_bridge_basis_axes(
            axes=[a, b],
            retained_axes=1,
        )
    )

    assert selected[0].source_axis_id == "a"


def test_later_basis_prefers_diversity():
    first = _axis(
        axis_id="first",
        rank=1,
        subject="binding",
        relation="AFFECTS",
        object_="sers",
        score=0.90,
    )

    near = _axis(
        axis_id="near",
        rank=2,
        subject="binding",
        relation="AFFECTS",
        object_="sers intensity",
        score=0.89,
    )

    diverse = _axis(
        axis_id="diverse",
        rank=3,
        subject="plasmon resonance",
        relation="VARIES_WITH",
        object_="aspect ratio",
        score=0.70,
    )

    selected = (
        select_diverse_bridge_basis_axes(
            axes=[
                first,
                near,
                diverse,
            ],
            retained_axes=2,
        )
    )

    assert (
        selected[0].source_axis_id
        == "first"
    )

    assert (
        selected[1].source_axis_id
        == "diverse"
    )


def test_build_returns_at_most_requested_axes():
    axes = [
        _axis(
            axis_id="a",
            rank=1,
            subject="a",
            relation="R",
            object_="b",
            score=0.9,
        ),
        _axis(
            axis_id="b",
            rank=2,
            subject="c",
            relation="S",
            object_="d",
            score=0.8,
        ),
    ]

    projected, records = (
        build_task_bridge_axes(
            axes=axes,
            requested_source="requested source",
            requested_target="requested target",
            retained_axes=3,
        )
    )

    assert len(projected) == 2
    assert len(records) == 2

    assert all(
        axis.proposed_subject
        == "requested source"
        for axis in projected
    )

    assert all(
        axis.proposed_object
        == "requested target"
        for axis in projected
    )
