from __future__ import annotations

import pytest

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    TaskBridgeCompositeCandidate,
)
from pipeline_core.discovery.task_bridge_composite_axis import (
    materialize_task_bridge_composite_axis,
)


def _source_axis() -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="axis:source",
        axis_rank=1,
        inspiration_id="inspiration:source",
        source_path_id="path:source",
        candidate_unit_id="unit:source",
        label="source relation",
        entry_anchor_id="entry",
        entry_anchor_label="old source",
        exit_anchor_id="exit",
        exit_anchor_label="old target",
        proposed_subject="structural order",
        proposed_relation="VARIES_WITH",
        proposed_object="size and shape uniformity",
        rendered_path="old path",
        source_mode="exploratory",
        exploration_score=0.5,
        candidate_unit_score=0.5,
        planner_score=0.7,
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


def _composite() -> TaskBridgeCompositeCandidate:
    return TaskBridgeCompositeCandidate(
        composite_id="composite:test",
        source_unit_id="unit:source",
        target_unit_id="unit:target",
        source_overlap_tokens=[
            "structural",
        ],
        target_overlap_tokens=[
            "plasmonic",
        ],
        source_mediator_tokens=[
            "shape",
            "size",
            "uniformity",
        ],
        target_mediator_tokens=[
            "composition",
            "shape",
            "size",
        ],
        shared_mediator_tokens=[
            "shape",
            "size",
        ],
        source_relation=(
            CandidateRelationView(
                unit_id="unit:source",
                proposed_subject=(
                    "superlattice structural order"
                ),
                proposed_relation="VARIES_WITH",
                proposed_object=(
                    "size and shape uniformity"
                ),
            )
        ),
        target_relation=(
            CandidateRelationView(
                unit_id="unit:target",
                proposed_subject=(
                    "plasmonic properties"
                ),
                proposed_relation="VARIES_WITH",
                proposed_object=(
                    "nanostructure size shape "
                    "composition arrangement"
                ),
            )
        ),
        compatibility_score=4.1,
    )


def test_materialization_preserves_task_nucleus():
    axis = materialize_task_bridge_composite_axis(
        composite=_composite(),
        source_axis=_source_axis(),
        requested_source=(
            "HaB binding sites structural motif"
        ),
        requested_target=(
            "SERS plasmonic behavior"
        ),
        axis_rank=1,
    )

    assert (
        axis.proposed_subject
        == "HaB binding sites structural motif"
    )

    assert (
        axis.proposed_object
        == "SERS plasmonic behavior"
    )

    assert (
        axis.proposed_relation
        == (
            "MAY_RELATE_TO_VIA_"
            "COMPOSED_CANDIDATE_BRIDGE"
        )
    )


def test_materialization_preserves_both_component_relations():
    axis = materialize_task_bridge_composite_axis(
        composite=_composite(),
        source_axis=_source_axis(),
        requested_source="requested source",
        requested_target="requested target",
        axis_rank=1,
    )

    assert (
        "superlattice structural order"
        in axis.rendered_path
    )

    assert (
        "plasmonic properties"
        in axis.rendered_path
    )

    assert (
        "shape, size"
        in axis.rendered_path
    )


def test_materialization_keeps_real_source_candidate_provenance():
    axis = materialize_task_bridge_composite_axis(
        composite=_composite(),
        source_axis=_source_axis(),
        requested_source="requested source",
        requested_target="requested target",
        axis_rank=1,
    )

    assert (
        axis.candidate_unit_id
        == "unit:source"
    )

    assert axis.requires_verification

    assert (
        "component_relations_inspiration_only"
        in axis.reason_codes
    )


def test_materialization_rejects_wrong_source_unit():
    wrong = _source_axis().model_copy(
        update={
            "candidate_unit_id":
                "unit:different"
        }
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        materialize_task_bridge_composite_axis(
            composite=_composite(),
            source_axis=wrong,
            requested_source="source",
            requested_target="target",
            axis_rank=1,
        )


def test_materialization_is_deterministic():
    kwargs = dict(
        composite=_composite(),
        source_axis=_source_axis(),
        requested_source="source",
        requested_target="target",
        axis_rank=1,
    )

    a = materialize_task_bridge_composite_axis(
        **kwargs
    )

    b = materialize_task_bridge_composite_axis(
        **kwargs
    )

    assert a.axis_id == b.axis_id
