from __future__ import annotations

import hashlib

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    TaskBridgeCompositeCandidate,
)


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(
            raw
        ).hexdigest()[:20]
    )


def materialize_task_bridge_composite_axis(
    *,
    composite: TaskBridgeCompositeCandidate,
    source_axis: DiscoveryAxis,
    requested_source: str,
    requested_target: str,
    axis_rank: int,
) -> DiscoveryAxis:
    """
    Materialize one inspiration-only source/target candidate
    composition as a task-preserving DiscoveryAxis.

    Epistemic contract
    ------------------
    - requested_source/requested_target define the question nucleus;
    - component candidate relations remain unverified inspiration;
    - shared mediator tokens justify search composition only;
    - no component relation is promoted to positive evidence;
    - resulting axis remains requires_verification=True.
    """

    requested_source = str(
        requested_source
    ).strip()

    requested_target = str(
        requested_target
    ).strip()

    if not requested_source:
        raise ValueError(
            "requested_source must be non-empty"
        )

    if not requested_target:
        raise ValueError(
            "requested_target must be non-empty"
        )

    if axis_rank < 1:
        raise ValueError(
            "axis_rank must be >= 1"
        )

    provenance_candidate_unit_id = (
        composite.provenance_candidate_unit_id
        or composite.source_unit_id
    )

    if (
        source_axis.candidate_unit_id
        != provenance_candidate_unit_id
    ):
        raise ValueError(
            "source_axis candidate unit does not "
            "match composite provenance candidate unit"
        )

    if not composite.shared_mediator_tokens:
        raise ValueError(
            "composite requires at least one "
            "shared mediator token"
        )

    source = composite.source_relation
    target = composite.target_relation

    is_component_topology = (
        composite.composition_mode
        ==
        "relation_component_topology_v1"
    )

    source_authority_label = (
        "CONFIRMED KNOWN COMPONENT"
        if (
            composite
            .source_component_authority
            == "confirmed_known"
        )
        else "CANDIDATE INSPIRATION COMPONENT"
    )

    target_authority_label = (
        "CONFIRMED KNOWN COMPONENT"
        if (
            composite
            .target_component_authority
            == "confirmed_known"
        )
        else "CANDIDATE INSPIRATION COMPONENT"
    )

    source_render_label = (
        source_authority_label
        if is_component_topology
        else "UNVERIFIED SOURCE RELATION"
    )

    target_render_label = (
        target_authority_label
        if is_component_topology
        else "UNVERIFIED TARGET RELATION"
    )

    bridge_label = (
        "relation-component topology"
        if is_component_topology
        else "composed candidate bridge"
    )

    source_descriptor = (
        "source component"
        if is_component_topology
        else "source candidate"
    )

    target_descriptor = (
        "target component"
        if is_component_topology
        else "target candidate"
    )

    source_sro = (
        f"{source.proposed_subject} | "
        f"{source.proposed_relation} | "
        f"{source.proposed_object}"
    )

    target_sro = (
        f"{target.proposed_subject} | "
        f"{target.proposed_relation} | "
        f"{target.proposed_object}"
    )

    mediator = ", ".join(
        composite.shared_mediator_tokens
    )

    axis_id = _stable_id(
        "task_bridge_composite_axis",
        composite.composite_id,
        requested_source,
        requested_target,
        axis_rank,
    )

    return DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=axis_rank,

        inspiration_id=(
            source_axis.inspiration_id
        ),

        source_path_id=(
            source_axis.source_path_id
        ),

        # Keep a real candidate-unit provenance anchor that
        # existing downstream machinery can resolve.
        candidate_unit_id=(
            source_axis.candidate_unit_id
        ),

        label=(
            f"{requested_source} → "
            f"{requested_target} via "
            f"{bridge_label} [{mediator}]; "
            f"{source_descriptor}: {source_sro}; "
            f"{target_descriptor}: {target_sro}"
        ),

        entry_anchor_id=(
            source_axis.entry_anchor_id
        ),

        entry_anchor_label=(
            requested_source
        ),

        exit_anchor_id=(
            source_axis.exit_anchor_id
        ),

        exit_anchor_label=(
            requested_target
        ),

        proposed_subject=(
            requested_source
        ),

        proposed_relation=(
            (
                "MAY_RELATE_TO_VIA_"
                "RELATION_COMPONENT_TOPOLOGY"
            )
            if is_component_topology
            else (
                "MAY_RELATE_TO_VIA_"
                "COMPOSED_CANDIDATE_BRIDGE"
            )
        ),

        proposed_object=(
            requested_target
        ),

        rendered_path=(
            f"{requested_source}"
            f" -> [{source_render_label}: "
            f"{source_sro}]"
            f" -> [SHARED MEDIATOR: {mediator}]"
            f" -> [{target_render_label}: "
            f"{target_sro}]"
            f" -> {requested_target}"
        ),

        source_mode=(
            (
                "task_conditioned_"
                "relation_component_topology"
            )
            if is_component_topology
            else (
                "task_conditioned_"
                "composite_bridge_projection"
            )
        ),

        exploration_score=float(
            source_axis.exploration_score
        ),

        candidate_unit_score=float(
            source_axis.candidate_unit_score
        ),

        planner_score=float(
            composite.compatibility_score
        ),

        mechanistic_continuity_band=(
            source_axis
            .mechanistic_continuity_band
        ),

        generic_entity_fraction=float(
            source_axis.generic_entity_fraction
        ),

        registry_hop_fraction=float(
            source_axis.registry_hop_fraction
        ),

        grounding_semantic_overlap=float(
            source_axis
            .grounding_semantic_overlap
        ),

        reaction_domain_switch_penalty=float(
            source_axis
            .reaction_domain_switch_penalty
        ),

        requires_verification=True,

        reason_codes=[
            *source_axis.reason_codes,

            *(
                composite.reason_codes
                if is_component_topology
                else []
            ),

            "task_bridge_composite_projection",
            "question_relation_nucleus_preserved",

            *(
                [
                    "relation_component_topology_projection",
                    "component_authority_preserved",
                    "topology_inspiration_only",
                ]
                if is_component_topology
                else [
                    "component_relations_inspiration_only",
                ]
            ),

            "shared_mediator_required",
            "composite_requires_verification",

            (
                "composite_id:"
                + composite.composite_id
            ),

            (
                (
                    "provenance_candidate_unit:"
                    if is_component_topology
                    else "source_candidate_unit:"
                )
                + provenance_candidate_unit_id
            ),

            (
                (
                    "target_component:"
                    if is_component_topology
                    else "target_candidate_unit:"
                )
                + composite.target_unit_id
            ),

            (
                "shared_mediators:"
                + ",".join(
                    composite.shared_mediator_tokens
                )
            ),

            *(
                [
                    (
                        "topology_id:"
                        + str(
                            composite.topology_id
                        )
                    ),
                    (
                        "source_component_id:"
                        + composite.source_component_id
                    ),
                    (
                        "target_component_id:"
                        + composite.target_component_id
                    ),
                    (
                        "source_component_authority:"
                        + composite.source_component_authority
                    ),
                    (
                        "target_component_authority:"
                        + composite.target_component_authority
                    ),
                ]
                if is_component_topology
                else []
            ),
        ],
    )
