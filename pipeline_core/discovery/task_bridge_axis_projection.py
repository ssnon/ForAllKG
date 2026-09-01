from __future__ import annotations

import hashlib
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.task_responsive_axis_shadow import (
    axis_signature_tokens,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskBridgeBasisSelection(StrictModel):
    source_axis_id: str
    source_axis_rank: int
    selected_rank: int
    planner_score: float
    min_basis_diversity_distance: float


class TaskBridgeProjectionRecord(StrictModel):
    source_axis_id: str
    projected_axis_id: str
    source_axis_rank: int
    projected_axis_rank: int

    requested_source: str
    requested_target: str

    bridge_subject: str
    bridge_relation: str
    bridge_object: str

    planner_score: float


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


def _jaccard_distance(
    left: Sequence[str],
    right: Sequence[str],
) -> float:
    a = set(left)
    b = set(right)

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return 1.0 - (
        len(a & b)
        / len(union)
    )


def select_diverse_bridge_basis_axes(
    *,
    axes: Sequence[DiscoveryAxis],
    retained_axes: int = 3,
) -> tuple[
    TaskBridgeBasisSelection,
    ...,
]:
    """
    Select scientifically different existing candidate relations
    as bridge inspiration.

    First choice preserves the best existing planner score.
    Subsequent choices prioritize distance from already selected
    candidate S/R/O signatures, then planner score, then original
    frozen axis rank.

    No semantic-distinctiveness or scientific-novelty outcome is
    consumed.
    """

    if retained_axes < 1:
        raise ValueError(
            "retained_axes must be >= 1"
        )

    by_id = {}

    for axis in axes:
        if axis.axis_id in by_id:
            raise ValueError(
                "duplicate axis_id"
            )
        by_id[axis.axis_id] = axis

    remaining = list(axes)
    selected: list[DiscoveryAxis] = []
    out: list[TaskBridgeBasisSelection] = []

    while (
        remaining
        and
        len(selected) < retained_axes
    ):
        ranked = []

        for axis in remaining:
            sig = axis_signature_tokens(
                axis
            )

            if not selected:
                distance = 1.0
            else:
                distance = min(
                    _jaccard_distance(
                        sig,
                        axis_signature_tokens(
                            prior
                        ),
                    )
                    for prior in selected
                )

            if not selected:
                key = (
                    float(axis.planner_score),
                    -int(axis.axis_rank),
                )
            else:
                key = (
                    float(distance),
                    float(axis.planner_score),
                    -int(axis.axis_rank),
                )

            ranked.append(
                (
                    key,
                    axis,
                    distance,
                )
            )

        ranked.sort(
            key=lambda row: row[0],
            reverse=True,
        )

        _key, winner, distance = ranked[0]

        selected.append(
            winner
        )

        remaining = [
            axis
            for axis in remaining
            if axis.axis_id != winner.axis_id
        ]

        out.append(
            TaskBridgeBasisSelection(
                source_axis_id=winner.axis_id,
                source_axis_rank=winner.axis_rank,
                selected_rank=len(out) + 1,
                planner_score=float(
                    winner.planner_score
                ),
                min_basis_diversity_distance=float(
                    distance
                ),
            )
        )

    return tuple(out)


def project_axis_to_task_bridge(
    *,
    source_axis: DiscoveryAxis,
    requested_source: str,
    requested_target: str,
    projected_rank: int,
) -> tuple[
    DiscoveryAxis,
    TaskBridgeProjectionRecord,
]:
    """
    Convert one existing discovery relation into a task-preserving
    *search projection*.

    The requested source/target define the question relation nucleus.
    The original candidate relation remains inspiration-only and is
    retained in the label/rendered path.

    This projection is NOT positive evidence that requested_source
    actually affects requested_target. It remains verification-required.
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

    if projected_rank < 1:
        raise ValueError(
            "projected_rank must be >= 1"
        )

    bridge_subject = str(
        source_axis.proposed_subject
    ).strip()

    bridge_relation = str(
        source_axis.proposed_relation
    ).strip()

    bridge_object = str(
        source_axis.proposed_object
    ).strip()

    bridge_text = (
        f"{bridge_subject} | "
        f"{bridge_relation} | "
        f"{bridge_object}"
    )

    axis_id = _stable_id(
        "task_bridge_axis",
        source_axis.axis_id,
        requested_source,
        requested_target,
        projected_rank,
    )

    projected = DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=projected_rank,

        inspiration_id=(
            source_axis.inspiration_id
        ),

        source_path_id=(
            source_axis.source_path_id
        ),

        candidate_unit_id=(
            source_axis.candidate_unit_id
        ),

        label=(
            f"{requested_source} → "
            f"{requested_target} "
            f"via candidate bridge: "
            f"{source_axis.label}"
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

        # Deliberately epistemically weak.
        # It states a search relation, not a scientific fact.
        proposed_relation=(
            "MAY_RELATE_TO_VIA_CANDIDATE_BRIDGE"
        ),

        proposed_object=(
            requested_target
        ),

        rendered_path=(
            f"{requested_source}"
            f" -> [UNVERIFIED BRIDGE: "
            f"{bridge_text}]"
            f" -> {requested_target}"
        ),

        source_mode=(
            "task_conditioned_bridge_projection"
        ),

        exploration_score=float(
            source_axis.exploration_score
        ),

        candidate_unit_score=float(
            source_axis.candidate_unit_score
        ),

        planner_score=float(
            source_axis.planner_score
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
            "task_bridge_projection",
            "question_relation_nucleus_preserved",
            "candidate_bridge_inspiration_only",
            "candidate_bridge_requires_verification",
            (
                "source_candidate_axis:"
                + source_axis.axis_id
            ),
        ],
    )

    record = TaskBridgeProjectionRecord(
        source_axis_id=(
            source_axis.axis_id
        ),

        projected_axis_id=axis_id,

        source_axis_rank=(
            source_axis.axis_rank
        ),

        projected_axis_rank=(
            projected_rank
        ),

        requested_source=(
            requested_source
        ),

        requested_target=(
            requested_target
        ),

        bridge_subject=(
            bridge_subject
        ),

        bridge_relation=(
            bridge_relation
        ),

        bridge_object=(
            bridge_object
        ),

        planner_score=float(
            source_axis.planner_score
        ),
    )

    return projected, record


def build_task_bridge_axes(
    *,
    axes: Sequence[DiscoveryAxis],
    requested_source: str,
    requested_target: str,
    retained_axes: int = 3,
) -> tuple[
    tuple[DiscoveryAxis, ...],
    tuple[TaskBridgeProjectionRecord, ...],
]:
    basis = select_diverse_bridge_basis_axes(
        axes=axes,
        retained_axes=retained_axes,
    )

    axis_by_id = {
        axis.axis_id: axis
        for axis in axes
    }

    projected = []
    records = []

    for row in basis:
        axis = axis_by_id[
            row.source_axis_id
        ]

        new_axis, record = (
            project_axis_to_task_bridge(
                source_axis=axis,
                requested_source=(
                    requested_source
                ),
                requested_target=(
                    requested_target
                ),
                projected_rank=(
                    row.selected_rank
                ),
            )
        )

        projected.append(
            new_axis
        )

        records.append(
            record
        )

    return (
        tuple(projected),
        tuple(records),
    )
