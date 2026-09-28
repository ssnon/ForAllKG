from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    DirectRelationPatternAssessment,
    DirectRelationPatternCandidate,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationArgumentSlot,
    RelationComponentAuthority,
    RelationComponentBindingView,
    RelationComponentView,
)
from pipeline_core.discovery.task_backbone_chain import _structured_task_binding
from pipeline_core.discovery.task_bridge_candidate_composition import lexical_tokens


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DirectTaskRelationProjectionMode = Literal[
    "strict_both_opposite_slots",
    "strict_source_anchor_opposite_slot",
    "strict_target_anchor_opposite_slot",
    "distinct_lexical_opposite_slots",
]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _slot_text(
    component: RelationComponentView,
    slot: RelationArgumentSlot,
) -> str:
    return component.subject if slot == "subject" else component.object


def _opposite_slot(slot: RelationArgumentSlot) -> RelationArgumentSlot:
    return "object" if slot == "subject" else "subject"


class LexicalRoleSignal(StrictModel):
    endpoint_text: str
    resolved: bool
    ambiguous: bool = False
    slot: RelationArgumentSlot | None = None
    overlap_tokens: list[str] = Field(default_factory=list)
    task_coverage: float = 0.0
    slot_coverage: float = 0.0

    @model_validator(mode="after")
    def validate_signal(self) -> "LexicalRoleSignal":
        if self.resolved and self.slot is None:
            raise ValueError("resolved lexical role signal requires a slot")
        if self.ambiguous and self.resolved:
            raise ValueError("ambiguous lexical role signal cannot be resolved")
        for value in (self.task_coverage, self.slot_coverage):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError("lexical role coverage must be within [0, 1]")
        return self


class DirectTaskRelationRoleProjection(StrictModel):
    schema_version: str = "direct-task-relation-role-projection-v1"

    requested_source: str
    requested_target: str

    source_role_slot: RelationArgumentSlot
    target_role_slot: RelationArgumentSlot

    source_role_text: str
    target_role_text: str

    projection_mode: DirectTaskRelationProjectionMode

    source_strict_binding: RelationComponentBindingView | None = None
    target_strict_binding: RelationComponentBindingView | None = None

    source_lexical_signal: LexicalRoleSignal
    target_lexical_signal: LexicalRoleSignal

    endpoint_equivalence_authorized: Literal[False] = False
    scientific_identity_asserted: Literal[False] = False
    synonym_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_projection(self) -> "DirectTaskRelationRoleProjection":
        if self.source_role_slot == self.target_role_slot:
            raise ValueError("direct task relation roles must occupy opposite slots")
        if not str(self.source_role_text).strip():
            raise ValueError("source role text must not be empty")
        if not str(self.target_role_text).strip():
            raise ValueError("target role text must not be empty")
        return self


class DirectTaskRelationBackboneView(StrictModel):
    """
    Shadow-only single-component task-relation backbone.

    This represents one CONFIRMED_KNOWN accepted RelationPattern that a
    stable full-question critic classified as DIRECT_ANSWER and whose
    relation arguments can be projected into opposite task roles.

    It creates role-projection authority only. It does not create endpoint
    equivalence, scientific identity, positive-premise, novelty, or
    production-selection authority.
    """

    schema_version: str = "direct-task-relation-backbone-v1"

    topology_id: str
    component: RelationComponentView
    candidate_id: str

    requested_source: str
    requested_target: str

    role_projection: DirectTaskRelationRoleProjection

    task_responsiveness_status: Literal["PASS"] = "PASS"
    task_responsiveness_role: Literal["DIRECT_ANSWER"] = "DIRECT_ANSWER"
    task_class: Literal["DIRECT"] = "DIRECT"
    decision_stable: Literal[True] = True

    epistemic_status: Literal["inspiration_only"] = "inspiration_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True

    endpoint_equivalence_authorized: Literal[False] = False
    scientific_identity_asserted: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False

    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_backbone(self) -> "DirectTaskRelationBackboneView":
        if self.component.authority != RelationComponentAuthority.CONFIRMED_KNOWN:
            raise ValueError(
                "direct task relation backbone requires CONFIRMED_KNOWN component authority"
            )

        projection = self.role_projection

        if projection.requested_source != self.requested_source:
            raise ValueError("source request lost during role projection")
        if projection.requested_target != self.requested_target:
            raise ValueError("target request lost during role projection")

        if projection.source_role_text != _slot_text(
            self.component,
            projection.source_role_slot,
        ):
            raise ValueError("source role projection text/slot mismatch")

        if projection.target_role_text != _slot_text(
            self.component,
            projection.target_role_slot,
        ):
            raise ValueError("target role projection text/slot mismatch")

        return self


def _lexical_role_signal(
    *,
    endpoint_text: str,
    component: RelationComponentView,
) -> LexicalRoleSignal:
    task_tokens = lexical_tokens(endpoint_text)
    rows = []

    for slot in ("subject", "object"):
        slot_text = _slot_text(component, slot)
        slot_tokens = lexical_tokens(slot_text)
        overlap = task_tokens & slot_tokens

        rows.append(
            (
                len(overlap),
                len(overlap) / max(len(task_tokens), 1),
                len(overlap) / max(len(slot_tokens), 1),
                slot,
                sorted(overlap),
            )
        )

    rows.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    best = rows[0]

    if best[0] == 0:
        return LexicalRoleSignal(
            endpoint_text=endpoint_text,
            resolved=False,
        )

    ambiguous = len(rows) > 1 and rows[0][0:3] == rows[1][0:3]

    if ambiguous:
        return LexicalRoleSignal(
            endpoint_text=endpoint_text,
            resolved=False,
            ambiguous=True,
            overlap_tokens=best[4],
            task_coverage=float(best[1]),
            slot_coverage=float(best[2]),
        )

    return LexicalRoleSignal(
        endpoint_text=endpoint_text,
        resolved=True,
        ambiguous=False,
        slot=best[3],
        overlap_tokens=best[4],
        task_coverage=float(best[1]),
        slot_coverage=float(best[2]),
    )


def _strict_binding(
    *,
    endpoint_text: str,
    component: RelationComponentView,
) -> RelationComponentBindingView | None:
    resolved = _structured_task_binding(
        component=component,
        task_endpoint=endpoint_text,
        endpoint_equivalences=(),
    )

    if resolved is None:
        return None

    binding, _mode = resolved
    return binding


def project_direct_task_relation_roles(
    *,
    component: RelationComponentView,
    requested_source: str,
    requested_target: str,
) -> DirectTaskRelationRoleProjection | None:
    source_strict = _strict_binding(
        endpoint_text=requested_source,
        component=component,
    )
    target_strict = _strict_binding(
        endpoint_text=requested_target,
        component=component,
    )

    source_lexical = _lexical_role_signal(
        endpoint_text=requested_source,
        component=component,
    )
    target_lexical = _lexical_role_signal(
        endpoint_text=requested_target,
        component=component,
    )

    source_slot = None
    target_slot = None
    mode = None

    if (
        source_strict is not None
        and target_strict is not None
        and source_strict.task_slot != target_strict.task_slot
    ):
        source_slot = source_strict.task_slot
        target_slot = target_strict.task_slot
        mode = "strict_both_opposite_slots"

    elif source_strict is not None and target_strict is None:
        inferred_target = _opposite_slot(source_strict.task_slot)

        if target_lexical.resolved and target_lexical.slot != inferred_target:
            return None

        source_slot = source_strict.task_slot
        target_slot = inferred_target
        mode = "strict_source_anchor_opposite_slot"

    elif target_strict is not None and source_strict is None:
        inferred_source = _opposite_slot(target_strict.task_slot)

        if source_lexical.resolved and source_lexical.slot != inferred_source:
            return None

        source_slot = inferred_source
        target_slot = target_strict.task_slot
        mode = "strict_target_anchor_opposite_slot"

    elif (
        source_lexical.resolved
        and target_lexical.resolved
        and source_lexical.slot != target_lexical.slot
    ):
        source_slot = source_lexical.slot
        target_slot = target_lexical.slot
        mode = "distinct_lexical_opposite_slots"

    if source_slot is None or target_slot is None or mode is None:
        return None

    return DirectTaskRelationRoleProjection(
        requested_source=requested_source,
        requested_target=requested_target,
        source_role_slot=source_slot,
        target_role_slot=target_slot,
        source_role_text=_slot_text(component, source_slot),
        target_role_text=_slot_text(component, target_slot),
        projection_mode=mode,
        source_strict_binding=source_strict,
        target_strict_binding=target_strict,
        source_lexical_signal=source_lexical,
        target_lexical_signal=target_lexical,
    )


def materialize_direct_task_relation_backbone(
    *,
    candidate: DirectRelationPatternCandidate,
    assessment: DirectRelationPatternAssessment,
    requested_source: str,
    requested_target: str,
) -> DirectTaskRelationBackboneView | None:
    if candidate.candidate_id != assessment.candidate_id:
        raise ValueError("candidate/assessment id mismatch")

    component = candidate.relation_component

    if component.authority != RelationComponentAuthority.CONFIRMED_KNOWN:
        return None

    if not assessment.decision_stable:
        return None

    if (
        assessment.stable_status != "PASS"
        or assessment.stable_role != "DIRECT_ANSWER"
        or assessment.task_class != "DIRECT"
    ):
        return None

    projection = project_direct_task_relation_roles(
        component=component,
        requested_source=requested_source,
        requested_target=requested_target,
    )

    if projection is None:
        return None

    return DirectTaskRelationBackboneView(
        topology_id=_stable_id(
            "direct_task_relation_backbone",
            candidate.candidate_id,
            component.component_id,
            requested_source,
            requested_target,
            projection.source_role_slot,
            projection.target_role_slot,
        ),
        component=component,
        candidate_id=candidate.candidate_id,
        requested_source=requested_source,
        requested_target=requested_target,
        role_projection=projection,
        reason_codes=[
            "confirmed_known_accepted_relationpattern",
            "stable_full_question_direct_answer",
            "opposite_slot_role_projection",
            "endpoint_equivalence_not_created",
            "scientific_identity_not_asserted",
            "shadow_only",
        ],
    )


def direct_backbone_role_texts(
    backbone: DirectTaskRelationBackboneView,
) -> dict[str, str]:
    return {
        "source": backbone.role_projection.source_role_text,
        "target": backbone.role_projection.target_role_text,
    }


__all__ = [
    "LexicalRoleSignal",
    "DirectTaskRelationRoleProjection",
    "DirectTaskRelationBackboneView",
    "project_direct_task_relation_roles",
    "materialize_direct_task_relation_backbone",
    "direct_backbone_role_texts",
]
