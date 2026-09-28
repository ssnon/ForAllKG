from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.direct_task_relation_backbone import DirectTaskRelationBackboneView
from pipeline_core.discovery.relation_component_composition import (
    RelationArgumentSlot,
    RelationComponentAuthority,
    RelationComponentView,
)
from pipeline_core.discovery.task_bridge_candidate_composition import lexical_tokens


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DirectAttachmentRole = Literal["source", "target"]


class DirectBackboneModifierEligibilityWitness(StrictModel):
    schema_version: str = "direct-backbone-modifier-eligibility-witness-v1"
    witness_id: str
    modifier_component_id: str
    validated_anchor_role: DirectAttachmentRole
    validated_anchor_slot: RelationArgumentSlot
    modifier_slot: RelationArgumentSlot
    validated_anchor_text: str
    modifier_text: str
    provenance_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    upstream_frozen_screen_pass: Literal[True] = True
    shadow_only: Literal[True] = True
    novelty_authority_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_witness(self):
        if not self.witness_id or not self.modifier_component_id:
            raise ValueError("modifier witness identity must be complete")
        if self.validated_anchor_slot == self.modifier_slot:
            raise ValueError("modifier anchor and modifier slots must be opposite")
        if not self.validated_anchor_text or not self.modifier_text:
            raise ValueError("modifier witness text must be complete")
        if not self.provenance_ids:
            raise ValueError("modifier eligibility witness requires provenance ids")
        return self


class DirectBackboneRoleBindingView(StrictModel):
    schema_version: str = "direct-backbone-role-binding-v1"
    anchor_role: DirectAttachmentRole
    modifier_anchor_slot: RelationArgumentSlot
    modifier_slot: RelationArgumentSlot
    backbone_role_text: str
    modifier_anchor_text: str
    modifier_text: str
    match_mode: Literal["exact_role_text", "compatible_role_text"]
    overlap_tokens: list[str] = Field(default_factory=list)
    compatibility_score: float = 0.0

    @model_validator(mode="after")
    def validate_binding(self):
        if self.modifier_anchor_slot == self.modifier_slot:
            raise ValueError("binding slots must be opposite")
        if not all(
            str(x).strip()
            for x in (self.backbone_role_text, self.modifier_anchor_text, self.modifier_text)
        ):
            raise ValueError("binding text must be complete")
        if not 0.0 <= float(self.compatibility_score) <= 1.0:
            raise ValueError("compatibility score must be within [0,1]")
        return self


class DirectHigherOrderTopologyCandidate(StrictModel):
    schema_version: str = "direct-higher-order-topology-candidate-v1"
    topology_id: str
    backbone_topology_id: str
    backbone: DirectTaskRelationBackboneView
    modifier_component: RelationComponentView
    modifier_eligibility: DirectBackboneModifierEligibilityWitness
    role_binding: DirectBackboneRoleBindingView
    epistemic_status: Literal["inspiration_only"] = "inspiration_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True
    topology_is_interaction_evidence: Literal[False] = False
    interaction_claim_authorized: Literal[False] = False
    endpoint_equivalence_authorized: Literal[False] = False
    scientific_identity_asserted: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology(self):
        if self.backbone_topology_id != self.backbone.topology_id:
            raise ValueError("backbone topology id mismatch")
        if self.modifier_component.component_id != self.modifier_eligibility.modifier_component_id:
            raise ValueError("modifier component/witness id mismatch")
        if self.modifier_component.authority != RelationComponentAuthority.CANDIDATE_INSPIRATION:
            raise ValueError("modifier must be CANDIDATE_INSPIRATION")
        if self.role_binding.anchor_role != self.modifier_eligibility.validated_anchor_role:
            raise ValueError("modifier anchor role mismatch")
        if self.role_binding.modifier_anchor_slot != self.modifier_eligibility.validated_anchor_slot:
            raise ValueError("modifier anchor slot mismatch")
        if self.role_binding.modifier_slot != self.modifier_eligibility.modifier_slot:
            raise ValueError("modifier slot mismatch")

        anchor_text = _slot_text(self.modifier_component, self.role_binding.modifier_anchor_slot)
        modifier_text = _slot_text(self.modifier_component, self.role_binding.modifier_slot)
        if _norm(anchor_text) != _norm(self.role_binding.modifier_anchor_text):
            raise ValueError("modifier anchor text mismatch")
        if _norm(modifier_text) != _norm(self.role_binding.modifier_text):
            raise ValueError("modifier text mismatch")

        expected = (
            self.backbone.role_projection.source_role_text
            if self.role_binding.anchor_role == "source"
            else self.backbone.role_projection.target_role_text
        )
        if _norm(expected) != _norm(self.role_binding.backbone_role_text):
            raise ValueError("backbone role text mismatch")
        return self


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _slot_text(component: RelationComponentView, slot: RelationArgumentSlot) -> str:
    return component.subject if slot == "subject" else component.object


def _role_compatibility(left: str, right: str):
    a = lexical_tokens(left)
    b = lexical_tokens(right)
    if not a or not b:
        return None
    shared = a & b
    if _norm(left) == _norm(right):
        return ("exact_role_text", sorted(shared), 1.0)
    if not shared:
        return None
    containment = len(shared) / min(len(a), len(b))
    if containment < 0.80:
        return None
    return ("compatible_role_text", sorted(shared), float(containment))


def _witness_from_audit_record(
    *,
    record: dict[str, Any],
    component: RelationComponentView,
    audit_source: str,
) -> DirectBackboneModifierEligibilityWitness:
    if not bool(record.get("passes_frozen_screen_mirror", False)):
        raise ValueError("modifier record did not pass frozen screen mirror")
    if list(record.get("reasons", [])):
        raise ValueError("passing modifier record carries rejection reasons")
    if str(record.get("component_id", "")) != component.component_id:
        raise ValueError("modifier audit component id mismatch")
    for key, value in (
        ("subject", component.subject),
        ("relation", component.relation),
        ("object", component.object),
    ):
        if _norm(record.get(key, "")) != _norm(value):
            raise ValueError("modifier audit relation identity mismatch")

    role = str(record.get("anchor_role", ""))
    anchor_slot = str(record.get("anchor_slot", ""))
    modifier_slot = str(record.get("modifier_slot", ""))
    if role not in {"source", "target"}:
        raise ValueError("unsupported direct-backbone anchor role")
    if anchor_slot not in {"subject", "object"} or modifier_slot not in {"subject", "object"}:
        raise ValueError("invalid modifier slots")
    if anchor_slot == modifier_slot:
        raise ValueError("modifier slots are not opposite")

    anchor_text = _slot_text(component, anchor_slot)
    modifier_text = _slot_text(component, modifier_slot)
    if _norm(anchor_text) != _norm(record.get("anchor_text", "")):
        raise ValueError("modifier anchor text mismatch")
    if _norm(modifier_text) != _norm(record.get("modifier_text", "")):
        raise ValueError("modifier text mismatch")

    return DirectBackboneModifierEligibilityWitness(
        witness_id=_stable_id(
            "direct_backbone_modifier_eligibility",
            component.component_id,
            role,
            anchor_slot,
            modifier_slot,
            anchor_text,
            modifier_text,
            audit_source,
        ),
        modifier_component_id=component.component_id,
        validated_anchor_role=role,
        validated_anchor_slot=anchor_slot,
        modifier_slot=modifier_slot,
        validated_anchor_text=anchor_text,
        modifier_text=modifier_text,
        provenance_ids=[
            str(component.provenance.source_id),
            "modifier_audit:" + str(audit_source),
        ],
        reason_codes=[
            "upstream_s201_frozen_screen_pass",
            "candidate_relation_identity_revalidated",
            "direct_backbone_role_anchor_required",
            "interaction_not_promoted_to_evidence",
            "shadow_only",
        ],
    )


def compose_direct_higher_order_topologies(
    *,
    backbones: Sequence[DirectTaskRelationBackboneView],
    modifier_components: Sequence[RelationComponentView],
    eligible_modifier_records: Sequence[dict[str, Any]],
    audit_source: str,
) -> tuple[DirectHigherOrderTopologyCandidate, ...]:
    by_id = {x.component_id: x for x in modifier_components}
    rows = []

    for record in eligible_modifier_records:
        component = by_id.get(str(record.get("component_id", "")))
        if component is None:
            continue

        witness = _witness_from_audit_record(
            record=record,
            component=component,
            audit_source=audit_source,
        )

        for backbone in backbones:
            role_text = (
                backbone.role_projection.source_role_text
                if witness.validated_anchor_role == "source"
                else backbone.role_projection.target_role_text
            )
            match = _role_compatibility(witness.validated_anchor_text, role_text)
            if match is None:
                continue

            mode, shared, score = match
            binding = DirectBackboneRoleBindingView(
                anchor_role=witness.validated_anchor_role,
                modifier_anchor_slot=witness.validated_anchor_slot,
                modifier_slot=witness.modifier_slot,
                backbone_role_text=role_text,
                modifier_anchor_text=witness.validated_anchor_text,
                modifier_text=witness.modifier_text,
                match_mode=mode,
                overlap_tokens=shared,
                compatibility_score=score,
            )

            rows.append(
                DirectHigherOrderTopologyCandidate(
                    topology_id=_stable_id(
                        "direct_higher_order_topology",
                        backbone.topology_id,
                        component.component_id,
                        witness.witness_id,
                    ),
                    backbone_topology_id=backbone.topology_id,
                    backbone=backbone,
                    modifier_component=component,
                    modifier_eligibility=witness,
                    role_binding=binding,
                    reason_codes=[
                        "direct_task_relation_backbone",
                        "candidate_modifier_frozen_screen_pass",
                        "backbone_role_anchor_compatible",
                        "structural_opportunity_only",
                        "interaction_requires_new_hypothesis",
                        "prior_art_review_required",
                        "n10_review_required",
                    ],
                )
            )

    rows.sort(
        key=lambda x: (
            x.backbone_topology_id,
            x.role_binding.anchor_role,
            x.modifier_component.component_id,
            x.topology_id,
        )
    )
    seen = set()
    out = []
    for row in rows:
        if row.topology_id in seen:
            continue
        seen.add(row.topology_id)
        out.append(row)
    return tuple(out)
