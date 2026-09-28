from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.direct_higher_order_topology import (
    DirectHigherOrderTopologyCandidate,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DirectSynthesisPremiseRole = Literal[
    "direct_task_relation",
    "modifier_relation",
]


class DirectHigherOrderSynthesisPremiseView(StrictModel):
    schema_version: str = "direct-higher-order-synthesis-premise-v1"
    premise_id: str
    premise_role: DirectSynthesisPremiseRole
    component_id: str
    subject: str
    relation: str
    object: str
    authority: RelationComponentAuthority
    provenance_source_id: str
    epistemic_use: Literal[
        "confirmed_known_component",
        "candidate_inspiration_component",
    ]

    @model_validator(mode="after")
    def validate_premise(self):
        required = (
            self.premise_id,
            self.component_id,
            self.subject,
            self.relation,
            self.object,
            self.provenance_source_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("synthesis premise requires complete identity")

        expected = (
            "confirmed_known_component"
            if self.authority
            == RelationComponentAuthority.CONFIRMED_KNOWN
            else "candidate_inspiration_component"
        )
        if self.epistemic_use != expected:
            raise ValueError(
                "premise epistemic use does not match relation authority"
            )
        return self


class DirectHigherOrderStructuralOpportunityView(StrictModel):
    schema_version: str = "direct-higher-order-structural-opportunity-v1"
    requested_source: str
    requested_target: str
    source_role_text: str
    target_role_text: str
    modifier_text: str
    modifier_anchor_role: Literal["source", "target"]
    modifier_anchor_text: str

    @model_validator(mode="after")
    def validate_opportunity(self):
        required = (
            self.requested_source,
            self.requested_target,
            self.source_role_text,
            self.target_role_text,
            self.modifier_text,
            self.modifier_anchor_text,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "structural opportunity requires complete role text"
            )
        return self


class DirectHigherOrderSynthesisGuardView(StrictModel):
    schema_version: str = "direct-higher-order-synthesis-guard-v1"
    preserve_recorded_relation_orientation: Literal[True] = True
    preserve_component_authority: Literal[True] = True
    preserve_component_provenance: Literal[True] = True
    topology_is_not_interaction_evidence: Literal[True] = True
    topology_is_not_novelty_evidence: Literal[True] = True
    endpoint_equivalence_assertion_authorized: Literal[False] = False
    scientific_identity_assertion_authorized: Literal[False] = False
    generated_interaction_must_remain_hypothesis: Literal[True] = True
    unsupported_direction_must_remain_open: Literal[True] = True
    prior_art_review_required: Literal[True] = True
    n10_novelty_review_required: Literal[True] = True
    discovery_axis_materialization_authorized: Literal[False] = False
    candidate_anchor_synthesis_authorized: Literal[False] = False
    llm_call_authorized: Literal[False] = False


class DirectHigherOrderSynthesisContext(StrictModel):
    schema_version: str = "direct-higher-order-synthesis-context-v1"
    context_id: str
    direct_higher_order_topology_id: str
    direct_backbone_topology_id: str
    requested_source: str
    requested_target: str
    direct_backbone_candidate_id: str
    modifier_component_id: str
    premises: list[
        DirectHigherOrderSynthesisPremiseView
    ] = Field(min_length=2, max_length=2)
    structural_opportunity: DirectHigherOrderStructuralOpportunityView
    guard: DirectHigherOrderSynthesisGuardView = Field(
        default_factory=DirectHigherOrderSynthesisGuardView
    )
    epistemic_status: Literal["inspiration_only"] = "inspiration_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True
    prompt_rendering_authorized: Literal[True] = True
    novelty_authority_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context(self):
        required = (
            self.context_id,
            self.direct_higher_order_topology_id,
            self.direct_backbone_topology_id,
            self.requested_source,
            self.requested_target,
            self.direct_backbone_candidate_id,
            self.modifier_component_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("synthesis context requires complete identity")

        by_role = {
            premise.premise_role: premise
            for premise in self.premises
        }
        if set(by_role) != {
            "direct_task_relation",
            "modifier_relation",
        }:
            raise ValueError(
                "context requires one direct-task and one modifier premise"
            )

        if (
            by_role["modifier_relation"].component_id
            != self.modifier_component_id
        ):
            raise ValueError("modifier premise lost component lineage")

        opportunity = self.structural_opportunity
        if (
            opportunity.requested_source != self.requested_source
            or opportunity.requested_target != self.requested_target
        ):
            raise ValueError(
                "structural opportunity lost requested task identity"
            )

        return self


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _premise_from_component(
    *,
    role: DirectSynthesisPremiseRole,
    component: RelationComponentView,
) -> DirectHigherOrderSynthesisPremiseView:
    epistemic_use = (
        "confirmed_known_component"
        if component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
        else "candidate_inspiration_component"
    )

    return DirectHigherOrderSynthesisPremiseView(
        premise_id=_stable_id(
            "direct_higher_order_synthesis_premise",
            role,
            component.component_id,
            component.provenance.source_id,
        ),
        premise_role=role,
        component_id=component.component_id,
        subject=component.subject,
        relation=component.relation,
        object=component.object,
        authority=component.authority,
        provenance_source_id=component.provenance.source_id,
        epistemic_use=epistemic_use,
    )


def direct_higher_order_synthesis_context(
    *,
    topology: DirectHigherOrderTopologyCandidate,
) -> DirectHigherOrderSynthesisContext:
    backbone = topology.backbone
    direct_component = backbone.component
    modifier_component = topology.modifier_component
    projection = backbone.role_projection

    return DirectHigherOrderSynthesisContext(
        context_id=_stable_id(
            "direct_higher_order_synthesis_context",
            topology.topology_id,
            backbone.topology_id,
            modifier_component.component_id,
        ),
        direct_higher_order_topology_id=topology.topology_id,
        direct_backbone_topology_id=backbone.topology_id,
        requested_source=backbone.requested_source,
        requested_target=backbone.requested_target,
        direct_backbone_candidate_id=backbone.candidate_id,
        modifier_component_id=modifier_component.component_id,
        premises=[
            _premise_from_component(
                role="direct_task_relation",
                component=direct_component,
            ),
            _premise_from_component(
                role="modifier_relation",
                component=modifier_component,
            ),
        ],
        structural_opportunity=DirectHigherOrderStructuralOpportunityView(
            requested_source=backbone.requested_source,
            requested_target=backbone.requested_target,
            source_role_text=projection.source_role_text,
            target_role_text=projection.target_role_text,
            modifier_text=topology.role_binding.modifier_text,
            modifier_anchor_role=topology.role_binding.anchor_role,
            modifier_anchor_text=topology.role_binding.modifier_anchor_text,
        ),
        reason_codes=[
            "direct_task_relation_orientation_preserved",
            "candidate_modifier_relation_orientation_preserved",
            "component_authority_preserved",
            "component_provenance_preserved",
            "structural_opportunity_not_interaction_evidence",
            "endpoint_equivalence_not_asserted",
            "scientific_identity_not_asserted",
            "prior_art_review_required",
            "n10_novelty_review_required",
            "llm_call_not_authorized",
        ],
    )


def build_direct_higher_order_synthesis_contexts(
    *,
    topologies: Sequence[DirectHigherOrderTopologyCandidate],
) -> tuple[DirectHigherOrderSynthesisContext, ...]:
    rows = []
    seen = set()

    for topology in topologies:
        if topology.topology_id in seen:
            raise ValueError("duplicate direct higher-order topology id")
        seen.add(topology.topology_id)
        rows.append(
            direct_higher_order_synthesis_context(
                topology=topology
            )
        )

    return tuple(rows)


def render_direct_higher_order_shadow_prompt(
    context: DirectHigherOrderSynthesisContext,
) -> str:
    premise_lines = []
    for index, premise in enumerate(context.premises, start=1):
        premise_lines.append(
            (
                f"{index}. [{premise.premise_role}; "
                f"authority={premise.authority.value}; "
                f"source={premise.provenance_source_id}] "
                f"{premise.subject} --{premise.relation}--> "
                f"{premise.object}"
            )
        )

    opportunity = context.structural_opportunity

    return "\n".join(
        [
            "DIRECT HIGHER-ORDER SHADOW SYNTHESIS CONTEXT",
            "",
            f"Requested source: {context.requested_source}",
            f"Requested target: {context.requested_target}",
            "",
            "RECORDED RELATION PREMISES "
            "(preserve exact orientation and authority):",
            *premise_lines,
            "",
            "TASK-ROLE PROJECTION "
            "(role assignment only; NOT scientific identity):",
            f"- Source-role expression: {opportunity.source_role_text}",
            f"- Target-role expression: {opportunity.target_role_text}",
            "",
            "STRUCTURAL OPPORTUNITY "
            "(NOT evidence of an interaction):",
            f"- Candidate modifier C: {opportunity.modifier_text}",
            (
                "- C is anchored to task role: "
                f"{opportunity.modifier_anchor_role}"
            ),
            (
                "- Recorded modifier anchor expression: "
                f"{opportunity.modifier_anchor_text}"
            ),
            "",
            "SHADOW SYNTHESIS CONSTRAINTS:",
            (
                "- Any claim that C conditions, moderates, interacts with, "
                "changes, strengthens, weakens, or mediates the direct "
                "source-target relation must remain a NEW HYPOTHESIS."
            ),
            "- The topology itself is not evidence that such an interaction exists.",
            "- Do not reverse or strengthen either recorded relation.",
            (
                "- Do not claim that the task-role expressions are "
                "scientifically identical or equivalent to the requested "
                "source/target wording."
            ),
            (
                "- Preserve the difference between CONFIRMED_KNOWN and "
                "CANDIDATE_INSPIRATION authority."
            ),
            (
                "- Keep interaction direction open unless a recorded premise "
                "explicitly supplies that direction."
            ),
            (
                "- Do not claim novelty from this topology. Prior-art review "
                "and N10 novelty review are required."
            ),
            (
                "- This is a shadow prompt rendering only; no LLM call is "
                "authorized by this context."
            ),
        ]
    )
