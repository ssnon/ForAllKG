from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_topology_carrier import (
    HigherOrderSynthesisCarrier,
    HigherOrderSynthesisLineageView,
)
from pipeline_core.discovery.higher_order_topology_composition import (
    HigherOrderAttachmentRole,
)
from pipeline_core.discovery.task_backbone_chain import (
    TaskBackboneChainView,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SynthesisPremiseRole = Literal[
    "source_backbone_relation",
    "middle_backbone_relation",
    "target_backbone_relation",
    "modifier_relation",
]


class HigherOrderSynthesisPremiseView(StrictModel):
    """
    One exact serialized relation supplied to shadow synthesis.

    The premise preserves the component's original subject/predicate/object
    orientation and epistemic authority. The synthesis boundary must not
    reverse, strengthen, or upgrade the recorded relation.
    """

    schema_version: str = "higher-order-synthesis-premise-v1"

    premise_id: str
    premise_role: SynthesisPremiseRole

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
    def validate_premise_authority(
        self,
    ) -> "HigherOrderSynthesisPremiseView":
        required = (
            self.premise_id,
            self.component_id,
            self.subject,
            self.relation,
            self.object,
            self.provenance_source_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order synthesis premise requires complete relation "
                "identity and provenance"
            )

        expected_use = (
            "confirmed_known_component"
            if self.authority
            == RelationComponentAuthority.CONFIRMED_KNOWN
            else "candidate_inspiration_component"
        )
        if self.epistemic_use != expected_use:
            raise ValueError(
                "higher-order synthesis premise epistemic use does not "
                "match component authority"
            )

        return self


class HigherOrderStructuralOpportunityView(StrictModel):
    """
    Role projection only.

    This structure says that the validated relations share a topology worth
    considering. It is explicitly not evidence that the modifier conditions,
    moderates, or changes the source-target relation.
    """

    schema_version: str = "higher-order-structural-opportunity-v1"

    requested_source: str
    requested_target: str

    source_side_mediator_text: str
    target_side_mediator_text: str

    modifier_text: str
    modifier_anchor_role: HigherOrderAttachmentRole
    modifier_anchor_text: str

    @model_validator(mode="after")
    def validate_opportunity_shape(
        self,
    ) -> "HigherOrderStructuralOpportunityView":
        required = (
            self.requested_source,
            self.requested_target,
            self.source_side_mediator_text,
            self.target_side_mediator_text,
            self.modifier_text,
            self.modifier_anchor_text,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order structural opportunity requires complete "
                "role text"
            )
        return self


class HigherOrderSynthesisGuardView(StrictModel):
    schema_version: str = "higher-order-synthesis-guard-v1"

    preserve_recorded_relation_orientation: Literal[True] = True
    preserve_component_authority: Literal[True] = True

    topology_is_not_interaction_evidence: Literal[True] = True
    topology_is_not_novelty_evidence: Literal[True] = True

    # Structural/lexical mediator compatibility does not authorize identity
    # between the source-side, target-side, and modifier-anchor expressions.
    mediator_identity_assertion_authorized: Literal[False] = False

    generated_interaction_must_remain_hypothesis: Literal[True] = True
    unsupported_direction_must_remain_open: Literal[True] = True

    prior_art_review_required: Literal[True] = True
    n10_novelty_review_required: Literal[True] = True

    discovery_axis_materialization_authorized: Literal[False] = False
    candidate_anchor_synthesis_authorized: Literal[False] = False
    llm_call_authorized: Literal[False] = False


class HigherOrderSynthesisContext(StrictModel):
    """
    Authority-safe prompt/context projection.

    This object is suitable for deterministic prompt rendering, but this stage
    does not authorize an LLM call. It preserves the topology-native carrier
    lineage and exposes exactly three relation premises: source-backbone,
    target-backbone, and modifier relation.
    """

    schema_version: str = "higher-order-synthesis-context-v1"

    context_id: str
    carrier_id: str
    higher_order_topology_id: str

    requested_source: str
    requested_target: str

    lineage: HigherOrderSynthesisLineageView

    premises: list[HigherOrderSynthesisPremiseView] = Field(
        min_length=3,
        max_length=4,
    )
    structural_opportunity: HigherOrderStructuralOpportunityView
    guard: HigherOrderSynthesisGuardView = Field(
        default_factory=HigherOrderSynthesisGuardView
    )

    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    requires_verification: Literal[
        True
    ] = True

    novelty_authority: Literal[
        False
    ] = False

    shadow_only: Literal[
        True
    ] = True

    prompt_rendering_authorized: Literal[
        True
    ] = True

    reason_codes: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_context_shape(
        self,
    ) -> "HigherOrderSynthesisContext":
        required = (
            self.context_id,
            self.carrier_id,
            self.higher_order_topology_id,
            self.requested_source,
            self.requested_target,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order synthesis context requires complete identity"
            )

        role_counts = {
            role: sum(
                1
                for premise in self.premises
                if premise.premise_role == role
            )
            for role in (
                "source_backbone_relation",
                "middle_backbone_relation",
                "target_backbone_relation",
                "modifier_relation",
            )
        }

        for required_role in (
            "source_backbone_relation",
            "target_backbone_relation",
            "modifier_relation",
        ):
            if role_counts[required_role] != 1:
                raise ValueError(
                    "higher-order synthesis context requires exactly one "
                    + required_role
                )

        expected_middle_count = (
            1
            if self.lineage.middle_component_id is not None
            else 0
        )
        if (
            role_counts["middle_backbone_relation"]
            != expected_middle_count
        ):
            raise ValueError(
                "higher-order synthesis middle-backbone premise/lineage mismatch"
            )

        by_role = {
            premise.premise_role: premise
            for premise in self.premises
        }
        expected_component_ids = {
            "source_backbone_relation":
                self.lineage.source_component_id,
            "target_backbone_relation":
                self.lineage.target_component_id,
            "modifier_relation":
                self.lineage.modifier_component_id,
        }
        expected_provenance_ids = {
            "source_backbone_relation":
                self.lineage.source_provenance_source_id,
            "target_backbone_relation":
                self.lineage.target_provenance_source_id,
            "modifier_relation":
                self.lineage.modifier_provenance_source_id,
        }

        if self.lineage.middle_component_id is not None:
            expected_component_ids[
                "middle_backbone_relation"
            ] = self.lineage.middle_component_id
            expected_provenance_ids[
                "middle_backbone_relation"
            ] = self.lineage.middle_provenance_source_id

        for role, component_id in expected_component_ids.items():
            premise = by_role[role]
            if premise.component_id != component_id:
                raise ValueError(
                    "higher-order synthesis premise lost component lineage: "
                    + role
                )
            if (
                premise.provenance_source_id
                != expected_provenance_ids[role]
            ):
                raise ValueError(
                    "higher-order synthesis premise lost provenance lineage: "
                    + role
                )

        if (
            self.higher_order_topology_id
            != self.lineage.higher_order_topology_id
        ):
            raise ValueError(
                "higher-order synthesis context topology lineage mismatch"
            )

        opportunity = self.structural_opportunity
        if (
            opportunity.modifier_text
            != self.lineage.modifier_text
        ):
            raise ValueError(
                "higher-order synthesis context modifier text mismatch"
            )
        if (
            opportunity.modifier_anchor_role
            != self.lineage.modifier_anchor_role
        ):
            raise ValueError(
                "higher-order synthesis context modifier role mismatch"
            )
        if (
            opportunity.modifier_anchor_text
            != self.lineage.modifier_anchor_text
        ):
            raise ValueError(
                "higher-order synthesis context modifier anchor mismatch"
            )

        return self


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")
    return (
        prefix
        + ":"
        + hashlib.sha256(raw).hexdigest()[:20]
    )


def _slot_text(
    component: RelationComponentView,
    slot: Literal["subject", "object"],
) -> str:
    return (
        component.subject
        if slot == "subject"
        else component.object
    )


def _premise_from_component(
    *,
    role: SynthesisPremiseRole,
    component: RelationComponentView,
) -> HigherOrderSynthesisPremiseView:
    epistemic_use: Literal[
        "confirmed_known_component",
        "candidate_inspiration_component",
    ] = (
        "confirmed_known_component"
        if component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
        else "candidate_inspiration_component"
    )

    return HigherOrderSynthesisPremiseView(
        premise_id=_stable_id(
            "higher_order_synthesis_premise",
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


def higher_order_synthesis_context(
    *,
    carrier: HigherOrderSynthesisCarrier,
) -> HigherOrderSynthesisContext:
    """
    Convert one validated topology-native carrier into a compact shadow
    synthesis context.

    No selection, ranking, relation reversal, candidate fabrication, novelty
    judgment, hypothesis generation, or LLM call occurs here.
    """

    topology = carrier.topology
    backbone = topology.backbone

    source_component = backbone.source_component
    target_component = backbone.target_component
    middle_component = (
        backbone.middle_component
        if isinstance(
            backbone,
            TaskBackboneChainView,
        )
        else None
    )
    modifier_component = topology.modifier_component

    source_mediator_text = _slot_text(
        source_component,
        backbone.source_binding.mediator_slot,
    )
    target_mediator_text = _slot_text(
        target_component,
        backbone.target_binding.mediator_slot,
    )

    premises = [
        _premise_from_component(
            role="source_backbone_relation",
            component=source_component,
        ),
        *(
            [
                _premise_from_component(
                    role="middle_backbone_relation",
                    component=middle_component,
                )
            ]
            if middle_component is not None
            else []
        ),
        _premise_from_component(
            role="target_backbone_relation",
            component=target_component,
        ),
        _premise_from_component(
            role="modifier_relation",
            component=modifier_component,
        ),
    ]

    return HigherOrderSynthesisContext(
        context_id=_stable_id(
            "higher_order_synthesis_context",
            carrier.carrier_id,
            topology.topology_id,
        ),
        carrier_id=carrier.carrier_id,
        higher_order_topology_id=topology.topology_id,
        requested_source=carrier.requested_source,
        requested_target=carrier.requested_target,
        lineage=carrier.lineage.model_copy(deep=True),
        premises=premises,
        structural_opportunity=HigherOrderStructuralOpportunityView(
            requested_source=carrier.requested_source,
            requested_target=carrier.requested_target,
            source_side_mediator_text=source_mediator_text,
            target_side_mediator_text=target_mediator_text,
            modifier_text=topology.role_binding.modifier_text,
            modifier_anchor_role=(
                topology.role_binding.modifier_anchor_role
            ),
            modifier_anchor_text=(
                topology.role_binding.modifier_anchor_text
            ),
        ),
        reason_codes=[
            "topology_native_shadow_synthesis_context",
            "exact_relation_orientation_preserved",
            "component_authority_preserved",
            "component_provenance_preserved",
            "interaction_not_promoted_to_evidence",
            "novelty_not_asserted",
            "prior_art_review_required",
            "n10_novelty_review_required",
            "llm_call_not_authorized",
        ],
    )


def build_higher_order_synthesis_contexts(
    *,
    carriers: Sequence[HigherOrderSynthesisCarrier],
) -> tuple[HigherOrderSynthesisContext, ...]:
    """
    Build one context per unique carrier, preserving input order.

    Duplicate carrier ids fail closed rather than being silently deduplicated.
    No quota or ranking is applied.
    """

    rows = []
    seen_carrier_ids = set()

    for carrier in carriers:
        if carrier.carrier_id in seen_carrier_ids:
            raise ValueError(
                "duplicate higher-order synthesis carrier id: "
                + carrier.carrier_id
            )
        seen_carrier_ids.add(carrier.carrier_id)
        rows.append(
            higher_order_synthesis_context(
                carrier=carrier
            )
        )

    return tuple(rows)


def render_higher_order_shadow_prompt(
    context: HigherOrderSynthesisContext,
) -> str:
    """
    Deterministically render an authority-safe prompt payload.

    Rendering is allowed; an LLM call is not. The text deliberately separates
    recorded relation premises from the unverified interaction hypothesis.
    """

    premise_lines = []
    for index, premise in enumerate(
        context.premises,
        start=1,
    ):
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

    return "\n".join([
        "TOPOLOGY-NATIVE SHADOW SYNTHESIS CONTEXT",
        "",
        f"Requested source: {context.requested_source}",
        f"Requested target: {context.requested_target}",
        "",
        "RECORDED RELATION PREMISES "
        "(preserve exact orientation and authority):",
        *premise_lines,
        "",
        "STRUCTURAL OPPORTUNITY "
        "(NOT evidence of interaction or novelty):",
        f"- Modifier C: {opportunity.modifier_text}",
        (
            "- Validated anchor role: "
            f"{opportunity.modifier_anchor_role}"
        ),
        f"- Validated anchor text: {opportunity.modifier_anchor_text}",
        (
            "- Source-side mediator expression: "
            f"{opportunity.source_side_mediator_text}"
        ),
        (
            "- Target-side mediator expression: "
            f"{opportunity.target_side_mediator_text}"
        ),
        "",
        "SHADOW SYNTHESIS CONSTRAINTS:",
        (
            "- Treat any claim that C conditions, moderates, interacts with, "
            "or changes the source-target relationship as a HYPOTHESIS, "
            "not as established evidence."
        ),
        (
            "- Do not reverse or strengthen any recorded predicate merely "
            "to make the hypothesis read more naturally."
        ),
        (
            "- The source-side mediator expression, target-side mediator "
            "expression, and modifier-anchor expression are structurally "
            "compatible relation arguments, not asserted identical entities. "
            "Keep them distinct unless an explicit mediator-equivalence "
            "witness is supplied; none is supplied by this context."
        ),
        (
            "- Do not invent a causal link between those mediator expressions "
            "and do not describe them as a shared/same mediator, mediator "
            "path, linked expressions, or as one expression connecting the "
            "others through an unsupported bridge."
        ),
        (
            "- Keep effect direction open unless a supplied premise "
            "explicitly supports that direction."
        ),
        (
            "- Do not claim that the components or their combination are "
            "novel from this topology alone."
        ),
        (
            "- Prior-art review and N10 novelty review are required before "
            "any positive novelty conclusion."
        ),
        (
            "- This is a shadow prompt rendering only; no LLM call is "
            "authorized by this context."
        ),
    ])
