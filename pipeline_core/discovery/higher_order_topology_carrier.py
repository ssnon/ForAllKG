from __future__ import annotations

import hashlib
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_topology_composition import (
    HigherOrderAttachmentRole,
    HigherOrderTopologyCandidate,
)
from pipeline_core.discovery.task_backbone_chain import (
    TaskBackboneChainView,
)
from pipeline_core.discovery.relation_component_composition import (
    EndpointBindingAuthority,
    RelationArgumentSlot,
    RelationComponentAuthority,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HigherOrderSynthesisLineageView(StrictModel):
    """
    Lossless lineage projection at the synthesis boundary.

    This view does not create a DiscoveryAxis, a candidate anchor, or novelty
    authority. It exposes only identities and authorities that already exist
    on the verified higher-order topology candidate.
    """

    schema_version: str = "higher-order-synthesis-lineage-v1"

    higher_order_topology_id: str
    backbone_topology_id: str

    source_component_id: str
    target_component_id: str
    modifier_component_id: str

    source_authority: RelationComponentAuthority
    target_authority: RelationComponentAuthority
    modifier_authority: RelationComponentAuthority

    source_provenance_source_id: str
    target_provenance_source_id: str

    middle_component_id: str | None = None
    middle_provenance_source_id: str | None = None

    modifier_provenance_source_id: str

    source_endpoint_binding_authority: EndpointBindingAuthority
    target_endpoint_binding_authority: EndpointBindingAuthority

    source_endpoint_equivalence_witness_id: str | None = None
    target_endpoint_equivalence_witness_id: str | None = None

    modifier_eligibility_witness_id: str
    modifier_anchor_role: HigherOrderAttachmentRole
    modifier_anchor_slot: RelationArgumentSlot
    modifier_slot: RelationArgumentSlot

    modifier_anchor_text: str
    modifier_text: str

    @model_validator(mode="after")
    def validate_lineage_shape(
        self,
    ) -> "HigherOrderSynthesisLineageView":
        required = (
            self.higher_order_topology_id,
            self.backbone_topology_id,
            self.source_component_id,
            self.target_component_id,
            self.modifier_component_id,
            self.source_provenance_source_id,
            self.target_provenance_source_id,
            self.modifier_provenance_source_id,
            self.modifier_eligibility_witness_id,
            self.modifier_anchor_text,
            self.modifier_text,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order synthesis lineage requires complete identities"
            )

        if (
            (self.middle_component_id is None)
            != (self.middle_provenance_source_id is None)
        ):
            raise ValueError(
                "middle backbone component/provenance lineage must be paired"
            )

        for authority, witness_id in (
            (
                self.source_endpoint_binding_authority,
                self.source_endpoint_equivalence_witness_id,
            ),
            (
                self.target_endpoint_binding_authority,
                self.target_endpoint_equivalence_witness_id,
            ),
        ):
            if authority == "equivalent" and not witness_id:
                raise ValueError(
                    "equivalent endpoint lineage requires witness id"
                )
            if authority != "equivalent" and witness_id is not None:
                raise ValueError(
                    "only equivalent endpoint lineage may carry witness id"
                )

        if self.modifier_anchor_slot == self.modifier_slot:
            raise ValueError(
                "carrier modifier anchor and modifier slots must be opposite"
            )

        return self


class HigherOrderSynthesisCarrier(StrictModel):
    """
    Topology-native, shadow-only carrier for the synthesis boundary.

    The carrier deliberately does not materialize a DiscoveryAxis and does not
    synthesize candidate-unit lineage. Synthesis may inspect this structure,
    but any generated scientific interaction remains a hypothesis requiring
    downstream verification and N10 novelty review.
    """

    schema_version: str = "higher-order-synthesis-carrier-v1"

    carrier_id: str
    requested_source: str
    requested_target: str

    topology: HigherOrderTopologyCandidate
    lineage: HigherOrderSynthesisLineageView

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

    discovery_axis_materialization_authorized: Literal[
        False
    ] = False

    candidate_anchor_synthesized: Literal[
        False
    ] = False

    reason_codes: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_carrier_fidelity(
        self,
    ) -> "HigherOrderSynthesisCarrier":
        if not str(self.carrier_id).strip():
            raise ValueError(
                "higher-order synthesis carrier_id must not be empty"
            )
        if not str(self.requested_source).strip():
            raise ValueError(
                "higher-order synthesis requested_source must not be empty"
            )
        if not str(self.requested_target).strip():
            raise ValueError(
                "higher-order synthesis requested_target must not be empty"
            )

        topology = self.topology
        backbone = topology.backbone
        lineage = self.lineage

        expected = {
            "higher_order_topology_id": topology.topology_id,
            "backbone_topology_id": topology.backbone_topology_id,
            "source_component_id": backbone.source_component.component_id,
            "target_component_id": backbone.target_component.component_id,
            "modifier_component_id": topology.modifier_component.component_id,
            "source_provenance_source_id": (
                backbone.source_component.provenance.source_id
            ),
            "target_provenance_source_id": (
                backbone.target_component.provenance.source_id
            ),
            "modifier_provenance_source_id": (
                topology.modifier_component.provenance.source_id
            ),
            "modifier_eligibility_witness_id": (
                topology.modifier_eligibility.witness_id
            ),
            "modifier_anchor_role": (
                topology.role_binding.modifier_anchor_role
            ),
            "modifier_anchor_slot": (
                topology.role_binding.modifier_anchor_slot
            ),
            "modifier_slot": topology.role_binding.modifier_slot,
            "modifier_anchor_text": (
                topology.role_binding.modifier_anchor_text
            ),
            "modifier_text": topology.role_binding.modifier_text,
        }

        if isinstance(
            backbone,
            TaskBackboneChainView,
        ):
            expected.update(
                {
                    "middle_component_id":
                        backbone.middle_component.component_id,
                    "middle_provenance_source_id":
                        backbone.middle_component.provenance.source_id,
                }
            )
        elif (
            lineage.middle_component_id is not None
            or lineage.middle_provenance_source_id is not None
        ):
            raise ValueError(
                "two-component backbone cannot carry middle-component lineage"
            )

        for field_name, value in expected.items():
            if getattr(lineage, field_name) != value:
                raise ValueError(
                    "higher-order synthesis lineage mismatch: "
                    + field_name
                )

        if (
            lineage.source_authority
            != backbone.source_component.authority
        ):
            raise ValueError(
                "higher-order synthesis source authority mismatch"
            )
        if (
            lineage.target_authority
            != backbone.target_component.authority
        ):
            raise ValueError(
                "higher-order synthesis target authority mismatch"
            )
        if (
            lineage.modifier_authority
            != topology.modifier_component.authority
        ):
            raise ValueError(
                "higher-order synthesis modifier authority mismatch"
            )

        source_binding = backbone.source_binding
        target_binding = backbone.target_binding

        if (
            lineage.source_endpoint_binding_authority
            != source_binding.binding_authority
        ):
            raise ValueError(
                "higher-order synthesis source endpoint authority mismatch"
            )
        if (
            lineage.target_endpoint_binding_authority
            != target_binding.binding_authority
        ):
            raise ValueError(
                "higher-order synthesis target endpoint authority mismatch"
            )
        if (
            lineage.source_endpoint_equivalence_witness_id
            != source_binding.equivalence_witness_id
        ):
            raise ValueError(
                "higher-order synthesis source endpoint witness mismatch"
            )
        if (
            lineage.target_endpoint_equivalence_witness_id
            != target_binding.equivalence_witness_id
        ):
            raise ValueError(
                "higher-order synthesis target endpoint witness mismatch"
            )

        if self.epistemic_status != topology.epistemic_status:
            raise ValueError(
                "higher-order synthesis epistemic status mismatch"
            )
        if self.requires_verification != topology.requires_verification:
            raise ValueError(
                "higher-order synthesis verification authority mismatch"
            )
        if self.novelty_authority != topology.novelty_authority:
            raise ValueError(
                "higher-order synthesis novelty authority mismatch"
            )
        if self.shadow_only != topology.shadow_only:
            raise ValueError(
                "higher-order synthesis shadow authority mismatch"
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


def topology_native_synthesis_carrier(
    *,
    topology: HigherOrderTopologyCandidate,
    requested_source: str,
    requested_target: str,
) -> HigherOrderSynthesisCarrier:
    """
    Project one verified higher-order topology to the synthesis boundary.

    This is a lineage-preserving projection only. It performs no ranking,
    selection, candidate fabrication, hypothesis generation, or novelty
    judgment.
    """

    backbone = topology.backbone
    source_component = backbone.source_component
    target_component = backbone.target_component
    modifier_component = topology.modifier_component
    middle_component = (
        backbone.middle_component
        if isinstance(
            backbone,
            TaskBackboneChainView,
        )
        else None
    )

    lineage = HigherOrderSynthesisLineageView(
        higher_order_topology_id=topology.topology_id,
        backbone_topology_id=topology.backbone_topology_id,
        source_component_id=source_component.component_id,
        target_component_id=target_component.component_id,
        modifier_component_id=modifier_component.component_id,
        source_authority=source_component.authority,
        target_authority=target_component.authority,
        modifier_authority=modifier_component.authority,
        source_provenance_source_id=(
            source_component.provenance.source_id
        ),
        target_provenance_source_id=(
            target_component.provenance.source_id
        ),
        middle_component_id=(
            None
            if middle_component is None
            else middle_component.component_id
        ),
        middle_provenance_source_id=(
            None
            if middle_component is None
            else middle_component.provenance.source_id
        ),
        modifier_provenance_source_id=(
            modifier_component.provenance.source_id
        ),
        source_endpoint_binding_authority=(
            backbone.source_binding.binding_authority
        ),
        target_endpoint_binding_authority=(
            backbone.target_binding.binding_authority
        ),
        source_endpoint_equivalence_witness_id=(
            backbone.source_binding.equivalence_witness_id
        ),
        target_endpoint_equivalence_witness_id=(
            backbone.target_binding.equivalence_witness_id
        ),
        modifier_eligibility_witness_id=(
            topology.modifier_eligibility.witness_id
        ),
        modifier_anchor_role=(
            topology.role_binding.modifier_anchor_role
        ),
        modifier_anchor_slot=(
            topology.role_binding.modifier_anchor_slot
        ),
        modifier_slot=topology.role_binding.modifier_slot,
        modifier_anchor_text=(
            topology.role_binding.modifier_anchor_text
        ),
        modifier_text=topology.role_binding.modifier_text,
    )

    return HigherOrderSynthesisCarrier(
        carrier_id=_stable_id(
            "higher_order_synthesis_carrier",
            topology.topology_id,
            requested_source,
            requested_target,
        ),
        requested_source=requested_source,
        requested_target=requested_target,
        topology=topology,
        lineage=lineage,
        reason_codes=[
            "topology_native_shadow_carrier",
            "component_authority_preserved",
            "component_provenance_preserved",
            "endpoint_fidelity_authority_preserved",
            "modifier_role_authority_preserved",
            "no_discovery_axis_materialization",
            "no_candidate_anchor_synthesis",
            "n10_novelty_authority_unchanged",
        ],
    )


def build_topology_native_synthesis_carriers(
    *,
    topologies: Sequence[HigherOrderTopologyCandidate],
    requested_source: str,
    requested_target: str,
) -> tuple[HigherOrderSynthesisCarrier, ...]:
    """
    Losslessly project every unique input topology, preserving input order.

    Duplicate topology ids fail closed rather than being silently deduplicated.
    No ranking or quota is applied.
    """

    rows = []
    seen_topology_ids = set()

    for topology in topologies:
        if topology.topology_id in seen_topology_ids:
            raise ValueError(
                "duplicate higher-order topology id at synthesis boundary: "
                + topology.topology_id
            )

        seen_topology_ids.add(topology.topology_id)
        rows.append(
            topology_native_synthesis_carrier(
                topology=topology,
                requested_source=requested_source,
                requested_target=requested_target,
            )
        )

    return tuple(rows)
