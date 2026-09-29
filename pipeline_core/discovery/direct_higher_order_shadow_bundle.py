from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderSynthesisContext,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _sha256_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class DirectHigherOrderStructuralRelationView(StrictModel):
    schema_version: Literal[
        "direct-higher-order-structural-relation-view-v1"
    ] = "direct-higher-order-structural-relation-view-v1"

    view_id: str
    direct_context_id: str
    direct_topology_id: str

    base_component_id: str
    base_subject: str
    base_relation: str
    base_object: str
    base_authority: Literal["confirmed_known"]

    modifier_component_id: str
    modifier_subject: str
    modifier_relation: str
    modifier_object: str
    modifier_authority: Literal["candidate_inspiration"]

    hypothesis_id: str
    full_higher_order_claim: str
    inferential_bridge: str

    structural_relation_form: Literal[
        "BASE_PLUS_MODIFIER_TO_GENERATED_FULL_CLAIM"
    ] = "BASE_PLUS_MODIFIER_TO_GENERATED_FULL_CLAIM"

    base_plus_modifier_is_interaction_evidence: Literal[False] = False
    structural_adapter_is_novelty_verdict: Literal[False] = False
    scientific_identity_asserted: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


class DirectHigherOrderSemanticDisposition(StrictModel):
    schema_version: Literal[
        "direct-higher-order-semantic-disposition-v1"
    ] = "direct-higher-order-semantic-disposition-v1"

    status: Literal[
        "NOT_RUN",
        "ACCEPTED",
        "REJECTED",
        "ERROR",
    ]
    hard_gate_passed: bool | None = None
    review_id: str | None = None
    run_id: str | None = None
    failure_stage: str | None = None
    error: str | None = None
    artifact_prefix: str | None = None

    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class DirectHigherOrderExternalNoveltyDisposition(StrictModel):
    schema_version: Literal[
        "direct-higher-order-external-novelty-disposition-v1"
    ] = "direct-higher-order-external-novelty-disposition-v1"

    status: Literal[
        "NOT_RUN",
        "COMPLETED",
        "SKIPPED_SEMANTIC_NOT_ACCEPTED",
        "ERROR",
    ]
    report_path: str | None = None
    query_plan_path: str | None = None
    prior_art_path: str | None = None
    card_statuses: list[str] = Field(default_factory=list)
    error: str | None = None

    search_bounded_only: Literal[True] = True
    positive_premise_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


class DirectHigherOrderConceptualKnownnessSlot(StrictModel):
    schema_version: Literal[
        "direct-higher-order-conceptual-knownness-slot-v1"
    ] = "direct-higher-order-conceptual-knownness-slot-v1"

    status: Literal[
        "NOT_RUN",
        "AVAILABLE",
        "ERROR",
    ] = "NOT_RUN"
    profile_path: str | None = None
    first_gap_level: str | None = None
    diagnostic_only: Literal[True] = True
    production_selection_authority: Literal[False] = False


class DirectHigherOrderShadowArmBundle(StrictModel):
    arm_index: int = Field(ge=1)
    direct_context_id: str
    direct_topology_id: str
    modifier_component_id: str
    modifier_text: str

    generation_status: str
    accepted_portfolio_id: str | None = None
    accepted_portfolio_sha256: str | None = None
    portfolio_artifact: str | None = None
    derived_context_artifact: str | None = None

    semantic: DirectHigherOrderSemanticDisposition
    external_novelty: DirectHigherOrderExternalNoveltyDisposition
    structural_views: list[
        DirectHigherOrderStructuralRelationView
    ] = Field(default_factory=list)
    conceptual_knownness: DirectHigherOrderConceptualKnownnessSlot = Field(
        default_factory=DirectHigherOrderConceptualKnownnessSlot
    )


class DirectHigherOrderShadowBundle(StrictModel):
    schema_version: Literal[
        "direct-higher-order-shadow-bundle-v1"
    ] = "direct-higher-order-shadow-bundle-v1"

    bundle_id: str
    bundle_sha256: str

    source_generation_report: str
    source_context_id: str
    source_context_sha256: str
    source_direct_relationpattern_report_id: str
    domain_profile_id: str

    arms: list[DirectHigherOrderShadowArmBundle] = Field(default_factory=list)

    semantic_attempted_count: int = Field(ge=0)
    semantic_accepted_count: int = Field(ge=0)
    external_novelty_completed_count: int = Field(ge=0)
    structural_view_count: int = Field(ge=0)

    conceptual_knownness_integrated: Literal[False] = False

    shadow_only: Literal[True] = True
    candidate_survival_authority: Literal[False] = False
    semantic_rejection_authority: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self):
        semantic_attempted = sum(
            row.semantic.status != "NOT_RUN"
            for row in self.arms
        )
        semantic_accepted = sum(
            row.semantic.status == "ACCEPTED"
            for row in self.arms
        )
        novelty_completed = sum(
            row.external_novelty.status == "COMPLETED"
            for row in self.arms
        )
        structural_views = sum(
            len(row.structural_views)
            for row in self.arms
        )

        if semantic_attempted != self.semantic_attempted_count:
            raise ValueError("semantic attempted count mismatch")
        if semantic_accepted != self.semantic_accepted_count:
            raise ValueError("semantic accepted count mismatch")
        if novelty_completed != self.external_novelty_completed_count:
            raise ValueError("external novelty completed count mismatch")
        if structural_views != self.structural_view_count:
            raise ValueError("structural view count mismatch")

        return self


def build_direct_higher_order_structural_views(
    *,
    direct_context: DirectHigherOrderSynthesisContext,
    portfolio: HypothesisPortfolio,
) -> tuple[DirectHigherOrderStructuralRelationView, ...]:
    by_role = {
        row.premise_role: row
        for row in direct_context.premises
    }

    direct = by_role["direct_task_relation"]
    modifier = by_role["modifier_relation"]

    direct_authority = getattr(
        direct.authority,
        "value",
        direct.authority,
    )
    modifier_authority = getattr(
        modifier.authority,
        "value",
        modifier.authority,
    )

    if str(direct_authority) != RelationComponentAuthority.CONFIRMED_KNOWN.value:
        raise ValueError("direct structural BASE must remain confirmed_known")
    if str(modifier_authority) != RelationComponentAuthority.CANDIDATE_INSPIRATION.value:
        raise ValueError("direct structural MODIFIER must remain candidate_inspiration")

    rows = []

    for card in portfolio.hypotheses:
        rows.append(
            DirectHigherOrderStructuralRelationView(
                view_id=_stable_id(
                    "direct_higher_order_structural_view",
                    direct_context.context_id,
                    direct_context.direct_higher_order_topology_id,
                    direct.component_id,
                    modifier.component_id,
                    card.hypothesis_id,
                ),
                direct_context_id=direct_context.context_id,
                direct_topology_id=direct_context.direct_higher_order_topology_id,
                base_component_id=direct.component_id,
                base_subject=direct.subject,
                base_relation=direct.relation,
                base_object=direct.object,
                base_authority="confirmed_known",
                modifier_component_id=modifier.component_id,
                modifier_subject=modifier.subject,
                modifier_relation=modifier.relation,
                modifier_object=modifier.object,
                modifier_authority="candidate_inspiration",
                hypothesis_id=card.hypothesis_id,
                full_higher_order_claim=card.hypothesis_statement,
                inferential_bridge=card.inferential_bridge,
            )
        )

    return tuple(rows)


def finalize_direct_higher_order_shadow_bundle(
    *,
    source_generation_report: str,
    source_context_id: str,
    source_context_sha256: str,
    source_direct_relationpattern_report_id: str,
    domain_profile_id: str,
    arms: Sequence[DirectHigherOrderShadowArmBundle],
) -> DirectHigherOrderShadowBundle:
    base = {
        "schema_version": "direct-higher-order-shadow-bundle-v1",
        "source_generation_report": source_generation_report,
        "source_context_id": source_context_id,
        "source_context_sha256": source_context_sha256,
        "source_direct_relationpattern_report_id": source_direct_relationpattern_report_id,
        "domain_profile_id": domain_profile_id,
        "arms": [
            row.model_dump(mode="json")
            for row in arms
        ],
        "semantic_attempted_count": sum(
            row.semantic.status != "NOT_RUN"
            for row in arms
        ),
        "semantic_accepted_count": sum(
            row.semantic.status == "ACCEPTED"
            for row in arms
        ),
        "external_novelty_completed_count": sum(
            row.external_novelty.status == "COMPLETED"
            for row in arms
        ),
        "structural_view_count": sum(
            len(row.structural_views)
            for row in arms
        ),
        "conceptual_knownness_integrated": False,
        "shadow_only": True,
        "candidate_survival_authority": False,
        "semantic_rejection_authority": False,
        "novelty_authority_created": False,
        "positive_premise_authority_created": False,
        "production_selection_authority": False,
        "stage8_input_changed": False,
    }

    bundle_id = _stable_id(
        "direct_higher_order_shadow_bundle",
        source_context_sha256,
        source_direct_relationpattern_report_id,
        *[
            (
                row.direct_topology_id
                + ":"
                + str(row.accepted_portfolio_id or "")
            )
            for row in arms
        ],
    )

    payload_for_hash = {
        **base,
        "bundle_id": bundle_id,
    }

    return DirectHigherOrderShadowBundle(
        **payload_for_hash,
        bundle_sha256=_sha256_json(payload_for_hash),
    )


__all__ = [
    "DirectHigherOrderStructuralRelationView",
    "DirectHigherOrderSemanticDisposition",
    "DirectHigherOrderExternalNoveltyDisposition",
    "DirectHigherOrderConceptualKnownnessSlot",
    "DirectHigherOrderShadowArmBundle",
    "DirectHigherOrderShadowBundle",
    "build_direct_higher_order_structural_views",
    "finalize_direct_higher_order_shadow_bundle",
]
