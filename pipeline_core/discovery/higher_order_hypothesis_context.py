from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderSynthesisContext,
    SynthesisPremiseRole,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.hypothesis_prompt import (
    HypothesisPrompt,
    HypothesisPromptAssembler,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_HIGHER_ORDER_RESTRICTIONS = (
    "higher_order_composition_inspiration_only",
    "confirmed_known_does_not_create_positive_premise_authority",
    "must_not_appear_in_premise_statement_ids",
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class HigherOrderHypothesisStatementLineage(StrictModel):
    schema_version: Literal[
        "higher-order-hypothesis-statement-lineage-v1"
    ] = "higher-order-hypothesis-statement-lineage-v1"

    statement_id: str
    higher_order_premise_id: str
    premise_role: SynthesisPremiseRole

    component_id: str
    provenance_source_id: str
    authority: RelationComponentAuthority

    @model_validator(mode="after")
    def validate_lineage(
        self,
    ) -> "HigherOrderHypothesisStatementLineage":
        required = (
            self.statement_id,
            self.higher_order_premise_id,
            self.component_id,
            self.provenance_source_id,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order hypothesis statement lineage requires "
                "complete identity and provenance"
            )
        if (
            self.authority
            != RelationComponentAuthority.CONFIRMED_KNOWN
        ):
            raise ValueError(
                "S24a1 production materialization accepts only "
                "CONFIRMED_KNOWN higher-order components"
            )
        return self


class HigherOrderHypothesisContextMaterialization(StrictModel):
    """Audit artifact for one shadow-only HypothesisContext projection."""

    schema_version: Literal[
        "higher-order-hypothesis-context-materialization-v1"
    ] = "higher-order-hypothesis-context-materialization-v1"

    materialization_id: str

    source_hypothesis_context_id: str
    source_hypothesis_context_sha256: str

    source_higher_order_context_id: str
    source_higher_order_topology_id: str
    source_higher_order_carrier_id: str

    derived_hypothesis_context_id: str
    derived_hypothesis_context_sha256: str

    canonical_positive_premise_statement_ids: list[str] = Field(
        min_length=1
    )
    generated_statement_lineage: list[
        HigherOrderHypothesisStatementLineage
    ] = Field(
        min_length=3,
        max_length=3,
    )

    shadow_only: Literal[True] = True
    llm_call_authorized: Literal[False] = False
    discovery_axis_materialization_authorized: Literal[False] = False
    candidate_anchor_synthesis_authorized: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    downstream_prior_art_review_required: Literal[True] = True
    downstream_n10_review_required: Literal[True] = True

    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_materialization(
        self,
    ) -> "HigherOrderHypothesisContextMaterialization":
        required = (
            self.materialization_id,
            self.source_hypothesis_context_id,
            self.source_hypothesis_context_sha256,
            self.source_higher_order_context_id,
            self.source_higher_order_topology_id,
            self.source_higher_order_carrier_id,
            self.derived_hypothesis_context_id,
            self.derived_hypothesis_context_sha256,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order hypothesis materialization requires "
                "complete source and derived lineage"
            )

        positive_ids = list(
            self.canonical_positive_premise_statement_ids
        )
        if len(positive_ids) != len(set(positive_ids)):
            raise ValueError(
                "duplicate canonical positive premise statement ID"
            )

        roles = [
            row.premise_role
            for row in self.generated_statement_lineage
        ]
        if len(roles) != len(set(roles)):
            raise ValueError(
                "duplicate higher-order generated premise role"
            )
        if set(roles) != {
            "source_backbone_relation",
            "target_backbone_relation",
            "modifier_relation",
        }:
            raise ValueError(
                "materialization requires exactly one source-backbone, "
                "target-backbone, and modifier lineage row"
            )

        statement_ids = [
            row.statement_id
            for row in self.generated_statement_lineage
        ]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError(
                "duplicate generated higher-order statement ID"
            )

        if set(statement_ids) & set(positive_ids):
            raise ValueError(
                "higher-order inspiration statement cannot be a canonical "
                "positive premise"
            )

        return self


class HigherOrderHypothesisContextProjection(StrictModel):
    schema_version: Literal[
        "higher-order-hypothesis-context-projection-v1"
    ] = "higher-order-hypothesis-context-projection-v1"

    context: HypothesisContext
    materialization: HigherOrderHypothesisContextMaterialization

    @model_validator(mode="after")
    def validate_projection(
        self,
    ) -> "HigherOrderHypothesisContextProjection":
        if (
            self.context.context_id
            != self.materialization.derived_hypothesis_context_id
        ):
            raise ValueError(
                "projection context/materialization context_id mismatch"
            )
        if (
            self.context.context_sha256
            != self.materialization.derived_hypothesis_context_sha256
        ):
            raise ValueError(
                "projection context/materialization context_sha256 mismatch"
            )

        statements = {
            row.statement_id: row
            for row in self.context.evidence_statements
        }
        if len(statements) != len(self.context.evidence_statements):
            raise ValueError(
                "derived HypothesisContext contains duplicate statement IDs"
            )

        positive_ids = {
            row.statement_id
            for row in self.context.evidence_statements
            if row.eligible_as_premise
        }
        if positive_ids != set(
            self.materialization.canonical_positive_premise_statement_ids
        ):
            raise ValueError(
                "derived context changed canonical positive-premise set"
            )

        for lineage in self.materialization.generated_statement_lineage:
            row = statements.get(lineage.statement_id)
            if row is None:
                raise ValueError(
                    "derived context missing generated higher-order statement"
                )
            if row.eligible_as_premise or row.eligible_as_gap:
                raise ValueError(
                    "higher-order generated statement gained premise/gap "
                    "authority"
                )
            if tuple(row.premise_restrictions) != _HIGHER_ORDER_RESTRICTIONS:
                raise ValueError(
                    "higher-order generated statement restriction drift"
                )

        return self


def _validate_source_context(
    context: HypothesisContext,
) -> list[str]:
    statement_ids = [
        row.statement_id
        for row in context.evidence_statements
    ]
    if len(statement_ids) != len(set(statement_ids)):
        raise ValueError(
            "source HypothesisContext contains duplicate statement IDs"
        )

    positive_ids = [
        row.statement_id
        for row in context.evidence_statements
        if row.eligible_as_premise
    ]
    if not positive_ids:
        raise ValueError(
            "higher-order shadow materialization requires at least one "
            "canonical eligible positive premise"
        )
    return positive_ids


def _validate_higher_order_context(
    context: HigherOrderSynthesisContext,
) -> None:
    if context.epistemic_status != "inspiration_only":
        raise ValueError(
            "higher-order context must remain inspiration_only"
        )
    if not context.requires_verification:
        raise ValueError(
            "higher-order context must require verification"
        )
    if context.novelty_authority:
        raise ValueError(
            "higher-order context cannot carry novelty authority"
        )
    if not context.shadow_only:
        raise ValueError(
            "higher-order production projection is shadow-only"
        )
    if not context.prompt_rendering_authorized:
        raise ValueError(
            "higher-order prompt rendering is not authorized"
        )

    guard = context.guard
    if not guard.topology_is_not_interaction_evidence:
        raise ValueError("topology cannot be interaction evidence")
    if not guard.topology_is_not_novelty_evidence:
        raise ValueError("topology cannot be novelty evidence")
    if guard.mediator_identity_assertion_authorized:
        raise ValueError(
            "mediator identity assertion must remain unauthorized"
        )
    if not guard.generated_interaction_must_remain_hypothesis:
        raise ValueError(
            "generated interaction must remain a hypothesis"
        )
    if not guard.prior_art_review_required:
        raise ValueError(
            "downstream prior-art review must remain required"
        )
    if not guard.n10_novelty_review_required:
        raise ValueError(
            "downstream N10 review must remain required"
        )
    if guard.discovery_axis_materialization_authorized:
        raise ValueError(
            "discovery-axis materialization must remain unauthorized"
        )
    if guard.candidate_anchor_synthesis_authorized:
        raise ValueError(
            "candidate-anchor synthesis must remain unauthorized"
        )
    if guard.llm_call_authorized:
        raise ValueError(
            "HigherOrderSynthesisContext itself must not authorize an LLM call"
        )

    for premise in context.premises:
        if (
            premise.authority
            != RelationComponentAuthority.CONFIRMED_KNOWN
            or premise.epistemic_use
            != "confirmed_known_component"
        ):
            raise ValueError(
                "S24a1 production materialization accepts only "
                "CONFIRMED_KNOWN higher-order components"
            )


def _restricted_statement(
    *,
    higher_order_context: HigherOrderSynthesisContext,
    premise,
) -> tuple[
    HypothesisEvidenceStatement,
    HigherOrderHypothesisStatementLineage,
]:
    statement_id = _stable_id(
        "stmt_ho_inspiration",
        higher_order_context.context_id,
        higher_order_context.higher_order_topology_id,
        premise.premise_role,
        premise.component_id,
        premise.provenance_source_id,
    )

    statement = HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=(
            "Composition inspiration relation only; not a positive premise "
            "for the generated source-target hypothesis: "
            f"{premise.subject} --{premise.relation}--> {premise.object}."
        ),
        epistemic_role="reported",
        claim_kind="higher_order_composition_inspiration",
        # The compact higher-order context preserves accepted-pattern source
        # IDs, not complete paper/chunk provenance. Do not fabricate paper IDs.
        paper_ids=[],
        scientific_support_node_ids=[],
        scientific_support_edge_ids=[],
        support_path_ids=[],
        alignment_path_ids=[],
        requires_verification=False,
        eligible_as_premise=False,
        eligible_as_gap=False,
        premise_restrictions=list(_HIGHER_ORDER_RESTRICTIONS),
    )

    lineage = HigherOrderHypothesisStatementLineage(
        statement_id=statement_id,
        higher_order_premise_id=premise.premise_id,
        premise_role=premise.premise_role,
        component_id=premise.component_id,
        provenance_source_id=premise.provenance_source_id,
        authority=premise.authority,
    )
    return statement, lineage


def materialize_higher_order_hypothesis_context(
    *,
    source_context: HypothesisContext,
    higher_order_context: HigherOrderSynthesisContext,
) -> HigherOrderHypothesisContextProjection:
    """
    Deterministically add exactly three restricted higher-order inspiration
    statements to a canonical HypothesisContext.

    No LLM call, ranking, novelty judgment, discovery-axis materialization,
    candidate-anchor synthesis, or production selection occurs here.
    """

    canonical_positive_ids = _validate_source_context(
        source_context
    )
    _validate_higher_order_context(
        higher_order_context
    )

    generated_statements: list[HypothesisEvidenceStatement] = []
    generated_lineage: list[
        HigherOrderHypothesisStatementLineage
    ] = []

    for premise in higher_order_context.premises:
        statement, lineage = _restricted_statement(
            higher_order_context=higher_order_context,
            premise=premise,
        )
        generated_statements.append(statement)
        generated_lineage.append(lineage)

    source_statement_ids = {
        row.statement_id
        for row in source_context.evidence_statements
    }
    generated_ids = {
        row.statement_id
        for row in generated_statements
    }
    collision = sorted(source_statement_ids & generated_ids)
    if collision:
        raise ValueError(
            "higher-order generated statement ID collision: "
            + ",".join(collision)
        )

    payload = source_context.model_dump(mode="json")
    payload.pop("context_sha256", None)

    derived_context_id = _stable_id(
        "hypothesis_context_ho_shadow",
        source_context.context_id,
        source_context.context_sha256,
        higher_order_context.context_id,
        higher_order_context.higher_order_topology_id,
    )
    payload["context_id"] = derived_context_id
    payload["evidence_statements"] = [
        *payload["evidence_statements"],
        *[
            row.model_dump(mode="json")
            for row in generated_statements
        ],
    ]

    derived_context_sha256 = _sha256_json(payload)
    derived_context = HypothesisContext(
        **payload,
        context_sha256=derived_context_sha256,
    )

    materialization = HigherOrderHypothesisContextMaterialization(
        materialization_id=_stable_id(
            "higher_order_hypothesis_context_materialization",
            source_context.context_sha256,
            higher_order_context.context_id,
            higher_order_context.higher_order_topology_id,
            derived_context_sha256,
        ),
        source_hypothesis_context_id=source_context.context_id,
        source_hypothesis_context_sha256=source_context.context_sha256,
        source_higher_order_context_id=higher_order_context.context_id,
        source_higher_order_topology_id=(
            higher_order_context.higher_order_topology_id
        ),
        source_higher_order_carrier_id=higher_order_context.carrier_id,
        derived_hypothesis_context_id=derived_context.context_id,
        derived_hypothesis_context_sha256=derived_context.context_sha256,
        canonical_positive_premise_statement_ids=(
            canonical_positive_ids
        ),
        generated_statement_lineage=generated_lineage,
        reason_codes=[
            "shadow_only_higher_order_hypothesis_context",
            "canonical_positive_premise_set_preserved",
            "higher_order_relations_restricted_nonpremise_only",
            "confirmed_known_does_not_create_positive_premise_authority",
            "topology_not_promoted_to_interaction_evidence",
            "topology_not_promoted_to_novelty_evidence",
            "no_discovery_axis_materialization",
            "no_candidate_anchor_synthesis",
            "no_llm_call_authorized_by_materialization",
            "downstream_prior_art_and_n10_required",
        ],
    )

    return HigherOrderHypothesisContextProjection(
        context=derived_context,
        materialization=materialization,
    )


class HigherOrderShadowHypothesisPromptAssembler(
    HypothesisPromptAssembler
):
    """
    Render the canonical prompt plus higher-order shadow focus.

    This assembler renders only. S24a2 must separately authorize any LLM call.
    """

    def __init__(
        self,
        *,
        higher_order_context: HigherOrderSynthesisContext,
        materialization: HigherOrderHypothesisContextMaterialization,
        statement_text_limit: int = 1100,
    ) -> None:
        super().__init__(
            statement_text_limit=statement_text_limit,
            max_hypotheses=1,
        )
        self.higher_order_context = higher_order_context
        self.materialization = materialization

    def build(
        self,
        context: HypothesisContext,
    ) -> HypothesisPrompt:
        if (
            context.context_id
            != self.materialization.derived_hypothesis_context_id
            or context.context_sha256
            != self.materialization.derived_hypothesis_context_sha256
        ):
            raise ValueError(
                "higher-order prompt assembler received the wrong derived "
                "HypothesisContext"
            )

        if (
            self.higher_order_context.context_id
            != self.materialization.source_higher_order_context_id
            or
            self.higher_order_context.higher_order_topology_id
            != self.materialization.source_higher_order_topology_id
        ):
            raise ValueError(
                "higher-order prompt/materialization provenance mismatch"
            )

        base = super().build(context)

        relation_lines = [
            (
                "- "
                + premise.premise_role
                + ": "
                + premise.subject
                + " --"
                + premise.relation
                + "--> "
                + premise.object
            )
            for premise in self.higher_order_context.premises
        ]
        canonical_ids = list(
            self.materialization.canonical_positive_premise_statement_ids
        )
        inspiration_ids = [
            row.statement_id
            for row in self.materialization.generated_statement_lineage
        ]
        opportunity = (
            self.higher_order_context.structural_opportunity
        )

        extra = "\n".join(
            [
                "",
                "HIGHER-ORDER SHADOW SYNTHESIS FOCUS",
                "===================================",
                (
                    "This section narrows generation focus only. It does not "
                    "upgrade evidence or create novelty authority."
                ),
                f"materialization_id: {self.materialization.materialization_id}",
                (
                    "higher_order_topology_id: "
                    + self.higher_order_context.higher_order_topology_id
                ),
                "",
                "REQUESTED RELATION:",
                f"- source: {self.higher_order_context.requested_source}",
                f"- target: {self.higher_order_context.requested_target}",
                "",
                "AUTOMATIC HIGHER-ORDER MODIFIER:",
                f"- modifier: {opportunity.modifier_text}",
                f"- anchor_role: {opportunity.modifier_anchor_role}",
                f"- anchor_text: {opportunity.modifier_anchor_text}",
                "",
                "RECORDED COMPOSITION RELATIONS:",
                *relation_lines,
                "",
                (
                    "These three relations are CONFIRMED_KNOWN structural "
                    "components. They are inspiration only for the proposed "
                    "higher-order interaction and are NOT positive premises "
                    "for that interaction."
                ),
                "",
                "CANONICAL POSITIVE PREMISE IDS:",
                *[f"- {statement_id}" for statement_id in canonical_ids],
                "",
                "RESTRICTED HIGHER-ORDER STATEMENT IDS:",
                *[f"- {statement_id}" for statement_id in inspiration_ids],
                "",
                "MANDATORY SHADOW RULES:",
                (
                    "- Generate at most ONE focused hypothesis centered on "
                    "the supplied modifier. Abstain if that cannot be done "
                    "without violating the evidence contract."
                ),
                (
                    "- premise_statement_ids may contain ONLY the canonical "
                    "positive premise IDs listed above."
                ),
                (
                    "- The restricted higher-order statement IDs MUST NOT "
                    "appear in premise_statement_ids or gap_statement_ids."
                ),
                (
                    "- Any claim that the modifier conditions, moderates, "
                    "interacts with, or changes the requested source-target "
                    "relationship is a HYPOTHESIS requiring verification."
                ),
                (
                    "- Do not claim that the topology, components, or their "
                    "combination are novel, unprecedented, established, "
                    "confirmed, or proven."
                ),
                (
                    "- Do not assert identity among the source-side mediator, "
                    "target-side mediator, and modifier-anchor expressions."
                ),
                (
                    "- Keep higher-order effect direction open unless an "
                    "exact canonical positive premise supports that direction; "
                    "otherwise use expected_direction='unspecified'."
                ),
                (
                    "- Prior-art review and N10 novelty review remain required "
                    "before any positive novelty conclusion."
                ),
                (
                    "- This prompt rendering does not itself authorize an "
                    "LLM call or production selection change."
                ),
            ]
        )

        return HypothesisPrompt.create(
            system_prompt=base.system_prompt,
            user_prompt=base.user_prompt + "\n" + extra,
        )
