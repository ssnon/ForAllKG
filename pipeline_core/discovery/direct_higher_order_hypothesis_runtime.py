from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderSynthesisContext,
    DirectSynthesisPremiseRole,
)
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_prompt import (
    HypothesisPrompt,
    HypothesisPromptAssembler,
)
from pipeline_core.discovery.hypothesis_runtime import (
    HypothesisMakerAgentRuntime,
    HypothesisMakerRunOutcome,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_RESTRICTIONS = (
    "direct_higher_order_composition_inspiration_only",
    "must_not_appear_in_premise_statement_ids",
    "must_not_appear_in_gap_statement_ids",
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


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


class DirectHigherOrderHypothesisStatementLineage(StrictModel):
    schema_version: Literal[
        "direct-higher-order-hypothesis-statement-lineage-v1"
    ] = "direct-higher-order-hypothesis-statement-lineage-v1"

    statement_id: str
    source_premise_id: str
    premise_role: DirectSynthesisPremiseRole

    component_id: str
    provenance_source_id: str
    authority: RelationComponentAuthority


class DirectHigherOrderHypothesisContextMaterialization(StrictModel):
    schema_version: Literal[
        "direct-higher-order-hypothesis-context-materialization-v1"
    ] = "direct-higher-order-hypothesis-context-materialization-v1"

    materialization_id: str

    source_hypothesis_context_id: str
    source_hypothesis_context_sha256: str

    source_direct_context_id: str
    source_direct_topology_id: str

    derived_hypothesis_context_id: str
    derived_hypothesis_context_sha256: str

    canonical_positive_premise_statement_ids: list[str] = Field(
        min_length=1
    )
    generated_statement_lineage: list[
        DirectHigherOrderHypothesisStatementLineage
    ] = Field(min_length=2, max_length=2)

    shadow_only: Literal[True] = True
    llm_call_authorized: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False

    downstream_prior_art_review_required: Literal[True] = True
    downstream_n10_review_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_materialization(self):
        roles = {
            row.premise_role
            for row in self.generated_statement_lineage
        }
        if roles != {
            "direct_task_relation",
            "modifier_relation",
        }:
            raise ValueError(
                "direct materialization requires exactly the two direct roles"
            )

        generated_ids = {
            row.statement_id
            for row in self.generated_statement_lineage
        }
        if generated_ids & set(
            self.canonical_positive_premise_statement_ids
        ):
            raise ValueError(
                "restricted direct statements cannot be canonical premises"
            )
        return self


class DirectHigherOrderHypothesisContextProjection(StrictModel):
    schema_version: Literal[
        "direct-higher-order-hypothesis-context-projection-v1"
    ] = "direct-higher-order-hypothesis-context-projection-v1"

    context: HypothesisContext
    materialization: DirectHigherOrderHypothesisContextMaterialization

    @model_validator(mode="after")
    def validate_projection(self):
        if (
            self.context.context_id
            != self.materialization.derived_hypothesis_context_id
            or self.context.context_sha256
            != self.materialization.derived_hypothesis_context_sha256
        ):
            raise ValueError(
                "derived context/materialization identity mismatch"
            )

        statements = {
            row.statement_id: row
            for row in self.context.evidence_statements
        }

        positive_ids = {
            row.statement_id
            for row in self.context.evidence_statements
            if row.eligible_as_premise
        }

        if positive_ids != set(
            self.materialization.canonical_positive_premise_statement_ids
        ):
            raise ValueError(
                "direct projection changed canonical positive-premise set"
            )

        for lineage in self.materialization.generated_statement_lineage:
            row = statements.get(lineage.statement_id)
            if row is None:
                raise ValueError(
                    "derived context missing restricted direct statement"
                )
            if row.eligible_as_premise or row.eligible_as_gap:
                raise ValueError(
                    "restricted direct statement gained premise/gap authority"
                )
            if tuple(row.premise_restrictions) != _RESTRICTIONS:
                raise ValueError(
                    "restricted direct statement restriction drift"
                )

        return self


def _validate_source_context(context: HypothesisContext) -> list[str]:
    positive_ids = [
        row.statement_id
        for row in context.evidence_statements
        if row.eligible_as_premise
    ]
    if not positive_ids:
        raise ValueError(
            "direct higher-order generation requires canonical positive premises"
        )
    return positive_ids


def _validate_direct_context(
    context: DirectHigherOrderSynthesisContext,
) -> None:
    if not context.shadow_only:
        raise ValueError(
            "direct higher-order context must remain shadow-only"
        )
    if not context.requires_verification:
        raise ValueError(
            "direct higher-order context must require verification"
        )
    if context.guard.llm_call_authorized:
        raise ValueError(
            "synthesis context itself must not authorize an LLM call"
        )
    if not context.guard.generated_interaction_must_remain_hypothesis:
        raise ValueError(
            "generated interaction must remain a hypothesis"
        )
    if not context.guard.prior_art_review_required:
        raise ValueError(
            "prior-art review must remain required"
        )
    if not context.guard.n10_novelty_review_required:
        raise ValueError(
            "N10 review must remain required"
        )

    by_role = {
        premise.premise_role: premise
        for premise in context.premises
    }

    direct = by_role["direct_task_relation"]
    modifier = by_role["modifier_relation"]

    if (
        direct.authority
        != RelationComponentAuthority.CONFIRMED_KNOWN
    ):
        raise ValueError(
            "direct task relation must remain CONFIRMED_KNOWN"
        )

    if (
        modifier.authority
        != RelationComponentAuthority.CANDIDATE_INSPIRATION
    ):
        raise ValueError(
            "S204 requires candidate-backed modifier inspiration"
        )


def _restricted_statement(
    *,
    context: DirectHigherOrderSynthesisContext,
    premise,
) -> tuple[
    HypothesisEvidenceStatement,
    DirectHigherOrderHypothesisStatementLineage,
]:
    statement_id = _stable_id(
        "stmt_direct_ho_inspiration",
        context.context_id,
        context.direct_higher_order_topology_id,
        premise.premise_role,
        premise.component_id,
        premise.provenance_source_id,
    )

    statement = HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=(
            "Direct higher-order composition inspiration only; "
            "not a positive premise or established interaction: "
            f"{premise.subject} --{premise.relation}--> {premise.object}."
        ),
        epistemic_role=(
            "reported"
            if premise.authority
            == RelationComponentAuthority.CONFIRMED_KNOWN
            else "unresolved"
        ),
        claim_kind="direct_higher_order_composition_inspiration",
        paper_ids=[],
        scientific_support_node_ids=[],
        scientific_support_edge_ids=[],
        support_path_ids=[],
        alignment_path_ids=[],
        requires_verification=True,
        eligible_as_premise=False,
        eligible_as_gap=False,
        premise_restrictions=list(_RESTRICTIONS),
    )

    lineage = DirectHigherOrderHypothesisStatementLineage(
        statement_id=statement_id,
        source_premise_id=premise.premise_id,
        premise_role=premise.premise_role,
        component_id=premise.component_id,
        provenance_source_id=premise.provenance_source_id,
        authority=premise.authority,
    )

    return statement, lineage


def materialize_direct_higher_order_hypothesis_context(
    *,
    source_context: HypothesisContext,
    direct_context: DirectHigherOrderSynthesisContext,
) -> DirectHigherOrderHypothesisContextProjection:
    canonical_positive_ids = _validate_source_context(
        source_context
    )
    _validate_direct_context(
        direct_context
    )

    generated_statements = []
    generated_lineage = []

    for premise in direct_context.premises:
        statement, lineage = _restricted_statement(
            context=direct_context,
            premise=premise,
        )
        generated_statements.append(statement)
        generated_lineage.append(lineage)

    source_ids = {
        row.statement_id
        for row in source_context.evidence_statements
    }
    generated_ids = {
        row.statement_id
        for row in generated_statements
    }

    if source_ids & generated_ids:
        raise ValueError(
            "direct higher-order generated statement id collision"
        )

    payload = source_context.model_dump(mode="json")
    payload.pop("context_sha256", None)

    derived_context_id = _stable_id(
        "hypothesis_context_direct_ho_shadow",
        source_context.context_id,
        source_context.context_sha256,
        direct_context.context_id,
        direct_context.direct_higher_order_topology_id,
    )

    payload["context_id"] = derived_context_id
    payload["evidence_statements"] = [
        *payload["evidence_statements"],
        *[
            row.model_dump(mode="json")
            for row in generated_statements
        ],
    ]

    derived_sha = _sha256_json(payload)

    derived = HypothesisContext(
        **payload,
        context_sha256=derived_sha,
    )

    materialization = DirectHigherOrderHypothesisContextMaterialization(
        materialization_id=_stable_id(
            "direct_ho_hypothesis_context_materialization",
            source_context.context_sha256,
            direct_context.context_id,
            derived_sha,
        ),
        source_hypothesis_context_id=source_context.context_id,
        source_hypothesis_context_sha256=source_context.context_sha256,
        source_direct_context_id=direct_context.context_id,
        source_direct_topology_id=(
            direct_context.direct_higher_order_topology_id
        ),
        derived_hypothesis_context_id=derived.context_id,
        derived_hypothesis_context_sha256=derived.context_sha256,
        canonical_positive_premise_statement_ids=(
            canonical_positive_ids
        ),
        generated_statement_lineage=generated_lineage,
    )

    return DirectHigherOrderHypothesisContextProjection(
        context=derived,
        materialization=materialization,
    )


class DirectHigherOrderShadowHypothesisPromptAssembler(
    HypothesisPromptAssembler
):
    def __init__(
        self,
        *,
        direct_context: DirectHigherOrderSynthesisContext,
        materialization: DirectHigherOrderHypothesisContextMaterialization,
        statement_text_limit: int = 1100,
    ) -> None:
        super().__init__(
            statement_text_limit=statement_text_limit,
            max_hypotheses=1,
        )
        self.direct_context = direct_context
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
                "direct shadow prompt received wrong derived context"
            )

        base = super().build(context)

        opportunity = self.direct_context.structural_opportunity

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
                + " [authority="
                + premise.authority.value
                + "]"
            )
            for premise in self.direct_context.premises
        ]

        positive_ids = (
            self.materialization.canonical_positive_premise_statement_ids
        )
        restricted_ids = [
            row.statement_id
            for row in self.materialization.generated_statement_lineage
        ]

        extra = "\n".join(
            [
                "",
                "DIRECT HIGHER-ORDER SHADOW GENERATION FOCUS",
                "==========================================",
                (
                    "Generate at most ONE falsifiable hypothesis. "
                    "The supplied topology is inspiration only."
                ),
                "",
                "REQUESTED TASK:",
                f"- source: {self.direct_context.requested_source}",
                f"- target: {self.direct_context.requested_target}",
                "",
                "TASK-ROLE EXPRESSIONS "
                "(role projection only; not identity/equivalence):",
                f"- source-role expression: {opportunity.source_role_text}",
                f"- target-role expression: {opportunity.target_role_text}",
                "",
                "RECORDED COMPOSITION RELATIONS:",
                *relation_lines,
                "",
                "CANDIDATE MODIFIER:",
                f"- modifier C: {opportunity.modifier_text}",
                f"- anchor role: {opportunity.modifier_anchor_role}",
                f"- anchor expression: {opportunity.modifier_anchor_text}",
                "",
                "CANONICAL POSITIVE PREMISE IDS:",
                *[
                    "- " + statement_id
                    for statement_id in positive_ids
                ],
                "",
                "RESTRICTED DIRECT-HIGHER-ORDER IDS:",
                *[
                    "- " + statement_id
                    for statement_id in restricted_ids
                ],
                "",
                "MANDATORY RULES:",
                (
                    "- premise_statement_ids may contain ONLY canonical "
                    "positive premise IDs listed above."
                ),
                (
                    "- restricted direct-higher-order IDs MUST NOT appear "
                    "in premise_statement_ids or gap_statement_ids."
                ),
                (
                    "- Center the proposed inferential bridge on whether C "
                    "conditions, moderates, changes, or bounds the requested "
                    "source-target relation. If that cannot be stated "
                    "falsifiably, abstain."
                ),
                (
                    "- Treat that C interaction as a NEW HYPOTHESIS, never "
                    "as established evidence."
                ),
                (
                    "- Do not reverse or strengthen either recorded relation."
                ),
                (
                    "- Do not assert scientific identity/equivalence between "
                    "task-role expressions and requested source/target wording."
                ),
                (
                    "- Preserve CONFIRMED_KNOWN versus CANDIDATE_INSPIRATION "
                    "authority. Candidate inspiration is not a positive premise."
                ),
                (
                    "- Preserve SYSTEM AND MATERIAL SCOPE. If a canonical "
                    "positive premise reports a mechanism for a different "
                    "material, layer, surface, reporter, substrate, or "
                    "experimental system than modifier C, state that difference "
                    "explicitly before using the premise as analogical "
                    "motivation."
                ),
                (
                    "- Do NOT write 'reported', 'established', 'demonstrated', "
                    "or equivalent fact language in a way that grammatically "
                    "attributes a cross-system mechanism to modifier C unless "
                    "an exact canonical positive premise reports that mechanism "
                    "for the same modifier identity and system."
                ),
                (
                    "- When transferring a mechanism across systems, separate "
                    "the clauses explicitly: first state what the other system "
                    "reported, then state that applying an analogous mechanism "
                    "to C is the NEW HYPOTHESIS. Do not compress those two "
                    "epistemic steps into one reported composite claim."
                ),
                (
                    "- A candidate modifier relation may motivate C, but it "
                    "does not authorize a candidate-specific mechanism as "
                    "reported fact. Any such mechanism must remain explicitly "
                    "proposed unless independently supported by an exact "
                    "canonical positive premise."
                ),
                (
                    "- Keep interaction direction open unless canonical "
                    "positive premises explicitly support a direction; "
                    "otherwise use expected_direction='unspecified'."
                ),
                (
                    "- Do not claim novelty. Prior-art and N10 review remain "
                    "mandatory downstream."
                ),
            ]
        )

        return HypothesisPrompt.create(
            system_prompt=base.system_prompt,
            user_prompt=base.user_prompt + "\n" + extra,
        )


class DirectHigherOrderShadowGenerationAuthorization(StrictModel):
    schema_version: Literal[
        "direct-higher-order-shadow-generation-authorization-v1"
    ] = "direct-higher-order-shadow-generation-authorization-v1"

    authorization_id: str

    materialization_id: str
    source_direct_context_id: str
    source_direct_topology_id: str
    derived_hypothesis_context_id: str
    derived_hypothesis_context_sha256: str

    llm_call_authorized: Literal[True] = True
    canonical_hypothesis_runtime_only: Literal[True] = True
    shadow_only: Literal[True] = True
    max_hypotheses: Literal[1] = 1

    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False
    external_novelty_bypass_authorized: Literal[False] = False
    n10_bypass_authorized: Literal[False] = False


def authorize_direct_higher_order_shadow_generation(
    *,
    projection: DirectHigherOrderHypothesisContextProjection,
    direct_context: DirectHigherOrderSynthesisContext,
) -> DirectHigherOrderShadowGenerationAuthorization:
    materialization = projection.materialization

    if materialization.llm_call_authorized:
        raise ValueError(
            "materialization itself must remain non-authorizing"
        )
    if direct_context.guard.llm_call_authorized:
        raise ValueError(
            "synthesis context itself must remain non-authorizing"
        )

    if (
        materialization.source_direct_context_id
        != direct_context.context_id
        or materialization.source_direct_topology_id
        != direct_context.direct_higher_order_topology_id
    ):
        raise ValueError(
            "direct generation authorization lineage mismatch"
        )

    return DirectHigherOrderShadowGenerationAuthorization(
        authorization_id=_stable_id(
            "direct_ho_shadow_generation_authorization",
            materialization.materialization_id,
            direct_context.context_id,
            direct_context.direct_higher_order_topology_id,
            projection.context.context_sha256,
        ),
        materialization_id=materialization.materialization_id,
        source_direct_context_id=direct_context.context_id,
        source_direct_topology_id=(
            direct_context.direct_higher_order_topology_id
        ),
        derived_hypothesis_context_id=projection.context.context_id,
        derived_hypothesis_context_sha256=projection.context.context_sha256,
    )


DirectHigherOrderShadowRunStatus = Literal[
    "proposed",
    "abstained",
    "canonical_rejected",
    "shadow_contract_rejected",
]


@dataclass(frozen=True)
class DirectHigherOrderShadowRunOutcome:
    projection: DirectHigherOrderHypothesisContextProjection
    authorization: DirectHigherOrderShadowGenerationAuthorization
    canonical_outcome: HypothesisMakerRunOutcome

    status: DirectHigherOrderShadowRunStatus
    shadow_contract_passed: bool
    shadow_failure_code: str | None = None
    shadow_failure_message: str | None = None

    @property
    def portfolio_for_downstream(
        self,
    ) -> HypothesisPortfolio | None:
        if self.status != "proposed":
            return None
        return self.canonical_outcome.accepted_portfolio


class DirectHigherOrderShadowHypothesisRuntime:
    def __init__(
        self,
        draft_backend: HypothesisDraftBackend,
        *,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError(
                "direct higher-order shadow supports max_repairs 0 or 1"
            )
        self.draft_backend = draft_backend
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.max_repairs = int(max_repairs)

    def _classify(
        self,
        *,
        projection: DirectHigherOrderHypothesisContextProjection,
        canonical: HypothesisMakerRunOutcome,
    ):
        portfolio = canonical.accepted_portfolio

        if portfolio is None:
            return (
                "canonical_rejected",
                False,
                "CANONICAL_RUNTIME_REJECTED",
                "canonical hypothesis runtime produced no accepted portfolio",
            )

        count = len(portfolio.hypotheses)

        if count == 0:
            return (
                "abstained",
                True,
                None,
                None,
            )

        if count != 1:
            return (
                "shadow_contract_rejected",
                False,
                "DIRECT_HO_CARDINALITY_VIOLATION",
                "direct higher-order shadow requires zero or one hypothesis",
            )

        card = portfolio.hypotheses[0]

        canonical_positive = set(
            projection.materialization
            .canonical_positive_premise_statement_ids
        )

        if not set(
            card.premise_statement_ids
        ).issubset(canonical_positive):
            return (
                "shadow_contract_rejected",
                False,
                "DIRECT_HO_NONCANONICAL_POSITIVE_PREMISE",
                "hypothesis used a premise outside canonical positive set",
            )

        restricted_ids = {
            row.statement_id
            for row in projection.materialization.generated_statement_lineage
        }

        if restricted_ids & set(
            card.premise_statement_ids
        ):
            return (
                "shadow_contract_rejected",
                False,
                "DIRECT_HO_INSPIRATION_USED_AS_POSITIVE_PREMISE",
                "restricted inspiration used as positive premise",
            )

        if restricted_ids & set(
            card.gap_statement_ids
        ):
            return (
                "shadow_contract_rejected",
                False,
                "DIRECT_HO_INSPIRATION_USED_AS_GAP",
                "restricted inspiration used as gap",
            )

        return (
            "proposed",
            True,
            None,
            None,
        )

    def run(
        self,
        *,
        projection: DirectHigherOrderHypothesisContextProjection,
        direct_context: DirectHigherOrderSynthesisContext,
        authorization: DirectHigherOrderShadowGenerationAuthorization,
    ) -> DirectHigherOrderShadowRunOutcome:
        if not authorization.llm_call_authorized:
            raise ValueError(
                "explicit direct higher-order LLM authorization required"
            )

        expected = {
            "materialization_id":
                projection.materialization.materialization_id,
            "source_direct_context_id":
                direct_context.context_id,
            "source_direct_topology_id":
                direct_context.direct_higher_order_topology_id,
            "derived_hypothesis_context_id":
                projection.context.context_id,
            "derived_hypothesis_context_sha256":
                projection.context.context_sha256,
        }

        for field_name, value in expected.items():
            if getattr(
                authorization,
                field_name,
            ) != value:
                raise ValueError(
                    "direct authorization binding mismatch: "
                    + field_name
                )

        assembler = DirectHigherOrderShadowHypothesisPromptAssembler(
            direct_context=direct_context,
            materialization=projection.materialization,
        )

        canonical_runtime = HypothesisMakerAgentRuntime(
            self.draft_backend,
            prompt_assembler=assembler,
            compiler=self.compiler,
            validator=self.validator,
            max_repairs=self.max_repairs,
        )

        canonical = canonical_runtime.run(
            projection.context
        )

        status, passed, code, message = self._classify(
            projection=projection,
            canonical=canonical,
        )

        return DirectHigherOrderShadowRunOutcome(
            projection=projection,
            authorization=authorization,
            canonical_outcome=canonical,
            status=status,
            shadow_contract_passed=passed,
            shadow_failure_code=code,
            shadow_failure_message=message,
        )
