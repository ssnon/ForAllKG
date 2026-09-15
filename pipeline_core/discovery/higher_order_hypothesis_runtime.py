from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_hypothesis_context import (
    HigherOrderHypothesisContextProjection,
    HigherOrderShadowHypothesisPromptAssembler,
)
from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderSynthesisContext,
)
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_runtime import (
    HypothesisMakerAgentRuntime,
    HypothesisMakerRunOutcome,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class HigherOrderShadowGenerationAuthorization(StrictModel):
    """
    Explicit run-level authority for one shadow generation call.

    The upstream HigherOrderSynthesisContext and its materialization remain
    non-authorizing. This token authorizes only delegation to the canonical
    HypothesisMakerAgentRuntime for the exact derived context bound here.
    """

    schema_version: Literal[
        "higher-order-shadow-generation-authorization-v1"
    ] = "higher-order-shadow-generation-authorization-v1"

    authorization_id: str

    materialization_id: str
    source_higher_order_context_id: str
    source_higher_order_topology_id: str

    derived_hypothesis_context_id: str
    derived_hypothesis_context_sha256: str

    llm_call_authorized: Literal[True] = True
    canonical_hypothesis_runtime_only: Literal[True] = True
    shadow_only: Literal[True] = True
    max_hypotheses: Literal[1] = 1

    novelty_authority: Literal[False] = False
    discovery_axis_materialization_authorized: Literal[False] = False
    candidate_anchor_synthesis_authorized: Literal[False] = False
    external_novelty_bypass_authorized: Literal[False] = False
    n10_bypass_authorized: Literal[False] = False
    production_selection_changed: Literal[False] = False

    downstream_prior_art_review_required: Literal[True] = True
    downstream_n10_review_required: Literal[True] = True

    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_authorization(
        self,
    ) -> "HigherOrderShadowGenerationAuthorization":
        required = (
            self.authorization_id,
            self.materialization_id,
            self.source_higher_order_context_id,
            self.source_higher_order_topology_id,
            self.derived_hypothesis_context_id,
            self.derived_hypothesis_context_sha256,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "higher-order shadow generation authorization requires "
                "complete lineage binding"
            )
        return self


def authorize_higher_order_shadow_generation(
    *,
    projection: HigherOrderHypothesisContextProjection,
    higher_order_context: HigherOrderSynthesisContext,
) -> HigherOrderShadowGenerationAuthorization:
    """
    Create explicit run-level LLM authority for one exact shadow projection.

    This does not mutate the source context, topology, materialization,
    production selector, or novelty state.
    """

    materialization = projection.materialization

    if materialization.llm_call_authorized:
        raise ValueError(
            "materialization must remain non-authorizing"
        )
    if not materialization.shadow_only:
        raise ValueError(
            "only shadow-only materialization may be authorized"
        )
    if materialization.production_selection_changed:
        raise ValueError(
            "shadow generation cannot start from a selection-mutating "
            "materialization"
        )

    if higher_order_context.guard.llm_call_authorized:
        raise ValueError(
            "HigherOrderSynthesisContext must remain non-authorizing"
        )
    if not higher_order_context.shadow_only:
        raise ValueError(
            "higher-order generation authorization is shadow-only"
        )
    if higher_order_context.novelty_authority:
        raise ValueError(
            "higher-order context cannot carry novelty authority"
        )

    if (
        higher_order_context.context_id
        != materialization.source_higher_order_context_id
    ):
        raise ValueError(
            "authorization higher-order context/materialization mismatch"
        )
    if (
        higher_order_context.higher_order_topology_id
        != materialization.source_higher_order_topology_id
    ):
        raise ValueError(
            "authorization higher-order topology/materialization mismatch"
        )
    if (
        projection.context.context_id
        != materialization.derived_hypothesis_context_id
        or projection.context.context_sha256
        != materialization.derived_hypothesis_context_sha256
    ):
        raise ValueError(
            "authorization projection/materialization mismatch"
        )

    authorization_id = _stable_id(
        "higher_order_shadow_generation_authorization",
        materialization.materialization_id,
        higher_order_context.context_id,
        higher_order_context.higher_order_topology_id,
        projection.context.context_id,
        projection.context.context_sha256,
    )

    return HigherOrderShadowGenerationAuthorization(
        authorization_id=authorization_id,
        materialization_id=materialization.materialization_id,
        source_higher_order_context_id=higher_order_context.context_id,
        source_higher_order_topology_id=(
            higher_order_context.higher_order_topology_id
        ),
        derived_hypothesis_context_id=projection.context.context_id,
        derived_hypothesis_context_sha256=(
            projection.context.context_sha256
        ),
        reason_codes=[
            "explicit_shadow_run_authorization",
            "canonical_hypothesis_runtime_only",
            "upstream_context_remains_non_authorizing",
            "materialization_remains_non_authorizing",
            "single_hypothesis_cardinality_required",
            "no_discovery_axis_materialization",
            "no_candidate_anchor_synthesis",
            "no_novelty_authority",
            "no_external_novelty_bypass",
            "no_n10_bypass",
            "no_production_selection_change",
        ],
    )


HigherOrderShadowRunStatus = Literal[
    "proposed",
    "abstained",
    "canonical_rejected",
    "shadow_contract_rejected",
]


@dataclass(frozen=True)
class HigherOrderShadowHypothesisRunOutcome:
    projection: HigherOrderHypothesisContextProjection
    authorization: HigherOrderShadowGenerationAuthorization
    canonical_outcome: HypothesisMakerRunOutcome

    status: HigherOrderShadowRunStatus
    shadow_contract_passed: bool
    shadow_failure_code: str | None = None
    shadow_failure_message: str | None = None

    @property
    def proposed(self) -> bool:
        return self.status == "proposed"

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"

    @property
    def portfolio_for_downstream(
        self,
    ) -> HypothesisPortfolio | None:
        if not self.proposed:
            return None
        return self.canonical_outcome.accepted_portfolio


class HigherOrderShadowHypothesisRuntime:
    """
    Shadow-only adapter over the canonical HypothesisMakerAgentRuntime.

    The adapter owns no alternate compiler, validator, repair policy, novelty
    policy, or selection policy. It binds a single higher-order focus prompt
    to the canonical runtime and applies only deterministic shadow-boundary
    checks after canonical compilation/validation.
    """

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
                "higher-order shadow generation supports max_repairs "
                "of 0 or 1 only"
            )
        self.draft_backend = draft_backend
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.max_repairs = int(max_repairs)

    def _validate_authorization_binding(
        self,
        *,
        projection: HigherOrderHypothesisContextProjection,
        higher_order_context: HigherOrderSynthesisContext,
        authorization: HigherOrderShadowGenerationAuthorization,
    ) -> None:
        materialization = projection.materialization

        expected = {
            "materialization_id": materialization.materialization_id,
            "source_higher_order_context_id": (
                higher_order_context.context_id
            ),
            "source_higher_order_topology_id": (
                higher_order_context.higher_order_topology_id
            ),
            "derived_hypothesis_context_id": projection.context.context_id,
            "derived_hypothesis_context_sha256": (
                projection.context.context_sha256
            ),
        }
        for field_name, expected_value in expected.items():
            if getattr(authorization, field_name) != expected_value:
                raise ValueError(
                    "higher-order shadow authorization binding mismatch: "
                    + field_name
                )

        expected_authorization_id = _stable_id(
            "higher_order_shadow_generation_authorization",
            materialization.materialization_id,
            higher_order_context.context_id,
            higher_order_context.higher_order_topology_id,
            projection.context.context_id,
            projection.context.context_sha256,
        )
        if authorization.authorization_id != expected_authorization_id:
            raise ValueError(
                "higher-order shadow authorization_id mismatch"
            )

        if not authorization.llm_call_authorized:
            raise ValueError(
                "shadow generation requires explicit LLM-call authorization"
            )
        if not authorization.canonical_hypothesis_runtime_only:
            raise ValueError(
                "shadow generation authorization must remain bound to the "
                "canonical hypothesis runtime"
            )
        if not authorization.shadow_only:
            raise ValueError(
                "shadow generation authorization must remain shadow-only"
            )
        if authorization.novelty_authority:
            raise ValueError(
                "shadow generation authorization cannot grant novelty "
                "authority"
            )
        if authorization.production_selection_changed:
            raise ValueError(
                "shadow generation authorization cannot mutate production "
                "selection"
            )
        if authorization.external_novelty_bypass_authorized:
            raise ValueError(
                "shadow generation cannot bypass external novelty review"
            )
        if authorization.n10_bypass_authorized:
            raise ValueError(
                "shadow generation cannot bypass N10 review"
            )

        # The run-level token is the only LLM authority introduced here.
        if materialization.llm_call_authorized:
            raise ValueError(
                "materialization unexpectedly gained LLM authority"
            )
        if higher_order_context.guard.llm_call_authorized:
            raise ValueError(
                "HigherOrderSynthesisContext unexpectedly gained LLM "
                "authority"
            )

    def _classify_canonical_outcome(
        self,
        *,
        projection: HigherOrderHypothesisContextProjection,
        canonical_outcome: HypothesisMakerRunOutcome,
    ) -> tuple[
        HigherOrderShadowRunStatus,
        bool,
        str | None,
        str | None,
    ]:
        portfolio = canonical_outcome.accepted_portfolio
        if portfolio is None:
            return (
                "canonical_rejected",
                False,
                "CANONICAL_HYPOTHESIS_RUNTIME_REJECTED",
                (
                    "canonical HypothesisMakerAgentRuntime did not produce "
                    "an accepted portfolio"
                ),
            )

        count = len(portfolio.hypotheses)
        if count == 0:
            return ("abstained", True, None, None)

        if count != 1:
            return (
                "shadow_contract_rejected",
                False,
                "HIGHER_ORDER_SHADOW_CARDINALITY_VIOLATION",
                (
                    "higher-order shadow generation requires zero "
                    "(abstention) or exactly one hypothesis"
                ),
            )

        card = portfolio.hypotheses[0]
        materialization = projection.materialization

        canonical_positive_ids = set(
            materialization.canonical_positive_premise_statement_ids
        )
        actual_premise_ids = set(card.premise_statement_ids)
        if not actual_premise_ids.issubset(canonical_positive_ids):
            return (
                "shadow_contract_rejected",
                False,
                "HIGHER_ORDER_NONCANONICAL_POSITIVE_PREMISE",
                (
                    "accepted higher-order hypothesis used a positive "
                    "premise outside the source context's canonical eligible "
                    "premise set"
                ),
            )

        generated_ids = {
            row.statement_id
            for row in materialization.generated_statement_lineage
        }
        if generated_ids & actual_premise_ids:
            return (
                "shadow_contract_rejected",
                False,
                "HIGHER_ORDER_INSPIRATION_USED_AS_POSITIVE_PREMISE",
                (
                    "restricted higher-order inspiration statement appeared "
                    "as a positive premise"
                ),
            )

        if generated_ids & set(card.gap_statement_ids):
            return (
                "shadow_contract_rejected",
                False,
                "HIGHER_ORDER_INSPIRATION_USED_AS_GAP",
                (
                    "restricted higher-order inspiration statement appeared "
                    "as a research-gap statement"
                ),
            )

        return ("proposed", True, None, None)

    def run(
        self,
        *,
        projection: HigherOrderHypothesisContextProjection,
        higher_order_context: HigherOrderSynthesisContext,
        authorization: HigherOrderShadowGenerationAuthorization,
    ) -> HigherOrderShadowHypothesisRunOutcome:
        self._validate_authorization_binding(
            projection=projection,
            higher_order_context=higher_order_context,
            authorization=authorization,
        )

        assembler = HigherOrderShadowHypothesisPromptAssembler(
            higher_order_context=higher_order_context,
            materialization=projection.materialization,
        )

        canonical_runtime = HypothesisMakerAgentRuntime(
            self.draft_backend,
            prompt_assembler=assembler,
            compiler=self.compiler,
            validator=self.validator,
            max_repairs=self.max_repairs,
        )
        canonical_outcome = canonical_runtime.run(
            projection.context
        )

        (
            status,
            passed,
            failure_code,
            failure_message,
        ) = self._classify_canonical_outcome(
            projection=projection,
            canonical_outcome=canonical_outcome,
        )

        return HigherOrderShadowHypothesisRunOutcome(
            projection=projection,
            authorization=authorization,
            canonical_outcome=canonical_outcome,
            status=status,
            shadow_contract_passed=passed,
            shadow_failure_code=failure_code,
            shadow_failure_message=failure_message,
        )
