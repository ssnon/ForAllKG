from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


N11MissingBridgePathClass = Literal[
    "DIRECT_SCIENTIFIC_CHAIN",
    "COMMON_ANCHOR_CONTEXT",
    "NAVIGATION_ONLY",
]


class N11MissingBridgeSearchRequirement(
    StrictModel
):
    """Epistemic constraints for the next grounded-bridge search.

    This contract describes what kind of evidence would be required
    to repair the missing lower-order bridge. It does not assert that
    such a relation exists.
    """

    search_objective: Literal[
        "FACTOR_TO_BASE_RELEVANT_STATE_OR_MECHANISM"
    ] = (
        "FACTOR_TO_BASE_RELEVANT_STATE_OR_MECHANISM"
    )

    required_grounding_mode: Literal[
        "EXPLICIT_TASK_CONNECTED_LOWER_ORDER_RELATION"
    ] = (
        "EXPLICIT_TASK_CONNECTED_LOWER_ORDER_RELATION"
    )

    allowed_path_classes: list[
        Literal[
            "DIRECT_SCIENTIFIC_CHAIN"
        ]
    ] = Field(
        default_factory=lambda: [
            "DIRECT_SCIENTIFIC_CHAIN"
        ]
    )

    blocked_path_classes: list[
        Literal[
            "COMMON_ANCHOR_CONTEXT",
            "NAVIGATION_ONLY",
        ]
    ] = Field(
        default_factory=lambda: [
            "COMMON_ANCHOR_CONTEXT",
            "NAVIGATION_ONLY",
        ]
    )

    common_anchor_context_is_sufficient: Literal[
        False
    ] = False

    navigation_only_is_sufficient: Literal[
        False
    ] = False

    must_preserve_factor_identity: Literal[
        True
    ] = True

    must_connect_to_base_relation_context: Literal[
        True
    ] = True

    must_not_assume_bridge_target_true: Literal[
        True
    ] = True

    opportunity_is_positive_evidence: Literal[
        False
    ] = False


class N11MissingBridgeOpportunity(
    StrictModel
):
    schema_version: Literal[
        "n11-missing-bridge-opportunity-v1"
    ] = "n11-missing-bridge-opportunity-v1"

    opportunity_id: str = Field(
        min_length=1
    )

    source_portfolio_id: str = Field(
        min_length=1
    )

    source_hypothesis_id: str = Field(
        min_length=1
    )

    source_claim_id: str = Field(
        min_length=1
    )

    source_execution_plan_id: str = Field(
        min_length=1
    )

    trigger_reason: Literal[
        "MISSING_LOWER_ORDER_BRIDGE"
    ] = "MISSING_LOWER_ORDER_BRIDGE"

    # Left-hand search anchor.
    factor_identity_terms: list[str] = Field(
        min_length=1
    )

    # Right-hand scientific context. This is NOT a newly asserted
    # factor->context relation.
    base_relation_terms: list[str] = Field(
        min_length=1
    )

    # Exact N10 texts retained only for audit/provenance.
    bridge_target_text_for_audit: str = Field(
        min_length=1
    )

    full_relation_text_for_audit: str = Field(
        min_length=1
    )

    bridge_retrieval_terms_for_audit: list[str]

    # Existing positive lower-order evidence remains separated by slot.
    established_base_work_ids: list[str] = Field(
        min_length=1
    )

    established_factor_work_ids: list[str] = Field(
        min_length=1
    )

    base_state: Literal[
        "ESTABLISHED"
    ] = "ESTABLISHED"

    factor_state: Literal[
        "ESTABLISHED"
    ] = "ESTABLISHED"

    bridge_state: Literal[
        "NOT_FOUND"
    ] = "NOT_FOUND"

    full_state: Literal[
        "NOT_FOUND"
    ] = "NOT_FOUND"

    # In the D1 trigger state cross-slot review cannot establish scope,
    # because BRIDGE has no positive evidence.
    relationship_review_performed: Literal[
        False
    ] = False

    relationship_scope_status: Literal[
        "UNASSESSED"
    ] = "UNASSESSED"

    relationship_bridge_kind_status: Literal[
        "UNASSESSED"
    ] = "UNASSESSED"

    search_requirement: (
        N11MissingBridgeSearchRequirement
    ) = Field(
        default_factory=(
            N11MissingBridgeSearchRequirement
        )
    )

    bridge_target_is_positive_evidence: Literal[
        False
    ] = False

    full_relation_is_positive_evidence: Literal[
        False
    ] = False

    production_authority: Literal[
        False
    ] = False


N11MissingBridgeCompilationStatus = Literal[
    "ELIGIBLE_FOR_GROUNDED_BRIDGE_SEARCH",
    "NOT_ELIGIBLE",
]


class N11MissingBridgeCompilation(
    StrictModel
):
    schema_version: Literal[
        "n11-missing-bridge-compilation-v1"
    ] = "n11-missing-bridge-compilation-v1"

    status: N11MissingBridgeCompilationStatus

    opportunity: (
        N11MissingBridgeOpportunity
        | None
    )

    reason_codes: list[str] = Field(
        min_length=1
    )

    production_authority: Literal[
        False
    ] = False

    @model_validator(
        mode="after"
    )
    def _status_consistency(
        self,
    ) -> "N11MissingBridgeCompilation":
        if (
            self.status
            == "ELIGIBLE_FOR_GROUNDED_BRIDGE_SEARCH"
            and self.opportunity is None
        ):
            raise ValueError(
                "eligible compilation requires opportunity"
            )

        if (
            self.status
            == "NOT_ELIGIBLE"
            and self.opportunity is not None
        ):
            raise ValueError(
                "non-eligible compilation cannot contain opportunity"
            )

        return self
