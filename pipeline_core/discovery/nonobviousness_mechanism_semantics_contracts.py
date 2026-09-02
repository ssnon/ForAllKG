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


MechanismSemanticRelation = Literal[
    "SAME_MECHANISM",
    "SUPPLEMENTAL_SUBSUMES_BASELINE",
    "BASELINE_SUBSUMES_SUPPLEMENTAL",
    "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
    "DISTINCT_MECHANISMS",
    "INSUFFICIENT_FOR_JUDGMENT",
]


MechanismSupplyGeometry = Literal[
    "DIRECT_SCIENTIFIC_CHAIN",
    "COMMON_ANCHOR_CONTEXT",
]


MechanismReviewConfidence = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
]


MechanismSearchOperator = Literal[
    "MECHANISM_AUGMENTATION",
    "RELATIVE_CONTRIBUTION_SHIFT",
    "PATHWAY_COMPETITION",
    "MECHANISM_SWITCH",
]


class MechanismSemanticDraft(
    StrictModel
):
    """Evidence-bounded semantic comparison of two mechanism components.

    The LLM classifies mechanism semantics and identifies components.
    It does not decide which N11 search operators are eligible.
    """

    classification: MechanismSemanticRelation

    shared_mechanistic_components: list[str] = Field(
        max_length=12,
    )

    baseline_only_components: list[str] = Field(
        max_length=12,
    )

    supplemental_only_components: list[str] = Field(
        max_length=12,
    )

    task_relation_grounded: bool

    reason_summary: str = Field(
        min_length=1
    )

    epistemic_cautions: list[str] = Field(
        max_length=12,
    )

    confidence: MechanismReviewConfidence

    @model_validator(mode="after")
    def _canonicalize_components(
        self,
    ) -> "MechanismSemanticDraft":
        for field_name in (
            "shared_mechanistic_components",
            "baseline_only_components",
            "supplemental_only_components",
            "epistemic_cautions",
        ):
            values = getattr(
                self,
                field_name,
            )

            cleaned = []
            seen = set()

            for value in values:
                value = " ".join(
                    str(value).split()
                )

                if (
                    not value
                    or value in seen
                ):
                    continue

                seen.add(value)
                cleaned.append(value)

            setattr(
                self,
                field_name,
                cleaned,
            )

        return self


class MechanismOperatorPolicyResult(
    StrictModel
):
    schema_version: Literal[
        "n11-mechanism-operator-policy-v1"
    ] = (
        "n11-mechanism-operator-policy-v1"
    )

    supply_geometry: MechanismSupplyGeometry

    semantic_classification: MechanismSemanticRelation

    task_relation_grounded: bool

    hypothesis_bound_gap_available: bool

    grounded_design_lever_available: bool

    explicit_competition_signal: bool

    explicit_switch_signal: bool

    eligible_operators: list[
        MechanismSearchOperator
    ] = Field(
        default_factory=list
    )

    blocked_operators: dict[
        str,
        list[str],
    ] = Field(
        default_factory=dict
    )

    shadow_only: Literal[True] = True

    scientific_selection_changed: Literal[
        False
    ] = False

    unresolved_relation_promoted_to_positive_evidence: Literal[
        False
    ] = False

    llm_has_operator_authority: Literal[
        False
    ] = False
