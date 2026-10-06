from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IdeaOriginKind = Literal[
    "FRONTIER",
    "EVOLUTION",
    "IMPORTED_REFRAME",
    "GENERATIONAL_OFFSPRING",
]

IdeaIdentityRelation = Literal[
    "SAME_IDEA",
    "DIFFERENT_IDEA",
    "INDETERMINATE",
]

IdeaGenealogyRelation = Literal[
    "REFINEMENT_OF",
    "CHILD_OF",
    "COMPOSED_FROM",
    "DERIVED_FROM",
    "IMPORTED_ROOT",
]

IdeaFacet = Literal[
    "CORE_COMMITMENTS",
    "SCOPE",
    "CONTRAST",
    "QUESTION",
]

IdeaFacetRelation = Literal[
    "PRESERVED",
    "REFINED",
    "ADDED",
    "REMOVED",
    "REPLACED",
    "INDETERMINATE",
]

FamilyRelation = Literal[
    "SAME_FAMILY",
    "ADJACENT_FAMILY",
    "DISTINCT_FAMILY",
    "INDETERMINATE",
]

SearchStatus = Literal[
    "ACTIVE",
    "ELITE",
    "ARCHIVED",
    "TERMINATED",
]


def _validate_unique_nonblank(values: list[str], label: str) -> list[str]:
    cleaned = [str(value or "").strip() for value in values]
    if any(not value for value in cleaned):
        raise ValueError(f"{label} must not contain blank values")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{label} must be unique")
    return cleaned


class ResearchIdeaKernel(StrictModel):
    """Operator- and provenance-independent semantic view of a research idea.

    The kernel is deliberately lightweight. It is not a scientific ontology and
    its hash is only an exact deterministic snapshot, not semantic identity
    authority.
    """

    schema_version: Literal[
        "research-idea-kernel-v1"
    ] = "research-idea-kernel-v1"

    canonical_intent: str = Field(min_length=1)
    core_scientific_commitments: list[str] = Field(min_length=1)
    scope_commitments: list[str] = Field(default_factory=list)
    contrastive_commitments: list[str] = Field(default_factory=list)
    question_commitment: str | None = None

    @model_validator(mode="after")
    def _validate_content(self) -> "ResearchIdeaKernel":
        self.core_scientific_commitments = _validate_unique_nonblank(
            self.core_scientific_commitments,
            "core_scientific_commitments",
        )
        self.scope_commitments = _validate_unique_nonblank(
            self.scope_commitments,
            "scope_commitments",
        )
        self.contrastive_commitments = _validate_unique_nonblank(
            self.contrastive_commitments,
            "contrastive_commitments",
        )
        if self.question_commitment is not None:
            question = self.question_commitment.strip()
            if not question:
                raise ValueError("question_commitment must be nonblank when present")
            self.question_commitment = question
        self.canonical_intent = self.canonical_intent.strip()
        return self


class ResearchIdeaNode(StrictModel):
    """Immutable-ish scientific identity and genealogy for generational search."""

    schema_version: Literal[
        "research-idea-node-v1"
    ] = "research-idea-node-v1"

    idea_id: str = Field(min_length=1)
    generation_index: int = Field(ge=0)
    parent_idea_ids: list[str] = Field(default_factory=list)

    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)

    origin_kind: IdeaOriginKind
    source_object_id: str = Field(min_length=1)
    source_parent_object_ids: list[str] = Field(default_factory=list)
    source_kind: str | None = None
    operator_id: str | None = None
    source_artifact_refs: list[str] = Field(default_factory=list)

    kernel: ResearchIdeaKernel
    kernel_sha256: str = Field(min_length=64, max_length=64)
    task_relation_mode: str = Field(min_length=1)

    differential_prediction: str = ""
    falsification_condition: str = ""
    discriminating_observation: str = ""
    projection_diagnostic_codes: list[str] = Field(default_factory=list)

    requires_verification: Literal[True] = True
    epistemic_status: Literal["INSPIRATION_ONLY"] = "INSPIRATION_ONLY"
    shadow_only: Literal[True] = True

    truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    semantic_identity_authority: Literal[False] = False
    generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_lineage(self) -> "ResearchIdeaNode":
        self.parent_idea_ids = _validate_unique_nonblank(
            self.parent_idea_ids,
            "parent_idea_ids",
        )
        self.source_parent_object_ids = _validate_unique_nonblank(
            self.source_parent_object_ids,
            "source_parent_object_ids",
        )
        self.source_artifact_refs = _validate_unique_nonblank(
            self.source_artifact_refs,
            "source_artifact_refs",
        )
        self.projection_diagnostic_codes = _validate_unique_nonblank(
            self.projection_diagnostic_codes,
            "projection_diagnostic_codes",
        )
        if self.idea_id in set(self.parent_idea_ids):
            raise ValueError("research idea cannot be its own parent")
        if self.generation_index == 0 and self.parent_idea_ids:
            raise ValueError("generation-0 research idea must not have parents")
        if self.origin_kind == "IMPORTED_REFRAME" and self.parent_idea_ids:
            raise ValueError(
                "existing imported reframes must not invent ResearchIdea parents"
            )
        return self


class SearchCostLedger(StrictModel):
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    retrieval_calls: int = Field(default=0, ge=0)
    graph_retraversal_calls: int = Field(default=0, ge=0)
    verification_calls: int = Field(default=0, ge=0)


class ResearchIdeaSearchState(StrictModel):
    """Mutable scheduler state kept separate from scientific idea identity."""

    schema_version: Literal[
        "research-idea-search-state-v1"
    ] = "research-idea-search-state-v1"

    idea_id: str = Field(min_length=1)
    visit_count: int = Field(default=0, ge=0)
    materialization_attempt_count: int = Field(default=0, ge=0)
    verification_attempt_count: int = Field(default=0, ge=0)
    search_status: SearchStatus = "ACTIVE"
    compute_spent: SearchCostLedger = Field(default_factory=SearchCostLedger)

    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaFacetAssessment(StrictModel):
    facet: IdeaFacet
    relation: IdeaFacetRelation
    similarity: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class ConceptualFamilyAssessment(StrictModel):
    idea_id_a: str = Field(min_length=1)
    idea_id_b: str = Field(min_length=1)
    relation: FamilyRelation
    similarity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal[
        "WEIGHTED_TOKEN_JACCARD_V1"
    ] = "WEIGHTED_TOKEN_JACCARD_V1"
    method_version: Literal[
        "research-idea-family-v1"
    ] = "research-idea-family-v1"

    diagnostic_only: Literal[True] = True
    family_assignment_authority: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class IdeaParentComparison(StrictModel):
    parent_idea_id: str = Field(min_length=1)
    identity_relation: IdeaIdentityRelation
    facet_assessments: list[IdeaFacetAssessment] = Field(min_length=4, max_length=4)
    family_assessment: ConceptualFamilyAssessment


class IdeaTransitionAssessment(StrictModel):
    schema_version: Literal[
        "idea-transition-assessment-v1"
    ] = "idea-transition-assessment-v1"

    proposed_idea_id: str = Field(min_length=1)
    parent_idea_ids: list[str] = Field(default_factory=list)
    parent_comparisons: list[IdeaParentComparison] = Field(default_factory=list)

    identity_relation: IdeaIdentityRelation
    genealogy_relation: IdeaGenealogyRelation
    operator_id: str | None = None
    operator_expectation_consistent: bool | None = None
    diagnostic_codes: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    semantic_identity_authority: Literal[False] = False
    family_assignment_authority: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    search_policy_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_shape(self) -> "IdeaTransitionAssessment":
        self.parent_idea_ids = _validate_unique_nonblank(
            self.parent_idea_ids,
            "parent_idea_ids",
        )
        self.diagnostic_codes = _validate_unique_nonblank(
            self.diagnostic_codes,
            "diagnostic_codes",
        )
        comparison_ids = [row.parent_idea_id for row in self.parent_comparisons]
        if comparison_ids != self.parent_idea_ids:
            raise ValueError(
                "parent_comparisons must align with parent_idea_ids in order"
            )
        if self.genealogy_relation == "IMPORTED_ROOT":
            if self.parent_idea_ids:
                raise ValueError("IMPORTED_ROOT must not have parents")
        elif not self.parent_idea_ids:
            raise ValueError("non-root transition requires at least one parent")
        return self


__all__ = [
    "ConceptualFamilyAssessment",
    "FamilyRelation",
    "IdeaFacetAssessment",
    "IdeaFacetRelation",
    "IdeaGenealogyRelation",
    "IdeaIdentityRelation",
    "IdeaOriginKind",
    "IdeaParentComparison",
    "IdeaTransitionAssessment",
    "ResearchIdeaKernel",
    "ResearchIdeaNode",
    "ResearchIdeaSearchState",
    "SearchCostLedger",
    "SearchStatus",
]
