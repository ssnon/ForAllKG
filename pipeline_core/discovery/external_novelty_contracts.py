from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExternalNoveltyStatus = Literal[
    "WELL_ESTABLISHED",
    "LITERATURE_SUPPORTED_EXTENSION",
    "NEW_COMBINATION_OF_KNOWN_EFFECTS",
    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    "PLAUSIBLY_NOVEL",
    "CONFLICTING_PRIOR_ART",
    "INSUFFICIENT_SEARCH_EVIDENCE",
]

NoveltyClaimKind = Literal[
    "mediator",
    "moderator_interaction",
    "context_condition",
    "pathway_competition",
    "descriptor_interaction",
    "distinctive_prediction",
    "mechanistic_link",
    "composite",
]

NoveltyClaimImportance = Literal["core", "supporting"]

NoveltySelectionRole = Literal[
    "NOVELTY_BEARING",
    "REQUIRED_ENABLING_RELATION",
    "TESTING_PREDICTION",
    "AUXILIARY",
]


NoveltyDiagnosticQueryKind = Literal[
    "NONE",
    "LOWER_ORDER_RELATION",
    "DIRECTIONAL_BOUNDARY",
]

NoveltyInferentialDistance = Literal[
    "LOCAL_REPHRASE",
    "SINGLE_KNOWN_STEP",
    "MULTI_STEP_COMPOSITION",
    "NEW_RELATIONAL_FORM",
    "NEW_REGIME_STRUCTURE",
]

NoveltyMechanisticNecessity = Literal[
    "NO_NEW_MECHANISM",
    "KNOWN_MECHANISM_REUSED",
    "NEW_BRIDGE_REQUIRED",
    "MECHANISM_SWITCH_REQUIRED",
]

NoveltyRegimeSpecificity = Literal[
    "NONE",
    "CONDITIONED",
    "THRESHOLD",
    "REVERSAL",
    "HYSTERESIS",
    "MECHANISM_SWITCH",
]

NoveltyCounterintuitiveness = Literal[
    "EXPECTED",
    "NONTRIVIAL",
    "COUNTER_TO_BASELINE",
]

NoveltyTestableDistinctiveness = Literal[
    "GENERIC",
    "COMPARATIVE",
    "QUANTITATIVE",
    "DISCRIMINATING_SIGNATURE",
]

NoveltyStructureFeature = Literal[
    "new_mechanism",
    "threshold",
    "regime_change",
    "reversal",
    "mechanism_switch",
    "inferential_distance",
    "mechanistic_necessity",
    "regime_specificity",
    "counterintuitiveness",
    "testable_distinctiveness",
]


class NoveltyStructureBasis(StrictModel):
    feature: NoveltyStructureFeature
    source_text: str = Field(min_length=1)


class NoveltyClaimScientificStructure(StrictModel):
    introduces_new_mechanism: bool = False
    introduces_threshold: bool = False
    introduces_regime_change: bool = False
    introduces_reversal: bool = False
    introduces_mechanism_switch: bool = False

    inferential_distance: NoveltyInferentialDistance = "LOCAL_REPHRASE"
    mechanistic_necessity: NoveltyMechanisticNecessity = "NO_NEW_MECHANISM"
    regime_specificity: NoveltyRegimeSpecificity = "NONE"
    counterintuitiveness: NoveltyCounterintuitiveness = "EXPECTED"
    testable_distinctiveness: NoveltyTestableDistinctiveness = "GENERIC"

    basis: list[NoveltyStructureBasis] = Field(default_factory=list)

PriorArtRelationship = Literal[
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "TITLE_ONLY_NEIGHBOR",
    "COMPONENT_ONLY",
    "LOWER_ORDER_RELATION_PRIOR_ART",
    "DIRECTIONAL_COUNTEREVIDENCE",
    "CONTEXTUAL_CONFLICT",
    "CONFLICTING_PRIOR_ART",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]

ClaimPriorArtStatus = Literal[
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
    "TITLE_ONLY_NEIGHBORS",
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
    "CONFLICTING_PRIOR_ART",
    "INSUFFICIENT_METADATA",
]


class NoveltyClaimDraft(StrictModel):
    local_id: str
    kind: NoveltyClaimKind
    importance: NoveltyClaimImportance = "core"
    novelty_selection_role: NoveltySelectionRole | None = None
    text: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    search_concepts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    distinguishing_terms: list[str] = Field(default_factory=list)
    prior_art_identity_terms: list[str] = Field(default_factory=list)
    relation_nucleus_terms: list[str] = Field(default_factory=list)

    # Exact hypothesis-source spans that explicitly state a
    # higher-order composed, mediated, linked, or joint relation.
    # This is provenance only, never novelty authority.
    higher_order_relation_basis: list[str] = Field(
        default_factory=list
    )

    # Structural provenance only. For a composite claim, these are
    # local IDs of separately emitted claims that constitute the
    # explicitly proposed higher-order relation.
    higher_order_component_local_ids: list[str] = Field(
        default_factory=list
    )

    required_bridge: str = ""
    predicted_observation: str = ""
    falsification_condition: str = ""
    scientific_structure: NoveltyClaimScientificStructure = Field(
        default_factory=NoveltyClaimScientificStructure
    )
    diagnostic_query_kind: NoveltyDiagnosticQueryKind = "NONE"
    diagnostic_search_query: str | None = None
    diagnostic_structural_terms: list[str] = Field(default_factory=list)
    diagnostic_relation_terms: list[str] = Field(default_factory=list)


class NoveltyClaimDecompositionDraft(StrictModel):
    claims: list[NoveltyClaimDraft] = Field(min_length=1)
    decomposition_notes: str = ""

    @model_validator(mode="after")
    def _validate_claim_set(self) -> "NoveltyClaimDecompositionDraft":
        ids = [row.local_id for row in self.claims]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate novelty-claim local_id"
            )

        if not any(
            row.importance == "core"
            for row in self.claims
        ):
            raise ValueError(
                "novelty-claim decomposition must contain "
                "at least one core claim"
            )

        return self


class NoveltyClaimInferenceProvenance(StrictModel):
    """Upstream inference context preserved for an atomic novelty claim.

    This record is provenance only. It does not establish the atomic
    claim, promote discovery-axis content into evidence, expand search
    vocabulary, or authorize novelty/non-obviousness conclusions.

    N9-D preserves the accepted Alpha4 inference context and, when a
    conservative lexical binding is available, narrows that context to
    the inference assertions that actually carry the atomic claim.
    """

    schema_version: Literal[
        "novelty-claim-inference-provenance-v1"
    ] = "novelty-claim-inference-provenance-v1"

    binding_scope: Literal[
        "HYPOTHESIS_REVIEW_CONTEXT",
        "ATOMIC_CLAIM_ASSERTION_BINDING",
    ] = "HYPOTHESIS_REVIEW_CONTEXT"

    final_hypothesis_id: str
    source_review_hypothesis_id: str
    axis_id: str
    review_status: str

    assertion_ids: list[str] = Field(
        default_factory=list
    )
    source_classes: list[str] = Field(
        default_factory=list
    )
    grounded_statement_ids: list[str] = Field(
        default_factory=list
    )
    axis_basis: list[str] = Field(
        default_factory=list
    )

    # Diagnostic provenance for deterministic assertion-to-atomic
    # binding. These fields never add scientific authority.
    binding_identity_terms: list[str] = Field(
        default_factory=list
    )
    binding_reason_codes: list[str] = Field(
        default_factory=list
    )


class NoveltyClaim(StrictModel):
    claim_id: str
    hypothesis_id: str
    claim_rank: int
    kind: NoveltyClaimKind
    importance: NoveltyClaimImportance
    novelty_selection_role: NoveltySelectionRole | None = None
    text: str
    rationale: str
    search_concepts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    distinguishing_terms: list[str] = Field(default_factory=list)
    prior_art_identity_terms: list[str] = Field(default_factory=list)
    relation_nucleus_terms: list[str] = Field(default_factory=list)

    # Validated exact-source provenance for an explicitly proposed
    # higher-order relation. Empty means no such relation was
    # conservatively preserved.
    higher_order_relation_basis: list[str] = Field(
        default_factory=list
    )

    # Canonical IDs of separately emitted component claims belonging
    # to this explicit higher-order claim. This expresses topology,
    # not evidence authority or selection outcome.
    higher_order_component_claim_ids: list[str] = Field(
        default_factory=list
    )

    required_bridge: str = ""
    predicted_observation: str = ""
    falsification_condition: str = ""
    scientific_structure: NoveltyClaimScientificStructure = Field(
        default_factory=NoveltyClaimScientificStructure
    )
    diagnostic_query_kind: NoveltyDiagnosticQueryKind = "NONE"
    diagnostic_search_query: str | None = None
    diagnostic_execution_query: str | None = None
    diagnostic_structural_terms: list[str] = Field(default_factory=list)
    diagnostic_relation_terms: list[str] = Field(default_factory=list)
    scientific_structure_reason_codes: list[str] = Field(default_factory=list)

    # Diagnostic-only validation results for explicit higher-order
    # relation provenance. These codes never establish novelty,
    # truth, prior-art status, or non-obviousness.
    higher_order_relation_reason_codes: list[str] = Field(
        default_factory=list
    )

    # Provenance only. This preserves the accepted Alpha4 inference
    # context through external novelty and N9. It is not scientific
    # authority and must not be used by itself to establish a claim.
    inference_provenance: NoveltyClaimInferenceProvenance | None = None

    # Diagnostic only. These codes record where an atomic
    # specification was absent or rejected while moving from the
    # decomposition draft into the canonical query-plan claim.
    #
    # They do not provide scientific authority and must never be used
    # to infer novelty or non-obviousness.
    specification_sanitization_reason_codes: list[str] = Field(
        default_factory=list
    )


class HypothesisNoveltyClaims(StrictModel):
    hypothesis_id: str
    title: str
    claims: list[NoveltyClaim] = Field(default_factory=list)
    decomposition_notes: str = ""


class LiteratureQuery(StrictModel):
    query_id: str
    hypothesis_id: str
    claim_id: str | None = None
    query_kind: Literal[
        "claim_primary",
        "claim_variant",
        "claim_diagnostic",
        "claim_exact_verification",
        "hypothesis_composite",
    ]
    query_text: str


class LiteratureQueryPlan(StrictModel):
    schema_version: Literal["literature-query-plan-v1"] = "literature-query-plan-v1"
    plan_id: str
    plan_sha256: str
    source_portfolio_id: str
    queries: list[LiteratureQuery] = Field(default_factory=list)
    claims: list[HypothesisNoveltyClaims] = Field(default_factory=list)
    policy_version: Literal["external-novelty-query-policy-v1"] = (
        "external-novelty-query-policy-v1"
    )


class PriorArtWork(StrictModel):
    work_id: str
    title: str
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    url: str | None = None
    open_access_url: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    citation_count: int | None = None
    providers: list[str] = Field(default_factory=list)
    provider_ids: dict[str, str] = Field(default_factory=dict)
    retrieval_query_ids: list[str] = Field(default_factory=list)
    retrieval_claim_ids: list[str] = Field(default_factory=list)


class QueryExecution(StrictModel):
    query_id: str
    provider: str
    success: bool
    result_count: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class PriorArtPacket(StrictModel):
    schema_version: Literal["prior-art-packet-v1"] = "prior-art-packet-v1"
    packet_id: str
    packet_sha256: str
    source_portfolio_id: str
    source_query_plan_id: str
    searched_at_utc: str
    providers_requested: list[str] = Field(default_factory=list)
    works: list[PriorArtWork] = Field(default_factory=list)
    executions: list[QueryExecution] = Field(default_factory=list)
    raw_work_count: int = 0
    canonical_work_count: int = 0
    deduplicated_work_count: int = 0
    supplementary_records_collapsed: int = 0
    epistemic_usage: Literal["prior_art_only_not_positive_premise"] = (
        "prior_art_only_not_positive_premise"
    )


class RankedPriorArtWork(StrictModel):
    work_id: str
    relevance_score: float
    semantic_similarity: float
    lexical_coverage: float
    reaction_domain_relevance: float = 0.5
    catalyst_scope_relevance: float = 0.5
    abstract_available: bool


class ClaimPriorArtCandidateSet(StrictModel):
    hypothesis_id: str
    claim_id: str
    ranked_works: list[RankedPriorArtWork] = Field(default_factory=list)


class PriorArtMatchDraft(StrictModel):
    work_id: str
    relationship: PriorArtRelationship
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class ClaimPriorArtReviewDraft(StrictModel):
    matches: list[PriorArtMatchDraft] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)


class PriorArtMatch(StrictModel):
    work_id: str
    relationship: PriorArtRelationship
    confidence: float
    rationale: str
    relevance_score: float
    semantic_similarity: float
    lexical_coverage: float
    reaction_domain_relevance: float = 0.5
    catalyst_scope_relevance: float = 0.5
    scope_compatible_for_conflict: bool = False
    scope_reason_codes: list[str] = Field(default_factory=list)
    title: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    abstract_available: bool = False


class ClaimSearchCoverage(StrictModel):
    claim_id: str
    query_count: int
    successful_query_count: int
    unique_work_count: int
    abstract_work_count: int
    reviewed_work_count: int


class ClaimPriorArtReview(StrictModel):
    hypothesis_id: str
    claim_id: str
    claim_text: str
    importance: NoveltyClaimImportance
    status: ClaimPriorArtStatus
    matches: list[PriorArtMatch] = Field(default_factory=list)
    coverage: ClaimSearchCoverage
    reason_codes: list[str] = Field(default_factory=list)
    reviewer_unknown_work_ids: list[str] = Field(default_factory=list)
    interpretation: str


class HypothesisSearchCoverage(StrictModel):
    hypothesis_id: str
    query_count: int
    successful_query_count: int
    provider_success_count: int
    unique_work_count: int
    abstract_work_count: int
    core_claim_count: int
    core_claims_with_minimum_abstract_coverage: int
    sufficient_for_absence_based_novelty: bool


class ExternalNoveltyPolicy(StrictModel):
    policy_version: Literal["external-novelty-policy-v1.1"] = (
        "external-novelty-policy-v1.1"
    )
    max_claims_per_hypothesis: int = 4
    max_queries_per_claim: int = 2
    max_ranked_works_per_claim: int = 8
    min_match_confidence: float = 0.65
    direct_match_confidence: float = 0.70
    min_unique_works_for_absence: int = 10
    min_abstract_works_for_absence: int = 5
    min_abstract_works_per_core_claim: int = 3
    min_successful_queries_for_absence: int = 2
    require_abstract_for_strong_match: bool = True
    require_abstract_for_partial_match: bool = True
    min_reaction_domain_for_conflict: float = 0.75
    min_catalyst_scope_for_conflict: float = 0.75


RelationalGapKind = Literal[
    "NONE",
    "HIGHER_ORDER_RELATIONAL_GAP",
]


class ExternalNoveltyCard(StrictModel):
    hypothesis_id: str
    title: str
    status: ExternalNoveltyStatus
    claim_reviews: list[ClaimPriorArtReview] = Field(default_factory=list)
    coverage: HypothesisSearchCoverage
    strongest_prior_art_work_ids: list[str] = Field(default_factory=list)
    contextual_conflict_work_ids: list[str] = Field(default_factory=list)
    lower_order_prior_art_work_ids: list[str] = Field(default_factory=list)
    lower_order_supported_core_claim_ids: list[str] = Field(default_factory=list)
    higher_order_relational_gap_claim_ids: list[str] = Field(default_factory=list)
    lower_order_core_prior_art_work_ids: list[str] = Field(default_factory=list)
    lower_order_core_unique_work_count: int = 0
    relational_gap_kind: RelationalGapKind = "NONE"
    directional_counterevidence_work_ids: list[str] = Field(default_factory=list)
    discovery_axis_id: str | None = None
    discovery_inspiration_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str
    search_limitations: list[str] = Field(default_factory=list)


class ExternalNoveltyReport(StrictModel):
    schema_version: Literal["external-novelty-report-v1"] = (
        "external-novelty-report-v1"
    )
    report_id: str
    report_sha256: str
    source_portfolio_id: str
    source_prior_art_packet_id: str
    searched_at_utc: str
    cards: list[ExternalNoveltyCard] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    policy: ExternalNoveltyPolicy
    external_novelty_claim_scope: Literal[
        "search-bounded_prior-art_assessment_not_literature-wide_proof"
    ] = "search-bounded_prior-art_assessment_not_literature-wide_proof"
    epistemic_usage: Literal["prior_art_only_not_positive_premise"] = (
        "prior_art_only_not_positive_premise"
    )
