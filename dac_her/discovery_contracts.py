from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DiscoveryPathType = Literal[
    "CANDIDATE_EXPLORATION",
    "SHARED_ENTITY_BRIDGE",
    "CROSS_PAPER_MECHANISTIC",
    "DIRECT_MECHANISTIC",
    "CROSS_PAPER_BRIDGE",
    "SCAFFOLD_NAVIGATION",
    "UNKNOWN",
]

MechanisticContinuityBand = Literal[
    "high",
    "medium",
    "low",
    "not_applicable",
]

SemanticDiversityMode = Literal[
    "node_embedding",
    "lexical_fallback",
    "disabled",
]


class DiscoveryScoreBreakdown(StrictModel):
    # v1 dimensions
    endpoint_relevance: float
    mechanistic_content: float
    cross_paper_span: float
    community_span: float
    relation_rarity: float
    exploratory_mode_bonus: float
    grounding_redundancy_penalty: float
    navigation_burden_penalty: float
    reverse_burden_penalty: float

    # v2.8.0-alpha2 dimensions. Defaults preserve compatibility with a1 bundles.
    mechanistic_continuity: float = 0.0
    semantic_grounding_redundancy_penalty: float = 0.0
    generic_entity_burden_penalty: float = 0.0
    registry_hop_penalty: float = 0.0

    # v2.8.0-alpha3 candidate-unit dimensions. These are traversal/discovery
    # quality heuristics, never novelty or evidentiary confidence scores.
    candidate_unit_quality: float = 0.0
    reaction_domain_switch_penalty: float = 0.0

    total: float


class DiscoveryInspiration(StrictModel):
    inspiration_id: str
    source_path_id: str
    source_corpus_id: str
    source_mode: str
    path_type: DiscoveryPathType = "UNKNOWN"
    paper_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    relation_sequence: list[str] = Field(default_factory=list)
    rendered_path: str
    exploration_score: float
    score_breakdown: DiscoveryScoreBreakdown
    reason_codes: list[str] = Field(default_factory=list)
    requires_verification: bool = False
    eligible_as_positive_premise: Literal[False] = False

    # Alpha2 auditable path-quality diagnostics.
    mechanism_before_alignment: bool = False
    mechanism_after_alignment: bool = False
    mechanistic_continuity_band: MechanisticContinuityBand = "not_applicable"
    generic_entity_fraction: float = 0.0
    max_generic_run_length: int = 0
    registry_hop_fraction: float = 0.0
    semantic_similarity_to_grounding: float = 0.0
    max_semantic_similarity_to_selected: float = 0.0
    semantic_diversity_mode: SemanticDiversityMode = "disabled"

    # Alpha3 candidate-unit lineage. Empty for ordinary mechanism paths.
    candidate_unit_id: str = ""
    candidate_unit_label: str = ""
    candidate_entry_anchor_id: str = ""
    candidate_entry_anchor_label: str = ""
    candidate_exit_anchor_id: str = ""
    candidate_exit_anchor_label: str = ""
    candidate_proposed_subject: str = ""
    candidate_proposed_relation: str = ""
    candidate_proposed_object: str = ""
    candidate_unit_score: float = 0.0
    reaction_domain_switch_penalty: float = 0.0


class DiscoveryBundle(StrictModel):
    schema_version: Literal["discovery-bundle-v1"] = "discovery-bundle-v1"
    bundle_id: str
    bundle_sha256: str
    corpus_id: str
    query_signature: str
    inspirations: list[DiscoveryInspiration] = Field(default_factory=list)
    source_traversal_files: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    selected_count: int = 0
    used_candidate_pool: bool = False
    warnings: list[str] = Field(default_factory=list)

    # How semantic deduplication was performed for this bundle.
    semantic_diversity_mode: SemanticDiversityMode = "disabled"
    semantic_model_name: str | None = None
    semantic_similarity_threshold: float = 0.88

    # Accept old a1 bundles while writing alpha2 bundles as v2.
    policy_version: Literal[
        "discovery-policy-v1",
        "discovery-policy-v2",
        "discovery-policy-v3",
    ] = "discovery-policy-v3"
