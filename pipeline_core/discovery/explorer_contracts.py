from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TraversalMode = Literal["evidence", "mechanism", "exploratory"]
ExplorerObjective = Literal[
    "map_evidence",
    "explain_connection",
    "compare_mechanisms",
    "identify_reported_design_levers",
]


class ExplorerTask(StrictModel):
    task_id: str
    question: str
    source_query: str | None = None
    target_query: str | None = None
    waypoint_query: str | None = None
    traversal_mode: TraversalMode
    objective: ExplorerObjective = "map_evidence"


class PaperScope(StrictModel):
    paper_id: str
    quality_status: Literal[
        "complete",
        "partial_acceptable",
        "partial_critical",
        "rejected",
        "unknown",
    ] = "unknown"
    source_token_coverage: float | None = None
    quarantine_token_fraction: float | None = None
    absence_claims_allowed: bool = False


class CorpusScope(StrictModel):
    corpus_id: str
    projection_mode: TraversalMode
    papers: list[PaperScope]
    substrate_version: str


class RetrievalSummary(StrictModel):
    algorithm: str
    effective_max_depth: int | None = None
    direct_concept_hit_count: int = 0
    returned_path_count: int = 0
    returned_path_type_counts: dict[str, int] = Field(default_factory=dict)
    endpoint_selector_enabled: bool | None = None
    waypoint_selector_enabled: bool | None = None
    candidate_path_count: int | None = None


class ExplorerPolicy(StrictModel):
    novel_hypotheses_allowed: Literal[False] = False
    scientific_direction_must_be_preserved: Literal[True] = True
    retrieval_similarity_is_scientific_confidence: Literal[False] = False
    absence_claims_packet_scoped: Literal[True] = True
    candidate_evidence_requires_verification: Literal[True] = True
    allowed_epistemic_roles: tuple[
        Literal[
            "reported",
            "evidence_synthesis",
            "navigation_note",
            "unresolved",
        ],
        ...,
    ] = (
        "reported",
        "evidence_synthesis",
        "navigation_note",
        "unresolved",
    )


class NodeEvidence(StrictModel):
    node_id: str
    node_type: str
    label: str
    node_text: str
    graph_layer: str = ""
    evidence_status: str = ""
    requires_verification: bool = False
    source_paper_id: str | None = None
    source_paper_ids: list[str] = Field(default_factory=list)
    extraction_quality_status: str | None = None
    absence_claims_allowed: bool = False


class EdgeEvidence(StrictModel):
    edge_id: str
    scientific_source: str
    relation: str
    scientific_target: str
    graph_layer: str = ""
    evidence_status: str = ""
    requires_verification: bool = False
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_pointers: list[dict[str, Any] | str] = Field(default_factory=list)
    supporting_node_ids: list[str] = Field(default_factory=list)
    derivation_rule: str | None = None
    evidence_pointer_source: Literal[
        "edge_sidecar",
        "traversal_selected_alternative",
        "edge_sidecar+traversal_selected_alternative",
        "derived_alignment",
        "missing",
    ] = "missing"
    provenance_status: Literal[
        "grounded_pointer",
        "derived_alignment",
        "missing_pointer",
    ] = "missing_pointer"


class EvidenceCatalog(StrictModel):
    nodes: dict[str, NodeEvidence]
    edges: dict[str, EdgeEvidence]


class AlignmentContext(StrictModel):
    context_id: str
    path_id: str
    hub_node_id: str
    hub_label: str | None = None
    hub_type: str | None = None
    alignment_edge_ids: list[str] = Field(default_factory=list)
    member_node_ids: list[str] = Field(default_factory=list)
    member_paper_ids: list[str] = Field(default_factory=list)
    traversed_entry_node_ids: list[str] = Field(default_factory=list)
    traversed_exit_node_ids: list[str] = Field(default_factory=list)


class ProvenanceSummary(StrictModel):
    strict_provenance: bool
    edge_count: int
    pointer_grounded_edge_count: int
    pointer_recovered_from_traversal_count: int
    derived_alignment_edge_count: int
    missing_pointer_edge_count: int
    materialized_node_count: int
    suppressed_alignment_member_node_count: int


class ExplorerDirectHit(StrictModel):
    hit_id: str
    node_id: str
    node_evidence_ref: str
    hit_tier: int
    quality_basis: str
    source_similarity: float | None = None
    target_similarity: float | None = None
    mechanism_bearing: bool = False
    requires_verification: bool = False


class EndpointView(StrictModel):
    source_node_id: str
    target_node_id: str
    source_label: str | None = None
    target_label: str | None = None
    source_similarity: float | None = None
    target_similarity: float | None = None
    semantic_tier: int | None = None
    pair_score: float | None = None
    source_exact: bool | None = None
    target_exact: bool | None = None


class WaypointView(StrictModel):
    node_id: str
    label: str | None = None
    semantic_tier: int | None = None
    semantic_similarity: float | None = None
    waypoint_rank: int | None = None


class ExplorerStep(StrictModel):
    navigation_source: str
    navigation_target: str
    traversal_direction: str
    scientific_source: str
    relation: str
    scientific_target: str
    selected_original_edge_id: str
    edge_evidence_ref: str
    edge_class: str = ""
    requires_verification: bool = False


class ExplorerPathQuality(StrictModel):
    path_type: str = "UNKNOWN"
    path_structure_type: str | None = None
    path_tags: list[str] = Field(default_factory=list)
    endpoint_semantic_tier: int | None = None
    endpoint_pair_score: float | None = None
    mechanism_edge_count: int = 0
    mechanism_node_count: int = 0
    mechanism_node_ids: list[str] = Field(default_factory=list)
    mechanistic_content: str | None = None
    mechanistic_content_basis: str | None = None
    mechanism_bearing: bool = False
    navigation_edge_fraction: float = 0.0
    reverse_fraction: float = 0.0
    candidate_fraction: float = 0.0
    endpoint_relevance: str | None = None
    navigation_burden: str | None = None
    reverse_burden: str | None = None
    visited_paper_count: int = 0
    shared_entity_bridge: bool = False


class ExplorerPath(StrictModel):
    path_id: str
    bundle_rank: int
    endpoint: EndpointView
    waypoint: WaypointView | None = None
    node_ids: list[str]
    steps: list[ExplorerStep]
    visited_paper_ids: list[str] = Field(default_factory=list)
    supporting_paper_ids: list[str] = Field(default_factory=list)
    hub_scope_paper_ids: list[str] = Field(default_factory=list)
    quality: ExplorerPathQuality


class GraphExplorerPacket(StrictModel):
    schema_version: Literal["graph-explorer-input-v1"] = "graph-explorer-input-v1"
    domain_profile_id: str
    packet_id: str
    packet_sha256: str
    task: ExplorerTask
    corpus: CorpusScope
    retrieval_summary: RetrievalSummary
    direct_concept_hits: list[ExplorerDirectHit]
    paths: list[ExplorerPath]
    evidence_catalog: EvidenceCatalog
    alignment_contexts: list[AlignmentContext] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary
    policy: ExplorerPolicy = Field(default_factory=ExplorerPolicy)


EpistemicRole = Literal[
    "reported",
    "evidence_synthesis",
    "navigation_note",
    "unresolved",
]
ClaimKind = Literal[
    "observation",
    "mechanism",
    "association",
    "comparison",
    "scope_limit",
    "retrieval_note",
]


class ExplorerStatement(StrictModel):
    statement_id: str
    text: str
    epistemic_role: EpistemicRole
    claim_kind: ClaimKind
    support_node_ids: list[str] = Field(default_factory=list)
    support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    support_direct_hit_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False
    scope: Literal["provided_evidence_packet"] = "provided_evidence_packet"


class MechanismRoute(StrictModel):
    route_id: str
    path_ids: list[str]
    statement_ids: list[str]
    mechanism_node_ids: list[str] = Field(default_factory=list)
    mechanism_edge_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    structural_type: Literal[
        "DIRECT_MECHANISTIC",
        "CROSS_PAPER_MECHANISTIC",
        "CROSS_PAPER_BRIDGE",
        "SHARED_ENTITY_BRIDGE",
        "SCAFFOLD_NAVIGATION",
        "CANDIDATE_EXPLORATION",
    ]
    navigation_heavy: bool = False
    uses_alignment: bool = False
    uses_reverse_navigation: bool = False
    requires_verification: bool = False


class MechanisticMotif(StrictModel):
    motif_id: str
    label: str
    statement_ids: list[str]
    mechanism_node_ids: list[str] = Field(default_factory=list)
    mechanism_edge_ids: list[str] = Field(default_factory=list)
    path_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    cross_paper: bool = False


class CrossPaperConnection(StrictModel):
    connection_id: str
    statement_ids: list[str]
    path_ids: list[str]
    paper_ids: list[str]
    uses_alignment: bool
    alignment_edge_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False


class EvidenceTension(StrictModel):
    tension_id: str
    statement_id: str
    side_a_statement_ids: list[str]
    side_b_statement_ids: list[str]
    tension_type: Literal[
        "context_dependency",
        "qualitative_difference",
        "quantitative_difference",
        "potential_conflict",
        "insufficient_context",
    ]
    paper_ids: list[str] = Field(default_factory=list)


class UnresolvedConnection(StrictModel):
    gap_id: str
    statement_id: str
    related_path_ids: list[str] = Field(default_factory=list)
    reason: Literal[
        "alignment_only",
        "navigation_heavy",
        "candidate_only",
        "missing_direct_relation_in_packet",
        "insufficient_provenance",
        "partial_source_scope",
        "insufficient_context",
    ]
    scope: Literal["provided_evidence_packet"] = "provided_evidence_packet"


class ReportedDesignLever(StrictModel):
    lever_id: str
    label: str
    statement_ids: list[str]
    mechanism_node_ids: list[str] = Field(default_factory=list)
    outcome_node_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)


class ExplorationReport(StrictModel):
    schema_version: Literal["exploration-report-v1"] = "exploration-report-v1"
    report_id: str
    task_id: str
    source_packet_sha256: str
    statements: list[ExplorerStatement]
    direct_findings: list[str] = Field(default_factory=list)
    mechanism_routes: list[MechanismRoute] = Field(default_factory=list)
    recurring_mechanistic_motifs: list[MechanisticMotif] = Field(default_factory=list)
    cross_paper_connections: list[CrossPaperConnection] = Field(default_factory=list)
    evidence_tensions: list[EvidenceTension] = Field(default_factory=list)
    unresolved_connections: list[UnresolvedConnection] = Field(default_factory=list)
    reported_design_levers: list[ReportedDesignLever] = Field(default_factory=list)
