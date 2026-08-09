from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


HypothesisType = Literal[
    "mechanistic_extension",
    "cross_evidence_synthesis",
    "design_lever_interaction",
    "descriptor_mediation",
    "context_dependency",
]

ExpectedDirection = Literal[
    "increase",
    "decrease",
    "shift",
    "non_monotonic",
    "qualitative_change",
    "unspecified",
]

CandidateDependency = Literal["none", "supporting", "essential"]


class HypothesisPolicy(StrictModel):
    generated_hypotheses_allowed: Literal[True] = True
    external_novelty_claims_allowed: Literal[False] = False
    experiment_protocols_allowed: Literal[False] = False
    unsupported_numeric_predictions_allowed: Literal[False] = False
    alignment_can_be_scientific_premise: Literal[False] = False
    unresolved_can_be_positive_premise: Literal[False] = False
    navigation_note_can_be_positive_premise: Literal[False] = False
    candidate_evidence_must_propagate: Literal[True] = True
    falsifiable_prediction_required: Literal[True] = True
    falsification_condition_required: Literal[True] = True
    source_report_must_validate: Literal[True] = True


class HypothesisEvidenceStatement(StrictModel):
    statement_id: str
    text: str
    epistemic_role: Literal[
        "reported",
        "evidence_synthesis",
        "navigation_note",
        "unresolved",
    ]
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    scientific_support_node_ids: list[str] = Field(default_factory=list)
    scientific_support_edge_ids: list[str] = Field(default_factory=list)
    support_path_ids: list[str] = Field(default_factory=list)
    alignment_path_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False
    eligible_as_premise: bool = False
    eligible_as_gap: bool = False
    premise_restrictions: list[str] = Field(default_factory=list)


class HypothesisRouteContext(StrictModel):
    route_id: str
    statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    structural_type: str
    uses_alignment: bool = False
    uses_reverse_navigation: bool = False
    navigation_heavy: bool = False
    requires_verification: bool = False


class HypothesisMotifContext(StrictModel):
    motif_id: str
    label: str
    statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    cross_paper: bool = False


class HypothesisDesignLeverContext(StrictModel):
    lever_id: str
    label: str
    statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)


class HypothesisGapContext(StrictModel):
    gap_id: str
    statement_id: str
    reason: str
    related_path_ids: list[str] = Field(default_factory=list)


class HypothesisContext(StrictModel):
    schema_version: Literal["hypothesis-context-v1"] = "hypothesis-context-v1"
    context_id: str
    context_sha256: str
    source_packet_id: str
    source_packet_sha256: str
    source_report_id: str
    source_report_sha256: str
    task_id: str
    question: str
    corpus_id: str
    evidence_statements: list[HypothesisEvidenceStatement]
    mechanism_routes: list[HypothesisRouteContext] = Field(default_factory=list)
    mechanistic_motifs: list[HypothesisMotifContext] = Field(default_factory=list)
    reported_design_levers: list[HypothesisDesignLeverContext] = Field(default_factory=list)
    research_gaps: list[HypothesisGapContext] = Field(default_factory=list)
    partial_absence_blocked_paper_ids: list[str] = Field(default_factory=list)
    policy: HypothesisPolicy = Field(default_factory=HypothesisPolicy)


class PredictedObservationDraft(StrictModel):
    local_id: str
    observable: str = Field(min_length=1)
    expected_direction: ExpectedDirection
    rationale: str = Field(min_length=1)


class FalsificationCriterionDraft(StrictModel):
    local_id: str
    observable: str = Field(min_length=1)
    falsifying_outcome: str = Field(min_length=1)


class HypothesisProposalDraft(StrictModel):
    local_id: str
    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    hypothesis_type: HypothesisType
    premise_statement_ids: list[str] = Field(min_length=1)
    gap_statement_ids: list[str] = Field(default_factory=list)
    inferential_bridge: str = Field(min_length=1)
    predicted_observations: list[PredictedObservationDraft] = Field(min_length=1)
    falsification_criteria: list[FalsificationCriterionDraft] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_local_ids(self) -> "HypothesisProposalDraft":
        prediction_ids = [x.local_id for x in self.predicted_observations]
        falsifier_ids = [x.local_id for x in self.falsification_criteria]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("duplicate predicted_observation local_id")
        if len(falsifier_ids) != len(set(falsifier_ids)):
            raise ValueError("duplicate falsification_criterion local_id")
        return self


class HypothesisPortfolioDraft(StrictModel):
    hypotheses: list[HypothesisProposalDraft] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _abstention_consistency(self) -> "HypothesisPortfolioDraft":
        if not self.hypotheses and not (self.abstention_reason or "").strip():
            raise ValueError("abstention_reason is required when no hypotheses are proposed")
        if self.hypotheses and self.abstention_reason is not None:
            raise ValueError("abstention_reason must be null when hypotheses are proposed")
        local_ids = [x.local_id for x in self.hypotheses]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("duplicate hypothesis local_id")
        return self


class PredictedObservation(StrictModel):
    observation_id: str
    observable: str
    expected_direction: ExpectedDirection
    rationale: str


class FalsificationCriterion(StrictModel):
    criterion_id: str
    observable: str
    falsifying_outcome: str


class HypothesisEvidenceProfile(StrictModel):
    premise_count: int
    gap_count: int
    source_paper_count: int
    candidate_premise_count: int
    reported_premise_count: int
    synthesis_premise_count: int


class HypothesisCard(StrictModel):
    schema_version: Literal["hypothesis-card-v1"] = "hypothesis-card-v1"
    hypothesis_id: str
    source_context_id: str
    source_context_sha256: str
    source_report_id: str
    source_report_sha256: str
    title: str
    hypothesis_statement: str
    hypothesis_type: HypothesisType
    premise_statement_ids: list[str]
    gap_statement_ids: list[str] = Field(default_factory=list)
    inferential_bridge: str
    predicted_observations: list[PredictedObservation]
    falsification_criteria: list[FalsificationCriterion]
    assumptions: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    gap_paper_ids: list[str] = Field(default_factory=list)
    cross_paper_synthesis: bool = False
    candidate_dependency: CandidateDependency = "none"
    evidence_profile: HypothesisEvidenceProfile
    status: Literal["hypothesized"] = "hypothesized"
    novelty_status: Literal["not_assessed"] = "not_assessed"


class HypothesisPortfolio(StrictModel):
    schema_version: Literal["hypothesis-portfolio-v1"] = "hypothesis-portfolio-v1"
    portfolio_id: str
    source_context_id: str
    source_context_sha256: str
    source_report_id: str
    source_report_sha256: str
    hypotheses: list[HypothesisCard] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _abstention_consistency(self) -> "HypothesisPortfolio":
        if not self.hypotheses and not (self.abstention_reason or "").strip():
            raise ValueError("abstention_reason is required when portfolio is empty")
        if self.hypotheses and self.abstention_reason is not None:
            raise ValueError("abstention_reason must be null when hypotheses exist")
        return self
