"""Small, untrusted-science Research Work protocol. No Level-II certification."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ResearchWorkRequest(Strict):
    research_question: str = Field(min_length=20)
    target_idea_id: str | None = None
    gap_or_assumption: str = Field(min_length=12)
    work_kind: Literal['REASON', 'READ_FROZEN_SOURCE']
    work_instructions: str = Field(min_length=16)
    available_inputs: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    expected_information: str = Field(min_length=15)
    stop_or_failure_condition: str = Field(min_length=15)

    @model_validator(mode='after')
    def check_sources(self):
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError('duplicate source IDs in work request')
        if len(set(self.available_inputs)) != len(self.available_inputs):
            raise ValueError('duplicate available input IDs')
        if self.work_kind == 'READ_FROZEN_SOURCE' and not self.source_ids:
            raise ValueError('READ_FROZEN_SOURCE requires source IDs')
        # Some model decisions put already-read sources in source_ids when
        # requesting REASON. Accept the original model-authored request *only*
        # after the stateful runner confirms EVERY such reference has been read.
        # No automatic reading, source promotion or model-output rewriting.
        return self


class WorkInterpretation(Strict):
    work_id: str = Field(min_length=6)
    product_sha256: str = Field(min_length=64, max_length=64)
    assessment: Literal['PROPOSED_CONSTRAINT', 'PROPOSED_COUNTEREXAMPLE', 'EVIDENCE_ACQUIRED',
                        'NO_PROGRESS', 'INSUFFICIENT_EVIDENCE']
    what_changed: str = Field(min_length=12)
    supported_by_artifact_refs: list[str] = Field(min_length=1)
    remaining_gap: str = Field(min_length=12)
    followup_reason: str = Field(min_length=12)
    external_scientific_validation: Literal[False] = False


class ProvisionalCandidateSnapshot(Strict):
    """Identifiable, unverified hypothesis snapshot, NOT a native ResearchIdea or Level II."""
    title: str = Field(min_length=8)
    causal_mechanism: str = Field(min_length=32)
    distinctive_prediction: str = Field(min_length=24)
    strong_conventional_null: str = Field(min_length=24)
    unresolved_physical_assumptions: list[str] = Field(min_length=1)
    supporting_work_ids: list[str] = Field(min_length=1)
    epistemic_status: Literal['UNVERIFIED_HYPOTHESIS'] = 'UNVERIFIED_HYPOTHESIS'


class ResearchDecision(Strict):
    action: Literal['READ_EVIDENCE', 'EXPLORE', 'CHALLENGE', 'PIVOT', 'DEEPEN', 'END']
    scientific_reason: str = Field(min_length=20)
    research_question_now: str = Field(min_length=20)
    work_request: ResearchWorkRequest | None = None
    interpretation: WorkInterpretation | None = None
    final_disposition: Literal['ABSTAIN', 'PROVISIONAL_RESEARCH_CANDIDATE'] | None = None
    final_justification: str | None = None
    provisional_candidate: ProvisionalCandidateSnapshot | None = None

    @model_validator(mode='after')
    def check_phase(self):
        if self.action == 'END':
            if self.work_request is not None or not self.final_disposition or not self.final_justification:
                raise ValueError('END requires justification/disposition and cannot request work')
            if (self.final_disposition == 'PROVISIONAL_RESEARCH_CANDIDATE') != (self.provisional_candidate is not None):
                raise ValueError('provisional END requires a full identifiable candidate snapshot; ABSTAIN cannot nominate')
        else:
            if self.work_request is None or self.final_disposition or self.final_justification or self.provisional_candidate:
                raise ValueError('non-END decision requires one WorkRequest without final disposition')
            if (self.action == 'READ_EVIDENCE') != (self.work_request.work_kind == 'READ_FROZEN_SOURCE'):
                raise ValueError('READ_EVIDENCE pairs with READ_FROZEN_SOURCE only')
        return self


class ReasonWorkContent(Strict):
    attempted_question: str = Field(min_length=12)
    premises_used: list[str] = Field(default_factory=list)
    reasoning_or_operations: list[str] = Field(default_factory=list)
    candidate_consequence: str | None = None
    counterexample_or_obstruction: str | None = None
    limits_and_uncertainties: list[str] = Field(default_factory=list)
    work_conclusion: Literal['DERIVATION_PROPOSED', 'COUNTEREXAMPLE_PROPOSED', 'INCONCLUSIVE']

    @model_validator(mode='after')
    def check_proposal(self):
        if self.work_conclusion == 'DERIVATION_PROPOSED' and not (self.candidate_consequence and self.reasoning_or_operations):
            raise ValueError('proposed derivation needs a candidate consequence and reasoning trace')
        if self.work_conclusion == 'COUNTEREXAMPLE_PROPOSED' and not (self.counterexample_or_obstruction and self.reasoning_or_operations):
            raise ValueError('proposed counterexample needs explicit obstruction and reasoning trace')
        return self


class ResearchWorkProduct(Strict):
    work_id: str
    work_kind: Literal['REASON', 'READ_FROZEN_SOURCE']
    execution_status: Literal['COMPLETED', 'FAILED', 'BLOCKED']
    inputs_sha256: str
    output_artifact_sha256: str
    origin: Literal['LLM_REASONING_UNVERIFIED', 'FROZEN_SOURCE_EXCERPT', 'EXECUTION_STATUS_RECEIPT']
    content: dict
    raw_response_sha256: str | None = None
    failure_reason: str | None = None
    scientific_validation: Literal[False] = False
