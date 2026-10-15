from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Mechanism(Strict):
    title: str = Field(min_length=8)
    distinct_causal_claim: str = Field(min_length=30)
    latent_state_variables: list[str] = Field(min_length=1, max_length=7)
    coupling_law: str = Field(min_length=15)
    quantitative_derivation: str = Field(min_length=30)
    physical_feasibility_constraints: list[str] = Field(min_length=1)
    novelty_risk: str = Field(min_length=20)
    strongest_existing_null: str = Field(min_length=20)
    discriminating_prediction: str = Field(min_length=25)
    independently_manipulated_variable: str = Field(min_length=10)
    independently_measured_outcome: str = Field(min_length=10)
    confounders: list[str] = Field(min_length=1)
    falsifier: str = Field(min_length=25)
    source_ids: list[str] = Field(default_factory=list)
    measurement_matrix: list[list[float]] = Field(default_factory=list)
    measurement_calibration: str = "NOT_CALIBRATED"
    source_support_scope: Literal["HYPOTHESIS_ONLY"] = "HYPOTHESIS_ONLY"

    @model_validator(mode="after")
    def check_matrix(self):
        if len(self.measurement_matrix) > 10 or any(len(row) != 2 for row in self.measurement_matrix):
            raise ValueError("measurement_matrix must be up to ten [EM_sensitivity, CHEM_sensitivity] pairs")
        return self

class Proposal(Strict):
    research_action: Literal["GENERATE_ALTERNATIVES", "REVISE_MODEL", "ABSTAIN"]
    scientific_reason: str = Field(min_length=15)
    mechanisms: list[Mechanism] = Field(default_factory=list, max_length=2)

class ChallengeRow(Strict):
    title: str
    physical_status: Literal["PLAUSIBLE_CONDITIONAL", "NEEDS_BOUNDS", "CONTRADICTORY", "UNKNOWN"]
    missing_or_invalid_derivation: str = Field(min_length=15)
    strongest_null_mimic: str = Field(min_length=20)
    truly_discriminating_after_null: str = Field(min_length=10)
    source_scope_limitations: str = Field(min_length=10)
    proposed_test_or_computation: str = Field(min_length=15)
    next_action_advice: Literal["REFRAME_ROOT", "REVISE_CAUSAL_MODEL", "GET_EVIDENCE", "ABSTAIN"]

class Challenge(Strict):
    rows: list[ChallengeRow]
    recurrent_failure_pattern: str = Field(min_length=15)

class Decision(Strict):
    action: Literal["REFRAME_ROOT", "REVISE_CAUSAL_MODEL", "GET_EVIDENCE", "STOP"]
    rationale: str = Field(min_length=20)
    target_title: str | None = None
    updated_open_question: str = Field(min_length=15)
    abandoned_assumption: str = Field(min_length=15)
    scientific_uncertainty: str = Field(min_length=15)

class Baseline(Strict):
    research_action: Literal["GENERATE_ALTERNATIVES", "ABSTAIN"]
    scientific_reason: str = Field(min_length=15)
    mechanisms: list[Mechanism] = Field(default_factory=list, max_length=2)
