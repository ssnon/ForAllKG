from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldCaseAudit


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ValidityDimension = Literal[
    "VARIABLE_IDENTITY",
    "MEASUREMENT_SCALE",
    "DIRECTIONAL_SUPPORT",
    "CAUSAL_MEDIATION",
    "COMPARISON_CONTEXT_COMPATIBILITY",
    "VARIANCE_SEMANTICS",
    "CONTROL_VARIABLE_ADEQUACY",
]
ValidityVerdict = Literal["PASS", "WARN", "FAIL", "UNKNOWN"]

VALIDITY_DIMENSIONS: tuple[ValidityDimension, ...] = (
    "VARIABLE_IDENTITY",
    "MEASUREMENT_SCALE",
    "DIRECTIONAL_SUPPORT",
    "CAUSAL_MEDIATION",
    "COMPARISON_CONTEXT_COMPATIBILITY",
    "VARIANCE_SEMANTICS",
    "CONTROL_VARIABLE_ADEQUACY",
)


class RelationValidityDimensionDraft(StrictModel):
    dimension: ValidityDimension
    verdict: ValidityVerdict
    rationale: str = Field(min_length=1)


class RelationValidityCandidateDraft(StrictModel):
    candidate_alias: str = Field(min_length=1)
    dimensions: list[RelationValidityDimensionDraft]
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def complete_dimensions(self) -> "RelationValidityCandidateDraft":
        observed = [x.dimension for x in self.dimensions]
        if len(observed) != len(set(observed)):
            raise ValueError("duplicate relation-validity dimension")
        if set(observed) != set(VALIDITY_DIMENSIONS):
            raise ValueError("incomplete relation-validity dimensions")
        return self


class RelationValidityBatchDraft(StrictModel):
    candidates: list[RelationValidityCandidateDraft]


class RelationValidityAudit(StrictModel):
    schema_version: Literal["adaptive-relation-validity-audit-v1"] = (
        "adaptive-relation-validity-audit-v1"
    )
    case_id: str
    model: str
    candidates: list[RelationValidityCandidateDraft]
    verdict_counts: dict[str, int]
    evaluation_only: Literal[True] = True
    rejection_authority: Literal[False] = False
    generation_feedback_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False


SYSTEM_PROMPT = """You are performing an evaluation-only scientific relation validity audit.
Do not assess novelty, rank candidates, rewrite hypotheses, or infer production value.
Judge whether the scientific relation as stated is coherent at the variable,
measurement-scale, directionality, causal, comparison, and variance levels.

Dimensions:
- VARIABLE_IDENTITY: are manipulated/explanatory and measured variables kept distinct and consistently defined?
- MEASUREMENT_SCALE: are spot-to-spot, particle-to-particle, sample-to-sample, batch-to-batch, temporal, and population-level quantities kept distinct?
- DIRECTIONAL_SUPPORT: if increase/decrease/non-monotonic/optimum language is used, is that direction supported by the supplied premises rather than merely plausible?
- CAUSAL_MEDIATION: does the proposed mediator/path connect variables without silently skipping required causal steps?
- COMPARISON_CONTEXT_COMPATIBILITY: are cross-system comparisons valid under compatible analyte, excitation, composition, sampling, and operating conditions, or are missing matched controls acknowledged?
- VARIANCE_SEMANTICS: are variance/CV/RSD/reproducibility concepts assigned to the correct source and level of variation?
- CONTROL_VARIABLE_ADEQUACY: are controls needed to identify the claimed relation specified or explicitly left as unresolved requirements?

Use PASS, WARN, FAIL, or UNKNOWN. A plausible hypothesis may FAIL one dimension.
Use UNKNOWN when supplied premises cannot resolve the issue. Ground judgments only
in the supplied candidate and premise text."""


def build_prompt(audit: AdaptiveYieldCaseAudit) -> str:
    unique: dict[str, dict] = {}
    for row in audit.candidates:
        if row.scientific_fingerprint in unique:
            continue
        unique[row.scientific_fingerprint] = {
            "candidate_alias": f"V{len(unique)+1:02d}",
            "title": row.title,
            "hypothesis_statement": row.hypothesis_statement,
            "inferential_bridge": row.inferential_bridge,
            "premises": [
                {"statement_id": sid, "text": text}
                for sid, text in zip(row.premise_statement_ids, row.premise_texts)
            ],
            "predictions": [x.model_dump(mode="json") for x in row.predictions],
            "falsifiers": [x.model_dump(mode="json") for x in row.falsifiers],
            "assumptions": row.assumptions,
        }
    payload = {"question": audit.question, "candidates": list(unique.values())}
    return (
        "SCIENTIFIC RELATION VALIDITY INPUT\n"
        "==================================\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nReturn one result per candidate_alias and all seven dimensions exactly once."
    )
