"""FAKG new-episode research focus: mechanistic explanation before generic identification.

Policy applies only to new episodes. It does not certify, validate or inject answers.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_ID = 'observation-mechanism-continuation-v1.7.0'
POLICY_SYSTEM_APPEND = """
NEW SCIENTIFIC STUDY (W07): OBSERVATION -> CAUSAL MECHANISM -> TESTABLE CONSEQUENCE.
The goal is a new physical understanding of Ye's reported SERS intermittency, not another prolonged campaign of Raman-only identifiability tests. Use the actual source pages to identify what needs explaining. Prior W06 hypotheses are unverified historical ideas, NOT facts, not CURRENT registered idea IDs, and not required next targets. Independently choose the question, the causal family, and whether continuing a prior hypothesis or opening a different account is scientifically valuable.

On each decision, assess the most important gap in the causal story: excitation -> a physically plausible and possibly stochastic transition or interaction -> mode-dependent response of molecular sites -> observed intermittent signal and source controls. A general state-variable label or a first-order mean relaxation equation alone does not generate irregular bursts. When feasible, choose a Research Work that could derive a new mechanism, physical feasibility bound, necessary/sufficient condition, or counterexample; work may be pure reasoning. Do not invent numerical experimental values, use physical assumptions not established by the source as claims, or assert physical originality without research.

Strong conventional nulls matter, but they must specify actual states, constraints, and joint predicted phenomena. A null's ability to mimic a single statistic with an arbitrary latent variable is not, by itself, decisive scientific refutation of a distinct physical causal pathway. Conversely, conditional predictions are not independently identifying without realistic competing models. Do not force a particular instrument, calculation, experiment, PVA interpretation, reservoir model, or answer.

Preserve the distinction between WORK COMPLETED, MODEL-INFERRED scientific progress, and externally validated new knowledge. A speculative ResearchIdea or exploratory Seed is welcome when actually warranted, but NEVER mandatory. If a Work materially changes a registered mechanism, you MAY use native REVISE or targeted Seed DEVELOP; if not, leave it unchanged. To target an existing current Seed, use its exact existing ID and its origin product ref in the new WorkRequest, not its former W06 ID. A prior episode ID is historical context only. Preserve provenance and use NULL for missing optional fields. END with ABSTAIN when warranted; no Level-II certification from model prose.

You MAY include research_focus_update (typed object or null) to record the current phenomenon, proposed causal chain and single key unknown. This is unverified working state, not scientific evidence, not a required form or discovery quota. Source_refs in a focus_update must be actually read source inventory IDs. If uncertain, use null and let the WorkRequest carry the question.
"""

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class ResearchFocus(Strict):
    question: str = Field(min_length=20)
    observation_refs: list[str] = Field(default_factory=list)
    causal_path: str = Field(min_length=25)
    blocking_unknown: str = Field(min_length=20)
    target_hypothesis_id: str | None = None
    reason_for_priority: str = Field(min_length=20)
    @model_validator(mode='after')
    def check_refs(self):
        if len(self.observation_refs)!=len(set(self.observation_refs)):
            raise ValueError('duplicate observation refs')
        return self


def focus_extension_schema(base_schema: dict) -> dict:
    import copy
    schema=copy.deepcopy(base_schema)
    sub=ResearchFocus.model_json_schema()
    schema['properties']['research_focus_update']={
        'anyOf':[{'type':'null'}, {k:v for k,v in sub.items() if k!='$defs'}]
    }
    if sub.get('$defs'):
        schema.setdefault('$defs',{}).update(sub['$defs'])
    # Optional field avoids adding another mandatory format-failure mode.
    return schema
