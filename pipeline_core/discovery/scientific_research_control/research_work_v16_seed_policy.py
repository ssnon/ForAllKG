"""W06: Mechanism construction and episode-local exploratory hypothesis seeds.

All scientific content is unverified. A seed is NOT a native ResearchIdea.
No W04/W05 hypothesis text is injected into new model calls.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_ID = 'mechanism-seed-development-v1.6.0'
POLICY_SYSTEM_APPEND = '''\nW06 MECHANISM CONSTRUCTION / EXPLORATORY SEED DEVELOPMENT:
Scientific objective: SEARCH AGGRESSIVELY, CLAIM CONSERVATIVELY. Build physically specified new explanations; do not turn every missing experiment into another generic identifiability exercise. After sufficient source grounding, consider an exploratory REASON Work on 1-3 alternative *causal constructions*: state variables, physical driving/coupling or transition, minimal dynamics, qualitative observables, obstacles, and strongest physically compatible null. A purely methodological test design, a relabeled latent variable, and a label such as "thermal effect" with no actual dynamics are NOT physical hypotheses. Never manufacture observations, equations, known novelty, or mechanisms to fill quotas.
Two independent records are possible: (A) a native ResearchIdea with strong conditional discriminator and existing schema, and (B) an exploratory SEED with a concrete physical process but incomplete discriminator. Seeds are UNVERIFIED research leads, not proven physics and not native ResearchIdeas. A missing discriminator alone is NOT a fatal objection; an established contradiction may be. A failed/null-identification REASON may just yield seed_update null. Do not invent a seed. The decision JSON response MUST include seed_update (null or one typed operation) and seed_work_target_id (null or an existing seed ID). This requirement forces accurate reporting, NOT seed creation.
After completed and interpreted REASON Work, optionally FORM a new seed that explains a phenomenon through explicitly named physical states and a driving/coupling pathway, plus objections and a next development question. Alternatively update a previously targeted seed with DEVELOP/CHALLENGE, DISCARD it if demonstrably untenable, or PROMOTE to a native ResearchIdea only alongside idea_update=PROPOSE with the original native high standard. Seed action must cite the exact previous REASON Work identity/digest. Nothing is automatically certified.
To research an existing seed, supply seed_work_target_id with an actual REASON work_request, and cite exact already-completed work artifact refs in available_inputs. The seed ID itself is NOT a source reference and is NOT a native target_idea_id: keep work_request.target_idea_id null. At the next Decision interpret that targeted Work; only then DEVELOP, CHALLENGE, DISCARD or PROMOTE that seed. Do not claim Seed Development merely because the question changed. A new seed ID is minted only upon Decision commit; do not target it in the same Decision. Keep changes substantive, clearly note null alternatives. A final END may ABSTAIN with unresolved seeds; that is scientifically legitimate. Never transfer seeds from previous experiments.''' 

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class SeedDraft(Strict):
    title: str = Field(min_length=10)
    causal_family: str = Field(min_length=12)
    physical_state_variables: list[str] = Field(min_length=1)
    physical_transition_or_coupling: str = Field(min_length=30)
    causal_mechanism: str = Field(min_length=40)
    phenomenon_explained: str = Field(min_length=25)
    conditional_observable: str = Field(min_length=20)
    strongest_competing_null: str = Field(min_length=25)
    known_objections: list[str] = Field(min_length=1)
    unresolved_physical_assumptions: list[str] = Field(min_length=1)
    next_development_question: str = Field(min_length=25)

class SeedMutation(Strict):
    kind: Literal['FORM','DEVELOP','CHALLENGE','DISCARD','PROMOTE']
    target_seed_id: str | None = None
    draft: SeedDraft | None = None
    supporting_work_id: str = Field(min_length=6)
    supporting_product_sha256: str = Field(min_length=64,max_length=64)
    rationale: str = Field(min_length=25)
    challenge_or_change: str | None = None

    @model_validator(mode='after')
    def form(self):
        if self.kind=='FORM' and (self.target_seed_id is not None or self.draft is None):
            raise ValueError('FORM requires new draft and no target ID')
        if self.kind!='FORM' and not self.target_seed_id:
            raise ValueError('seed mutation must target existing seed')
        if self.kind=='DEVELOP' and self.draft is None:
            raise ValueError('DEVELOP requires fully revised draft')
        if self.kind in ('CHALLENGE','DISCARD') and (self.draft is not None or not self.challenge_or_change or len(self.challenge_or_change)<25):
            raise ValueError('CHALLENGE/DISCARD require explicit critique without new draft')
        if self.kind=='PROMOTE' and self.draft is not None:
            raise ValueError('PROMOTE uses native ResearchIdea draft, not a new seed draft')
        return self


def seed_extension_schema(base_decision_schema: dict) -> dict:
    """Add two required structured fields to model-facing schema, not base event schema."""
    import copy
    schema=copy.deepcopy(base_decision_schema)
    defs=SeedMutation.model_json_schema()['$defs']
    seed_schema=SeedMutation.model_json_schema()
    schema['$defs'].update(defs)
    # SeedMutation schema references #/$defs/SeedDraft already installed.
    schema['properties']['seed_update']={'anyOf':[{'type':'null'},
        {k:v for k,v in seed_schema.items() if k!='$defs'}]}
    schema['properties']['seed_work_target_id']={'anyOf':[{'type':'null'},{'type':'string','minLength':8}]}
    schema['required']=list(dict.fromkeys([*schema.get('required',[]),'seed_update','seed_work_target_id']))
    return schema
