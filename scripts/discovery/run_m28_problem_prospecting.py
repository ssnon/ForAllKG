"""M28 add-only source-grounded research *problem* prospecting and Case B handoff.

No new controller, no synthetic KG. All paid LLM calls route through the existing
scientific_discovery_core.engine._call immutable call ledger. This experimental
adapter does not certify scientific truth or novelty.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "fakg-m28-problem-prospecting-v1"
STAGES = ("prospect", "select")

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Candidate(Strict):
    local_id: str = Field(min_length=2)
    domain: str = Field(min_length=8)
    source_ids: list[str] = Field(min_length=1, max_length=6)
    source_reported_observation: str = Field(min_length=30)
    research_question: str = Field(min_length=30)
    why_not_preanswered_by_sources: str = Field(min_length=30)
    strongest_known_or_simple_explanation: str = Field(min_length=25)
    explanatory_opportunity: str = Field(min_length=30)
    first_independent_check: str = Field(min_length=25)
    likely_confounder_or_obstruction: str = Field(min_length=25)
    prior_art_risk: str = Field(min_length=20)

class Prospect(Strict):
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)
    why_these_candidates: str = Field(min_length=25)
    abstention_or_scope_limit: str = Field(min_length=20)

class Review(Strict):
    local_id: str
    source_fidelity_0to3: int = Field(ge=0, le=3)
    explanatory_opening_0to3: int = Field(ge=0, le=3)
    feasible_independent_test_0to3: int = Field(ge=0, le=3)
    preexisting_answer_risk_0to3: int = Field(ge=0, le=3)
    strongest_null_mimic: str = Field(min_length=25)
    potential_value_and_failure_mode: str = Field(min_length=30)

class Selection(Strict):
    decision: Literal["SELECT", "ABSTAIN"]
    selected_id: str | None = None
    reviews: list[Review]
    selection_reason: str = Field(min_length=35)
    next_research_action: str = Field(min_length=25)
    strongest_unresolved_obstacle: str = Field(min_length=25)
    no_discovery_claim: Literal[True] = True
    @model_validator(mode="after")
    def check(self):
        if self.decision == "ABSTAIN" and self.selected_id is not None:
            raise ValueError("ABSTAIN_WITH_SELECTION")
        if self.decision == "SELECT" and not self.selected_id:
            raise ValueError("SELECT_WITHOUT_ID")
        return self

class Baseline(Strict):
    decision: Literal["SELECT", "ABSTAIN"]
    selected_problem: Candidate | None = None
    rationale: str = Field(min_length=35)
    no_discovery_claim: Literal[True] = True
    @model_validator(mode="after")
    def check(self):
        if (self.decision == "SELECT") != (self.selected_problem is not None):
            raise ValueError("BASELINE_SELECTION_INCONSISTENT")
        return self


def canon(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha(x):
    return hashlib.sha256(canon(x)).hexdigest()

def file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def new(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def validate_sources(x):
    if x.get("schema_version") != "fakg-m28-source-cards-v1":
        raise ValueError("INVALID_M28_SOURCE_SCHEMA")
    cards = x.get("source_cards", [])
    if len(cards) != 6 or len({c["id"] for c in cards}) != 6:
        raise ValueError("EXPECTED_SIX_UNIQUE_CARDS")
    counts = Counter(c["domain"] for c in cards)
    if sorted(counts.values()) != [2,2,2]:
        raise ValueError("EXPECTED_THREE_FIELDS_TWO_SOURCES_EACH")
    for c in cards:
        for field in ("title", "doi", "url", "methods", "source_reported_findings", "source_scope"):
            if not isinstance(c.get(field), str) or not c[field].strip():
                raise ValueError(f"INVALID_CARD_{c['id']}_{field}")
        if not c["url"].startswith("https://doi.org/"):
            raise ValueError("DOI_URL_REQUIRED")
    return cards

def settings(out):
    out = Path(out)
    manifest = read(out/"manifest.json")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("INVALID_M28_MANIFEST")
    evidence_file = out/"evidence.json"
    if file_sha(evidence_file) != manifest["evidence_bytes_sha256"]:
        raise ValueError("FROZEN_EVIDENCE_CHANGED")
    validate_sources(read(evidence_file))
    return manifest,read(evidence_file)

def stages_prompt(stage, evidence, previous=None):
    system = ("You are an autonomous scientific research agent. Search aggressively; claim conservatively. "
      "Here you must DISCOVER which scientific question merits investigation, not answer an assigned problem. "
      "The input contains publisher-abstract-level paraphrases of six published studies in three fields. "
      "Each scientific fact must trace to an actual source ID; no source is asserted to contradict another. "
      "Distinct organisms, materials, conditions and methods cannot be silently equated. "
      "Avoid known-effect relabeling, pseudo-novelty, and purely observational correlations presented as mechanisms. "
      "Identify an explanation-level opportunity only if scientific evidence warrants further research. "
      "You may abstain; no forced discovery. Never fabricate experimental measurements, theoretical results or prior-art novelty. "
      "Research hypotheses and questions are speculative, not established by the cited work. "
      "For each candidate, domain MUST equal the exact domain string of its cited source cards; "
      "source_ids MUST be copied verbatim from supplied IDs and all must share the candidate domain. "
      "Respond with one strict JSON object and no markdown.")
    if stage == "prospect":
        task = ("Prospect across ALL three research fields before choosing up to three candidate research questions. "
                "Do not simply rephrase the titles, write a paper review, or copy a known answer. "
                "For each proposed question state actual observations, existing null, potential new explanatory space, "
                "a discriminating independent test and confounding; record prior-art risk. "
                "Propose ZERO candidates if nothing scientifically worth pursuing follows from these abstract summaries.")
        cls = Prospect
    elif stage == "select":
        task = ("Critically assess every prior candidate against a strong conventional explanation. "
                "Do not treat a plausible problem as already unsolved or novel. "
                "Select exactly one scientifically productive research question, or ABSTAIN. "
                "Compare source quality, nontriviality and independent measurability; do not choose by prose novelty. "
                "All reviews MUST use the exact local_id strings of the prior candidates. "
                "No new scientific mechanism is claimed at this stage.")
        cls = Selection
    elif stage == "baseline":
        task = ("In ONE independent self-contained response with no feedback or previous proposals, "
                "find and select one research question with genuine explanatory potential from these six source cards, "
                "or ABSTAIN. Include scientific controls and a strong null. Do not claim that any question is novel.")
        cls = Baseline
    else:
        raise ValueError("BAD_STAGE")
    user = {"task":stage,"scientific_goal":task,"source_cards":evidence["source_cards"],
            "source_depth":evidence["source_depth"],"previous_prospect":previous,
            "response_json_schema":cls.model_json_schema()}
    return system,json.dumps(user,ensure_ascii=False,sort_keys=True),cls

def validate_candidate_refs(candidates,evidence):
    ids={c["id"] for c in evidence["source_cards"]}
    field={c["id"]:c["domain"] for c in evidence["source_cards"]}
    seen=[]
    for cand in candidates:
        if cand.local_id in seen: raise ValueError("DUPLICATE_CANDIDATE_ID")
        seen.append(cand.local_id)
        if len(set(cand.source_ids))!=len(cand.source_ids) or not set(cand.source_ids)<=ids:
            raise ValueError("UNKNOWN_OR_REPEATED_SOURCE_ID")
        if any(field[i] != cand.domain for i in cand.source_ids):
            raise ValueError("CANDIDATE_DOMAIN_SOURCE_MISMATCH")

def validate_stage(stage,out,evidence):
    if stage == "prospect":
        if not (out/"calls/prospect/response.json").is_file():
            raise ValueError("PROSPECT_NOT_COMPLETE")
        p=Prospect.model_validate(read(out/"calls/prospect/response.json"))
        validate_candidate_refs(p.candidates,evidence)
        return p
    if stage == "select":
        p=validate_stage("prospect",out,evidence)
        s=Selection.model_validate(read(out/"calls/select/response.json"))
        expected={x.local_id for x in p.candidates}
        if {x.local_id for x in s.reviews}!=expected or len(s.reviews)!=len(expected):
            raise ValueError("NOT_ALL_CANDIDATES_REVIEWED")
        if s.selected_id is not None and s.selected_id not in expected:
            raise ValueError("UNKNOWN_SELECTION")
        return s
    if stage.startswith("baseline_"):
        b=Baseline.model_validate(read(out/"calls"/stage/"response.json"))
        if b.selected_problem:validate_candidate_refs([b.selected_problem],evidence)
        return b
    raise ValueError("INVALID_STAGE")

def get_call_args(stage,out,evidence):
    prev=None
    if stage == "select":
        prev=validate_stage("prospect",out,evidence).model_dump(mode="json")
    system,user,cls=stages_prompt("baseline" if stage.startswith("baseline_") else stage,evidence,prev)
    m=read(out/"manifest.json")
    request={"stage":stage,"system":system,"user":user,"schema":cls.__name__,
             "model":m["model"],"temperature":m["temperature"],"max_tokens":m["max_tokens"]}
    return system,user,cls,request

def verify(out):
    out=Path(out);m,e=settings(out)
    status={}
    for stage in (*STAGES,"baseline_01","baseline_02"):
        d=out/"calls"/stage
        if not d.exists():status[stage]="NOT_STARTED";continue
        sys,user,cls,expected=get_call_args(stage,out,e)
        if read(d/"request.json")!=expected:
            raise ValueError(f"{stage}: REQUEST_DRIFT")
        if read(d/"attempt.json").get("request_sha256")!=sha(expected):
            raise ValueError(f"{stage}: ATTEMPT_DRIFT")
        if not (d/"receipt.json").is_file():
            status[stage]="RAW_SAVED_NEEDS_RECOVERY" if (d/"raw_response.json").is_file() else "ATTEMPT_NO_RAW_DO_NOT_RETRY"
            continue
        receipt=read(d/"receipt.json")
        if receipt.get("request_sha256")!=sha(expected) or receipt.get("status")!="COMPLETED":
            raise ValueError(f"{stage}: RECEIPT_DRIFT")
        if sha(read(d/"raw_response.json"))!=receipt.get("original_raw_response_sha256"):
            raise ValueError(f"{stage}: RAW_DRIFT")
        if sha(read(d/"response.json"))!=receipt.get("validated_response_sha256"):
            raise ValueError(f"{stage}: NORMALIZED_DRIFT")
        validate_stage(stage,out,e)
        status[stage]="VERIFIED"
    return {"status":"VERIFIED_NO_CALL","scientific_validation":False,"steps":status,
            "evidence_sha256":m["evidence_bytes_sha256"],"paid_calls":0}

def prepare(source,out,model,temperature,max_tokens):
    out=Path(out);source=Path(source)
    validate_sources(read(source))
    if out.exists():raise FileExistsError("NEW_RUN_DIRECTORY_REQUIRED")
    out.mkdir(parents=True,exist_ok=False)
    new(out/"evidence.json",read(source))
    new(out/"manifest.json",{"schema_version":SCHEMA,"evidence_bytes_sha256":file_sha(out/"evidence.json"),
        "model":model,"temperature":temperature,"max_tokens":max_tokens,
        "stage_budget_primary":2,"stage_budget_baseline":2,"scientific_truth_certified":False,
        "novelty_certified":False,"source_depth":"abstract_paraphrases", "source_original_path":str(source)})
    return verify(out)

def next_stage(out,allow_paid,baseline=False):
    out=Path(out);m,e=settings(out)
    v=verify(out)
    order=("baseline_01","baseline_02") if baseline else STAGES
    stage=next((s for s in order if v["steps"][s]!="VERIFIED"),None)
    if stage is None:return {"status":"ALREADY_COMPLETE_NO_CALL","verified":v}
    if v["steps"][stage]!="NOT_STARTED":
        raise RuntimeError(f"{stage}: SAVED_ATTEMPT_PRESENT_NO_RETRY; RUN recovery diagnostics")
    if not allow_paid or os.environ.get("FAKG_RESEARCH_ALLOW_PAID")!="YES":
        raise PermissionError("PAID_BLOCKED: --allow-paid AND FAKG_RESEARCH_ALLOW_PAID=YES REQUIRED")
    # Import and reuse Case B's ledger and OpenRouter backend. No duplicate billing path.
    from pipeline_core.discovery.scientific_discovery_core.engine import _call, OpenRouter
    sys,user,cls,_=get_call_args(stage,out,e)
    backend=OpenRouter()
    _call(out,stage,cls,sys,user,backend,max_calls=4,settings=m)
    validate_stage(stage,out,e)
    return {"status":"STAGE_COMMITTED_ONE_CALL","stage":stage,"result":verify(out)}

def recovery(out,stage):
    out=Path(out);m,e=settings(out)
    if stage not in (*STAGES,"baseline_01","baseline_02"):
        raise ValueError("UNKNOWN_STAGE")
    d=out/"calls"/stage
    if not (d/"raw_response.json").is_file():
        raise RuntimeError("BILLING_UNCERTAIN_NO_RETRY")
    sys,user,cls,request=get_call_args(stage,out,e)
    if read(d/"request.json")!=request or read(d/"attempt.json").get("request_sha256")!=sha(request):
        raise ValueError("REQUEST_MISMATCH")
    from pipeline_core.discovery.scientific_discovery_core.engine import _validate_preserving_science
    raw=read(d/"raw_response.json")
    parsed,diagnostics=_validate_preserving_science(raw,cls)
    if diagnostics:raise ValueError("UNEXPECTED_RECOVERY_DIAGNOSTICS")
    value=parsed.model_dump(mode="json")
    # Validate all scientific references and selection BEFORE writing normalized.
    if stage=="prospect":validate_candidate_refs(parsed.candidates,e)
    elif stage=="select":
        p=validate_stage("prospect",out,e)
        ids=[x.local_id for x in p.candidates]
        if sorted(x.local_id for x in parsed.reviews)!=sorted(ids) or len(parsed.reviews)!=len(ids):
            raise ValueError("MISSING_REVIEWS")
        if parsed.selected_id and parsed.selected_id not in ids:raise ValueError("UNKNOWN_SELECTION")
    elif parsed.selected_problem:validate_candidate_refs([parsed.selected_problem],e)
    if not (d/"response.json").exists():new(d/"response.json",value)
    elif read(d/"response.json")!=value:raise ValueError("EXISTING_NORMALIZED_MISMATCH")
    if not (d/"receipt.json").exists():
        new(d/"receipt.json",{"status":"COMPLETED","request_sha256":sha(request),
          "usage":"RECOVERED_RAW_NO_NEW_CALL","parsed_schema":cls.__name__,
          "paid_attempt_count_for_stage":1,"original_raw_response_sha256":sha(raw),
          "validated_response_sha256":sha(value),"quarantined_matrix_count":0})
    return {"status":"RECOVERED_NO_CALL","stage":stage,"verify":verify(out)}

def export_problem(out,destination):
    out=Path(out);_,e=settings(out)
    v=verify(out)
    if v["steps"]["select"]!="VERIFIED":raise ValueError("SELECTION_NOT_COMPLETE")
    s=validate_stage("select",out,e)
    if s.decision!="SELECT":return {"status":"ABSTAIN_NO_PROBLEM_EXPORTED"}
    p=validate_stage("prospect",out,e)
    chosen=next(x for x in p.candidates if x.local_id==s.selected_id)
    cards={c["id"]:c for c in e["source_cards"]}
    problem={"schema_version":"fakg-open-problem-v1",
        "research_question":chosen.research_question,
        "problem_discovery_origin":"M28_AUTONOMOUS_SELECTION_NOT_HUMAN_ASSIGNED",
        "selected_prospect_id":chosen.local_id,
        "scientific_context":{"observations":chosen.source_reported_observation,
           "why_not_preanswered":chosen.why_not_preanswered_by_sources,
           "strongest_known_or_simple_explanation":chosen.strongest_known_or_simple_explanation,
           "explanatory_opportunity_UNVERIFIED":chosen.explanatory_opportunity,
           "first_independent_check":chosen.first_independent_check,
           "confounding":chosen.likely_confounder_or_obstruction,
           "prior_art_risk":chosen.prior_art_risk},
        "source_depth":"abstract_paraphrases_only",
        "references":[{**cards[i],"source_support_scope":"SOURCE_SUMMARY_ONLY"} for i in chosen.source_ids],
        "independent_scientific_validation":False,
        "m28_selection_sha256":sha(s.model_dump(mode="json")),
        "m28_evidence_bytes_sha256":v["evidence_sha256"]}
    dest=Path(destination)
    if dest.exists():
        if read(dest)!=problem:raise FileExistsError("DESTINATION_EXISTS_WITH_DIFFERENT_DATA")
        return {"status":"EXPORTED_IDEMPOTENT","path":str(dest)}
    new(dest,problem)
    return {"status":"CASEB_PROBLEM_EXPORTED_NO_CALL","path":str(dest),
            "research_question":chosen.research_question,"source_ids":chosen.source_ids}

def parse():
    a=argparse.ArgumentParser(description="M28 source-grounded autonomous problem prospecting")
    sub=a.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare");p.add_argument("--evidence",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    p.add_argument("--model",default="openai/gpt-5.6-luna");p.add_argument("--temperature",type=float,default=0.5);p.add_argument("--max-tokens",type=int,default=6500)
    for name in ("verify","preview","next","next-baseline"):
        p=sub.add_parser(name);p.add_argument("--out",required=True,type=Path)
        if name.startswith("next"):p.add_argument("--allow-paid",action="store_true")
    p=sub.add_parser("recover");p.add_argument("--out",required=True,type=Path);p.add_argument("--stage",required=True,choices=(*STAGES,"baseline_01","baseline_02"))
    p=sub.add_parser("export-problem");p.add_argument("--out",required=True,type=Path);p.add_argument("--dest",required=True,type=Path)
    return a.parse_args()

def main():
    a=parse()
    if a.cmd=="prepare":res=prepare(a.evidence,a.out,a.model,a.temperature,a.max_tokens)
    elif a.cmd=="verify":res=verify(a.out)
    elif a.cmd=="preview":
        v=verify(a.out);s=next((x for x in STAGES if v["steps"][x]!="VERIFIED"),None)
        res={"status":"PREVIEW_NO_CALL","next_primary_stage":s,"stage_state":v["steps"],
             "source_count":6,"paid_calls_if_next":1 if s and v["steps"][s]=="NOT_STARTED" else 0}
    elif a.cmd in ("next","next-baseline"):
        res=next_stage(a.out,a.allow_paid,baseline=(a.cmd=="next-baseline"))
    elif a.cmd=="recover":res=recovery(a.out,a.stage)
    elif a.cmd=="export-problem":res=export_problem(a.out,a.dest)
    else:raise ValueError("UNKNOWN_CMD")
    print(json.dumps(res,indent=2,ensure_ascii=False,sort_keys=True))

if __name__=="__main__":main()
