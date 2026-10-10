"""Opt-in adversarial M6.1 development (invoked from repo root with python -m)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.research_idea_scientific_development_m6 import load, save_new, file_sha
from pipeline_core.discovery.research_idea_scientific_adversarial_m61 import (
    SCHEMA, ROLES, prepare_challenges, validate_cases, critic_prompt, branch_prompt,
    build_analyst_seed_deltas, materialize_branches, merge_model_attacks,
    validate_delta, _validate_attack, print_witness_checks, export_evolution_feedback,
)


def _prepaid(args, case_count: int, input_files: list[str]) -> Path:
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("Paid calls require both opt-in authorization flags")
    if args.max_new_calls != case_count or not (1 <= case_count <= 2):
        raise ValueError("Paid pilot requires 1..2 cases and exactly equal max-new-calls")
    if not os.environ.get(args.api_key_env):
        raise ValueError(f"Missing {args.api_key_env}; no call made")
    dst=Path(args.out_dir).resolve()
    if any(dst==Path(x).resolve() or dst in Path(x).resolve().parents or Path(x).resolve() in dst.parents for x in input_files):
        raise ValueError("output must be separated from source paths")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError("paid output is not empty; do not retry already charged calls")
    return dst


def _call_client(args):
    from openai import OpenAI  # Only in explicit paid modes, not offline
    return OpenAI(api_key=os.environ[args.api_key_env],base_url=args.base_url,
                  timeout=args.timeout,max_retries=0)


def _request(client,args,system,user):
    kwargs={"model":args.model,"messages":[{"role":"system","content":system},
             {"role":"user","content":user}],"response_format":{"type":"json_object"}}
    if args.temperature is not None:kwargs["temperature"]=args.temperature
    r=client.chat.completions.create(**kwargs)
    usage=r.usage.model_dump() if r.usage else None
    return r.choices[0].message.content,{"provider_response_id":getattr(r,"id",None),
                                         "usage":usage,"model":args.model}


def critique_paid(args) -> dict:
    cases=validate_cases(load(args.cases))
    dst=_prepaid(args,len(cases),[args.cases]); client=_call_client(args);dst.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,case in enumerate(cases):
        system,user=critic_prompt(case)
        content,metadata=_request(client,args,system,user)
        save_new(dst/f"CALL_{i+1}_CRITIQUE_RAW.json",{"content":content,"metadata":metadata,
                       "source_proposal_sha256":case["source_proposal_sha256"],
                       "parent_idea_id":case["parent_idea_id"]})
        doc=json.loads(content)
        if set(doc)!={"attacks"} or not isinstance(doc["attacks"],list) or not 1<=len(doc["attacks"])<=3:
            raise ValueError("LLM critique format invalid after paid call; keep raw receipt")
        for j,attack in enumerate(doc["attacks"]):
            attack["attack_id"]="m61_model_attack:"+file_sha(dst/f"CALL_{i+1}_CRITIQUE_RAW.json")[:12]+f"_{j}"
            _validate_attack(attack,case)
        records.append({"parent_idea_id":case["parent_idea_id"],
                        "source_proposal_sha256":case["source_proposal_sha256"],
                        "attacks":doc["attacks"]})
    out={"schema_version":SCHEMA,"status":"M61_MODEL_ATTACKS_PROPOSED_SCIENCE_UNREVIEWED",
         "cases_sha256":file_sha(args.cases),"attack_sets":records,
         "generation_info":{"new_logical_calls":len(records),"model":args.model},
         "authority":{"scientific_truth_authority":False,"novelty_certified":False,
                      "production_selection_authority":False}}
    save_new(dst/"M61_MODEL_ATTACKS.json",out)
    return {"status":out["status"],"new_logical_calls":len(records),"output":str(dst/"M61_MODEL_ATTACKS.json")}


def branch_paid(args) -> dict:
    cases=validate_cases(load(args.cases))
    dst=_prepaid(args,len(cases),[args.cases]);client=_call_client(args);dst.mkdir(parents=True,exist_ok=True)
    deltas=[]
    for i,case in enumerate(cases):
        if not case["attacks"]:raise ValueError("cannot branch from no scientific attacks")
        system,user=branch_prompt(case)
        content,metadata=_request(client,args,system,user)
        save_new(dst/f"CALL_{i+1}_BRANCH_RAW.json",{"content":content,"metadata":metadata,
                    "source_proposal_sha256":case["source_proposal_sha256"],
                    "parent_idea_id":case["parent_idea_id"]})
        doc=json.loads(content)
        if set(doc)!={"deltas"} or not isinstance(doc["deltas"],list) or len(doc["deltas"])!=3:
            raise ValueError("LLM branch format invalid after paid call; keep raw receipt")
        if {d.get("branch_role") for d in doc["deltas"]}!=set(ROLES):
            raise ValueError("LLM must produce three distinct branching roles")
        for d in doc["deltas"]:validate_delta(d,case,"MODEL_GENERATED")
        deltas.extend(doc["deltas"])
    out={"schema_version":SCHEMA,"status":"M61_DIVERGENT_DELTAS_PROPOSED_SCIENCE_UNREVIEWED",
         "cases_sha256":file_sha(args.cases),"deltas":deltas,
         "generation_info":{"new_logical_calls":len(cases),"model":args.model},
         "authority":{"scientific_truth_authority":False,"novelty_certified":False,
                      "production_selection_authority":False}}
    save_new(dst/"M61_MODEL_DELTAS.json",out)
    return {"status":out["status"],"new_logical_calls":len(cases),"output":str(dst/"M61_MODEL_DELTAS.json")}


def main():
    ap=argparse.ArgumentParser(description="M6.1: scientific attack and divergent research development")
    sub=ap.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("challenge");p.add_argument("--developed",required=True);p.add_argument("--out-dir",required=True)
    p=sub.add_parser("analyst-seed");p.add_argument("--cases",required=True);p.add_argument("--out-dir",required=True)
    p=sub.add_parser("merge-model-attacks");p.add_argument("--cases",required=True);p.add_argument("--model-attacks",required=True);p.add_argument("--out-dir",required=True)
    p=sub.add_parser("materialize");p.add_argument("--cases",required=True);p.add_argument("--deltas",required=True);p.add_argument("--out-dir",required=True)
    p=sub.add_parser("export-feedback");p.add_argument("--cases",required=True);p.add_argument("--population",required=True);p.add_argument("--out-dir",required=True)
    for mode in ("critique-paid","branch-paid"):
        p=sub.add_parser(mode)
        p.add_argument("--cases",required=True);p.add_argument("--out-dir",required=True)
        p.add_argument("--model",required=True);p.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
        p.add_argument("--base-url",default="https://openrouter.ai/api/v1")
        p.add_argument("--temperature",type=float,default=0.3)
        p.add_argument("--timeout",type=float,default=180.0)
        p.add_argument("--max-new-calls",type=int,default=2)
        p.add_argument("--allow-paid",action="store_true")
        p.add_argument("--i-authorize-new-llm-calls",action="store_true")
    args=ap.parse_args()
    if args.mode=="challenge":
        r=prepare_challenges(args.developed,args.out_dir)
        print(json.dumps({"status":r["status"],"cases":r["case_count"],
                          "attacks":sum(len(c["attacks"]) for c in r["cases"]),
                          "physics_checks":print_witness_checks(),"out_dir":args.out_dir},indent=2))
    elif args.mode=="analyst-seed":
        r=build_analyst_seed_deltas(args.cases,args.out_dir)
        print(json.dumps({"status":r["status"],"analyst_scaffolds":len(r["deltas"]),"out_dir":args.out_dir},indent=2))
    elif args.mode=="merge-model-attacks":
        r=merge_model_attacks(args.cases,args.model_attacks,args.out_dir)
        print(json.dumps({"status":r["status"],"enriched_attacks":sum(len(c["attacks"]) for c in r["cases"]),"out_dir":args.out_dir},indent=2))
    elif args.mode=="materialize":
        r=materialize_branches(args.cases,args.deltas,args.out_dir)
        print(json.dumps({"status":r["status"],"parents":r["parent_count"],"forks":r["fork_count"],"out_dir":args.out_dir},indent=2))
    elif args.mode=="export-feedback":
        r=export_evolution_feedback(args.cases,args.population,args.out_dir)
        print(json.dumps({"status":r["status"],"hints":r["hint_count"],"out_dir":args.out_dir},indent=2))
    elif args.mode=="critique-paid":
        print(json.dumps(critique_paid(args),indent=2))
    elif args.mode=="branch-paid":
        print(json.dumps(branch_paid(args),indent=2))

if __name__=="__main__":main()
