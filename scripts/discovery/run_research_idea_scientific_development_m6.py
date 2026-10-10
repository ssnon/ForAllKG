"""CLI for opt-in M6 development; all modes except run-paid are offline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline_core.discovery.research_idea_scientific_development_m6 import (
    SCHEMA, prepare, develop, load, save_new, prompt_for_task, validate_proposal, file_sha, verify_native_sources, prompt_for_revision,
)


def run_paid(args: argparse.Namespace) -> dict:
    """One LLM call per selected source idea, strictly capped. No implicit retries."""
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("paid execution requires both explicit consent flags")
    if not args.model or not args.api_key_env:
        raise ValueError("model and API key env must be specified")
    if not (1 <= args.max_new_calls <= 2):
        raise ValueError("M6 initial paid pilot is capped at 2 new logical calls")
    prepared = load(args.prepared)
    if prepared.get("status") != "M6_CHALLENGE_PREPARED_NO_MODEL_CALLS":
        raise ValueError("invalid frozen source preparation")
    if len(prepared["tasks"]) != args.max_new_calls:
        raise ValueError("pilot paid task count must equal --max-new-calls; prepare exactly 1 or 2 seeds")
    dst = Path(args.out_dir).resolve()
    if Path(args.prepared).resolve() == dst or dst in Path(args.prepared).resolve().parents:
        raise ValueError("output location invalid")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError("paid output must be a new empty directory; do not overwrite")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f"missing API key env {args.api_key_env}")
    from openai import OpenAI  # deferred import, never for offline
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout, max_retries=0)
    recorded = []
    dst.mkdir(parents=True, exist_ok=True)
    for task in prepared["tasks"][:args.max_new_calls]:
        system, user = prompt_for_task(task)
        kwargs = dict(model=args.model, messages=[{"role":"system","content":system},{"role":"user","content":user}],
                      response_format={"type":"json_object"})
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        save_new(dst / (task["task_id"].replace(":","_")+".raw.json"),
                 {"task_id":task["task_id"],"prompt_sha256":file_sha(args.prepared),
                  "content":content,"model":args.model,
                  "provider_response_id":getattr(resp,"id",None),
                  "reported_usage":resp.usage.model_dump() if resp.usage else None})
        doc = json.loads(content)
        if set(doc)!={"proposals"} or len(doc["proposals"]) != 1:
            raise ValueError(f"malformed model proposal after paid call: {task['task_id']}")
        p=doc["proposals"][0]
        validate_proposal(p,{task["parent_idea_id"]})
        recorded.append(p)
    if len(recorded) != len(prepared["tasks"]):
        return {"status":"M6_PAID_PARTIAL_NO_MERGED_DRAFTS", "new_logical_calls":len(recorded)}
    save_new(dst / "M6_PAID_DRAFTS.json", {"schema_version":SCHEMA,"proposals":recorded,
        "generation_info":{"paid_calls":len(recorded), "model":args.model,
                           "scientific_truth_authority":False, "novelty_certified":False}})
    return {"status":"M6_PAID_DRAFTS_WRITTEN_SCIENCE_UNREVIEWED",
            "new_logical_calls":len(recorded),"output":str(dst/"M6_PAID_DRAFTS.json")}


def revise_paid(args: argparse.Namespace) -> dict:
    """Second bounded paid call after first-pass M6 critique; no silent retries."""
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("paid revision requires both explicit consent flags")
    if not 1 <= args.max_new_calls <= 2:
        raise ValueError("bounded revision cap is 1..2 calls")
    prepared = load(args.prepared)
    developed = load(args.developed)
    if (prepared.get("schema_version") != SCHEMA or
            developed.get("schema_version") != SCHEMA or
            developed.get("source_prepared_sha256") != file_sha(args.prepared)):
        raise ValueError("developed run and prepared inputs are not linked")
    tasks = {t["parent_idea_id"]: t for t in prepared["tasks"]}
    branches = developed["development_branches"]
    if len(branches) != len(tasks) or len(branches) != args.max_new_calls:
        raise ValueError("revision count must equal --max-new-calls, at most 2")
    dst = Path(args.out_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError("revision paid output must be an empty new directory")
    if not os.environ.get(args.api_key_env):
        raise ValueError("missing API key env")
    from openai import OpenAI
    client=OpenAI(api_key=os.environ[args.api_key_env],base_url=args.base_url,
                  timeout=args.timeout,max_retries=0)
    dst.mkdir(parents=True,exist_ok=True)
    revisions=[]
    for branch in branches[:args.max_new_calls]:
        t=tasks[branch["parent_idea_id"]]
        system, user = prompt_for_revision(branch,t)
        kwargs=dict(model=args.model,messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    response_format={"type":"json_object"})
        if args.temperature is not None:kwargs["temperature"]=args.temperature
        resp=client.chat.completions.create(**kwargs)
        content=resp.choices[0].message.content
        save_new(dst/(branch["branch_id"].replace(":","_")+".raw.json"),
                 {"branch_id":branch["branch_id"],"content":content,"model":args.model,
                  "provider_response_id":getattr(resp,"id",None),
                  "reported_usage":resp.usage.model_dump() if resp.usage else None})
        doc=json.loads(content)
        if set(doc)!={"proposals"} or len(doc["proposals"])!=1:
            raise ValueError("invalid returned revision envelope")
        p=doc["proposals"][0]
        validate_proposal(p,{branch["parent_idea_id"]})
        if p["local_id"]!=branch["initial_proposal"]["local_id"] or p["source_of_proposal"]!="MODEL_REVISED":
            raise ValueError("revision must preserve branch identity and declare provenance")
        revisions.append(p)
    if len(revisions)!=len(branches):
        return {"status":"M6_REVISION_PARTIAL_NO_MERGED_FILE", "new_logical_calls":len(revisions)}
    output=dst/"M6_PAID_REVISIONS.json"
    save_new(output,{"schema_version":SCHEMA,"proposals":revisions,
                     "generation_info":{"paid_calls":len(revisions),"model":args.model,
                                        "scientific_truth_authority":False,"novelty_certified":False}})
    return {"status":"M6_PAID_REVISIONS_WRITTEN_SCIENCE_UNREVIEWED",
            "new_logical_calls":len(revisions), "output":str(output)}


def main() -> None:
    ap=argparse.ArgumentParser(description="M6 scientific development - opt-in, source preserving")
    sub=ap.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--packet",required=True);p.add_argument("--assessments",required=True);p.add_argument("--out-dir",required=True)
    p.add_argument("--idea-id",action="append",default=None,help="Repeat to select frozen parent ideas")
    p.add_argument("--raw-g3");p.add_argument("--raw-g4")
    p=sub.add_parser("develop")
    p.add_argument("--prepared",required=True);p.add_argument("--drafts",required=True);p.add_argument("--revisions")
    p.add_argument("--out-dir",required=True)
    p=sub.add_parser("run-paid")
    p.add_argument("--prepared",required=True);p.add_argument("--out-dir",required=True)
    p.add_argument("--model",required=True);p.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    p.add_argument("--base-url",default="https://openrouter.ai/api/v1")
    p.add_argument("--temperature",type=float,default=0.2);p.add_argument("--timeout",type=float,default=120.0)
    p.add_argument("--max-new-calls",type=int,default=2)
    p.add_argument("--allow-paid",action="store_true");p.add_argument("--i-authorize-new-llm-calls",action="store_true")
    p=sub.add_parser("revise-paid")
    p.add_argument("--prepared",required=True);p.add_argument("--developed",required=True)
    p.add_argument("--out-dir",required=True);p.add_argument("--model",required=True)
    p.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    p.add_argument("--base-url",default="https://openrouter.ai/api/v1")
    p.add_argument("--temperature",type=float,default=0.2);p.add_argument("--timeout",type=float,default=120.0)
    p.add_argument("--max-new-calls",type=int,default=2)
    p.add_argument("--allow-paid",action="store_true");p.add_argument("--i-authorize-new-llm-calls",action="store_true")
    x=ap.parse_args()
    if x.mode=="prepare":
        if bool(x.raw_g3) != bool(x.raw_g4):
            raise ValueError("pass both --raw-g3 and --raw-g4, or neither")
        if x.raw_g3:
            print(json.dumps(verify_native_sources(x.packet,x.raw_g3,x.raw_g4),indent=2))
        r=prepare(x.packet,x.assessments,x.out_dir,x.idea_id)
        print(json.dumps({"status":r["status"],"selected_ideas":len(r["tasks"]),"out":x.out_dir},indent=2))
    elif x.mode=="develop":
        r=develop(x.prepared,x.drafts,x.out_dir,x.revisions)
        print(json.dumps({"status":r["status"],"branches":r["branch_count"],
                          "metrics":r["metrics_non_scientific"],"out":x.out_dir},indent=2))
    elif x.mode=="run-paid":
        print(json.dumps(run_paid(x),indent=2))
    elif x.mode=="revise-paid":
        print(json.dumps(revise_paid(x),indent=2))

if __name__=="__main__":main()
