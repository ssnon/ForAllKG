#!/usr/bin/env python3
"""M10 SERS: verified G6 -> falsifiable novelty seeds -> bounded paid diversification -> literature queue.

Provider calls are always opt-in and receipts are write-once. Never certifies novelty.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
from pathlib import Path
from pipeline_core.discovery.sers_academic_novelty_v1 import (
    PRIOR_ART, benchmark, make_tasks, read_json, write_new,
    reference_hypothesis, review_queue, validate_candidate, verify_g6, digest)


def prepare(a):
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError("Refuse nonempty experiment directory")
    g6=read_json(a.g6_json)
    nodes=verify_g6(g6)
    tasks=make_tasks(nodes)
    seed=reference_hypothesis()
    seed["candidate_id"]="SERS_HYPOTHESIS_SEED_001_NOT_NOVELTY_CERTIFIED"
    seed["prior_art_queries"]=[
        "SERS time-reversal asymmetric cross-correlation vibrational bands detailed balance",
        "surface enhanced Raman anti-symmetric lag correlation molecular kinetic cycle",
        "single molecule Raman time-reversal irreversibility non-equilibrium spectral cross correlation",
        "SERS cross spectrum imaginary part adsorption site cycling driven nonequilibrium",
        "SERS cyclic chemical state kinetics reversible drive lagged band covariance",
        "plasmon enhanced vibrational spectroscopy antisymmetric temporal correlation stochastic thermodynamics",
    ]
    index={"status":"PREPARED_SEARCH_ONLY_NO_NOVELTY_CLAIM", "g6_sha256":digest(g6),
           "g6_report_id":g6["report_id"], "parent_ids":list(nodes),
           "tasks":tasks,"max_new_provider_calls_per_approved_command":2,
           "candidate_count":1,"source_physics_is_known":True,
           "source_prior_art_is_abstract_or_metadata_level_only":True}
    write_new(a.out_dir/"M10_SPRINT_PLAN.json",index)
    write_new(a.out_dir/"M10_ANALYTICAL_BENCHMARK.json",benchmark())
    write_new(a.out_dir/"M10_SEED_CANDIDATE.json",seed)
    write_new(a.out_dir/"M10_PRELIMINARY_PRIOR_ART.json",PRIOR_ART)
    print(json.dumps({"status":"M10_PREPARED_WITH_REAL_NATIVE_G6_AND_ANALYTICAL_NULLS",
        "new_provider_calls":0,"out":str(a.out_dir),"planned_optional_generation_calls":len(tasks),
        "source_g6_ideas":list(nodes)},ensure_ascii=False,indent=2))


def _receipt_check(task, receipt_path, model):
    """Refuse silently changed/alien receipts before considering a new provider call."""
    receipt = read_json(receipt_path)
    tid = task["task_id"]
    if (receipt.get("task_id") != tid or
            receipt.get("prompt_sha256") != digest(task) or
            receipt.get("model") != model):
        raise ValueError("M10_RECEIPT_IDENTITY_OR_PROMPT_MISMATCH:" + tid)
    if not isinstance(receipt.get("raw_response"), str):
        raise ValueError("M10_RECEIPT_RAW_RESPONSE_MISSING:" + tid)
    return receipt


def pending_generation_tasks(plan, root, model):
    """Unparsed paid receipt is a completed attempt, NEVER an eligible new call."""
    todo, unresolved = [], []
    for task in plan["tasks"]:
        tid = task["task_id"]
        intent = root / "intents" / (tid + ".json")
        receipt = root / "receipts" / (tid + ".json")
        parsed = root / "parsed" / (tid + ".json")
        if receipt.exists():
            _receipt_check(task, receipt, model)
            if not intent.is_file():
                raise RuntimeError("Paid receipt lacks attempt intent: " + str(receipt))
            if parsed.is_file():
                validate_candidate(read_json(parsed), tid)
            else:
                unresolved.append(tid)
            continue
        if parsed.exists():
            raise RuntimeError("Parsed candidate without receipt: " + str(parsed))
        if intent.exists():
            raise RuntimeError("Unresolved paid intent; do not auto-retry: " + str(intent))
        todo.append((task, intent, receipt, parsed))
    return todo, unresolved


def recover_saved(a):
    """Inspect already-paid receipts OFFLINE; do not invent scientific equations."""
    plan = read_json(a.plan)
    if plan.get("status") != "PREPARED_SEARCH_ONLY_NO_NOVELTY_CLAIM":
        raise ValueError("Unknown plan status")
    root = a.plan.parent
    lockpath = root / "model_lock.json"
    if not lockpath.is_file():
        raise FileNotFoundError("Model lock required for receipt authenticity")
    lock = read_json(lockpath)
    model = lock["model"]
    rows = []
    for task in plan["tasks"]:
        tid = task["task_id"]
        receipt_path = root / "receipts" / (tid + ".json")
        parsed_path = root / "parsed" / (tid + ".json")
        if not receipt_path.exists():
            rows.append({"task_id":tid, "status":"NOT_YET_CALLED"})
            continue
        receipt = _receipt_check(task, receipt_path, model)
        row = {"task_id":tid,"finish_reason":receipt.get("finish_reason"),
               "receipt_sha256":__import__("hashlib").sha256(receipt_path.read_bytes()).hexdigest(),
               "new_provider_calls":0}
        if parsed_path.is_file():
            validate_candidate(read_json(parsed_path), tid)
            row["status"]="ALREADY_PARSED_VALID"
            rows.append(row)
            continue
        if receipt.get("finish_reason") != "stop":
            row.update(status="INCOMPLETE_RESPONSE_NO_RECALL",error="finish_reason != stop")
            rows.append(row)
            continue
        try:
            candidate = json.loads(receipt["raw_response"])
            if not isinstance(candidate,dict):
                raise ValueError("Candidate response not a JSON object")
            row["raw_candidate_keys"] = sorted(candidate.keys())
            # Only rename a verbatim model-supplied field. NEVER fabricate formulae.
            aliases = ('model_equations', 'conditional_equations', 'governing_equations')
            matched = [key for key in aliases if key in candidate and candidate[key]]
            if not candidate.get('equations') and len(matched) == 1:
                source_equations = candidate[matched[0]]
                from pipeline_core.discovery.sers_academic_novelty_v1 import _valid_equations, _has_mathematics
                if not _valid_equations(source_equations) or not _has_mathematics(source_equations):
                    raise ValueError('Equation alias has no explicit mathematical expression')
                candidate = dict(candidate)
                candidate['equations'] = source_equations
                row['verbatim_field_alias'] = matched[0] + '->equations'
            validated = validate_candidate(candidate, tid)
            validated["candidate_id"] = "M10_SPECULATIVE_" + tid
            validated["source_g6"] = task["source_g6"]
            if row.get("verbatim_field_alias"):
                validated["offline_receipt_recovery"]={
                    "type":"VERBATIM_FIELD_ALIAS_ONLY",
                    "source_field":matched[0],
                    "receipt_sha256":row["receipt_sha256"],
                    "scientific_equation_invented":False,
                }
            write_new(parsed_path, validated)
            row["status"]="RECOVERED_OFFLINE_VALID"
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
            row.update(status="SAVED_RESPONSE_SCIENTIFICALLY_INCOMPLETE",
                       validation_error=str(exc),
                       raw_candidate_unverified=json.loads(receipt["raw_response"]) if
                           _valid_json_object(receipt["raw_response"]) else {"raw_text":receipt["raw_response"]})
        rows.append(row)
    report={"status":"M10_RECEIPT_RECOVERY_AUDIT_ONLY_NO_PROVIDER_CALLS",
            "scientific_truth_authority":False,"novelty_certified":False,
            "user_must_review_any_unverified_candidates":True,
            "results":rows,"new_provider_calls":0}
    write_new(a.out,report)
    print(json.dumps({"status":report["status"],"results":[
        {k:v for k,v in r.items() if k not in {"raw_candidate_unverified"}} for r in rows],
        "audit_file":str(a.out)},ensure_ascii=False,indent=2))


def _valid_json_object(raw):
    try:
        return isinstance(json.loads(raw),dict)
    except (ValueError,TypeError):
        return False


def paid(a):
    if not(a.allow_paid and a.i_authorize_new_llm_calls):
        raise ValueError("Explicit two-flag paid authorization required")
    if not 1<=a.max_new_calls<=2 or not 1000<=a.max_output_tokens<=6000:
        raise ValueError("Bounds: calls 1..2, tokens 1000..6000")
    plan=read_json(a.plan)
    root=a.plan.parent
    if plan["status"]!="PREPARED_SEARCH_ONLY_NO_NOVELTY_CLAIM":raise ValueError("Unknown plan status")
    if not os.getenv(a.api_key_env):raise RuntimeError("Missing API key")
    lock={"model":a.model,"max_output_tokens":a.max_output_tokens,"temperature":0.3,
          "base_url":a.base_url}
    lock_path=root/"model_lock.json"
    if lock_path.exists():
        if read_json(lock_path)!=lock:raise ValueError("Model lock mismatch")
    else:write_new(lock_path,lock)
    todo, unresolved = pending_generation_tasks(plan, root, a.model)
    for tid in unresolved:
        print("M10_SAVED_UNPARSED_RECEIPT_SKIPPED_NO_RECALL=" + tid)
    if not todo:
        print("M10_ALL_GENERATION_RECEIPTS_ALREADY_SAVED_NO_PROVIDER_CALLS")
        return
    from openai import OpenAI
    client=OpenAI(api_key=os.environ[a.api_key_env],base_url=a.base_url,timeout=180,max_retries=0)
    n=0
    for task,intent,receipt,parsed in todo[:a.max_new_calls]:
        write_new(intent,{"status":"PAID_PROVIDER_CALL_ATTEMPTED_NO_AUTO_RETRY",
                          "task_id":task["task_id"],"prompt_sha256":digest(task),"model":a.model})
        reply=client.chat.completions.create(model=a.model,temperature=.3,max_tokens=a.max_output_tokens,
            response_format={"type":"json_object"},messages=[
                {"role":"system","content":task["system"]},{"role":"user","content":task["user"]}])
        n+=1
        raw=reply.choices[0].message.content or ""
        reason=reply.choices[0].finish_reason
        write_new(receipt,{"task_id":task["task_id"],"model":a.model,
                           "prompt_sha256":digest(task),"finish_reason":reason,
                           "usage": reply.usage.model_dump(mode="json") if reply.usage else None,
                           "raw_response":raw,"scientific_truth_authority":False})
        if reason!="stop":raise RuntimeError(f"RAW_RECEIPT_SAVED_NONSTOP_FINISH_{task['task_id']}; no automatic retry")
        try:
            obj=validate_candidate(json.loads(raw),task["task_id"])
        except Exception as e:
            raise RuntimeError(f"RAW_RECEIPT_SAVED_INVALID_CANDIDATE_{task['task_id']}: {e}; no automatic retry") from e
        obj["candidate_id"]="M10_SPECULATIVE_"+task["task_id"]
        obj["source_g6"]=task["source_g6"]
        write_new(parsed,obj)
        print(json.dumps({"generated_search_only":task["task_id"],"novelty_certified":False},ensure_ascii=False))
    print(json.dumps({"status":"M10_BOUNDED_PROVIDER_CALLS_COMPLETE","new_provider_calls":n},ensure_ascii=False))


def packet(a):
    root=a.plan.parent
    plan=read_json(a.plan)
    seed=read_json(root/"M10_SEED_CANDIDATE.json")
    candidates=[seed]
    missing=[]
    for task in plan["tasks"]:
        tid=task["task_id"]
        parsed=root/"parsed"/(tid+".json")
        if parsed.is_file():
            c=read_json(parsed)
            validate_candidate(c,tid)
            candidates.append(c)
        else:missing.append(tid)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError("Refuse nonempty packet directory")
    out={"status":"LITERATURE_CHALLENGE_PENDING_NOT_NOVELTY_CERTIFIED",
         "source_g6_report_id":plan["g6_report_id"],"generated_candidate_count":len(candidates)-1,
         "missing_paid_candidates":missing,"candidates":candidates,
         "source_prior_art":PRIOR_ART,"novelty_certified":False,
         "interpretation":"The seed is theory-guided; extra candidates are LLM speculative. Prior art requires full-text/claim-level adjudication and independent experts."}
    write_new(a.out_dir/"M10_NOVEL_HYPOTHESIS_CANDIDATE_PACKET.json",out)
    queue=review_queue(candidates)
    filename=a.out_dir/"M10_CLAIM_LEVEL_LITERATURE_REVIEW_QUEUE.csv"
    with filename.open("x",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(queue[0]))
        writer.writeheader();writer.writerows(queue)
    print(json.dumps({"status":"M10_CANDIDATES_PREPARED_NOT_NOVELTY_CERTIFIED",
        "candidates":len(candidates),"literature_queries":len(queue),
        "packet":str(a.out_dir/"M10_NOVEL_HYPOTHESIS_CANDIDATE_PACKET.json")},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--g6-json",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    a.set_defaults(func=prepare)
    a=sub.add_parser("generate-paid")
    a.add_argument("--plan",type=Path,required=True)
    a.add_argument("--model",required=True)
    a.add_argument("--max-new-calls",type=int,default=1)
    a.add_argument("--max-output-tokens",type=int,default=4500)
    a.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    a.add_argument("--base-url",default="https://openrouter.ai/api/v1")
    a.add_argument("--allow-paid",action="store_true")
    a.add_argument("--i-authorize-new-llm-calls",action="store_true")
    a.set_defaults(func=paid)
    a=sub.add_parser("recover-saved")
    a.add_argument("--plan",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    a.set_defaults(func=recover_saved)
    a=sub.add_parser("packet")
    a.add_argument("--plan",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    a.set_defaults(func=packet)
    args=p.parse_args();args.func(args)

if __name__=="__main__":main()
