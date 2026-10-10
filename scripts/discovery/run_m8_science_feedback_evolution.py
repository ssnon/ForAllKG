#!/usr/bin/env python3
"""M8 matched G6 opt-in; prepare/preview/replay/score free, generate-paid explicit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.m8_feedback_paired_evolution import (
    ARMS, TARGETS, canonical, digest, read, write_new, sha_file,
    make_experiment, checked_plan, make_prompt, parse_draft,
    native_components, receipt_paths, create_blind_packet, score_blind,
)


def prepare(args):
    sources={"g5":args.g5,"m7":args.m7_report,"m7_plan":args.m7_plan,"reviewed":args.reviewed}
    for p in sources.values():
        if not p.is_file():raise FileNotFoundError(p)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Refusing nonempty experiment directory")
    experiment=make_experiment(raw=read(args.g5),m7=read(args.m7_report),
        m7_plan=read(args.m7_plan),reviewed=read(args.reviewed),source_files=sources)
    # Check exact native schema before writing any experiment artifact.
    plan,parallel,parents=native_components(experiment)
    if len(plan.tasks)!=2 or len(parents)!=2:raise RuntimeError("Native G6 plan failure")
    write_new(args.out_dir/"M8_EXPERIMENT_PLAN.json", experiment)
    write_new(args.out_dir/"M8_EXPERIMENT_PREFLIGHT.json",{
        "status":"M8_G6_NATIVE_PLAN_PREFLIGHT_PASS_NO_PROVIDER_CALLS",
        "parents":[{"label":p["label"],"idea_id":p["idea_id"]} for p in experiment["parents"]],
        "matched_arms":list(ARMS),"max_paid_logical_calls":4,
        "native_task_count":2,"official_scheduler_executed":False,
        "production_population_mutated":False,"new_provider_calls":0,
    })
    print(json.dumps({"status":"M8_PREPARED_NATIVE_G6_NO_API_CALLS",
       "experiment_id":experiment["experiment_id"],"out":str(args.out_dir),"planned_paid_calls":4},indent=2))


def preview(args):
    cfg=checked_plan(args.plan)
    plan,par,nodes=native_components(cfg)
    for arm in ARMS:
        for parent in cfg["parents"]:
            prompt=make_prompt(cfg=cfg,parent=parent,arm=arm)
            print(f"{arm} | {parent['label']} | task={prompt['task_id']} | prompt_chars={len(prompt['user'])}")
    print("M8_MATCHED_G6_NATIVE_PLAN_AND_PROMPT_PREVIEW_PASS_NO_PROVIDER_CALLS")


def _paid_output(raw, prompt, cfg, parent, model, finish_reason, usage):
    draft=parse_draft(raw,parent,cfg)
    return draft.model_dump(mode="json")


def paid(args):
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("Explicit paid approval is required")
    if args.arm not in ARMS or not 1<=args.max_new_calls<=2:
        raise ValueError("Arm or max call count invalid")
    cfg=checked_plan(args.plan); root=args.plan.parent
    if not os.getenv(args.api_key_env):raise RuntimeError(f"No API key {args.api_key_env}")
    native_components(cfg) # Validate native plan before cost.
    # Once any receipt or uncertain attempt exists, no model changes or implicit reruns.
    lock_path=root/"M8_MODEL_LOCK.json"
    lock={"model":args.model,"max_completion_tokens":args.max_output_tokens,
          "temperature":0,"base_url":args.base_url or "https://openrouter.ai/api/v1"}
    if lock_path.is_file():
        if read(lock_path)!=lock:raise ValueError("Model/limit drift violates matched trial")
    else:write_new(lock_path,lock)
    entries=receipt_paths(root,cfg)
    for arm,parent,receipt,draft in entries:
        if receipt.exists() != draft.exists():
            raise RuntimeError(f"Unresolved or incomplete receipt (never auto-recall): {receipt}")
        intent=root/"intents"/arm/(parent["label"]+".json")
        if intent.exists() and not receipt.exists():
            raise RuntimeError(f"Prior attempt may have reached provider; inspect before rerunning: {intent}")
    eligible=[(parent,receipt,draft) for arm,parent,receipt,draft in entries if arm==args.arm and not receipt.is_file()]
    if not eligible:
        print(json.dumps({"status":"ARM_ALREADY_DONE_NO_CALLS","arm":args.arm,"new_provider_calls":0}))
        return
    # Always initialize transport only after zero-cost safety checks.
    from openai import OpenAI
    client=OpenAI(api_key=os.environ[args.api_key_env],base_url=lock["base_url"],timeout=180,max_retries=0)
    calls=0
    for parent,receipt,draft in eligible[:args.max_new_calls]:
        prompt=make_prompt(cfg=cfg,parent=parent,arm=args.arm)
        intent=root/"intents"/args.arm/(parent["label"]+".json")
        write_new(intent,{"status":"PROVIDER_CALL_ATTEMPTED_NO_AUTO_RETRY",
            "model":args.model,"prompt_sha256":digest(prompt),"arm":args.arm,
            "label":parent["label"],"new_calls_authorized":True})
        try:
            reply=client.chat.completions.create(
                model=args.model,messages=[{"role":"system","content":prompt["system"]},
                                           {"role":"user","content":prompt["user"]}],
                temperature=0,max_tokens=args.max_output_tokens)
            calls+=1
        except Exception as exc:
            print(f"M8_PROVIDER_ERROR_INTENT_PRESERVED_{parent['label']}: {type(exc).__name__}: {exc}")
            raise
        raw=reply.choices[0].message.content or ""
        fin=reply.choices[0].finish_reason
        usage=reply.usage.model_dump(mode="json") if reply.usage else None
        write_new(receipt,{"arm":args.arm,"label":parent["label"],
            "prompt_sha256":digest(prompt),"model":args.model,
            "finish_reason":fin,"raw_response":raw,"usage":usage,
            "positive_premise_authority":False,"scientific_truth_authority":False})
        if fin!="stop":
            raise RuntimeError(f"M8_RAW_RECEIPT_SAVED_NONSTOP_FINISH:{parent['label']}:{fin}. No re-call.")
        try:
            parsed=_paid_output(raw,prompt,cfg,parent,args.model,fin,usage)
        except Exception as exc:
            raise RuntimeError(f"M8_RECEIPT_SAVED_DRAFT_INVALID:{parent['label']}:{exc}. No re-call.") from exc
        write_new(draft,parsed)
        print(json.dumps({"arm":args.arm,"label":parent["label"],
                          "status":"VALID_DRAFT_AND_RAW_RECEIPT_SAVED","candidate_count":len(parsed["candidates"])},ensure_ascii=False))
    print(json.dumps({"status":"M8_MATCHED_G6_PAID_ARM_PROGRESS","arm":args.arm,
                      "new_provider_calls_this_command":calls},indent=2))


class SavedDraftBackend:
    """No provider; returns immutable cached paid drafts to actual native G6 executor."""
    def __init__(self, root, arm, cfg):
        self.root=root;self.arm=arm;self.cfg=cfg
        self.by_task={p["task_id"]:p for p in cfg["parents"]}
        self.calls=[]

    def generate(self,prompt):
        from pipeline_core.discovery.research_idea_offspring_execution import (
            GenerationalOffspringBatchDraft, OffspringGeneration)
        parent=self.by_task[prompt.task_id]
        path=self.root/"drafts"/self.arm/(parent["label"]+".json")
        draft=GenerationalOffspringBatchDraft.model_validate(read(path))
        self.calls.append(prompt.task_id)
        return OffspringGeneration(draft=draft,input_tokens=0,output_tokens=0)

    def repair(self,*a,**kw):raise RuntimeError("M8_SEMANTIC_RETRIES_DISABLED")


def replay(args):
    cfg=checked_plan(args.plan);root=args.plan.parent
    entries=receipt_paths(root,cfg)
    if any(not r.exists() or not d.exists() for _,_,r,d in entries):
        raise RuntimeError("All 4 bounded provider receipts and drafts required")
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import execute_epistemic_generation
    plan,parallel,nodes=native_components(cfg)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Never overwrite a native comparison output directory")
    staged={}
    for arm in ARMS:
        backend=SavedDraftBackend(root,arm,cfg)
        native,prompts=execute_epistemic_generation(
            plan=plan,parallel_report=parallel,parent_by_id=nodes,
            research_question="M8_G6_MATCHED_SCIENTIFIC_FEEDBACK_SEARCH_NO_CLAIM_AUTHORITY",
            backend=backend,semantic_retry_limit=0)
        if len(backend.calls)!=2 or set(backend.calls)!={p["task_id"] for p in cfg["parents"]}:
            raise RuntimeError("Native executor did not execute exact matched G6 tasks")
        if native.generation_index!=6 or native.raw_offspring_count>2:
            raise RuntimeError("Native G6 budget breach")
        staged[arm]=native.model_dump(mode="json")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:write_new(args.out_dir/f"M8_G6_NATIVE_{arm}.json",staged[arm])
    create_blind_packet(cfg,staged,args.out_dir)
    comparison={"status":"M8_MATCHED_G6_NATIVE_REPLAY_AND_BLIND_PACKET_READY",
        "source_experiment_id":cfg["experiment_id"],
        "per_arm_source_parents":2,"paired_provider_receipts":4,
        "arms":{arm:{"native_genuine_child_count":row["genuine_child_count"],
            "raw_offspring_count":row["raw_offspring_count"],
            "disposition_counts":row["disposition_counts"],
            "compile_failures":[{"task_id":r["task_id"],"issues":r["compile_issue_codes"],"decision":r["decision"]} for r in row["run_records"]]}
            for arm,row in staged.items()},
        "warning":"Native semantic DIFFERENT_IDEA is not science novelty. Blind human-science assessment pending.",
        "official_scheduler_executed":False,"scientific_truth_authority":False,
        "positive_premise_authority":False,"novelty_certified":False,
        "canonical_graph_mutated":False,"production_selection_authority":False,
        "new_provider_calls_in_replay":0}
    write_new(args.out_dir/"M8_OPERATIONAL_COMPARISON.json",comparison)
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
    print("M8_BLIND_PACKET="+str(args.out_dir/"M8_BLINDED_SCIENCE_PACKET.json"))


def scored(args):
    result=score_blind(args.reviewed_csv,args.unblind_key,args.out)
    print(json.dumps({"status":result["status"],"out":str(args.out)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare");p.add_argument("--g5",type=Path,required=True)
    p.add_argument("--m7-report",type=Path,required=True)
    p.add_argument("--m7-plan",type=Path,required=True)
    p.add_argument("--reviewed",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.set_defaults(func=prepare)
    p=sub.add_parser("preview");p.add_argument("--plan",type=Path,required=True);p.set_defaults(func=preview)
    p=sub.add_parser("generate-paid");p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--model",required=True);p.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    p.add_argument("--base-url",default="https://openrouter.ai/api/v1")
    p.add_argument("--max-new-calls",type=int,default=2)
    p.add_argument("--max-output-tokens",type=int,default=4500)
    p.add_argument("--allow-paid",action="store_true");p.add_argument("--i-authorize-new-llm-calls",action="store_true")
    p.set_defaults(func=paid)
    p=sub.add_parser("replay-native");p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.set_defaults(func=replay)
    p=sub.add_parser("score-blind");p.add_argument("--reviewed-csv",type=Path,required=True)
    p.add_argument("--unblind-key",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.set_defaults(func=scored)
    args=parser.parse_args();args.func(args)

if __name__=="__main__":main()
