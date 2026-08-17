from __future__ import annotations
import argparse, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    PAPER_REVIEW_SYSTEM_PROMPT, FINAL_ADJUDICATOR_SYSTEM_PROMPT,
    FreshCPaperReview, FreshCFinalAdjudication,
)
from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import (
    C1AR1_CORPUS, atomic_json, build_final_prompt, build_target_boundaries,
    canonical_json_sha256, format_paper_prompt, load_object, review_payload_sha,
    validate_final_against_reviews, validate_frozen_lineage,
    validate_pages_manifest, validate_review_grounding,
)
from dac_her.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR, DEFAULT_PROTOCOL_PATH, DEFAULT_RUN_DIR,
    DEFAULT_SCHEMA_QUALIFICATION_DIR, final_recovery_transport_schema,
    paper_recovery_transport_schema, validate_parent_failure_state,
    validate_protocol, validate_runtime_env,
)

MODEL = "openai/gpt-5.6-luna"

def root():
    return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"], text=True).strip())

def client(env):
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=env["base_url"],
                  timeout=300.0, max_retries=0)

def call(c, model_cls, schema, system, user, max_tokens):
    r = c.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        seed=0, max_tokens=max_tokens,
        response_format={"type":"json_schema","json_schema":{
            "name":model_cls.__name__,"strict":True,"schema":schema}},
        extra_body={"reasoning":{"effort":"medium","exclude":True},
                    "provider":{"only":["openai"],"allow_fallbacks":False,
                                "require_parameters":True,"data_collection":"deny"}},
    )
    if r.model != MODEL:
        raise RuntimeError("Recovery served-model drift")
    parsed = model_cls.model_validate_json(r.choices[0].message.content)
    u = r.usage
    return parsed, {"requested_model":MODEL,"served_model":r.model,
                    "input_tokens":u.prompt_tokens if u else None,
                    "output_tokens":u.completion_tokens if u else None,
                    "total_tokens":u.total_tokens if u else None,
                    "finish_reason":r.choices[0].finish_reason}

def qualify():
    rt = root(); p = validate_protocol(rt / DEFAULT_PROTOCOL_PATH)
    validate_parent_failure_state(rt); env = validate_runtime_env()
    out = rt / DEFAULT_SCHEMA_QUALIFICATION_DIR
    if out.exists(): raise FileExistsError("Recovery schema qualification exists")
    paper, usage = call(
        client(env), FreshCPaperReview, paper_recovery_transport_schema(),
        PAPER_REVIEW_SYSTEM_PROMPT,
        "Synthetic recovery-schema qualification only. No Fresh-C text or real "
        "scientific hypothesis content is present. Return reserve_index=1, "
        "canonical_id='doi:synthetic-recovery', materialization_mode='DIRECT_ORIGINAL'. "
        "Assess both required hypothesis IDs as DIRECT_PRIOR_ART with one synthetic "
        "evidence locator each on page 1. Set every verbatim_quote to null and all "
        "guard flags false.",
        3500,
    )
    if any(ev.verbatim_quote is not None for a in paper.assessments for ev in a.evidence):
        raise RuntimeError("Recovery schema did not force quote null")
    q = {"schema_version":"sers-fresh-c-c1b2-r1-schema-qualification-v1",
         "protocol_id":p["protocol_id"],"protocol_sha256":p["protocol_sha256"],
         "recovery_schema_adapter_id":p["recovery_schema_adapter_id"],
         "paper_recovery_transport_schema_sha256":p["paper_recovery_transport_schema_sha256"],
         "requested_model":MODEL,"served_model":usage["served_model"],
         "paper_schema_passed":True,"verbatim_quote_returned_null":True,
         "usage":usage,"network_calls":1,"llm_calls":1,
         "fresh_c_scientific_text_used":False,
         "scientific_hypothesis_text_used":False,
         "scientific_adjudication_performed":False,
         "recovery_live_authorized":False,"stop":True}
    q["qualification_sha256"] = canonical_json_sha256(q)
    q["qualification_id"] = "sers_fresh_c_c1b2_r1_schema_qualification_v1:" + q["qualification_sha256"][:20]
    out.mkdir(parents=True, exist_ok=False); atomic_json(out/"qualification_result.json", q)
    print("Fresh-C C1B.2-R1 quote-null schema qualification")
    print(f"Qualification ID: {q['qualification_id']}")
    print(f"Qualification SHA256: {q['qualification_sha256']}")
    print("Recovery paper schema passed: True")
    print("verbatim_quote returned null: True")
    print("Network/LLM calls: 1/1")
    print("Fresh-C scientific text used: False")
    print("Scientific hypothesis text used: False")
    print("Scientific adjudication performed: False")
    print("Recovery live authorized: False")
    print("STOP: True")
    return 0

def state(rt):
    from scripts.verify_sers_fresh_c_c1b2_r1_recovery_protocol_freeze_v1 import main as verify
    verify(); p=validate_protocol(rt/DEFAULT_PROTOCOL_PATH)
    f=validate_parent_failure_state(rt); env=validate_runtime_env()
    lineage=validate_frozen_lineage(rt)
    targets=build_target_boundaries(lineage["r2_report"])
    if (rt/DEFAULT_RUN_DIR).exists(): raise RuntimeError("Recovery run directory exists")
    return p,f,env,targets

def preflight():
    rt=root(); p,f,env,targets=state(rt)
    print("Fresh-C C1B.2-R1 guarded recovery preflight")
    print(f"Protocol ID: {p['protocol_id']}")
    print("Parent failure: VERIFIED")
    print("Original completed reviews: 0/25")
    print("Original scientific calls attempted: 1")
    print("Scientific prompts/targets unchanged: True")
    print("Failed parent response reused: False")
    print("Exact frozen papers: 25/25")
    print("Recovery order: 1..25")
    print("Verbatim quote evidence enabled: False")
    print("Expected recovery LLM/network calls: 26/26")
    print("Recovery live authorized: False")
    print("Automatic post-recovery transition: False")
    print("STOP: True")
    print("Preflight: PASS")
    return 0

def live():
    rt=root(); p,f,env,targets=state(rt)
    d=rt/DEFAULT_RUN_DIR; d.mkdir(parents=True,exist_ok=False)
    rd=d/"paper_reviews"; rd.mkdir()
    atomic_json(d/"C1B2_R1_RECOVERY_STARTED.json",{
        "schema_version":"sers-fresh-c-c1b2-r1-recovery-started-v1",
        "started_at_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_id":p["protocol_id"],"protocol_sha256":p["protocol_sha256"],
        "parent_scientific_read_already_occurred":True,
        "this_is_new_fresh_reserve_consumption":False,
        "same_recovery_epoch_rerun_allowed":False,
        "failure_restores_freshness":False,
        "failure_authorizes_tuning_on_fresh_c":False,
        "failed_parent_response_reused":False,
        "verbatim_quote_evidence_enabled":False,"stop":True})
    c=client(env); completed=[]; reviews=[]; rows=[]; calls=0
    try:
        corpus=load_object(rt/C1AR1_CORPUS); by={x["reserve_index"]:x for x in corpus["records"]}
        for rec in f["records"]:
            pages=validate_pages_manifest(rt,by[rec["reserve_index"]])
            prompt=format_paper_prompt(target_boundaries=targets,
                reserve_index=rec["reserve_index"],canonical_id=rec["canonical_id"],
                materialization_mode=rec["materialization_mode"],pages_manifest=pages)
            calls += 1
            review,usage=call(c,FreshCPaperReview,paper_recovery_transport_schema(),
                              PAPER_REVIEW_SYSTEM_PROMPT,prompt,3500)
            if any(ev.verbatim_quote is not None for a in review.assessments for ev in a.evidence):
                raise ValueError("Recovery quote-null invariant violated")
            validate_review_grounding(review,expected_record=rec,pages_manifest=pages)
            payload={"schema_version":"sers-fresh-c-c1b2-r1-paper-review-record-v1",
                     "reserve_index":rec["reserve_index"],"canonical_id":rec["canonical_id"],
                     "review":review.model_dump(mode="json"),
                     "review_sha256":review_payload_sha(review),"usage":usage,
                     "verbatim_quote_evidence_enabled":False,
                     "failed_parent_response_reused":False}
            payload["record_sha256"]=canonical_json_sha256(payload)
            path=rd/f"reserve_c_{rec['reserve_index']:03d}.json"; atomic_json(path,payload)
            completed.append(rec["reserve_index"]); reviews.append(review)
            rows.append({"reserve_index":rec["reserve_index"],"canonical_id":rec["canonical_id"],
                         "record_path":str(path.relative_to(rt)),
                         "record_sha256":payload["record_sha256"]})
            print(f"[C1B.2-R1] paper {rec['reserve_index']:02d}/25 reviewed | {rec['canonical_id']}")
        if completed != list(range(1,26)): raise RuntimeError("Recovery order incomplete")
        calls += 1
        final,usage=call(c,FreshCFinalAdjudication,final_recovery_transport_schema(),
                         FINAL_ADJUDICATOR_SYSTEM_PROMPT,
                         build_final_prompt(target_boundaries=targets,reviews=reviews),5000)
        validate_final_against_reviews(final,reviews)
        fp={"schema_version":"sers-fresh-c-c1b2-r1-final-adjudication-v1",
            "adjudication":final.model_dump(mode="json"),"usage":usage,
            "recovery_after_parent_validation_failure":True,
            "new_fresh_reserve_claimed":False,"failed_parent_response_reused":False,
            "verbatim_quote_evidence_enabled":False}
        fp["record_sha256"]=canonical_json_sha256(fp); atomic_json(d/"final_adjudication.json",fp)
        run={"schema_version":"sers-fresh-c-c1b2-r1-scientific-recovery-run-v1",
             "protocol_id":p["protocol_id"],"protocol_sha256":p["protocol_sha256"],
             "parent_c1b2_failed":True,"parent_completed_paper_reviews":0,
             "parent_scientific_call_attempts":1,
             "recovery_after_validation_failure":True,"new_fresh_reserve_claimed":False,
             "failed_parent_response_reused":False,"paper_review_records":rows,
             "paper_review_calls":25,"final_adjudication_calls":1,
             "recovery_llm_calls":calls,"recovery_network_calls":calls,
             "all_25_papers_processed":True,"scientific_adjudication_performed":True,
             "external_literature_used":False,"count_threshold_used":False,
             "hypothesis_rewrite_performed":False,"hypothesis_upgrade_performed":False,
             "h2_resurrected":False,"verbatim_quote_evidence_enabled":False,
             "same_recovery_epoch_rerun_allowed":False,
             "automatic_post_recovery_transition_allowed":False,"stop":True}
        run["run_sha256"]=canonical_json_sha256(run)
        run["run_id"]="sers_fresh_c_c1b2_r1_scientific_recovery_run_v1:"+run["run_sha256"][:20]
        atomic_json(d/"run_manifest.json",run)
        atomic_json(d/"C1B2_R1_RECOVERY_COMPLETE.json",
                    {"run_id":run["run_id"],"run_sha256":run["run_sha256"],
                     "all_25_papers_processed":True,"scientific_adjudication_performed":True,
                     "new_fresh_reserve_claimed":False,"same_recovery_epoch_rerun_allowed":False,
                     "automatic_post_recovery_transition_allowed":False,"stop":True})
        print("Fresh-C C1B.2-R1 scientific recovery complete")
        print(f"Run ID: {run['run_id']}"); print(f"Run SHA256: {run['run_sha256']}")
        print("Paper reviews: 25/25"); print("Final adjudication: True")
        print(f"Recovery LLM/network calls: {calls}/{calls}")
        print("New Fresh-C reserve claimed: False")
        print("Failed parent response reused: False")
        print("Verbatim quote evidence enabled: False")
        print("Same recovery-epoch rerun allowed: False")
        print("Automatic post-recovery transition: False"); print("STOP: True")
        return 0
    except Exception as exc:
        atomic_json(d/"C1B2_R1_RECOVERY_FAILED.json",
                    {"schema_version":"sers-fresh-c-c1b2-r1-recovery-failed-v1",
                     "failed_at_utc":datetime.now(timezone.utc).isoformat(),
                     "protocol_id":p["protocol_id"],"protocol_sha256":p["protocol_sha256"],
                     "error_type":type(exc).__name__,"error_summary":str(exc)[:1000],
                     "completed_reserve_indexes":completed,"completed_paper_reviews":len(completed),
                     "recovery_llm_call_attempts":calls,"recovery_network_call_attempts":calls,
                     "new_fresh_reserve_claimed":False,"same_recovery_epoch_rerun_allowed":False,
                     "failure_restores_freshness":False,
                     "failure_authorizes_tuning_on_fresh_c":False,
                     "automatic_post_recovery_transition_allowed":False,"stop":True})
        print("Fresh-C C1B.2-R1 scientific recovery: FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Completed paper reviews: {len(completed)}/25")
        print(f"Recovery LLM/network attempts: {calls}/{calls}")
        print("Same recovery-epoch rerun allowed: False"); print("STOP: True")
        raise

def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic-schema-qualification",action="store_true")
    g.add_argument("--preflight",action="store_true")
    g.add_argument("--confirm-recovery",action="store_true")
    a=p.parse_args()
    if a.synthetic_schema_qualification:return qualify()
    if a.preflight:return preflight()
    return live()

if __name__=="__main__": raise SystemExit(main())
