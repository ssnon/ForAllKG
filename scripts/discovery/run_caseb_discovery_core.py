"""Case B open problem native FAKG scientific discovery research loop."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_discovery_core.engine import (
    OpenRouter, export_native_roots, import_roots_to_native_sis, prepare, read_json, run, verify, sha, write_new, inspect_failed_baseline, recover_failed_baseline)


def main():
    p=argparse.ArgumentParser(description="FAKG open problem scientific discovery research loop (experimental)")
    sub=p.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("prepare")
    b.add_argument("--problem",required=True,type=Path)
    b.add_argument("--arm",required=True,choices=["ROOT_NO_KG","ROOT_KG","LINEAGE_KG","LLM_BASELINE"])
    b.add_argument("--kg-packet",type=Path)
    b.add_argument("--parent-execution",type=Path)
    b.add_argument("--out",required=True,type=Path)
    b.add_argument("--model",default="openai/gpt-5.6-luna")
    b.add_argument("--temperature",type=float,default=0.55)
    b.add_argument("--max-tokens",type=int,default=5000)
    b.add_argument("--rounds",type=int,default=2)
    b.add_argument("--enable-web-retrieval",action="store_true")
    v=sub.add_parser("verify");v.add_argument("--out",required=True,type=Path)
    r=sub.add_parser("run");r.add_argument("--out",required=True,type=Path)
    r.add_argument("--allow-paid",action="store_true")
    r.add_argument("--max-calls",required=True,type=int)
    r.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    r.add_argument("--base-url",default="https://openrouter.ai/api/v1")
    e=sub.add_parser("export");e.add_argument("--out",required=True,type=Path)
    e.add_argument("--native",action="store_true",help="Generate native ResearchIdeaNode-shaped shadow export")
    n=sub.add_parser("import-sis");n.add_argument("--out",required=True,type=Path)
    n.add_argument("--context",required=True,type=Path,help="Genuine Case B FAKG hypothesis.context.json")
    ins=sub.add_parser("inspect-failed-baseline");ins.add_argument("--out",required=True,type=Path);ins.add_argument("--stage",default="baseline_01")
    rec=sub.add_parser("recover-failed-baseline");rec.add_argument("--out",required=True,type=Path);rec.add_argument("--stage",default="baseline_01")
    rec.add_argument("--quarantine-invalid-matrices",action="store_true",required=True)
    a=p.parse_args()
    if a.cmd=="prepare":
        result=prepare(problem_file=a.problem, arm=a.arm, out=a.out,
                       graph_packet=a.kg_packet,parent_execution=a.parent_execution,
                       model=a.model, temperature=a.temperature,max_tokens=a.max_tokens,
                       rounds=a.rounds,allow_web=a.enable_web_retrieval)
    elif a.cmd=="verify":
        result={"status":"INPUT_SHA_VERIFIED", "manifest":verify(a.out),"paid_calls":0}
    elif a.cmd=="import-sis":
        result=import_roots_to_native_sis(out=a.out,context_file=a.context)
    elif a.cmd=="inspect-failed-baseline":
        result=inspect_failed_baseline(a.out,stage=a.stage)
    elif a.cmd=="recover-failed-baseline":
        result=recover_failed_baseline(a.out,stage=a.stage)
    elif a.cmd=="run":
        backend=OpenRouter(api_key_env=a.api_key_env,base_url=a.base_url)
        result=run(a.out,backend,allow_paid=a.allow_paid,max_calls=a.max_calls)
    else:
        trace=read_json(a.out/"research_trace.json")
        if a.native:
            result=export_native_roots(a.out)
        else:
            # The blind export does not reveal arm, scores, or research question phrasing.
            rows=[]
            for step in trace["rounds"]:
                mechanisms=step.get("proposals",{}).get("mechanisms",[]) if "proposals" in step else step.get("mechanisms",[])
                for idea in mechanisms:
                    rows.append({"blind_mechanism_id":"M"+sha(idea)[:12],"mechanism":idea})
            result={"schema_version":"fakg-blind-discovery-review-v1", "hypotheses":rows,
                    "scientific_authority":False}
            write_new(a.out/"blind_review.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
