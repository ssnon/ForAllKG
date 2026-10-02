
from __future__ import annotations
import argparse, json
from pathlib import Path
from pipeline_core.discovery.sers_feedback_input_resolution import resolve_inputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root",required=True,type=Path)
    p.add_argument("--abl",required=True,type=Path)
    p.add_argument("--portfolio",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    payload=resolve_inputs(run_root=a.run_root,abl=a.abl,portfolio_path=a.portfolio)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Resolved SERS closed-loop inputs")
    for k,v in payload.items():
        if k not in {"schema_version"}: print(k,":",v)
    return 0
if __name__=="__main__": raise SystemExit(main())
