
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.sers_closeout_v8 import build_closeout,build_n9_projection,compare_existing_certification,render_md

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def main():
    q=argparse.ArgumentParser()
    for name in ("portfolio","aggregation_a","aggregation_b","readiness","selection_audit","stability","source_binding","sentinel","search_root","output_dir"):
        q.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    a=q.parse_args()
    c=build_closeout(load(a.portfolio),load(a.aggregation_a),load(a.aggregation_b),load(a.readiness),load(a.selection_audit),load(a.stability),load(a.source_binding),load(a.sentinel))
    p=build_n9_projection(c)
    b=compare_existing_certification(p,a.search_root)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write(a.output_dir/"sers_closeout.freeze.json",c)
    write(a.output_dir/"sers_residual_aware_n9_projection.shadow.json",p)
    write(a.output_dir/"sers_n9_baseline_comparison.shadow.json",b)
    (a.output_dir/"SERS_CLOSEOUT_REPORT.md").write_text(render_md(c),encoding="utf-8")
    print("===== SERS CLOSEOUT V8 =====")
    print("states:",c["state_counts"])
    print("authority-ready shadow:",c["authority_ready_shadow_count"])
    print("selection completeness:",c["selection_completeness_pass"])
    print("review stability:",c["review_stability_pass"])
    print("N9 projected:",p["projected_count"],"held:",p["held_count"])
    print("baseline comparison:",b["comparison_state"])
    for r in c["hypotheses"]:
        print(r["hypothesis_id"],"|",r["final_epistemic_state"],"| components=",f'{r["component_backed_count"]}/{r["component_count"]}',"| readiness=",r["authority_readiness_state"])
    print("SHADOW_ONLY=True")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0
if __name__=="__main__": raise SystemExit(main())
