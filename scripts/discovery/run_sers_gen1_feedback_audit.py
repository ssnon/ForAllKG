
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import audit_gen1

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generation-report",required=True,type=Path)
    p.add_argument("--external-report",required=True,type=Path)
    p.add_argument("--aggregation",required=True,type=Path)
    p.add_argument("--cohort-audit",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    report=audit_gen1(
      generation_report=load(a.generation_report),
      external_report=load(a.external_report),
      aggregation=load(a.aggregation),
      cohort_audit=load(a.cohort_audit),
    )
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("Closed-loop Gen1 audit")
    print("closed_loop_completed:",report["closed_loop_completed"])
    print("accepted_generation_count:",report["accepted_generation_count"])
    print("known_axis_repeat_count:",report["known_axis_repeat_count"])
    print("outcomes:",report["outcome_counts"])
    for row in report["rows"]:
      print(row["source_hypothesis_id"],"|",row["route"],"|",row["outcome"],"| gen1=",row["gen1_hypothesis_id"],"| external=",row["fresh_external_status"],"| residual=",row["residual_dispositions"])
    print("SHADOW_ONLY=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0
if __name__=="__main__": raise SystemExit(main())
