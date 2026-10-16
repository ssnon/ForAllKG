"""Native-lineage claim research: selected native Work, autonomous tools, native WorkProduct."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import scientific_claim_development as a
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as old
from pipeline_core.discovery.scientific_research_control import research_work_v18_feedback_runner as native


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for name in ('verify','prepare-work','preview','next','recover','finish','commit-native','native-inspect','next-decision'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name in ('next','next-decision'):p.add_argument('--allow-paid',action='store_true')
    p=sub.add_parser('prepare');p.add_argument('--reference-native',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    x=ap.parse_args()
    if x.cmd=='prepare':r=a.prepare(x.reference_native,x.out)
    elif x.cmd=='verify':r=a.verify(x.out)
    elif x.cmd=='prepare-work':r=a.prepare_work(x.out)
    elif x.cmd=='preview':r=a.preview(x.out)
    elif x.cmd=='next':r=a.next_step(x.out,allow_paid=x.allow_paid)
    elif x.cmd=='recover':r=a.recover(x.out)
    elif x.cmd=='finish':r=a.finish(x.out)
    elif x.cmd=='commit-native':r=a.commit_native(x.out)
    elif x.cmd=='native-inspect':r=native.inspect(old.native_dir(x.out))
    elif x.cmd=='next-decision':
        a.verify(x.out)
        r=native.decision(old.native_dir(x.out),allow_paid=x.allow_paid,max_new_calls=1)
        a.verify(x.out)
    else:raise AssertionError(x.cmd)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
