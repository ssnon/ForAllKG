"""Native FAKG W08-lineage literature-research Work with real OpenAlex search."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import scientific_literature_work as a
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as lineage
from pipeline_core.discovery.scientific_research_control import research_work_v18_feedback_runner as native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('verify','prepare-work','preview','next','recover','finish','commit-native','native-inspect','next-decision'):
        p=sub.add_parser(name);p.add_argument('--out',required=True,type=Path)
        if name in ('next','next-decision'):p.add_argument('--allow-paid',action='store_true')
    p=sub.add_parser('source-decision');p.add_argument('--reference-native',required=True,type=Path)
    p.add_argument('--allow-paid',action='store_true')
    p=sub.add_parser('prepare');p.add_argument('--reference-native',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--no-literature',action='store_true',help='Control arm: forbid new external literature retrieval')
    x=parser.parse_args()
    if x.cmd=='prepare':r=a.prepare(x.reference_native,x.out,literature_enabled=not x.no_literature)
    elif x.cmd=='source-decision':
        context=native.verify(x.reference_native)
        pending=context['state'].get('pending_work')
        if pending and pending['status']=='REQUESTED':
            r={'status':'SOURCE_ALREADY_HAS_REQUESTED_WORK_NO_CALL','work_id':pending['work_id'],
               'work_kind':pending['request']['work_kind'],'question':pending['request']['research_question']}
        elif context['state']['ended']:
            r={'status':'SOURCE_RESEARCH_ALREADY_ENDED_NO_CALL'}
        else:
            r=native.decision(x.reference_native,allow_paid=x.allow_paid,max_new_calls=1)
    elif x.cmd=='verify':r=a.verify(x.out)
    elif x.cmd=='prepare-work':r=a.prepare_work(x.out)
    elif x.cmd=='preview':r=a.preview(x.out)
    elif x.cmd=='next':r=a.next_step(x.out,allow_paid=x.allow_paid)
    elif x.cmd=='recover':r=a.recover(x.out)
    elif x.cmd=='finish':r=a.finish(x.out)
    elif x.cmd=='commit-native':r=a.commit_native(x.out)
    elif x.cmd=='native-inspect':r=native.inspect(lineage.native_dir(x.out))
    elif x.cmd=='next-decision':
        a.verify(x.out)
        r=native.decision(lineage.native_dir(x.out),allow_paid=x.allow_paid,max_new_calls=1)
        a.verify(x.out)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
