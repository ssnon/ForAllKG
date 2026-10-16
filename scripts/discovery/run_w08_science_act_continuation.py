"""W08 lineage-preserving Science Act: native Decision, optional research instrument, native WorkProduct."""
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as a
from pipeline_core.discovery.scientific_research_control import research_work_v18_feedback_runner as v18


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ('verify','inspect','choose','prepare-work','preview','next','recover','finish','commit-native','next-decision','native-inspect','native-frozen-read'):
        p=sub.add_parser(name);p.add_argument('--out',required=True,type=Path)
        if name in ('choose','next','next-decision'):
            p.add_argument('--allow-paid',action='store_true')
    p=sub.add_parser('prepare');p.add_argument('--reference-w08',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    x=ap.parse_args();out=x.out
    if x.command=='prepare':r=a.prepare(x.reference_w08,out)
    elif x.command=='verify':r=a.verify(out)
    elif x.command=='inspect':r=a.inspect(out)
    elif x.command=='choose':r=a.choose(out,allow_paid=x.allow_paid)
    elif x.command=='prepare-work':r=a.prepare_work(out)
    elif x.command=='preview':r=a.preview(out)
    elif x.command=='next':r=a.next_step(out,allow_paid=x.allow_paid)
    elif x.command=='recover':r=a.recover(out)
    elif x.command=='finish':r=a.finish(out)
    elif x.command=='commit-native':r=a.commit_native(out)
    elif x.command=='native-inspect':r=v18.inspect(a.native_dir(out))
    elif x.command=='native-frozen-read':
        r=v18.execute(a.native_dir(out));a.verify(out)
    elif x.command=='next-decision':
        a.verify(out)
        r=v18.decision(a.native_dir(out),allow_paid=x.allow_paid,max_new_calls=1)
        a.verify(out)
    else:raise AssertionError(x.command)
    print(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
