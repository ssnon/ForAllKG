"""Add-only, in-process compatibility for model-specific CTMC protocol groups.

`fork` replays ONLY the last rejected CTMC tool call into a NEW research root.
The original research/receipts/raw are unchanged. All other commands run the
existing scientific work runner with a scoped CTMC adapter. No new paid call
unless `next --allow-paid` is explicitly invoked.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

from pipeline_core.discovery.scientific_research_control import scientific_execution_probe as probe
from pipeline_core.discovery.scientific_research_control import scientific_execution_work as work


BASE_EVALUATE=probe.evaluate

def group_compatible_evaluate(request):
    """Model-absent protocol controls are inert, but wholly unknown keys still fail.

    The adapter does NOT infer scientific mechanisms, rate laws, or emissions.
    """
    original_generator=probe.generator
    original_evaluate=BASE_EVALUATE
    if isinstance(request,dict) and isinstance(request.get('models'),list):
        # Retain original model and protocol validation, including duplicate edges.
        validated_models=[probe.parse_model(m) for m in request['models']]
        group_union={g for m in validated_models for _,_,_,g in m['edges']}
        for p in request.get('protocols',[]):
            parsed=probe.parse_protocol(p)
            for segment in parsed['segments']:
                unknown=set(segment['multipliers'])-group_union
                if unknown:
                    raise ValueError('PROTOCOL_REFERENCES_UNKNOWN_GROUP:'+str(sorted(unknown)))
    else:
        return original_evaluate(request)

    def generator_for_union(model,multipliers):
        model_groups={g for _,_,_,g in model['edges']}
        applicable={k:v for k,v in multipliers.items() if k in model_groups}
        return original_generator(model,applicable)
    try:
        probe.generator=generator_for_union
        return original_evaluate(request)
    finally:
        probe.generator=original_generator


@contextmanager
def compatible_scope():
    original=probe.evaluate
    probe.evaluate=group_compatible_evaluate
    try:
        yield
    finally:
        probe.evaluate=original


def _h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fork(source:Path,target:Path):
    source=Path(source).expanduser().resolve()
    target=Path(target).expanduser().resolve()
    if target.exists():raise FileExistsError('FORK_TARGET_ALREADY_EXISTS')
    if target==source or source in target.parents or target in source.parents:
        raise ValueError('FORK_PATHS_MUST_BE_SEPARATE')
    before=work.verify(source)
    if before['incomplete'] or before['finalized'] or before['last_action']!='CONTINUE':
        raise ValueError('SOURCE_MUST_BE_COMPLETE_NONFINAL_CONTINUE')
    step=before['steps']
    p=source/f'step_{step:04d}'
    norm=work.read(p/'normalized.json')
    tool=norm.get('tool_request')
    original_result=work.read(p/'tool_result.json')['tool_result']
    if not tool or tool.get('kind')!='ctmc_transient':raise ValueError('NO_CTMC_TO_REPLAY')
    if not original_result or original_result.get('status')!='TOOL_REQUEST_REJECTED':
        raise ValueError('LAST_TOOL_RESULT_NOT_REJECTED')
    if not str(original_result.get('reason','')).startswith('PROTOCOL_REFERENCES_UNKNOWN_GROUP:'):
        raise ValueError('NOT_A_GROUP_COMPAT_REJECTION')
    original_raw_sha=_h(p/'raw_response.json')
    original_result_sha=_h(p/'tool_result.json')
    # Evaluate prior to cloning: no side effects, no LLM, no filesystem mutation.
    with compatible_scope():
        computed=work.dispatch(tool,work.rows(source)[0][:-1])
    if computed.get('status')!='COMPUTED_CONDITIONAL_MODEL_RESULT':
        raise RuntimeError('REPLAY_DID_NOT_COMPUTE:'+str(computed.get('reason')))
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,target,symlinks=False)
    try:
        dst=target/f'step_{step:04d}'
        # Update ONLY the fork's tool-result and the consequent receipt hash.
        (dst/'tool_result.json').write_text(json.dumps({'tool_result':computed},indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
        receipt=work.read(dst/'receipt.json')
        receipt['tool_result_sha256']=_h(dst/'tool_result.json')
        (dst/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
        provenance={
            'type':'REPLAYED_MODEL_SPECIFICATION_IN_FORK',
            'scientific_validation':False,
            'paid_calls_for_replay':0,
            'original_run_untouched':True,
            'source':str(source),
            'source_steps':step,
            'replay_step':step,
            'source_tool_status':original_result['status'],
            'source_tool_reason':original_result['reason'],
            'source_raw_sha256':original_raw_sha,
            'fork_raw_sha256':_h(dst/'raw_response.json'),
            'source_tool_result_sha256':original_result_sha,
            'fork_tool_result_sha256':_h(dst/'tool_result.json'),
            'interpretation':'A model-specific protocol group is a no-op where that group is absent. Unknown-to-all groups remain invalid. Results are theoretical consequences of submitted assumptions, not experimental findings.',
        }
        (target/'ctmc_replay_provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
        verified=work.verify(target)
        if work.verify(source)!=before or _h(p/'raw_response.json')!=original_raw_sha or _h(p/'tool_result.json')!=original_result_sha:
            raise RuntimeError('SOURCE_CHANGED_DURING_REPLAY')
    except BaseException:
        shutil.rmtree(target)
        raise
    return {'status':'CTMC_REPLAY_FORK_VERIFIED_NO_PAID_CALL',
            'source_untouched':True,'replayed_step':step,
            'computed_tool_status':computed['status'],
            'fork':str(target),'fork_steps':verified['steps'],
            'original_result_remains_rejected':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('fork','verify','preview','next','recover','trajectory','finish'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-paid',action='store_true')
    args=parser.parse_args()
    if args.command=='fork':
        if args.source is None:parser.error('fork requires --source')
        result=fork(args.source,args.out)
    else:
        with compatible_scope():
            if args.command=='verify':result=work.verify(args.out)
            elif args.command=='preview':result=work.preview(args.out)
            elif args.command=='next':result=work.next_step(args.out,allow_paid=args.allow_paid)
            elif args.command=='recover':result=work.recover(args.out)
            elif args.command=='trajectory':result=work.trajectory(args.out)
            else:result=work.finish(args.out)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))

if __name__=='__main__':main()
