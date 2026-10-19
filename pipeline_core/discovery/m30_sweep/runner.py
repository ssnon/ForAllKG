"""M30 controlled capability sweep. Uses native FAKG Case B ledger/backend;
research prompts are a common experimental adapter, NOT production Research Control.
All interventions are inert JSON, never model-generated executable Python.
"""
from __future__ import annotations
import argparse,csv,hashlib,html,json,os,random,traceback
from collections import defaultdict
from pathlib import Path
from typing import Any,Literal
from pydantic import BaseModel,ConfigDict,Field
from pipeline_core.discovery.scientific_discovery_core import engine as core
from . import oracle

PROTOCOL='m30-capability-sweep-v1'
ARMS=('PLAIN','FAKG_ADAPTER','PLAIN_TOOL','FAKG_ADAPTER_TOOL')
STAGES=('proposal','revision','final')
REPS=2
class Strict(BaseModel):model_config=ConfigDict(extra='forbid')
class Proposal(Strict):
    hypothesis:str=Field(min_length=12)
    substantive_reasoning:str=Field(min_length=12)
    candidate:dict
    null_or_alternative:str=Field(min_length=8)
    proposed_probe:dict=Field(default_factory=dict)
    predicted_probe_result:str=Field(min_length=8)
    limitations:str=Field(min_length=8)
class Revision(Strict):
    what_changed_and_why:str=Field(min_length=12)
    hypothesis:str=Field(min_length=12)
    candidate:dict
    strongest_counterargument:str=Field(min_length=8)
    proposed_probe:dict=Field(default_factory=dict)
    predicted_probe_result:str=Field(min_length=8)
class Final(Strict):
    hypothesis:str=Field(min_length=12)
    candidate:dict
    core_consequence:str=Field(min_length=8)
    strongest_remaining_obstacle:str=Field(min_length=8)
    update_from_evidence:str=Field(min_length=8)
    novelty_or_truth_certified:Literal[False]=False
MODEL={'proposal':Proposal,'revision':Revision,'final':Final}

def filehash(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    core.write_new(p,obj)
def read(p):return core.read_json(p)
def task_dir(out,task,arm,rep):return out/'episodes'/f'{task}__{arm}__r{rep}'
def episode_ids():return [(t,a,r) for t in oracle.TASKS for a in ARMS for r in range(1,REPS+1)]
def _filelocks():return {'oracle_sha256':filehash(Path(oracle.__file__)),'runner_sha256':filehash(Path(__file__)),'native_core_sha256':filehash(Path(core.__file__))}

def prepare(out:Path,model='openai/gpt-5.6-luna',temperature=.7,max_tokens=6000):
    if out.exists():raise FileExistsError('M30_REFUSE_OVERWRITE_SWEEP')
    if not 0<=temperature<=1.5 or not 800<=max_tokens<=20000:raise ValueError('MODEL_SETTINGS_OUTSIDE_RANGE')
    rows=episode_ids();order=list(range(len(rows)));random.Random(20261017).shuffle(order)
    manifest={'schema_version':PROTOCOL,'model':model,'temperature':temperature,'max_tokens':max_tokens,
              'arms':list(ARMS),'tasks':list(oracle.TASKS),'reps':REPS,'stages':list(STAGES),
              'n_episodes':len(rows),'max_paid_calls':len(rows)*len(STAGES),
              'paired_execution_order':[list(rows[i]) for i in order],
              'science_truth_certified':False,'reference': 'SELF_CONTAINED_SYNTHETIC_ORACLES',
              'adapter_disclosure':'FAKG_ADAPTER reuses native FAKG CaseB model ledger/backend and a stage-wise scientific critique pattern, but is NOT the production Research Control / KG / SIS pipeline.',
              'conditions_note':'identical prompt data, model and calls; tool conditions receive up to two chosen experiments; no-tool conditions receive no oracle observations.',
              **_filelocks()}
    out.mkdir(parents=True,exist_ok=False);put(out/'manifest.json',manifest)
    for task,arm,rep in rows:
        p=task_dir(out,task,arm,rep)
        public=oracle.public_problem(task)
        put(p/'problem.json',public)
        put(p/'episode.json',{'task':task,'arm':arm,'rep':rep,'public_problem_sha256':core.sha(public),
                              'prompt_contract':'m30-unified-three-stage-scientific-work-v1','novelty_authority':False})
    return {'status':'M30_PREPARED_NO_PAID_CALLS','episodes':len(rows),'maximum_paid_attempts':len(rows)*3,'out':str(out)}

def _manifest(out):
    m=read(out/'manifest.json')
    if m.get('schema_version')!=PROTOCOL:raise ValueError('M30_INVALID_MANIFEST')
    for k,v in _filelocks().items():
        if m.get(k)!=v:raise ValueError('M30_SOURCE_LOCK_CHANGED:'+k)
    if m['tasks']!=oracle.TASKS or tuple(m['arms'])!=ARMS or m['reps']!=REPS:raise ValueError('M30_DESIGN_CHANGED')
    return m

def _episode(out,task,arm,rep):
    d=task_dir(out,task,arm,rep)
    e=read(d/'episode.json');prob=read(d/'problem.json')
    if e['public_problem_sha256']!=core.sha(prob) or prob!=oracle.public_problem(task):raise ValueError('M30_PUBLIC_PROBLEM_CHANGED')
    if e['task']!=task or e['arm']!=arm or e['rep']!=rep:raise ValueError('M30_EPISODE_MISMATCH')
    return d,prob

def _stage_state(d,stage):
    p=d/'calls'/stage
    receipt=p/'receipt.json'; raw=p/'raw_response.json'; attempt=p/'attempt.json';req=p/'request.json';norm=p/'response.json'
    if receipt.exists():
        if not all(f.is_file() for f in (raw,attempt,req,norm)):raise ValueError('M30_INCOMPLETE_RECEIPT:'+str(p))
        r=read(receipt)
        if (r.get('status')!='COMPLETED' or r.get('request_sha256')!=core.sha(read(req)) or
            r.get('original_raw_response_sha256')!=core.sha(read(raw)) or
            r.get('validated_response_sha256')!=core.sha(read(norm))):raise ValueError('M30_RECEIPT_HASH_MISMATCH:'+str(p))
        MODEL[stage].model_validate(read(norm));return 'VERIFIED'
    if raw.exists():return 'RAW_SAVED_RECOVERY_REQUIRED'
    if attempt.exists():return 'AMBIGUOUS_PAID_ATTEMPT_MANUAL_REVIEW'
    if p.exists() and any(p.iterdir()):return 'PARTIAL_CALL_FILES_MANUAL_REVIEW'
    return 'NOT_STARTED'

def _probe_state(d,stage,arm):
    if 'TOOL' not in arm or stage=='final':return 'NOT_APPLICABLE'
    p=d/'probes'/stage/'probe.json'
    if not p.exists():return 'NOT_STARTED'
    v=read(p);given=read(d/'calls'/stage/'response.json')
    if v.get('request_sha256')!=core.sha({'candidate':given['candidate'],'proposed_probe':given['proposed_probe']}):raise ValueError('PROBE_INPUT_CHANGED')
    actual=oracle.experiment(read(d/'episode.json')['task'],given['proposed_probe'],given['candidate'])
    if v['result']!=actual:raise ValueError('PROBE_RESULT_CHANGED')
    return 'VERIFIED'

def state(out):
    m=_manifest(out)
    complete=0;pauses=defaultdict(int);episodes=[];calls=0;probes=0
    for task,arm,rep in episode_ids():
        d,_=_episode(out,task,arm,rep);s={v:_stage_state(d,v) for v in STAGES};p={v:_probe_state(d,v,arm) for v in STAGES[:2]}
        calls+=sum(v!='NOT_STARTED' for v in s.values());probes+=sum(v=='VERIFIED' for v in p.values())
        done=all(v=='VERIFIED' for v in s.values()) and all(v in ('VERIFIED','NOT_APPLICABLE') for v in p.values())
        if done:complete+=1
        if any(v not in ('VERIFIED','NOT_STARTED') for v in s.values()):pauses['CALL_RECOVERY_OR_REVIEW_REQUIRED']+=1
        episodes.append({'id':d.name,'task':task,'arm':arm,'rep':rep,'stages':s,'probes':p,'complete':done})
    return {'status':'M30_VERIFIED_READ_ONLY','episodes_total':len(episodes),'episodes_completed':complete,'paid_attempts_recorded':calls,
            'tool_probes_completed':probes,'blocked_states':dict(pauses),'source_history_preserved':True,
            'scientific_discovery_certified':False}

def stage_prompt(problem,arm,stage,prev1=None,prev2=None,probe1=None,probe2=None):
    base=("You are studying an externally specified scientific or mathematical problem. Your job is to produce a genuine "+
          "testable explanation or valid construction, not a novel-sounding gap. Infer from observations. "+
          "The candidate is executable inert JSON, NOT code. AST uses only documented fields/ops. "+
          "Your final candidate is scored on hidden conditions. Never claim external scientific novelty. "+
          "You may admit uncertainty, but provide your best candidate. Only supplied experimental observations are evidence. "+
          "Return a single JSON object conforming exactly to the response_json_schema.")
    fakg=arm.startswith('FAKG')
    if fakg:
        # Explicitly derived from FAKG Case B's published _SYSTEM and Proposal->Challenge->Decision structure;
        # we do NOT present this adapter as native Research Control execution.
        base+=' Search aggressively; claim conservatively. Treat strong known models as nulls; compare explanatory power with the strongest alternative. Scientific criticism is not a source of new observations. Separate mathematical implications from evidence. Change approach if a candidate is falsified.'
    if stage=='proposal':
        inst='Form a substantive first explanation/construction and concrete candidate. Provide a discriminating probe request and predict its result BEFORE seeing an oracle response.'
    elif stage=='revision':
        inst=('ADVERSARIAL CHALLENGE: independently construct a strong null or counterexample, examine the actual observed tool results if any, then revise the candidate or defend it with specific reasoning.' if fakg else
              'Continue researching freely. Use earlier reasoning and any actual observed probe. Improve the candidate in whatever way seems productive.')
    else:
        inst=('FINAL RESEARCH DECISION: incorporate challenge and tool feedback. Do not overclaim. Produce the strongest grounded candidate and identify the remaining obstruction.' if fakg else
              'Make the final best attempt. Integrate all available information, propose your strongest final candidate and state remaining doubts.')
    if 'TOOL' in arm:inst+=' You have at most one automatic experimental probe after stage proposal and one after revision; propose a bounded JSON experiment. It returns the actual oracle outcome after your model call. No more probes after final.'
    else:inst+=' You have no executable oracle or intervention tool. You can still reason about potential tests, but they will NOT be executed in this condition.'
    payload={'scientific_problem_and_public_evidence':problem,'phase':stage,'instruction':inst,
             'previous_proposal':prev1,'previous_revision':prev2,'verified_probe_after_proposal':probe1,
             'verified_probe_after_revision':probe2,'response_json_schema':MODEL[stage].model_json_schema()}
    return base,json.dumps(payload,ensure_ascii=False,sort_keys=True)

def preview_episode(out,task,arm,rep):
    _manifest(out);d,problem=_episode(out,task,arm,rep)
    for stage in STAGES:
        s=_stage_state(d,stage)
        if s not in ('VERIFIED','NOT_STARTED'):return {'episode':d.name,'status':'BLOCKED','reason':s,'stage':stage}
        if s=='VERIFIED':
            ps=_probe_state(d,stage,arm)
            if ps=='NOT_STARTED':return {'episode':d.name,'status':'NEED_PROBE_NO_MODEL_CALL','stage':stage}
            continue
        return {'episode':d.name,'status':'NEXT_MODEL_CALL','stage':stage}
    return {'episode':d.name,'status':'COMPLETE','stage':None}

def _commit_probe(d,stage,task,arm):
    if 'TOOL' not in arm or stage=='final':return {'status':'NO_TOOL_ARM'}
    given=read(d/'calls'/stage/'response.json');p=d/'probes'/stage/'probe.json'
    outcome=oracle.experiment(task,given['proposed_probe'],given['candidate'])
    payload={'schema_version':'m30-deterministic-probe-v1',
             'request_sha256':core.sha({'candidate':given['candidate'],'proposed_probe':given['proposed_probe']}),
             'result':outcome,'model_charged':False}
    if p.exists():
        if read(p)!=payload:raise ValueError('PROBE_CONFLICT')
    else:put(p,payload)
    return payload

def next_episode(out,task,arm,rep,allow_paid=False,backend=None):
    st=preview_episode(out,task,arm,rep)
    d,problem=_episode(out,task,arm,rep)
    if st['status']=='COMPLETE':return {'status':'ALREADY_COMPLETE','episode':st['episode']}
    if st['status']=='BLOCKED':raise RuntimeError('PAID_CALL_AMBIGUOUS_NO_RETRY '+str(st))
    stage=st['stage']
    if st['status']=='NEED_PROBE_NO_MODEL_CALL':
        return {'status':'PROBE_COMMITTED_NO_PAID_CALL','episode':st['episode'],'stage':stage,'probe':_commit_probe(d,stage,task,arm)}
    if not allow_paid or os.getenv('FAKG_M30_AUTHORIZE_PAID')!='YES':raise PermissionError('M30_EXPLICIT_PAID_AUTHORIZATION_REQUIRED')
    p1=read(d/'calls/proposal/response.json') if stage!='proposal' else None
    p2=read(d/'calls/revision/response.json') if stage=='final' else None
    tool1=read(d/'probes/proposal/probe.json')['result'] if 'TOOL' in arm and stage!='proposal' else None
    tool2=read(d/'probes/revision/probe.json')['result'] if 'TOOL' in arm and stage=='final' else None
    sy,us=stage_prompt(problem,arm,stage,p1,p2,tool1,tool2)
    m=_manifest(out)
    settings={k:m[k] for k in ('model','temperature','max_tokens')}
    if backend is None:backend=core.OpenRouter(api_key_env='OPENROUTER_API_KEY')
    result=core._call(d,stage,MODEL[stage],sy,us,backend,max_calls=3,settings=settings)
    res={'status':'MODEL_STAGE_COMMITTED','episode':st['episode'],'stage':stage,'candidate_sha256':core.sha(result.candidate)}
    if 'TOOL' in arm and stage!='final':res['probe']=_commit_probe(d,stage,task,arm)
    return res

def recover(out,task,arm,rep):
    _manifest(out);d,_=_episode(out,task,arm,rep);st=preview_episode(out,task,arm,rep)
    if st['status']=='NEED_PROBE_NO_MODEL_CALL':return _commit_probe(d,st['stage'],task,arm)
    if st['status']!='BLOCKED' or st['reason']!='RAW_SAVED_RECOVERY_REQUIRED':
        raise RuntimeError('NO_VALID_RAW_TO_RECOVER: '+str(st))
    stage=st['stage'];p=d/'calls'/stage;raw=read(p/'raw_response.json')
    parsed=MODEL[stage].model_validate(raw).model_dump(mode='json')
    req=read(p/'request.json');attempt=read(p/'attempt.json')
    if attempt['request_sha256']!=core.sha(req):raise ValueError('RECOVERY_REQUEST_HASH_CHANGED')
    if (p/'response.json').exists():
        if read(p/'response.json')!=parsed:raise ValueError('RECOVERY_EXISTING_NORMALIZED_CONFLICT')
    else:put(p/'response.json',parsed)
    put(p/'receipt.json',{'status':'COMPLETED','request_sha256':core.sha(req),
                          'usage':{'recovered_usage_unknown':True},'parsed_schema':MODEL[stage].__name__,
                          'paid_attempt_count_for_stage':1,
                          'original_raw_response_sha256':core.sha(raw),'validated_response_sha256':core.sha(parsed),
                          'quarantined_matrix_count':0,'recovered_without_paid_call':True})
    if 'TOOL' in arm and stage!='final':_commit_probe(d,stage,task,arm)
    return {'status':'RECOVERED_FROM_SAVED_RAW_NO_PAID_CALL','episode':st['episode'],'stage':stage}

def run_batch(out,max_new_model_calls,allow_paid=False):
    if not allow_paid or os.getenv('FAKG_M30_AUTHORIZE_PAID')!='YES':raise PermissionError('M30_EXPLICIT_PAID_AUTHORIZATION_REQUIRED')
    m=_manifest(out);n=0;errors=[]
    for task,arm,rep in m['paired_execution_order']:
        while n<max_new_model_calls:
            st=preview_episode(out,task,arm,rep)
            if st['status']=='COMPLETE':break
            if st['status']=='BLOCKED':errors.append({'episode':st['episode'],'reason':st['reason']});break
            try:
                r=next_episode(out,task,arm,rep,allow_paid=True)
                if r['status']=='MODEL_STAGE_COMMITTED':n+=1
            except Exception as e:
                errors.append({'episode':st['episode'],'stage':st['stage'],'error':f'{type(e).__name__}:{str(e)[:300]}'})
                break
    return {'status':'BATCH_PAUSED_OR_COMPLETED','new_model_calls_in_invocation':n,'failed_or_blocked':errors[:100], **state(out)}

def report(out):
    m=_manifest(out);summary=state(out)
    records=[];by=defaultdict(list);traces=[]
    for task,arm,rep in episode_ids():
        d,_=_episode(out,task,arm,rep);st=preview_episode(out,task,arm,rep)
        if st['status']=='COMPLETE':
            result=oracle.final_score(task,read(d/'calls/final/response.json')['candidate'])
            baseline=oracle.final_score(task,read(d/'calls/proposal/response.json')['candidate'])
            row={'task':task,'family':task.split('_')[0],'arm':arm,'rep':rep,'status':'COMPLETE',
                 'score':result['score'],'first_score':baseline['score'], 'change':round(result['score']-baseline['score'],6),
                 'metric':result['metric'],'score_details':json.dumps(result,ensure_ascii=False,sort_keys=True)}
            by[(task,arm)].append(result['score'])
            a=read(d/'calls/proposal/response.json');b=read(d/'calls/revision/response.json');c=read(d/'calls/final/response.json')
            pr1=read(d/'probes/proposal/probe.json')['result'] if 'TOOL' in arm else None
            pr2=read(d/'probes/revision/probe.json')['result'] if 'TOOL' in arm else None
            traces.append({'task':task,'arm':arm,'rep':rep,'initial_hypothesis':a['hypothesis'],
                           'initial_candidate':json.dumps(a['candidate'],ensure_ascii=False,sort_keys=True),
                           'initial_prediction':a['predicted_probe_result'],
                           'first_probe':json.dumps(pr1,ensure_ascii=False) if pr1 else '',
                           'what_changed':b['what_changed_and_why'],
                           'revised_candidate':json.dumps(b['candidate'],ensure_ascii=False,sort_keys=True),
                           'second_probe':json.dumps(pr2,ensure_ascii=False) if pr2 else '',
                           'final_hypothesis':c['hypothesis'],'final_candidate':json.dumps(c['candidate'],ensure_ascii=False,sort_keys=True),
                           'remaining_obstacle':c['strongest_remaining_obstacle'],
                           'initial_score':baseline['score'],'final_score':result['score'],
                           'score_change':round(result['score']-baseline['score'],6)})
        else:
            row={'task':task,'family':task.split('_')[0],'arm':arm,'rep':rep,'status':st['status'],
                 'score':'','first_score':'','change':'','metric':'','score_details':str(st)}
        records.append(row)
    # Deliberately output a per-problem table, NOT one grand cross-domain discovery score.
    lines=['# M30 Scientific Capability Sweep — Observed Results','',
           '**Scope:** controlled synthetic benchmarks, NOT newly discovered scientific laws.',
           '**FAKG arms:** staged FAKG-derived experimental adapters using native Case B LLM ledger; NOT the full production KG/SIS pipeline.',
           f"**Episodes complete:** {summary['episodes_completed']} / {summary['episodes_total']}",
           f"**Recorded model attempts:** {summary['paid_attempts_recorded']}",
           '', '|Problem|Plain|FAKG-adapter|Plain + Tool|FAKG-adapter + Tool|', '|---|---:|---:|---:|---:|']
    for task in oracle.TASKS:
        def avg(arm):
            a=by.get((task,arm),[])
            return f'{sum(a)/len(a):.3f} ({len(a)}/2)' if a else '—'
        lines.append('|'+task+'|'+ '|'.join(avg(a) for a in ARMS)+'|')
    lines.extend(['','## Interpretation cautions','',
                 '- Comparisons are paired within each task; scores across task families are NOT commensurate measures of novelty.',
                 '- Experimental adapters test research-control prompts and tool access, not complete native FAKG vs a production alternative.',
                 '- A repeated experiment result can improve prediction without establishing a truly new causal mechanism.',
                 '- Tool arms see up to two additional oracle observations; C vs D isolates the adapter conditional on those opportunities.',
                 '- Repetitions are few; treat observed differences as diagnostic, not statistically established superiority.',
                 '- Invalid model JSON must never cause automatic duplicate paid calls.'])
    t=out/'reports';t.mkdir(exist_ok=True)
    # Reports can be refreshed. Never overwrite original provider receipts.
    with (t/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Within-repetition paired differences; missingness is explicit and not zero-filled.
    completed_scores={(row['task'],row['arm'],row['rep']):float(row['score']) for row in records if row['status']=='COMPLETE'}
    contrast_pairs=[('STRUCTURE_NO_TOOL','FAKG_ADAPTER','PLAIN'),('TOOL_PLAIN','PLAIN_TOOL','PLAIN'),
                    ('STRUCTURE_WITH_TOOL','FAKG_ADAPTER_TOOL','PLAIN_TOOL'),('TOOL_FAKG','FAKG_ADAPTER_TOOL','FAKG_ADAPTER')]
    contrasts=[]
    for task in oracle.TASKS:
        for rep in range(1,REPS+1):
            for label,arm_a,arm_b in contrast_pairs:
                va=completed_scores.get((task,arm_a,rep));vb=completed_scores.get((task,arm_b,rep))
                contrasts.append({'task':task,'rep':rep,'contrast':label,'arm_a':arm_a,'arm_b':arm_b,
                                  'delta_a_minus_b':round(va-vb,6) if va is not None and vb is not None else '',
                                  'status':'PAIRED' if va is not None and vb is not None else 'INCOMPLETE_PAIR'})
    with (t/'paired_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
    lines.extend(['','## Paired contrasts (mean of available complete repeated pairs; diagnostic only)','',
                  '|Problem|Structure: B−A|Feedback: C−A|Structure with tools: D−C|Feedback with structure: D−B|',
                  '|---|---:|---:|---:|---:|'])
    for task in oracle.TASKS:
        vals=[]
        for label,_,_ in contrast_pairs:
            d=[float(x['delta_a_minus_b']) for x in contrasts if x['task']==task and x['contrast']==label and x['status']=='PAIRED']
            vals.append(f'{sum(d)/len(d):+.3f} ({len(d)}/2)' if d else '—')
        lines.append('|'+task+'|'+'|'.join(vals)+'|')
    if traces:
        with (t/'research_traces.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(traces[0]));w.writeheader();w.writerows(traces)
    # Human-readable local HTML, without remote dependencies or external scripts.
    table_rows=[]
    for row in records:
        e=html.escape
        task=e(str(row['task']));arm=e(str(row['arm']));rep=e(str(row['rep']))
        score=e(str(row['score']));start=e(str(row['first_score']));change=e(str(row['change']))
        status=e(str(row['status']));detail=e(str(row['score_details']))
        tag='good' if row['status']=='COMPLETE' else 'incomplete'
        table_rows.append(f'<tr class="{tag}" data-family="{e(str(row["family"]))}" data-arm="{arm}"><td>{task}</td><td>{arm}</td><td>{rep}</td><td>{status}</td><td>{start}</td><td>{score}</td><td>{change}</td><td><details><summary>Detail</summary><pre>{detail}</pre></details></td></tr>')
    arm_opts=''.join(f'<option value="{a}">{a}</option>' for a in ARMS)
    fams=sorted({r['family'] for r in records})
    fam_opts=''.join(f'<option value="{f}">{f}</option>' for f in fams)
    dashboard='''<!doctype html><html lang="en"><meta charset="utf-8"><title>M30 Sweep Results</title>
<style>body{font:14px/1.5 system-ui,Arial,sans-serif;max-width:1400px;margin:35px auto;padding:0 16px;color:#222}
h1{font-size:26px}h2{font-size:18px} .muted{color:#666}.filters{display:flex;gap:12px;margin:18px 0}
select{padding:8px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #ddd;text-align:left;padding:9px;vertical-align:top}
th{position:sticky;top:0;background:#f1f3f5}tr.incomplete{color:#777}td pre{white-space:pre-wrap;word-break:break-word;max-width:380px}
summary{cursor:pointer} a{color:#1366a2}.stats{background:#f3f4f6;padding:12px}</style>
<h1>M30 Scientific Capability Sweep</h1><p class="muted">Controlled synthetic tasks. FAKG-labelled conditions are a research-control adapter, not a full native FAKG E2E run. No Level II certification.</p>
<p class="stats">Episodes complete: <b>__COMPLETED__ / __TOTAL__</b> | Recorded paid attempts: <b>__ATTEMPTS__</b> | Probes executed: <b>__PROBES__</b></p>
<h2>Experiment-level results</h2><p>Filter by family and research condition. All scores are relative within each problem; compare <a href="scores.csv">scores.csv</a>, <a href="paired_contrasts.csv">paired_contrasts.csv</a> or <a href="research_traces.csv">research_traces.csv</a> for full research changes.</p>
<div class="filters"><label>Family <select id="family"><option value="">All</option>__FAMILY_OPTIONS__</select></label>
<label>Arm <select id="arm"><option value="">All</option>__ARM_OPTIONS__</select></label></div>
<table><thead><tr><th>Problem</th><th>Condition</th><th>Rep</th><th>Status</th><th>Initial score</th><th>Final score</th><th>Delta</th><th>Metric details</th></tr></thead>
<tbody>__ROWS__</tbody></table>
<script>function filter(){let f=document.getElementById('family').value,a=document.getElementById('arm').value;
for(let r of document.querySelectorAll('tbody tr'))r.style.display=(!f||r.dataset.family===f)&&(!a||r.dataset.arm===a)?'':'none';}
document.getElementById('family').onchange=filter;document.getElementById('arm').onchange=filter;</script></html>'''
    for a,b in {'__COMPLETED__':str(summary['episodes_completed']),'__TOTAL__':str(summary['episodes_total']),
                '__ATTEMPTS__':str(summary['paid_attempts_recorded']),'__PROBES__':str(summary['tool_probes_completed']),
                '__FAMILY_OPTIONS__':fam_opts,'__ARM_OPTIONS__':arm_opts,'__ROWS__':'\n'.join(table_rows)}.items():dashboard=dashboard.replace(a,b)
    (t/'DASHBOARD.html').write_text(dashboard,encoding='utf-8')
    (t/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (t/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'status':'REPORTS_GENERATED_NO_MODEL_CALL','report_dir':str(t),**summary}

def cli():
    p=argparse.ArgumentParser(description='M30 multitype synthetic science capability sweep, add-only / no native source changes')
    sub=p.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('prepare');x.add_argument('--out',type=Path,required=True);x.add_argument('--model',default='openai/gpt-5.6-luna');x.add_argument('--temperature',type=float,default=.7);x.add_argument('--max-tokens',type=int,default=6000)
    for cmd in ['verify','report','preview','next','recover','run-batch']:
        x=sub.add_parser(cmd);x.add_argument('--out',type=Path,required=True)
        if cmd in ['preview','next','recover']:
            x.add_argument('--task',required=True,choices=oracle.TASKS);x.add_argument('--arm',required=True,choices=ARMS);x.add_argument('--rep',type=int,required=True,choices=[1,2])
        if cmd in ['next','run-batch']:x.add_argument('--allow-paid',action='store_true')
        if cmd=='run-batch':x.add_argument('--max-new-model-calls',type=int,default=240)
    args=p.parse_args()
    if args.cmd=='prepare':r=prepare(args.out,model=args.model,temperature=args.temperature,max_tokens=args.max_tokens)
    elif args.cmd=='verify':r=state(args.out)
    elif args.cmd=='report':r=report(args.out)
    elif args.cmd=='preview':r=preview_episode(args.out,args.task,args.arm,args.rep)
    elif args.cmd=='next':r=next_episode(args.out,args.task,args.arm,args.rep,allow_paid=args.allow_paid)
    elif args.cmd=='recover':r=recover(args.out,args.task,args.arm,args.rep)
    elif args.cmd=='run-batch':r=run_batch(args.out,args.max_new_model_calls,allow_paid=args.allow_paid)
    else:raise RuntimeError('BAD_CMD')
    print(json.dumps(r,indent=2,ensure_ascii=False,sort_keys=True))

if __name__=='__main__':cli()
