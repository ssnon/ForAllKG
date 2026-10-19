"""M30-R: opt-in scientific Work with executable synthetic feedback, committed to generic native v1.2.

NOT the full v1.8 Seed/Focus policy. No scientific truth certification.
Sealed synthetic oracle is a deterministic benchmark instrument, not a lab.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from . import m30r_generic_v12 as native
from .m30r_generic_v12_tools import REASON_SYSTEM
from .research_work_v12_contracts import ReasonWorkContent

PROTOCOL='m30r-generic-science-act-v1'
MAX_WORK_STEPS=3
ALLOWED_TASKS=frozenset(['dyn_predator','dyn_resource','sidon_25','sidon_40','law_saturation','law_inhibition','dfa_mod3','dfa_suffix01','chem_blocking','chem_dualpath'])

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class Step(Strict):
    operation:Literal['EXPLORE','CONSTRUCT','DERIVE','CHALLENGE','DEVELOP','REVISE','PIVOT','RETURN']
    scientific_work_content:ReasonWorkContent
    research_delta:str=Field(min_length=16)
    next_action:Literal['CONTINUE','RETURN']
    next_research_question:str|None=None
    why_next_action:str=Field(min_length=16)
    probe_request:dict|None=None
    prospective_probe_prediction:str|None=None
    @model_validator(mode='after')
    def check(self):
        if self.next_action=='CONTINUE' and not self.next_research_question:
            raise ValueError('CONTINUE_NEEDS_NEXT_QUESTION')
        if self.probe_request is not None:
            if self.next_action!='CONTINUE':raise ValueError('TOOL_REQUIRES_NEXT_INTERPRETATION')
            if not self.prospective_probe_prediction or len(self.prospective_probe_prediction)<16:
                raise ValueError('PROSPECTIVE_PREDICTION_BEFORE_TOOL_REQUIRED')
        return self

def sha(obj:Any)->str:return native.sha(obj)
def hfile(p:Path)->str:return native.file_sha(Path(p))
def read(p:Path)->Any:return native.read(Path(p))
def put(p:Path,obj:Any)->None:native.write_new(Path(p),obj)
def now():return datetime.now(timezone.utc).isoformat()
def source_root():return Path(__file__).resolve().parents[3]
def run_root(out:Path)->Path:
    out=Path(out).expanduser().resolve()
    permitted=(source_root()/'experiments/m30r_science_act/runs').resolve()
    if out==permitted or permitted not in out.parents:raise ValueError('OUTSIDE_M30R_SCIENCE_RUNS')
    return out

def oracle_module():
    from pipeline_core.discovery.m30_sweep import oracle
    return oracle

def read_lock(out:Path)->dict:return read(run_root(out)/'lock.json')
def native_path(out:Path)->Path:return run_root(out)/'native'
def work_path(out:Path,work_id:str|None=None)->Path:
    if work_id is None:
        c=native.verify(native_path(out))
        pending=c['state']['pending_work']
        if not pending:raise ValueError('NO_ACTIVE_WORK_ID')
        work_id=pending['work_id']
    if not isinstance(work_id,str) or not work_id.startswith('work_') or '/' in work_id:
        raise ValueError('INVALID_WORK_ID')
    return run_root(out)/'research_works'/work_id

def prepare(packet:Path,out:Path,*,tool_task:str|None=None,model='openai/gpt-5.6-luna',max_decisions:int=8)->dict:
    out=run_root(out)
    if out.exists():raise FileExistsError('M30R_RUN_ALREADY_EXISTS')
    if tool_task is not None and tool_task not in ALLOWED_TASKS:raise ValueError('UNKNOWN_SYNTHETIC_TOOL_TASK')
    packet=Path(packet).resolve()
    evidence=read(packet)
    if tool_task is not None:
        if evidence['problem_kind']!='SYNTHETIC_CONTROLLED' or evidence['problem_id']!='M30R_'+tool_task.upper():
            raise ValueError('ORACLE_TASK_NOT_BOUND_TO_FROZEN_PROBLEM')
        oracle=oracle_module()
        if not hasattr(oracle,'experiment') or not hasattr(oracle,'public_problem'):
            raise ValueError('ORACLE_INCOMPATIBLE')
        expected=oracle.public_problem(tool_task)
        try:observed=json.loads(evidence['sources']['pages'][0]['text'])
        except (KeyError,ValueError,TypeError) as exc:raise ValueError('OBSERVATION_NOT_MACHINE_CHECKABLE') from exc
        if observed.get('observed_data')!=expected['observed_data'] or observed.get('scientific_problem')!=expected['scientific_problem']:
            raise ValueError('FROZEN_OBSERVATIONS_DO_NOT_MATCH_SEALED_ORACLE')
        oracle_hash=hfile(Path(oracle.__file__))
    else:oracle_hash=None
    out.mkdir(parents=True,exist_ok=False)
    try:
        native.prepare_generic(packet_file=packet,out=out/'native',model=model,max_decisions=max_decisions)
        lock={'protocol':PROTOCOL,'packet_sha256':hfile(packet), 'problem_id':evidence['problem_id'],
              'native_manifest_sha256':sha(read(out/'native/manifest.json')),
              'native_source_sha256':hfile(Path(native.__file__)),
              'work_source_sha256':hfile(Path(__file__)),
              'oracle_source_sha256':oracle_hash,'tool_task':tool_task,
              'tool_status':('M30_DETERMINISTIC_SYNTHETIC_ORACLE' if tool_task else 'NO_ORACLE'),
              'native_policy':'GENERIC_V12_NOT_V18_SEED_FOCUS',
              'scientific_validation':False}
        put(out/'lock.json',lock)
        verify(out)
    except BaseException:
        import shutil
        import stat
        shutil.rmtree(out)
        raise
    return {'status':'M30R_GENERIC_NATIVE_PREPARED_NO_CALL','problem_id':evidence['problem_id'],
            'tool_task':tool_task,'native_policy':lock['native_policy'],'scientific_validation':False}

def _tool_request_ok(lock:dict,request:Any)->dict:
    if not lock['tool_task']:raise ValueError('ORACLE_NOT_ENABLED')
    if not isinstance(request,dict) or set(request)!={'candidate','intervention'}:
        raise ValueError('EXPLICIT_CANDIDATE_AND_INTERVENTION_REQUIRED')
    inter=request['intervention']
    if not isinstance(inter,dict):raise ValueError('INTERVENTION_MUST_BE_OBJECT')
    task=lock['tool_task']; family=task.split('_')[0]
    keyset=({'x0','y0','u'} if family=='dyn' else
            {'x','z'} if family=='law' else {'c','o','r'} if family=='chem' else set())
    if set(inter)!=keyset:raise ValueError('INTERVENTION_FIELDS_MUST_BE_COMPLETE: '+str(sorted(keyset)))
    if family in ('dyn','law','chem'):
        for k,v in inter.items():
            if type(v) not in (float,int) or not -0.25<=v<=3:
                raise ValueError('INTERVENTION_NUMBER_INVALID: '+k)
    cand=request['candidate']
    if not isinstance(cand,dict):raise ValueError('CANDIDATE_MUST_BE_OBJECT')
    if family=='dyn' and set(cand)!={'dx','dy'}:raise ValueError('COMPLETE_DX_DY_REQUIRED')
    if family=='sidon' and set(cand)!={'set'}:raise ValueError('SIDON_SET_REQUIRED')
    if family=='dfa' and set(cand)!={'transitions','accepting'}:raise ValueError('COMPLETE_DFA_REQUIRED')
    return {'task':task,'candidate':cand,'intervention':inter}

def run_tool(lock:dict,req:Any)->dict:
    try:
        arguments=_tool_request_ok(lock,req)
        response=oracle_module().experiment(arguments['task'],arguments['intervention'],arguments['candidate'])
        return {'status':response['status'], 'arguments':arguments,
                'result':response.get('result'), 'error':response.get('error'),
                'source':'DETERMINISTIC_SYNTHETIC_M30_ORACLE_NOT_EMPIRICAL'}
    except (ValueError,TypeError,KeyError,OverflowError,AttributeError) as exc:
        return {'status':'REJECTED_INVALID_REQUEST','error':str(exc)[:600],
                'source':'DETERMINISTIC_SYNTHETIC_M30_ORACLE_NOT_EMPIRICAL'}

def _request(out:Path,rows:list[dict],n:int,wd:Path)->dict:
    ctx=read(wd/'context.json')
    lock=read_lock(out)
    payload={'native_work_request':ctx['native_work_request'],
             'research_problem':ctx['research_problem'],
             'native_work_payload':ctx['native_work_payload'],
             'research_step':n,'max_steps':MAX_WORK_STEPS,
             'prior_research_steps':rows,'tool_available':bool(lock['tool_task']) and n<MAX_WORK_STEPS,
             'tool_task':lock['tool_task'] if n<MAX_WORK_STEPS else None,
             'response_schema':Step.model_json_schema(),'scientific_certification':False}
    sys=REASON_SYSTEM+'\\n'+'''Perform substantive research on the assigned question. You may construct, derive, refute, or return without a result. Distinguish scientific premises, predictions, and unknowns. All prior work is UNVERIFIED. The model, not the runtime, must explicitly choose any scientific intervention. An instrument validates only a submitted synthetic candidate; it never proves real physical truth or novelty. Never invent an experiment, validation, or source. Supply all JSON fields, including nullable fields, exactly.'''
    if lock['tool_task'] and n<MAX_WORK_STEPS:
        sys+='\\n'+'''OPTIONAL DETERMINISTIC SYNTHETIC BENCHMARK ORACLE: submit probe_request as {"candidate":{...},"intervention":{...}}. Supply ALL intervention fields: dyn requires x0,y0,u; law requires x,z; chem requires c,o,r; sidon/dfa require {}. Candidate: dynamics {dx:AST,dy:AST}, laws/chem AST expression directly, sidon {set:[...]}, DFA {transitions:[[...]],accepting:[...]}. Missing parameters will be REJECTED, never defaulted. Include prospective_probe_prediction BEFORE seeing a response. If tool is not useful, set probe_request:null and prospective_probe_prediction:null. The next research step receives the ACTUAL oracle result and can interpret or revise. Synthetic benchmark truth is not peer reviewed empirical science.'''
    else:sys+='\\n'+'''No tool call is available in this step. Set probe_request:null and prospective_probe_prediction:null.'''
    if n==MAX_WORK_STEPS:sys+='\\nFinal step: next_action MUST be RETURN. Summarize strongest genuinely supported result and limitations.'
    if rows and rows[-1]['instrument_result'] is not None:
        sys+='\\nThe previous instrument result is included in prior_research_steps. Compare it to your explicit prior prediction. An invalid request is not a scientific falsification.'
    return {'model':ctx['model'],'temperature':ctx['temperature'],
            'max_tokens':ctx['max_tokens'],
            'messages':[{'role':'system','content':sys},
                        {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}

def _rows(out:Path,wd:Path):
    rows=[];pending=None
    if not wd.exists():return rows,pending
    for n in range(1,MAX_WORK_STEPS+1):
        sd=wd/f'step_{n:04d}'
        if not sd.exists():continue
        if n!=len(rows)+1 or (rows and rows[-1]['normalized']['next_action']=='RETURN'):
            raise ValueError('NONCONTIGUOUS_RESEARCH_STEPS')
        req=_request(out,rows,n,wd)
        if read(sd/'request.json')!=req:raise ValueError('RESEARCH_REQUEST_CHANGED')
        att=read(sd/'attempt.json')
        if att['request_sha256']!=sha(req) or att['prior_receipt_sha256']!=(sha(rows[-1]['receipt']) if rows else None):
            raise ValueError('RESEARCH_ATTEMPT_CHAIN_TAMPERED')
        if not (sd/'raw_response.json').is_file() or not all((sd/x).is_file() for x in ('normalized.json','instrument_result.json','receipt.json')):
            pending=sd;break
        raw=read(sd/'raw_response.json');v=Step.model_validate_json(raw['raw_text']).model_dump(mode='json')
        if n==MAX_WORK_STEPS and (v['next_action']!='RETURN' or v['probe_request'] is not None):
            raise ValueError('FINAL_RESEARCH_STEP_INVALID')
        if v['probe_request'] is not None and not read_lock(out)['tool_task']:
            raise ValueError('NO_INSTRUMENT_FOR_PROBE')
        if read(sd/'normalized.json')!=v:raise ValueError('NORMALIZED_RESEARCH_CHANGED')
        result=run_tool(read_lock(out),v['probe_request']) if v['probe_request'] is not None else None
        if read(sd/'instrument_result.json')!={'instrument_result':result}:raise ValueError('INSTRUMENT_RESULT_CHANGED')
        receipt={'number':n,'request_sha256':sha(req),'raw_sha256':hfile(sd/'raw_response.json'),
                 'normalized_sha256':hfile(sd/'normalized.json'), 'instrument_sha256':hfile(sd/'instrument_result.json'),
                 'prior_receipt_sha256':sha(rows[-1]['receipt']) if rows else None}
        if read(sd/'receipt.json')!=receipt:raise ValueError('RESEARCH_RECEIPT_CHANGED')
        rows.append({'normalized':v,'instrument_result':result,'receipt':receipt})
    return rows,pending

def verify(out:Path)->dict:
    out=run_root(out);lock=read_lock(out)
    if lock['protocol']!=PROTOCOL or lock['native_policy']!='GENERIC_V12_NOT_V18_SEED_FOCUS':
        raise ValueError('M30R_LOCK_PROTOCOL_MISMATCH')
    if hfile(Path(native.__file__))!=lock['native_source_sha256'] or hfile(Path(__file__))!=lock['work_source_sha256']:
        raise ValueError('RESEARCH_IMPLEMENTATION_CHANGED')
    if lock['tool_task'] and hfile(Path(oracle_module().__file__))!=lock['oracle_source_sha256']:
        raise ValueError('SYNTHETIC_ORACLE_SOURCE_CHANGED')
    c=native.verify(native_path(out))
    if sha(c['manifest'])!=lock['native_manifest_sha256'] or c['manifest']['generic_problem_id']!=lock['problem_id']:
        raise ValueError('NATIVE_PROBLEM_LOCK_CHANGED')
    active=c['state']['pending_work']
    history={w['work_id']:w for w in c['state']['work_history']}
    total_steps=0;tools=0;unfinished=[];num_works=0
    root=out/'research_works'
    if root.exists():
        for wd in sorted(root.iterdir()):
            if not wd.is_dir() or not wd.name.startswith('work_'):raise ValueError('UNEXPECTED_WORK_DIRECTORY')
            anchor=read(wd/'work_lock.json');ctx=read(wd/'context.json')
            if wd.name!=anchor['work_id'] or ctx['native_work_request']!=anchor['native_work_request']:
                raise ValueError('NATIVE_WORK_IDENTITY_CHANGED')
            if anchor['native_request_sha256']!=sha(read(wd/'native_request.json')) or anchor['context_sha256']!=sha(ctx):
                raise ValueError('WORK_INPUT_LOCK_CHANGED')
            if active and active['work_id']==wd.name and active['status']=='REQUESTED':
                if native._make_call_request(c,'work')!=read(wd/'native_request.json') or sha(c['state'])!=anchor['native_state_sha256']:
                    raise ValueError('ACTIVE_NATIVE_REQUEST_CHANGED')
            elif wd.name not in history or history[wd.name]['execution_status']!='COMPLETED':
                raise ValueError('COMMITTED_WORK_NOT_FOUND')
            rows,pending=_rows(out,wd)
            total_steps+=len(rows);tools+=sum(1 for x in rows if x['instrument_result'] is not None)
            if pending:unfinished.append(str(pending))
            completed=wd/'final_content.json'
            if completed.exists():
                if pending or not rows or (rows[-1]['normalized']['next_action']!='RETURN' and len(rows)<MAX_WORK_STEPS):
                    raise ValueError('WORK_FINISHED_WITH_PENDING_STEPS')
                if read(completed)!=rows[-1]['normalized']['scientific_work_content']:
                    raise ValueError('FINAL_SCIENTIFIC_CONTENT_CHANGED')
            if wd.name in history:
                if not completed.exists() or history[wd.name]['content']!=read(completed):
                    raise ValueError('NATIVE_WORK_CONTENT_NOT_FROM_RESEARCH')
            num_works+=1
    return {'status':'M30R_GENERIC_NATIVE_REPLAY_VERIFIED_NO_CALL','native_event_count':c['state']['event_count'],
            'native_decision_count':c['state']['decision_count'],'native_work_count':len(c['state']['work_history']),
            'native_interpretation_count':len(c['state']['interpretations']),'native_ended':c['state']['ended'],
            'companion_works':num_works,'work_steps':total_steps,'research_pending':bool(unfinished),
            'tool_results':tools,'tool_enabled':bool(lock['tool_task']),
            'scientific_validation':False,'native_policy':lock['native_policy']}

def next_decision(out:Path,*,allow_paid=False,backend=None)->dict:
    out=run_root(out);v=verify(out)
    if v['research_pending']:raise RuntimeError('RECOVER_PENDING_RESEARCH_FIRST')
    c=native.verify(native_path(out));p=c['state']['pending_work']
    if p and p['status']=='REQUESTED':
        raise ValueError('COMPLETE_OR_EXECUTE_PENDING_NATIVE_WORK_FIRST')
    r=native.decision(native_path(out),allow_paid=allow_paid,backend=backend)
    verify(out)
    return r

def prepare_work(out:Path)->dict:
    out=run_root(out);verify(out);c=native.verify(native_path(out));p=c['state']['pending_work']
    if not p or p['status']!='REQUESTED':raise ValueError('NO_REQUESTED_NATIVE_WORK')
    if p['request']['work_kind']=='READ_FROZEN_SOURCE':
        r=native.execute(native_path(out));verify(out);return r
    wd=work_path(out)
    if wd.exists():raise FileExistsError('M30R_WORK_ALREADY_PREPARED')
    native_req=native._make_call_request(c,'work')
    ctx={'native_work_request':p['request'],'research_problem':c['manifest']['question'],
         'native_work_payload':native.work_payload(c),'model':c['manifest']['model'],
         'temperature':c['manifest']['temperature'],'max_tokens':c['manifest']['max_tokens']}
    wd.mkdir(parents=True)
    put(wd/'context.json',ctx)
    put(wd/'native_request.json',native_req)
    put(wd/'work_lock.json',{'work_id':p['work_id'],'native_work_request':p['request'],
          'native_request_sha256':sha(native_req),'context_sha256':sha(ctx),
          'native_state_sha256':sha(c['state']),'scientific_validation':False})
    verify(out)
    return {'status':'M30R_NATIVE_WORK_PREPARED_NO_CALL','work_id':p['work_id'],'max_steps':MAX_WORK_STEPS}

def preview(out:Path)->dict:
    out=run_root(out);v=verify(out);c=native.verify(native_path(out));p=c['state']['pending_work']
    if p and p['status']=='REQUESTED':
        if p['request']['work_kind']=='READ_FROZEN_SOURCE':return {'next':'READ_FROZEN_SOURCE_NO_CALL',**v}
        if not work_path(out).exists():return {'next':'PREPARE_WORK_NO_CALL',**v}
        rows,pending=_rows(out,work_path(out))
        if pending:return {'next':'RECOVER_WORK_NO_CALL',**v}
        if (work_path(out)/'final_content.json').exists():return {'next':'COMMIT_WORK_NO_CALL',**v}
        if len(rows)==MAX_WORK_STEPS or (rows and rows[-1]['normalized']['next_action']=='RETURN'):
            return {'next':'FINISH_WORK_NO_CALL',**v}
        return {'next':'WORK_MODEL_CALL','step':len(rows)+1,**v}
    if c['state']['ended']:return {'next':'COMPLETE',**v}
    return {'next':'DECISION_MODEL_CALL',**v}

def next_work(out:Path,*,allow_paid=False,backend=None)->dict:
    out=run_root(out);pv=preview(out)
    if pv['next']!='WORK_MODEL_CALL':raise ValueError('WORK_NOT_READY: '+pv['next'])
    native._require_paid(backend,allow_paid,1)
    rows,_=_rows(out,work_path(out));n=len(rows)+1;sd=work_path(out)/f'step_{n:04d}'
    sd.mkdir(parents=True,exist_ok=False)
    req=_request(out,rows,n,work_path(out))
    put(sd/'request.json',req)
    put(sd/'attempt.json',{'request_sha256':sha(req),
                          'prior_receipt_sha256':sha(rows[-1]['receipt']) if rows else None,
                          'status':'UNKNOWN_BILLING_UNTIL_RAW_SAVED','utc':now()})
    raw,usage,rid=native._invoke(req,backend)
    put(sd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:r=recover_work(out)
    except Exception as exc:raise RuntimeError('RAW_SAVED_NO_AUTO_RETRY; USE_RECOVER_WORK') from exc
    r['new_paid_calls_this_command']=int(backend is None)
    return r

def recover_work(out:Path)->dict:
    out=run_root(out);verify(out);rows,pending=_rows(out,work_path(out))
    if pending is None:raise ValueError('NO_PENDING_RESEARCH_CALL')
    sd=pending
    if not (sd/'raw_response.json').is_file():raise RuntimeError('RAW_NOT_SAVED_BILLING_UNKNOWN_NO_RETRY')
    raw=read(sd/'raw_response.json');step=Step.model_validate_json(raw['raw_text']).model_dump(mode='json')
    if sd.name==f'step_{MAX_WORK_STEPS:04d}' and (step['next_action']!='RETURN' or step['probe_request'] is not None):
        raise ValueError('FINAL_STEP_CANNOT_CONTINUE_OR_PROBE')
    if step['probe_request'] is not None and not read_lock(out)['tool_task']:
        raise ValueError('PROBE_SUBMITTED_IN_NO_TOOL_ARM')
    result=run_tool(read_lock(out),step['probe_request']) if step['probe_request'] is not None else None
    if not (sd/'normalized.json').exists():put(sd/'normalized.json',step)
    if not (sd/'instrument_result.json').exists():put(sd/'instrument_result.json',{'instrument_result':result})
    rec={'number':len(rows)+1,'request_sha256':read(sd/'attempt.json')['request_sha256'],
         'raw_sha256':hfile(sd/'raw_response.json'),'normalized_sha256':hfile(sd/'normalized.json'),
         'instrument_sha256':hfile(sd/'instrument_result.json'),
         'prior_receipt_sha256':sha(rows[-1]['receipt']) if rows else None}
    if not (sd/'receipt.json').exists():put(sd/'receipt.json',rec)
    verify(out)
    return {'status':'M30R_RESEARCH_STEP_COMMITTED_NO_RETRY','step':len(rows)+1,
            'next_action':step['next_action'],'tool_status':result['status'] if result else None}

def finish_work(out:Path)->dict:
    out=run_root(out);verify(out);rows,pending=_rows(out,work_path(out))
    if pending or not rows or (len(rows)<MAX_WORK_STEPS and rows[-1]['normalized']['next_action']!='RETURN'):
        raise ValueError('RESEARCH_NOT_FINISHED')
    f=work_path(out)/'final_content.json'
    if f.exists():raise FileExistsError('FINAL_ALREADY_FROZEN')
    put(f,rows[-1]['normalized']['scientific_work_content'])
    verify(out)
    return {'status':'M30R_SCIENTIFIC_WORK_FROZEN_NO_CALL','research_steps':len(rows)}

def commit_work(out:Path)->dict:
    out=run_root(out);verify(out);wd=work_path(out)
    if not (wd/'final_content.json').exists():raise ValueError('FREEZE_WORK_FIRST')
    original=read(wd/'native_request.json')
    work=read(wd/'final_content.json')
    raw=json.dumps(work,ensure_ascii=False,sort_keys=True)
    def saved_backend(req):
        if req!=original:raise ValueError('NATIVE_WORK_REQUEST_CHANGED')
        return raw,None,'m30r-preserved-research-work'
    r=native.execute(native_path(out),backend=saved_backend)
    verify(out)
    return {'status':'M30R_NATIVE_WORK_PRODUCT_COMMITTED_NO_NEW_PAID_CALL','native':r}
