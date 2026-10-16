"""Native W08 Research Decision -> optional instrument Work -> native WorkProduct.

Read-only source episode. All new native events occur in a verified independent fork.
This module adds no scientific certification and does not silently alter v1.8 protocols.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from . import research_work_v18_feedback_runner as native
from . import research_work_v12_stable_runner as base
from .research_work_v12_contracts import ReasonWorkContent
from .research_work_v12_stable_tools import REASON_SYSTEM
from . import science_act_probe

PROTOCOL = 'w08-native-science-act-continuation-v1'
MAX_STEPS = 3
PREFIX = 'science_act_continuation'
SYSTEM = '''You are an independent scientific researcher executing the CURRENT FAKG ResearchWorkRequest,
which was selected by FAKG Research Control. The supplied scientific history and previous
model-derived explanations are UNVERIFIED; never treat them as measured facts. Search
aggressively, claim conservatively. Grounding controls claims, not imagination.

Do real theoretical research rather than merely describing a research plan. You can propose,
explore, derive, challenge, revise, pivot within the assigned question, or conclude inconclusive.
At every step decide whether further work is scientifically worthwhile. Previous computations
are mathematical statements about the submitted hypothetical models, not real observations.
If a proposed mechanism is not distinguished by the instrument's outputs, acknowledge that
fact when it matters; never claim causal identification from conditional mathematics.
An error or a valid failure is a scientific outcome. Never claim Level II or lab validation.
Output one JSON object matching the supplied schema. You need not use an instrument.
'''
TOOL = '''OPTIONAL bounded scientific computation: finite CTMC stationary occupancy, episode
entry flux/durations, bounded parameter sweeps. You must construct any mechanistic CTMC
yourself. The calculation is NOT evidence for physical reality or novelty. Use when the
research question and a specified state model justify it. Output a complete inert JSON
probe_request: {"models":[{"name":"candidate","states":["D","B"],
"transitions":[{"source":"D","target":"B","rate":1.0,"group":"forward"},
{"source":"B","target":"D","rate":2.0,"group":"backward"}],
"observables":[{"name":"bright","bright_states":["B"]}]}],
"sweep":{"rate_group":"forward","multipliers":[0.5,1,2]}}.
If finite CTMC is a poor fit, set probe_request to null and research freely.
Do not prescribe a physical mechanism merely because this tool is present.
'''

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Step(Strict):
    operation: Literal['EXPLORE','CONSTRUCT','DERIVE','CHALLENGE','DEVELOP','REVISE','PIVOT','RETURN']
    scientific_work_content: ReasonWorkContent
    research_delta: str = Field(min_length=16)
    next_action: Literal['CONTINUE','RETURN']
    next_research_question: str | None = None
    why_next_action: str = Field(min_length=16)
    probe_request: dict | None = None
    @model_validator(mode='after')
    def _check(self):
        if self.next_action == 'CONTINUE' and (not self.next_research_question or len(self.next_research_question)<16):
            raise ValueError('CONTINUE_REQUIRES_SCIENTIFIC_NEXT_QUESTION')
        if self.probe_request is not None and self.next_action!='CONTINUE':
            raise ValueError('PROBE_REQUIRES_FOLLOWUP_INTERPRETATION')
        return self


def root(): return Path(__file__).resolve().parents[3]
def digest(obj): return base.sha(obj)
def hfile(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return base.read(p)
def put(p,obj): base.write_new(p,obj)
def now(): return datetime.now(timezone.utc).isoformat()

def safe_path(out:Path):
    out=Path(out).expanduser().resolve()
    area=(root()/'experiments'/PREFIX/'runs').resolve()
    if out==area or area not in out.parents:
        raise ValueError('OUTPUT_MUST_BE_INSIDE_EXPERIMENT_RUNS')
    return out

def native_dir(out):return safe_path(out)/'native'
def companion(out):return safe_path(out)/'companion'

def prepare(reference:Path, out:Path):
    ref=Path(reference).expanduser().resolve();out=safe_path(out)
    if out.exists():raise FileExistsError('EXPERIMENT_ALREADY_EXISTS_NO_OVERWRITE')
    if out==ref or out in ref.parents or ref in out.parents:
        raise ValueError('REFERENCE_AND_OUTPUT_MUST_BE_INDEPENDENT')
    c=native.verify(ref);s=c['state']
    if s['ended'] or s['decision_count']>=c['manifest']['max_decisions']:
        raise ValueError('REFERENCE_ALREADY_ENDED_OR_AT_DECISION_BUDGET')
    if any(x.is_symlink() for x in ref.rglob('*')):raise ValueError('REFERENCE_HAS_SYMLINKS')
    p=s.get('pending_work')
    if p and p['status']=='REQUESTED' and any((ref/'work'/p['work_id']/x).exists() for x in ('attempt.json','raw_response.json','llm_request.json','product.json')):
        raise ValueError('REFERENCE_HAS_UNSETTLED_WORK_ATTEMPT')
    parent=out.parent;parent.mkdir(parents=True,exist_ok=True)
    # Copy the original W08 event chain verbatim; do not rewrite or rebase its past.
    out.mkdir(mode=0o700)
    try:
        shutil.copytree(ref,out/'native')
        copied=native.verify(out/'native')
        if digest(copied['state'])!=digest(s):raise ValueError('FORK_STATE_DIFFERS')
        previous={str(x.relative_to(ref)):hfile(x) for x in ref.rglob('*') if x.is_file() and x.name!='state.json'}
        put(out/'fork_lock.json',{'schema':PROTOCOL,'reference_abs_path':str(ref),
            'reference_state_sha256':digest(s),'initial_files_sha256':previous,
            'native_source_sha256':hfile(Path(native.__file__)),
            'executor_source_sha256':hfile(Path(__file__)),
            'instrument_source_sha256':hfile(Path(science_act_probe.__file__)),
            'scientific_validation':False,'source_unmodified':True,
            'fork_native_prior_count':s['decision_count']})
        verify(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    return {'status':'W08_VERIFIED_FORK_PREPARED_NO_CALL','source':str(ref),'fork':str(out),
            'decisions_inherited':s['decision_count'],'pending_work_kind':p['request']['work_kind'] if p and p['status']=='REQUESTED' else None}

def verify(out:Path):
    out=safe_path(out);lock=read(out/'fork_lock.json')
    if lock['schema']!=PROTOCOL or lock['native_source_sha256']!=hfile(Path(native.__file__)) or lock['executor_source_sha256']!=hfile(Path(__file__)) or lock['instrument_source_sha256']!=hfile(Path(science_act_probe.__file__)):
        raise ValueError('EXECUTOR_OR_NATIVE_CODE_CHANGED')
    rootnative=native_dir(out)
    for name,old_sha in lock['initial_files_sha256'].items():
        file=rootnative/name
        if not file.is_file() or hfile(file)!=old_sha:
            raise ValueError('INHERITED_NATIVE_HISTORY_CHANGED: '+name)
    c=native.verify(rootnative)
    d=companion(out)
    if d.exists():
        dirs=[x for x in d.iterdir() if x.is_dir()]
        if len(dirs)>1:raise ValueError('MORE_THAN_ONE_WORK_IN_SINGLE_PILOT')
        for wd in dirs:
            _verify_work(out,wd,c)
    return {'status':'VERIFIED_NO_CALL','inherited_state_sha256':lock['reference_state_sha256'],
        'native_decisions':c['state']['decision_count'],'native_works':len(c['state']['work_history']),
        'native_pending':(c['state'].get('pending_work') or {}).get('status'),
        'companion_works':[x.name for x in d.iterdir() if x.is_dir()] if d.exists() else [],
        'native_ended':c['state']['ended'],'scientific_validation':False}

def _valid_pending(c):
    p=c['state'].get('pending_work')
    if not p or p['status']!='REQUESTED':raise ValueError('NO_PENDING_NATIVE_WORK')
    if p['request']['work_kind']!='REASON':raise ValueError('USE_NATIVE_EXECUTE_FOR_FROZEN_READ')
    return p

def choose(out:Path,*,allow_paid=False,backend=None):
    verify(out)
    c=native.verify(native_dir(out))
    p=c['state'].get('pending_work')
    if p and p['status']=='REQUESTED':
        return {'status':'EXISTING_NATIVE_DECISION_RETAINED_NO_CALL','work_id':p['work_id'],
                'question':p['request']['research_question'],'work_kind':p['request']['work_kind']}
    if c['state']['ended']:raise ValueError('NATIVE_EPISODE_ENDED')
    return native.decision(native_dir(out),allow_paid=allow_paid,max_new_calls=1,backend=backend)

def prepare_work(out:Path):
    verify(out);c=native.verify(native_dir(out));p=_valid_pending(c)
    wd=companion(out)/p['work_id']
    if wd.exists():raise FileExistsError('WORK_ALREADY_PREPARED')
    if (native_dir(out)/'work'/p['work_id']/'attempt.json').exists():
        raise ValueError('NATIVE_WORK_ALREADY_ATTEMPTED')
    req=native._make_call_request(c,'work')
    wd.mkdir(parents=True)
    put(wd/'lock.json',{'schema':PROTOCOL,'work_id':p['work_id'],
        'native_state_before_work_sha256':digest(c['state']),
        'native_request_sha256':digest(req),'max_steps':MAX_STEPS,
        'native_work_request':p['request'],'native_model':c['manifest']['model'],
        'scientific_validation':False})
    put(wd/'native_request.json',req)
    verify(out)
    return {'status':'NATIVE_WORK_INSTRUMENT_PREPARED_NO_CALL','work_id':p['work_id'],
            'research_question':p['request']['research_question'],'max_steps':MAX_STEPS}

def _work_dir(out):
    d=companion(out)
    dirs=list(d.iterdir()) if d.exists() else []
    if len(dirs)!=1 or not dirs[0].is_dir():raise ValueError('PREPARE_ONE_WORK_FIRST')
    return dirs[0]

def _request(lock,base_request,prior,step):
    payload={'original_native_work_request':base_request,
       'research_step':step,'max_steps':MAX_STEPS,
       'prior_research_steps':prior,
       'independent_scientific_instrument_available':step<MAX_STEPS,
       'response_schema':Step.model_json_schema(),
       'science_certification':False}
    sys=REASON_SYSTEM+'\n'+SYSTEM
    if step<MAX_STEPS:sys+='\n'+TOOL
    else:sys+='\nFinal available research step. probe_request MUST be null. Synthesize the best current\nscientific result including any error, counterexample, or inconclusiveness.\n'
    return {'model':lock['native_model'],'temperature':base_request['temperature'],
            'max_tokens':base_request['max_tokens'],
            'messages':[{'role':'system','content':sys},
                        {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}

def _probe(step):
    if step['probe_request'] is None:return None
    try:
        result=science_act_probe.evaluate(step['probe_request'])
        return {'status':'COMPUTED','result':result}
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return {'status':'REJECTED_INVALID_MODEL','error_type':type(exc).__name__,'error':str(exc)[:600]}

def _rows(wd):
    rows=[];pending=None
    for n in range(1,MAX_STEPS+1):
        sd=wd/f'step_{n:04d}'
        if not sd.exists():continue
        if n!=len(rows)+1 or (rows and rows[-1]['normalized']['next_action']=='RETURN'):
            raise ValueError('INVALID_RESEARCH_STEP_SEQUENCE')
        req=_request(read(wd/'lock.json'),read(wd/'native_request.json'),rows,n)
        if read(sd/'request.json')!=req:raise ValueError('STEP_REQUEST_REPLAY_MISMATCH')
        att=read(sd/'attempt.json')
        if att['request_sha256']!=digest(req) or att['previous_receipt_sha256']!=(digest(rows[-1]['receipt']) if rows else None):
            raise ValueError('STEP_ATTEMPT_CHAIN_CHANGED')
        if not (sd/'raw_response.json').is_file():
            pending=str(sd);break
        if not (sd/'normalized.json').is_file() or not (sd/'instrument_result.json').is_file() or not (sd/'receipt.json').is_file():
            pending=str(sd);break
        raw=read(sd/'raw_response.json')
        norm=Step.model_validate_json(raw['raw_text']).model_dump(mode='json')
        if n==MAX_STEPS and norm['probe_request'] is not None:raise ValueError('LAST_STEP_HAS_UNINTERPRETED_PROBE')
        if read(sd/'normalized.json')!=norm:raise ValueError('NORMALIZED_CHANGED')
        computed=_probe(norm)
        if read(sd/'instrument_result.json')!={'instrument_result':computed}:
            raise ValueError('INSTRUMENT_RESULT_CHANGED')
        rec={'number':n,'request_sha256':digest(req),'raw_sha256':hfile(sd/'raw_response.json'),
             'normalized_sha256':hfile(sd/'normalized.json'),
             'instrument_sha256':hfile(sd/'instrument_result.json'),
             'previous_receipt_sha256':digest(rows[-1]['receipt']) if rows else None}
        if read(sd/'receipt.json')!=rec:raise ValueError('RECEIPT_CHANGED')
        rows.append({'normalized':norm,'instrument_result':computed,'receipt':rec})
    return rows,pending

def _verify_work(out,wd,c):
    lock=read(wd/'lock.json');req=read(wd/'native_request.json')
    if lock['schema']!=PROTOCOL or lock['work_id']!=wd.name or lock['native_request_sha256']!=digest(req):
        raise ValueError('WORK_LOCK_INCONSISTENT')
    p=c['state'].get('pending_work')
    this_pending=p and p['work_id']==lock['work_id']
    if this_pending and p['request']!=lock['native_work_request']:
        raise ValueError('PENDING_WORK_CHANGED')
    if this_pending and p['status']=='REQUESTED':
        if digest(c['state'])!=lock['native_state_before_work_sha256'] or native._make_call_request(c,'work')!=req:
            raise ValueError('NATIVE_WORK_REQUEST_CHANGED')
    else:
        prior=[x for x in c['state']['work_history'] if x['work_id']==lock['work_id']]
        if len(prior)!=1 or prior[0]['execution_status']!='COMPLETED':
            raise ValueError('NATIVE_WORK_NOT_FOUND_AFTER_FOLLOWUP')
        native_work_request=read(native_dir(out)/'work'/lock['work_id']/'request.json')
        if native_work_request!=lock['native_work_request']:
            raise ValueError('NATIVE_REQUEST_ARTIFACT_CHANGED')
    rows,pending=_rows(wd)
    f=wd/'final_content.json'
    if f.exists():
        if pending or not rows or (len(rows)<MAX_STEPS and rows[-1]['normalized']['next_action']!='RETURN'):
            raise ValueError('FINAL_BEFORE_RESEARCH_FINISHED')
        if read(f)!=rows[-1]['normalized']['scientific_work_content']:
            raise ValueError('FINAL_CONTENT_NOT_LAST_MODEL_OUTPUT')
    if not (this_pending and p['status']=='REQUESTED'):
        product=read(native_dir(out)/'work'/lock['work_id']/'product.json')
        if not f.exists() or product['execution_status']!='COMPLETED' or product['content']!=read(f):
            raise ValueError('NATIVE_PRODUCT_NOT_COMPANION_SCIENCE')
    return {'steps':len(rows),'incomplete':pending,'finished':f.exists(),
            'native_status':p['status'] if this_pending else 'FOLLOWED_BY_NEXT_DECISION'}

def inspect(out:Path):
    v=verify(out);d=companion(out);details={}
    if d.exists():
        c=native.verify(native_dir(out))
        for wd in d.iterdir():
            if wd.is_dir():details[wd.name]=_verify_work(out,wd,c)
    return v|{'research_work':details}

def preview(out:Path):
    verify(out);wd=_work_dir(out);c=native.verify(native_dir(out))
    if c['state']['pending_work']['status']!='REQUESTED':raise ValueError('WORK_ALREADY_COMMITTED')
    rows,pending=_rows(wd)
    if pending:raise RuntimeError('UNSETTLED_RESEARCH_CALL_USE_RECOVER')
    if rows and rows[-1]['normalized']['next_action']=='RETURN':raise ValueError('RESEARCHER_RETURNED')
    if len(rows)>=MAX_STEPS:raise ValueError('RESEARCH_BUDGET_REACHED')
    req=_request(read(wd/'lock.json'),read(wd/'native_request.json'),rows,len(rows)+1)
    return {'status':'PREVIEW_NO_CALL','next_step':len(rows)+1,'request_sha256':digest(req),
            'remaining_paid_calls':MAX_STEPS-len(rows),'work_id':wd.name}

def recover(out:Path):
    verify(out);wd=_work_dir(out);rows,pending=_rows(wd)
    if not pending:raise ValueError('NO_INCOMPLETE_RESEARCH_ATTEMPT')
    sd=Path(pending)
    if not (sd/'raw_response.json').is_file():
        raise RuntimeError('RAW_NOT_SAVED_BILLING_UNCERTAIN_DO_NOT_RETRY')
    raw=read(sd/'raw_response.json')
    obj=Step.model_validate_json(raw['raw_text']).model_dump(mode='json')
    if sd.name==f'step_{MAX_STEPS:04d}' and obj['probe_request'] is not None:
        raise ValueError('LAST_STEP_HAS_UNINTERPRETED_PROBE')
    probe=_probe(obj)
    if not (sd/'normalized.json').exists():put(sd/'normalized.json',obj)
    if not (sd/'instrument_result.json').exists():put(sd/'instrument_result.json',{'instrument_result':probe})
    old=rows[-1] if rows else None
    rec={'number':len(rows)+1,'request_sha256':read(sd/'attempt.json')['request_sha256'],
         'raw_sha256':hfile(sd/'raw_response.json'),'normalized_sha256':hfile(sd/'normalized.json'),
         'instrument_sha256':hfile(sd/'instrument_result.json'),
         'previous_receipt_sha256':digest(old['receipt']) if old else None}
    if not (sd/'receipt.json').exists():put(sd/'receipt.json',rec)
    verify(out)
    return {'status':'SCIENTIFIC_STEP_RECOVERED_NO_CALL','step':len(rows)+1,
            'next_action':obj['next_action'],'instrument_status':probe['status'] if probe else None}

def next_step(out:Path,*,allow_paid=False,backend=None):
    pv=preview(out);base._require_paid(backend,allow_paid,1)
    wd=_work_dir(out);rows,_=_rows(wd);n=len(rows)+1
    sd=wd/f'step_{n:04d}';sd.mkdir(parents=True,exist_ok=False)
    request=_request(read(wd/'lock.json'),read(wd/'native_request.json'),rows,n)
    put(sd/'request.json',request)
    put(sd/'attempt.json',{'request_sha256':digest(request),
        'previous_receipt_sha256':digest(rows[-1]['receipt']) if rows else None,
        'status':'UNKNOWN_BILLING_UNTIL_RAW_SAVED','time_utc':now()})
    raw,usage,rid=base._invoke(request,backend)
    put(sd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:r=recover(out)
    except Exception as exc:raise RuntimeError('RAW_SAVED_NO_RETRY_USE_RECOVER') from exc
    r['new_paid_calls_this_command']=int(backend is None)
    return r

def finish(out:Path):
    verify(out);wd=_work_dir(out);rows,pending=_rows(wd)
    if pending or not rows or (len(rows)<MAX_STEPS and rows[-1]['normalized']['next_action']!='RETURN'):
        raise ValueError('RESEARCH_NOT_FINISHED')
    f=wd/'final_content.json'
    if f.exists():raise FileExistsError('FINAL_CONTENT_ALREADY_FROZEN')
    put(f,rows[-1]['normalized']['scientific_work_content'])
    verify(out)
    return {'status':'SCIENTIFIC_WORK_FROZEN_NO_CALL','work_id':wd.name,
            'steps':len(rows),'scientific_validation':False}

def commit_native(out:Path):
    verify(out);wd=_work_dir(out);f=wd/'final_content.json'
    if not f.exists():raise ValueError('FINISH_SCIENTIFIC_WORK_BEFORE_NATIVE_COMMIT')
    c=native.verify(native_dir(out));p=_valid_pending(c)
    if p['work_id']!=wd.name:raise ValueError('NATIVE_WORK_ID_CHANGED')
    # Native's backend API accepts a single completed JSON answer. The paid research
    # calls and any calculations are already immutable in the sidecar. This native
    # transition therefore requires no paid call and uses unchanged Work contracts.
    final=json.dumps(read(f),ensure_ascii=False,sort_keys=True)
    def saved_science_only(request):
        if digest(request)!=read(wd/'lock.json')['native_request_sha256']:
            raise ValueError('NATIVE_REQUEST_NOT_FROZEN')
        return final,None,'science-act-companion-saved-research-no-new-call'
    r=native.execute(native_dir(out),backend=saved_science_only,max_new_calls=1)
    verify(out)
    return {'status':'NATIVE_WORK_COMMITTED_FROM_SCIENTIFIC_RESEARCH_NO_PAID_CALL',
            'work_id':wd.name,'native_event':r['event'],'research_steps':len(_rows(wd)[0]),
            'scientific_validation':False}
