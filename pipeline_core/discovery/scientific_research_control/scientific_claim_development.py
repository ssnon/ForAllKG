"""Scientific Claim Development as a native W08-lineage Research Work executor.

New behavior: freely chosen claim development / constructive witness attempts with
optional exact finite-observable or CTMC calculations. NOT a claim certification gate.

Uses the existing lineage-fork and native decision/work-product machinery. A new
verified fork is prepared from an already native-committed prior episode.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
from . import w08_science_act_continuation as lineage
from . import finite_observable_probe
from . import science_act_probe
from . import research_work_v18_feedback_runner as native
from . import research_work_v12_stable_runner as base
from .research_work_v12_contracts import ReasonWorkContent
from .research_work_v12_stable_tools import REASON_SYSTEM

PROTOCOL='scientific-claim-development-native-v1'
MAX_STEPS=4
SYS='''You are executing a scientific ResearchWorkRequest independently selected by FAKG native
Research Control, with original source excerpts and earlier unverified research history.
Your aim is new scientific understanding (Level II: a discovery-worthy hypothesis), NOT
record management or safe-sounding prose. Search aggressively; claim conservatively.
Grounding controls claims, not imagination. Find and develop potentially bold mechanisms.

Research creatively. Make mechanistic models, derive nontrivial constraints, construct
counterexamples, run optional computations, falsify, revise or pivot as fits the question.
Never assume that a physically named mechanism implies different mathematical outputs.
A strong mathematical statement ("no invariant", "all models", "construct counterexample")
needs a demonstrable mathematical argument or actual witness. If not obtained, treat it as
an OPEN claim and pursue a useful avenue when one remains; or honestly return inconclusive.
A result of a hypothetical instrument is NOT experimental evidence. Distinguish exact
conditional results, heuristic reasoning, and literature-grounded observations.
Do not automatically stop after a plausible explanation. Do not mindlessly extend when
no scientific progress is achievable. You may freely RETURN at any research step.

You may use tools ONLY through an inert JSON probe_request. For finite observable comparison,
the instrument exhaustively enumerates (using floating arithmetic) probabilities of observations in finite model classes;
it does NOT compute event rates, durations, plume dynamics, or establish physical reality.
CTMC remains available only when kinetics are scientifically relevant. All tools optional.
Respond with exactly one JSON object matching the supplied response_schema.
'''
TOOL='''OPTIONAL instruments, never required:
1) probe_request={"kind":"finite_observable","payload":{"models":[
  {"name":"independent","conditions":{"c1":{"kind":"independent_binary_sites",
   "background":[0],"sites":[{"p_on":0.2,"emission":[1]},{"p_on":0.2,"emission":[1]}]}}},
  {"name":"shared","conditions":{"c1":{"kind":"finite_shared_states",
   "states":[{"probability":0.64,"observable":[0]},{"probability":0.36,"observable":[2]}]}}}],
   "conditions":["c1"],"thresholds":[0,1]}}.
Every model must specify exactly matching condition labels; all emissions are bounded
integer vectors and all probabilities finite. Comparison yields exact instantaneous
threshold exceedance and cross-mode covariance; it does NOT model event timing.
2) probe_request={"kind":"ctmc","payload":<existing science_act_probe data-only schema>}
  where CTMC schema has models and sweep (rate_group, multipliers). CTMC computes stationary
  occupancies and entry flux for explicit user-defined generators.
3) probe_request={"kind":"ctmc_threshold","payload":{"models":[
 {"name":"candidate","conditions":{"c1":{"states":[
 {"name":"D","observable":[0]},{"name":"B","observable":[2]}],
 "transitions":[{"source":"D","target":"B","rate":1.0},
 {"source":"B","target":"D","rate":2.0}]}}}],
 "conditions":["c1"],"thresholds":[0,1]}}.
 This uses the existing stationary CTMC solver, then computes threshold-defined
 entry flux, occupancy, mean durations and covariances for finite emissions.
 Assumes direct state observation, not noisy or binned experimental detection.
Request one probe at a time. For a probe, next_action MUST be CONTINUE so you can
interpret the returned result in another genuine research step. No probe in last step.
'''

class Strict(BaseModel):model_config=ConfigDict(extra='forbid')

class ResearchStep(Strict):
    operation:Literal['EXPLORE','MODEL','DERIVE','CONSTRUCT','TEST','CHALLENGE','REVISE','PIVOT','RETURN']
    scientific_work_content:ReasonWorkContent
    focal_claim:str=Field(min_length=12)
    claim_maturity:Literal['EXPLORATORY','CONDITIONAL_DERIVATION','CONSTRUCTIVE_WITNESS_PROPOSED','COMPUTATION_CHECKED','REFUTED','INCONCLUSIVE']
    research_delta:str=Field(min_length=15)
    next_action:Literal['CONTINUE','RETURN']
    next_research_question:str|None=None
    why_next_action:str=Field(min_length=15)
    probe_request:dict|None=None
    @model_validator(mode='after')
    def check(self):
        if self.next_action=='CONTINUE' and (not self.next_research_question or len(self.next_research_question)<14):
            raise ValueError('CONTINUE_REQUIRES_RESEARCH_QUESTION')
        if self.probe_request is not None and self.next_action!='CONTINUE':
            raise ValueError('PROBE_NEEDS_FOLLOWUP_INTERPRETATION')
        return self

def root(out):return lineage.safe_path(out)
def wd(out):return root(out)/'claim_work'
def put(p,obj):lineage.put(p,obj)
def read(p):return lineage.read(p)
def hashobj(x):return lineage.digest(x)
def hfile(p):return lineage.hfile(p)

def _check_probe(payload):
    if payload is None:return None
    if not isinstance(payload,dict) or set(payload)!={'kind','payload'}:
        raise ValueError('INVALID_TOOL_DISPATCH_KEYS')
    kind=payload['kind']
    try:
        if kind=='finite_observable':r=finite_observable_probe.evaluate(payload['payload'])
        elif kind=='ctmc':r=science_act_probe.evaluate(payload['payload'])
        elif kind=='ctmc_threshold':r=finite_observable_probe.evaluate_kinetic(payload['payload'])
        else:raise ValueError('UNKNOWN_INSTRUMENT')
        return {'status':'COMPUTED','kind':kind,'result':r}
    except (ValueError,TypeError,OverflowError,KeyError) as exc:
        return {'status':'REJECTED_INVALID_MODEL','kind':kind,'error_type':type(exc).__name__,'error':str(exc)[:450]}

def prepare(reference:Path,out:Path):
    """Reuse original fork validation, copy native event chain, not earlier sidecar execution."""
    out=root(out)
    if wd(out).exists():raise FileExistsError('CLAIM_RESEARCH_ALREADY_PREPARED')
    source=native.verify(Path(reference).expanduser().resolve())
    prior_state=source['state']
    if prior_state['decision_count']<5 or not any(
        x['work_id']=='work_0004_1c8f243a3943' and x['execution_status']=='COMPLETED'
        for x in prior_state['work_history']):
        raise ValueError('COMPLETE_W08_WORK4_AND_AUTONOMOUS_DECISION5_FIRST')
    sp=prior_state.get('pending_work')
    if not sp or sp['status']!='REQUESTED' or sp['request']['work_kind']!='REASON':
        raise ValueError('NEED_NATIVE_RESEARCH_DECISION_REQUESTING_REASON')
    r=lineage.prepare(reference,out)
    return r|{'claim_development':'NATIVE_WORK_CHOSEN_READY_TO_PREPARE'}

def prepare_work(out:Path):
    lineage.verify(out)
    out=root(out); c=native.verify(lineage.native_dir(out)); p=lineage._valid_pending(c)
    dest=wd(out)
    if dest.exists():raise FileExistsError('CLAIM_WORK_ALREADY_PREPARED')
    if (lineage.native_dir(out)/'work'/p['work_id']/'attempt.json').exists():
        raise ValueError('NATIVE_WORK_ALREADY_ATTEMPTED')
    req=native._make_call_request(c,'work')
    dest.mkdir(parents=True)
    put(dest/'lock.json',{'protocol':PROTOCOL,'work_id':p['work_id'],
         'native_state_sha256':hashobj(c['state']),
         'native_request_sha256':hashobj(req),'native_work_request':p['request'],
         'model':c['manifest']['model'],'max_steps':MAX_STEPS,
         'module_sha256':hfile(Path(__file__)),
         'finite_probe_sha256':hfile(Path(finite_observable_probe.__file__)),
         'scientific_validation':False})
    put(dest/'native_request.json',req)
    return {'status':'CLAIM_RESEARCH_PREPARED_NO_CALL','work_id':p['work_id'],
            'research_question':p['request']['research_question'],'max_steps':MAX_STEPS}

def _request(lock,native_request,rows,n):
    payload={'native_selected_work':native_request,
        'scientific_goal':'Develop actual scientific content; do not just classify claims.',
        'step':n,'max_steps':MAX_STEPS,
        'earlier_scientific_steps':rows,
        'response_schema':ResearchStep.model_json_schema()}
    sys=REASON_SYSTEM+'\n'+SYS
    if n<MAX_STEPS:sys+='\n'+TOOL
    else:sys+='\nFinal allowed step: probe_request MUST be null. Report scientific content even if inconclusive.\n'
    return {'model':lock['model'],'temperature':native_request['temperature'],
        'max_tokens':native_request['max_tokens'],
        'messages':[{'role':'system','content':sys},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}

def _rows(out):
    d=wd(out); lock=read(d/'lock.json');req=read(d/'native_request.json')
    rows=[];incomplete=None
    existing={p.name for p in d.iterdir() if p.is_dir() and p.name.startswith('step_')}
    valid={f'step_{i:04d}' for i in range(1,MAX_STEPS+1)}
    if not existing <= valid:raise ValueError('UNKNOWN_RESEARCH_STEP_DIR')
    for i in range(1,MAX_STEPS+1):
        p=d/f'step_{i:04d}'
        if not p.exists():
            if any(f'step_{j:04d}' in existing for j in range(i+1,MAX_STEPS+1)):
                raise ValueError('RESEARCH_STEP_GAP')
            break
        if rows and rows[-1]['normalized']['next_action']=='RETURN':
            raise ValueError('STEP_AFTER_VOLUNTARY_RETURN')
        expected=_request(lock,req,rows,i)
        if read(p/'request.json')!=expected:raise ValueError('RESEARCH_REQUEST_CHANGED')
        a=read(p/'attempt.json')
        prev=hashobj(rows[-1]['receipt']) if rows else None
        if a['request_sha256']!=hashobj(expected) or a['previous_receipt_sha256']!=prev:
            raise ValueError('RESEARCH_ATTEMPT_CHANGED')
        if not (p/'raw_response.json').exists():incomplete=str(p);break
        if not all((p/x).is_file() for x in ('normalized.json','instrument_result.json','receipt.json')):
            incomplete=str(p);break
        raw=read(p/'raw_response.json')
        step=ResearchStep.model_validate_json(raw['raw_text']).model_dump(mode='json')
        if i==MAX_STEPS and step['probe_request'] is not None:raise ValueError('UNINTERPRETED_LAST_PROBE')
        if step!=read(p/'normalized.json'):raise ValueError('SCIENTIFIC_RESULT_CHANGED')
        r=_check_probe(step['probe_request'])
        if read(p/'instrument_result.json')!={'instrument_result':r}:
            raise ValueError('INSTRUMENT_RESULT_CHANGED')
        receipt={'step':i,'request_sha256':hashobj(expected),
           'raw_sha256':hfile(p/'raw_response.json'),
           'normalized_sha256':hfile(p/'normalized.json'),
           'instrument_sha256':hfile(p/'instrument_result.json'),
           'previous_receipt_sha256':prev}
        if read(p/'receipt.json')!=receipt:raise ValueError('RESEARCH_RECEIPT_CHANGED')
        rows.append({'normalized':step,'instrument_result':r,'receipt':receipt})
    return rows,incomplete

def verify(out):
    l=lineage.verify(out);d=wd(out)
    if not d.exists():return l|{'claim_work':'NOT_PREPARED'}
    lock=read(d/'lock.json');nr=read(d/'native_request.json')
    if lock['protocol']!=PROTOCOL or lock['module_sha256']!=hfile(Path(__file__)) or lock['finite_probe_sha256']!=hfile(Path(finite_observable_probe.__file__)):
        raise ValueError('CLAIM_WORK_SOURCE_CHANGED')
    if lock['native_request_sha256']!=hashobj(nr):raise ValueError('NATIVE_REQUEST_CHANGED')
    c=native.verify(lineage.native_dir(out));p=c['state'].get('pending_work')
    if p and p['status']=='REQUESTED' and p['work_id']==lock['work_id']:
        if hashobj(c['state'])!=lock['native_state_sha256'] or native._make_call_request(c,'work')!=nr:
            raise ValueError('NATIVE_WORK_CONTEXT_CHANGED')
    else:
        completed=[x for x in c['state']['work_history'] if x['work_id']==lock['work_id'] and x['execution_status']=='COMPLETED']
        if len(completed)!=1:raise ValueError('CLAIM_WORK_NOT_FOUND_IN_NATIVE_HISTORY')
        prod=read(lineage.native_dir(out)/'work'/lock['work_id']/'product.json')
        if not (d/'final_content.json').is_file() or prod['content']!=read(d/'final_content.json'):
            raise ValueError('NATIVE_PRODUCT_DIFFERS_FROM_CLAIM_RESEARCH')
    rows,pending=_rows(out)
    if (d/'final_content.json').exists():
        if pending or not rows or (rows[-1]['normalized']['next_action']!='RETURN' and len(rows)!=MAX_STEPS):
            raise ValueError('PREMATURE_FINALIZATION')
        if read(d/'final_content.json')!=rows[-1]['normalized']['scientific_work_content']:
            raise ValueError('FINAL_CONTENT_CHANGED')
    return l|{'claim_work':{'work_id':lock['work_id'],'steps':len(rows),'incomplete':pending,
       'last_action':rows[-1]['normalized']['next_action'] if rows else None,
       'last_claim_maturity':rows[-1]['normalized']['claim_maturity'] if rows else None,
       'finalized':(d/'final_content.json').is_file()}}

def preview(out):
    v=verify(out);d=wd(out)
    if v.get('claim_work')=='NOT_PREPARED':raise ValueError('PREPARE_WORK_FIRST')
    if v['claim_work']['finalized']:raise ValueError('RESEARCH_ALREADY_FROZEN')
    c=native.verify(lineage.native_dir(out));p=lineage._valid_pending(c)
    if p['work_id']!=read(d/'lock.json')['work_id']:raise ValueError('WORK_ID_CHANGED')
    rows,pending=_rows(out)
    if pending:raise RuntimeError('UNSETTLED_ATTEMPT_NO_RETRY')
    if rows and rows[-1]['normalized']['next_action']=='RETURN':raise ValueError('RESEARCHER_RETURNED')
    if len(rows)>=MAX_STEPS:raise ValueError('RESEARCH_BUDGET_REACHED')
    request=_request(read(d/'lock.json'),read(d/'native_request.json'),rows,len(rows)+1)
    return {'status':'PREVIEW_NO_CALL','step':len(rows)+1,'work_id':p['work_id'],
        'request_sha256':hashobj(request),'remaining_paid_calls':MAX_STEPS-len(rows)}

def recover(out):
    verify(out);rows,pending=_rows(out)
    if not pending:raise ValueError('NO_RECOVERY_NEEDED')
    d=Path(pending)
    if not (d/'raw_response.json').is_file():raise RuntimeError('BILLING_UNCERTAIN_DO_NOT_RETRY')
    norm=ResearchStep.model_validate_json(read(d/'raw_response.json')['raw_text']).model_dump(mode='json')
    if len(rows)+1==MAX_STEPS and norm['probe_request'] is not None:
        raise ValueError('UNINTERPRETED_LAST_PROBE')
    r=_check_probe(norm['probe_request'])
    if not (d/'normalized.json').exists():put(d/'normalized.json',norm)
    if not (d/'instrument_result.json').exists():put(d/'instrument_result.json',{'instrument_result':r})
    prev=hashobj(rows[-1]['receipt']) if rows else None
    rec={'step':len(rows)+1,'request_sha256':read(d/'attempt.json')['request_sha256'],
      'raw_sha256':hfile(d/'raw_response.json'),
      'normalized_sha256':hfile(d/'normalized.json'),
      'instrument_sha256':hfile(d/'instrument_result.json'),
      'previous_receipt_sha256':prev}
    if not (d/'receipt.json').exists():put(d/'receipt.json',rec)
    verify(out)
    return {'status':'SCIENTIFIC_CLAIM_STEP_RECOVERED_NO_CALL','step':len(rows)+1,
        'next_action':norm['next_action'],'claim_maturity':norm['claim_maturity'],
        'instrument_status':r['status'] if r else None}

def next_step(out,*,allow_paid=False,backend=None):
    pv=preview(out);base._require_paid(backend,allow_paid,1)
    d=wd(out);rows,_=_rows(out);n=len(rows)+1
    step=d/f'step_{n:04d}';step.mkdir(parents=True,exist_ok=False)
    req=_request(read(d/'lock.json'),read(d/'native_request.json'),rows,n)
    put(step/'request.json',req)
    put(step/'attempt.json',{'request_sha256':hashobj(req),
         'previous_receipt_sha256':hashobj(rows[-1]['receipt']) if rows else None,
         'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED'})
    raw,usage,rid=base._invoke(req,backend)
    put(step/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:r=recover(out)
    except Exception as exc:raise RuntimeError('RAW_SAVED_USE_RECOVER_NO_RETRY') from exc
    return r|{'new_paid_calls_this_command':int(backend is None)}

def finish(out):
    v=verify(out);d=wd(out)
    if v['claim_work']['finalized']:raise FileExistsError('FINAL_ALREADY_FROZEN')
    rows,pending=_rows(out)
    if pending or not rows or (rows[-1]['normalized']['next_action']!='RETURN' and len(rows)!=MAX_STEPS):
        raise ValueError('RESEARCH_NOT_FINISHED')
    put(d/'final_content.json',rows[-1]['normalized']['scientific_work_content'])
    verify(out)
    return {'status':'CLAIM_RESEARCH_FROZEN_NO_CALL','steps':len(rows),
         'claim_maturity_self_reported':rows[-1]['normalized']['claim_maturity'],
         'scientific_validation':False}

def commit_native(out):
    verify(out);d=wd(out)
    if not (d/'final_content.json').is_file():raise ValueError('FINISH_FIRST')
    c=native.verify(lineage.native_dir(out));p=lineage._valid_pending(c)
    if p['work_id']!=read(d/'lock.json')['work_id']:raise ValueError('WORK_ID_CHANGED')
    raw=json.dumps(read(d/'final_content.json'),ensure_ascii=False,sort_keys=True)
    def saved_research(request):
        if hashobj(request)!=read(d/'lock.json')['native_request_sha256']:
            raise ValueError('NATIVE_REQUEST_DIFFERS')
        return raw,None,'claim-development-saved-research-no-new-call'
    r=native.execute(lineage.native_dir(out),backend=saved_research,max_new_calls=1)
    verify(out)
    return {'status':'NATIVE_CLAIM_WORK_COMMITTED_NO_PAID_CALL',
        'work_id':p['work_id'],'event':r['event'],'scientific_validation':False}
