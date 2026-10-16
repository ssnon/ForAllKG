"""M19-E2 paired study: shared initial idea, then follow-up without vs with CTMC tool.

The optional bounded tool is NOT available to the model as direct arbitrary code execution.
No existing episodes are changed. Every paid attempt is immutable and never automatically retried.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any,Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from . import research_work_v12_stable_runner as base
from . import m19_e2_ctmc as calc

SCHEMA='m19-e2-paired-grounded-development-json-v2'
PHASES=('initial','control','tool')
MODEL='openai/gpt-5.6-luna'
QUESTION=('Investigate a nontrivial, conditional scientific relation for threshold-detected bright events in a finite, immobilized ensemble of independently reversible on/off sites. '
          'Explore what observable event statistics can and cannot imply about molecular dynamics or a shared state. Construct a specific mathematical model or alternative, derive at least one consequence, and state what would remain unknown for Ye SERS. '
          'Do not assume the model fits Ye, and do not invent experimental evidence. You may choose whether a small exact numerical check is scientifically useful.')
INSTRUCTION=('You are investigating a scientific question, not merely reviewing a written claim. '
 'Actively construct or derive a meaningful conditional relation or causal alternative. You may test, revise, retain, or abandon it based on evidence. '
 'Distinguish mathematical model assumptions, new inference and original experimental facts. Do not assert scientific or empirical certification. '
 'A bounded independent calculator can evaluate exact stationary bright fraction, bright-event entry rate and mean bright interval for 1–3 independent binary continuous-time Markov sites. '
 'Only on_rates, off_rates, positive intensity weights and a total-intensity threshold are accepted. '
 'If you believe that calculation would check a claim worth investigating, provide a concrete model_spec; otherwise set it null. '
 'The existence of a calculator does not force you to use it. '
 'Respond with exactly one valid JSON object matching the supplied response schema. Use JSON object format, not Markdown. ')


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class Initial(Strict):
    question_refined:str=Field(min_length=15)
    model_and_assumptions:list[str]=Field(min_length=2)
    derived_claim:str=Field(min_length=30)
    mathematical_reasoning:list[str]=Field(min_length=1)
    candidate_consequence:str=Field(min_length=20)
    limitations:list[str]=Field(min_length=1)
    proposed_calculator_spec:dict[str,Any]|None
    claim_to_check:str|None
    next_scientific_question:str=Field(min_length=20)
    @model_validator(mode='after')
    def check(self):
        if self.proposed_calculator_spec is not None:
            if not self.claim_to_check or len(self.claim_to_check)<15:raise ValueError('SPEC_REQUIRES_CLAIM')
        return self

class Follow(Strict):
    refined_scientific_claim:str=Field(min_length=30)
    changed_or_retained_why:str=Field(min_length=30)
    mathematical_or_physical_reasoning:list[str]=Field(min_length=1)
    remaining_uncertainties:list[str]=Field(min_length=1)
    new_conditional_prediction:str=Field(min_length=20)
    next_research_action:str=Field(min_length=15)
    scientific_validation:Literal[False]


def repo_root()->Path:return Path(__file__).resolve().parents[3]
def guard(out:Path)->Path:
    repo=repo_root();root=(repo/'experiments'/'m19_e2'/'runs').resolve()
    out=out.expanduser().resolve()
    if root not in out.parents:raise ValueError('OUTPUT_OUTSIDE_REPO_M19_E2_RUNS')
    return repo

def _load(out:Path)->dict:return base.read(out/'lock.json')
def _initial_request(lock:dict)->dict:
    return {'model':lock['model'],'temperature':lock['temperature'],'max_tokens':lock['max_tokens'],
      'messages':[{'role':'system','content':INSTRUCTION},
       {'role':'user','content':base.canon({'research_question':QUESTION,'required_response_schema':Initial.model_json_schema()}).decode('utf-8')}],
      'role':'m19_e2_initial'}

def _follow_request(lock:dict,first:dict,tooldata:dict,phase:str)->dict:
    payload={'research_question':QUESTION,'original_unverified_scientific_result':first,
      'instruction':'Further develop or revise this SAME scientific result. Carry out actual mathematical or physical research, not just tone changes. Any claim requiring missing observations must remain conditional. The initial result is not certified. Do not assert that tool access by itself establishes real physics.',
      'response_schema':Follow.model_json_schema()}
    if phase=='tool':
        payload['external_calculator_result']=tooldata
        payload['external_calculator_label']='SEPARATELY_COMPUTED_MATHEMATICAL_CONSTRAINT_NOT_A_DISCOVERY'
    elif phase=='control':
        payload['external_calculator_result']=None
        payload['external_calculator_label']='NO_EXTERNAL_FEEDBACK_IN_THIS_ARM'
    else:raise ValueError('INVALID_FOLLOW_PHASE')
    return {'model':lock['model'],'temperature':lock['temperature'],'max_tokens':lock['max_tokens'],
      'messages':[{'role':'system','content':INSTRUCTION},
                  {'role':'user','content':base.canon(payload).decode('utf-8')}],
      'role':'m19_e2_'+phase}


def _assert_json_mode_ready(request:dict)->None:
    """Provider transport preflight: json_object format requires literal 'json' in input messages.

    Must run before attempt dir / billing metadata is written, so an invalid
    request can never consume an attempt or create an UNKNOWN_BILLING episode.
    """
    messages=request.get('messages')
    if not isinstance(messages,list) or not messages:
        raise ValueError('JSON_MODE_PREFLIGHT_MISSING_MESSAGES')
    joined='\n'.join(str(x.get('content','')) for x in messages if isinstance(x,dict))
    if 'json' not in joined.lower():
        raise ValueError('JSON_MODE_PREFLIGHT_MISSING_JSON_KEYWORD')
    if not isinstance(messages[0],dict) or 'json' not in str(messages[0].get('content','')).lower():
        raise ValueError('JSON_MODE_PREFLIGHT_MISSING_EXPLICIT_SYSTEM_JSON_INSTRUCTION')

def prepare(out:Path,*,model:str=MODEL,temperature:float=0,max_tokens:int=4000)->dict:
    guard(out)
    if out.exists():raise FileExistsError('EXPERIMENT_EXISTS')
    if not model or not 0<=temperature<=1 or max_tokens<1000 or max_tokens>16000:
        raise ValueError('INVALID_EXPERIMENT_CONFIG')
    lock={'schema':SCHEMA,'model':model,'temperature':float(temperature),'max_tokens':max_tokens,
          'question_sha256':base.sha(QUESTION),'instruction_sha256':base.sha(INSTRUCTION),
          'runner_source_sha256':base.file_sha(Path(__file__)),
          'calculator_source_sha256':base.file_sha(Path(calc.__file__)),
          'base_source_sha256':base.file_sha(Path(base.__file__)),
          'max_total_paid_attempts':3,'scientific_validation':False,
          'protocol':'V2 JSON-mode request compatibility repair; one shared initial output and paired independent CONTROL/TOOL responses; no scientific prompt-policy change other than transport-format requirement.'}
    out.mkdir(parents=True,exist_ok=False)
    base.write_new(out/'lock.json',lock)
    verify(out)
    return {'status':'PREPARED_NO_CALL','out':str(out),'model':model}

def _parse(raw:str,phase:str)->dict:
    obj=base._raw_obj(raw)
    if phase=='initial':return Initial.model_validate(obj).model_dump(mode='json')
    return Follow.model_validate(obj).model_dump(mode='json')

def _calculation_record(spec:dict|None)->dict:
    if spec is None:return {'status':'NOT_REQUESTED','result':None}
    try:return {'status':'EXACT_CTMC_RESULT','result':calc.evaluate(spec)}
    except ValueError as exc:
        return {'status':'UNSUPPORTED_MODEL_SPEC','result':None,'reason':str(exc)}

def _request(out:Path,phase:str)->dict:
    lock=_load(out)
    if phase=='initial':return _initial_request(lock)
    initial=base.read(out/'initial'/'normalized.json')
    feedback=base.read(out/'tool_check.json')
    return _follow_request(lock,initial,feedback,phase)

def _attempt_files(out:Path,phase:str):
    return out/phase

def verify(out:Path)->dict:
    guard(out)
    lock=_load(out)
    if lock['schema']!=SCHEMA or lock['runner_source_sha256']!=base.file_sha(Path(__file__)) or lock['calculator_source_sha256']!=base.file_sha(Path(calc.__file__)) or lock['base_source_sha256']!=base.file_sha(Path(base.__file__)) or lock['question_sha256']!=base.sha(QUESTION) or lock['instruction_sha256']!=base.sha(INSTRUCTION):
        raise ValueError('LOCKED_SOURCES_CHANGED')
    status={};attempts=0;normalized={}
    for phase in PHASES:
        d=out/phase
        if not d.exists():continue
        if phase!='initial' and ('initial' not in normalized or not (out/'tool_check.json').is_file()):
            raise ValueError('FOLLOW_WITHOUT_FROZEN_SHARED_INITIAL')
        if not d.is_dir() or not (d/'request.json').is_file() or not (d/'attempt.json').is_file():
            raise ValueError('INCOMPLETE_METADATA')
        req=_request(out,phase)
        if base.read(d/'request.json')!=req:raise ValueError('REQUEST_TAMPERED')
        att=base.read(d/'attempt.json')
        if att['request_sha256']!=base.sha(req):raise ValueError('ATTEMPT_TAMPERED')
        attempts+=1
        if not (d/'raw_response.json').is_file():
            if (d/'normalized.json').exists() or (d/'receipt.json').exists():raise ValueError('NO_RAW_BUT_RESULT_EXISTS')
            status[phase]='UNKNOWN_BILLING_NO_RAW_NO_RETRY';continue
        if not (d/'normalized.json').is_file() or not (d/'receipt.json').is_file():
            status[phase]='SAVED_RAW_OFFLINE_RECOVER_REQUIRED';continue
        obj=_parse(base.read(d/'raw_response.json')['raw_text'],phase)
        if base.read(d/'normalized.json')!=obj:raise ValueError('NORMALIZED_TAMPERED')
        receipt=base.read(d/'receipt.json')
        if receipt!={'request_file_sha256':base.file_sha(d/'request.json'),
            'attempt_file_sha256':base.file_sha(d/'attempt.json'),
            'raw_file_sha256':base.file_sha(d/'raw_response.json'),
            'normalized_file_sha256':base.file_sha(d/'normalized.json'),
            'response_id':base.read(d/'raw_response.json').get('response_id')}:
            raise ValueError('RECEIPT_TAMPERED')
        status[phase]='COMPLETE';normalized[phase]=obj
    if attempts>3:raise ValueError('ATTEMPT_CEILING_EXCEEDED')
    cp=out/'tool_check.json'
    if cp.exists():
        if 'initial' not in normalized:raise ValueError('CALCULATOR_WITHOUT_VERIFIED_INITIAL')
        spec=normalized['initial']['proposed_calculator_spec']
        expected=_calculation_record(spec)
        if base.read(cp)!=expected:raise ValueError('CALC_RESULT_TAMPERED')
    elif any(p in status for p in ('control','tool')):
        raise ValueError('FOLLOW_WITHOUT_CALC_RECORD')
    fp=out/'final_report.json'
    if fp.exists():
        if status.get('control')!='COMPLETE' or status.get('tool')!='COMPLETE':raise ValueError('REPORT_WITHOUT_COMPLETE_ARMS')
        if base.read(fp)!=_report(out,status,normalized,attempts):raise ValueError('REPORT_TAMPERED')
    return {'status':'VERIFIED_NO_CALL','phases':status,'attempt_count':attempts,
      'tool_check':base.read(cp)['status'] if cp.exists() else 'NOT_RUN',
      'report_exists':fp.is_file(),'model':lock['model'],'scientific_validation':False}

def preview(out:Path,phase:str)->dict:
    v=verify(out)
    if phase not in PHASES:raise ValueError('UNKNOWN_PHASE')
    if phase in v['phases']:raise ValueError('PHASE_ALREADY_ATTEMPTED_NEVER_RETRY')
    if phase!='initial' and (v['phases'].get('initial')!='COMPLETE' or v['tool_check']=='NOT_RUN'):
        raise ValueError('NEED_VERIFIED_SHARED_INITIAL_AND_TOOL_CHECK')
    if any(s!='COMPLETE' for s in v['phases'].values()):raise ValueError('UNRESOLVED_ATTEMPT_BLOCKS_NEW_CALLS')
    if v['report_exists']:raise ValueError('REPORT_ALREADY_FINAL')
    req=_request(out,phase)
    _assert_json_mode_ready(req)
    return {'status':'PREVIEW_NO_CALL','phase':phase,'request_sha256':base.sha(req),'model':req['model'], 'json_mode_ready':True}

def next_step(out:Path,phase:str,*,allow_paid:bool=False,max_total_paid_attempts:int=3,backend=None)->dict:
    if not 1<=max_total_paid_attempts<=3:raise ValueError('INVALID_PAID_CEILING')
    pv=preview(out,phase)
    if verify(out)['attempt_count']>=max_total_paid_attempts:raise RuntimeError('PAID_CEILING_REACHED')
    base._require_paid(backend,allow_paid,1)
    d=_attempt_files(out,phase);d.mkdir(parents=True,exist_ok=False)
    req=_request(out,phase)
    base.write_new(d/'request.json',req)
    base.write_new(d/'attempt.json',{'request_sha256':base.sha(req),'started_utc':datetime.now(timezone.utc).isoformat(),
       'status':'ATTEMPT_STARTED_BILLING_UNKNOWN_UNTIL_RAW'})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(d/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        _commit(out,phase)
    except Exception as e:
        raise RuntimeError('SAVED_RAW_RECOVER_OFFLINE_NO_RECALL: '+str(d)) from e
    return {'status':'COMMITTED','phase':phase,'paid_attempts_total':verify(out)['attempt_count']}

def _commit(out:Path,phase:str)->dict:
    d=_attempt_files(out,phase)
    raw=base.read(d/'raw_response.json')
    obj=_parse(raw['raw_text'],phase)
    if not (d/'normalized.json').exists():base.write_new(d/'normalized.json',obj)
    elif base.read(d/'normalized.json')!=obj:raise ValueError('EXISTING_NORMALIZED_DIFFERS')
    receipt={'request_file_sha256':base.file_sha(d/'request.json'),
      'attempt_file_sha256':base.file_sha(d/'attempt.json'),
      'raw_file_sha256':base.file_sha(d/'raw_response.json'),
      'normalized_file_sha256':base.file_sha(d/'normalized.json'),
      'response_id':raw.get('response_id')}
    if not (d/'receipt.json').exists():base.write_new(d/'receipt.json',receipt)
    elif base.read(d/'receipt.json')!=receipt:raise ValueError('EXISTING_RECEIPT_DIFFERS')
    return obj

def recover(out:Path)->dict:
    v=verify(out)
    missing=[k for k,s in v['phases'].items() if s!='COMPLETE']
    if not missing:return {'status':'NOTHING_TO_RECOVER_NO_CALL'}
    if len(missing)!=1 or v['phases'][missing[0]]!='SAVED_RAW_OFFLINE_RECOVER_REQUIRED':
        raise ValueError('UNKNOWN_BILLING_OR_AMBIGUOUS_RECOVERY_REFUSED')
    obj=_commit(out,missing[0]);verify(out)
    return {'status':'RECOVERED_NO_PAID','phase':missing[0]}

def calculate(out:Path)->dict:
    v=verify(out)
    if (out/'tool_check.json').exists():raise FileExistsError('CHECK_ALREADY_PERFORMED')
    if v['phases'].get('initial')!='COMPLETE':raise ValueError('INITIAL_NOT_COMMITTED')
    spec=base.read(out/'initial'/'normalized.json')['proposed_calculator_spec']
    result=_calculation_record(spec)
    base.write_new(out/'tool_check.json',result)
    verify(out)
    return {'status':result['status'],'no_paid_calls':True,'calculator_result_sha256':base.sha(result)}

def _report(out:Path,status:dict,normalized:dict,attempts:int)->dict:
    rows={k:{'response':normalized[k], 'request_sha256':base.file_sha(out/k/'request.json'),
       'raw_sha256':base.file_sha(out/k/'raw_response.json')} for k in PHASES}
    return {'schema':SCHEMA,'scientific_validation':False,'attempts':attempts,
            'tool_check':base.read(out/'tool_check.json'),
            'shared_initial_and_paired_followups':rows,
            'interpretation':'Paired tool-feedback effect on a fixed initial hypothesis only; not evidence of autonomous Level II discovery, and not a validated FAKG architecture improvement.'}

def finalize(out:Path)->dict:
    v=verify(out)
    if v['report_exists']:raise FileExistsError('ALREADY_FINALIZED')
    if set(v['phases'])!=set(PHASES) or any(s!='COMPLETE' for s in v['phases'].values()):
        raise ValueError('INCOMPLETE_PAIRED_PROBE')
    norm={k:base.read(out/k/'normalized.json') for k in PHASES}
    data=_report(out,v['phases'],norm,v['attempt_count'])
    base.write_new(out/'final_report.json',data)
    verify(out)
    return {'status':'FINALIZED_NO_CALL','scientific_validation':False}
