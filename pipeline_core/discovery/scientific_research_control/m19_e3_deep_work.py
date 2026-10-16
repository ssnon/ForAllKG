"""M19-E3 Work4: isolated self-directed scientific micro-research on native D4.

Never modifies the source E3 fork or W08. ResearchWorkProduct is a candidate
until separately reviewed; native Work event is NOT committed by this runner.
No evaluator-discovered Markov answers are present in any scientific prompt.
"""
from __future__ import annotations
import copy, json, os, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from . import m19_e3_autonomy as entry
from . import research_work_v12_stable_runner as base
from . import research_work_v18_feedback_runner as v18
from .research_work_v12_contracts import ReasonWorkContent, ResearchWorkProduct

SCHEMA='m19-e3-autonomous-selected-deep-work-v1'
D4_SHA='217556c19507121aca52655e0af59e5e101e68a6d173c408ad825d65cbcc184a'
MAX_STEPS=3
NATIVE='native_work_episode'

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class MicroOutput(Strict):
    operation: Literal['CONSTRUCT','DERIVE','EXPLORE','CHALLENGE','DEVELOP','PIVOT_RECOMMEND','RETURN']
    research_delta: str=Field(min_length=20)
    provisional_work_content: ReasonWorkContent
    next_step: Literal['CONTINUE','RETURN']
    next_question_or_test: str | None=None
    reason_for_choice: str=Field(min_length=20)
    source_or_prior_work_refs: list[str]=Field(default_factory=list)
    @model_validator(mode='after')
    def check_step(self):
        if self.next_step=='CONTINUE' and (not self.next_question_or_test or len(self.next_question_or_test.strip())<15):
            raise ValueError('CONTINUE_REQUIRES_RESEARCH_QUESTION')
        if len(set(self.source_or_prior_work_refs))!=len(self.source_or_prior_work_refs):
            raise ValueError('DUPLICATE_SOURCE_REFS')
        return self

SYSTEM_APPEND='''
M19 E3 SCIENTIFIC DEEP WORK. Return one valid JSON object matching the supplied micro_output_schema.
This is a scientific investigation of the Research Control's actual WorkRequest, NOT an exercise in producing longer critiques.
Actively construct, mathematically derive, expand, challenge or revise a useful causal/statistical relationship.
Use the actual finite-ensemble assumptions and distinguish a derived equality from an intuitive analogy.
For any claimed necessity or scaling relation, show the physical/mathematical assumptions and the relevant derivation or a precise obstruction. You may discover no valid invariant and report INCONCLUSIVE.
Do not claim an independently measured Ye result, new experiment, or Level-II discovery. Prior W06-W08 outputs are unverified work, not ground truth.
Choose next_step=CONTINUE only if a specific productive remaining research action is identified; otherwise RETURN, including after an honest failed derivation.
No mandatory GENERATE-CRITIQUE-REVISE ritual, no mechanism selected for you, no forced number of calls.
Do not assume that a paper/Work ID in source context is a valid citation; use only IDs in the provided eligible evidence inventory. You may provide source_or_prior_work_refs=[]; unknown IDs are preserved as unsupported, never treated as evidence.
JSON output shape: {"operation":"DERIVE","research_delta":"...","provisional_work_content":{"attempted_question":"...","premises_used":[],"reasoning_or_operations":[],"candidate_consequence":null,"counterexample_or_obstruction":null,"limits_and_uncertainties":[],"work_conclusion":"INCONCLUSIVE"},"next_step":"RETURN","next_question_or_test":null,"reason_for_choice":"...","source_or_prior_work_refs":[]}.
'''

def repo_root()->Path:return Path(__file__).resolve().parents[3]

def guard(out:Path)->Path:
    root=(repo_root()/'experiments'/'m19_e3_work'/'runs').resolve()
    p=out.expanduser().resolve()
    if p==root or root not in p.parents:raise ValueError('OUTSIDE_REPO_M19_E3_WORK_RUNS')
    return p

def tree_shas(p:Path)->dict:
    if not p.is_dir() or p.is_symlink():raise ValueError('EXPECTED_REAL_SOURCE_DIR')
    d={}
    for f in sorted(p.rglob('*')):
        if f.is_symlink():raise ValueError('SYMLINK_IN_RESEARCH_SOURCE')
        if f.is_file():d[f.relative_to(p).as_posix()]=base.file_sha(f)
    return d

def _pin_e3(e3:Path)->tuple[dict,dict]:
    status=entry.verify(e3)
    if status['status']!='DECISION4_COMMITTED_VERIFIED' or status['fork_state_sha256']!=D4_SHA or status['pending_work_kind']!='REASON':
        raise ValueError('E3_D4_STATE_NOT_PINNED_RESEARCH_CHOICE')
    c=v18.verify(e3/entry.FORK_SUBDIR)
    s=c['state'];p=s.get('pending_work') or {}
    if p.get('status')!='REQUESTED' or p['request']['work_kind']!='REASON':
        raise ValueError('E3_NOT_PENDING_REASON_WORK')
    return c,status

def prepare(*,e3:Path,out:Path)->dict:
    dest=guard(out)
    e3=e3.expanduser().resolve(strict=True)
    if dest.exists():raise FileExistsError('WORK_TRIAL_ALREADY_EXISTS_NO_OVERWRITE')
    c,z=_pin_e3(e3)
    src=e3/entry.FORK_SUBDIR
    srcfiles=tree_shas(src)
    native_req=v18._make_call_request(c,'work')
    if 'json' not in ' '.join(x['content'] for x in native_req['messages']).lower():
        raise ValueError('ORIGINAL_WORK_JSON_MODE_NOT_READY')
    if native_req['input_state_sha256']!=D4_SHA:raise ValueError('ORIGINAL_WORK_STATE_SHA_MISMATCH')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.preparing')
    if tmp.exists():raise FileExistsError('STALE_PREPARATION_NEEDS_MANUAL_REVIEW')
    tmp.mkdir(mode=0o700)
    try:
        shutil.copytree(src,tmp/NATIVE)
        if tree_shas(tmp/NATIVE)!=srcfiles:raise ValueError('CLONED_E3_WORK_SOURCE_MISMATCH')
        if base.sha(v18.verify(tmp/NATIVE)['state'])!=D4_SHA:raise ValueError('CLONED_STATE_MISMATCH')
        payload=json.loads(native_req['messages'][1]['content'])
        refs=set(payload['work_request']['available_inputs'])|set(payload['work_request']['source_ids'])
        refs|={x['artifact_ref'] for x in payload['read_work_products']}
        refs|={x['id'] for x in payload['read_scientific_sources']}
        # Valid WorkProduct IDs (no newly invented current-work or future-work IDs).
        lock={'schema':SCHEMA,'e3_path':str(e3), 'source_e3_file_shas':srcfiles,
             'selected_d4_state_sha256':D4_SHA,
             'work_id':c['state']['pending_work']['work_id'],
             'work_request_sha256':base.sha(c['state']['pending_work']['request']),
             'native_work_request':native_req,
             'native_work_request_sha256':base.sha(native_req),
             'model':native_req['model'],'max_steps':MAX_STEPS,
             'eligible_evidence_refs':sorted(refs),
             'base_source_sha256':base.file_sha(Path(base.__file__)),
             'v18_source_sha256':base.file_sha(Path(v18.__file__)),
             'e3_source_sha256':base.file_sha(Path(entry.__file__)),
             'work_runner_source_sha256':base.file_sha(Path(__file__)),
             'scientific_validation':False,
             'external_evaluator_answers_injected':False,
             'native_work_event_committed':False}
        base.write_new(tmp/'lock.json',lock)
        tmp.rename(dest)
    except Exception:raise
    verify(dest)
    return {'status':'PREPARED_D4_NATIVE_WORK_REQUEST_NO_CALL','out':str(dest),'work_id':lock['work_id'],
            'research_question':payload['work_request']['research_question'],
            'd4_sha256':D4_SHA, 'model':native_req['model'],'scientific_validation':False}

def step_dir(out:Path,i:int)->Path:return out/'micro_steps'/f'step_{i:04d}'

def _request(lock:dict, completed:list, index:int)->dict:
    native=lock['native_work_request']
    payload=json.loads(native['messages'][1]['content'])
    payload['micro_output_schema']=MicroOutput.model_json_schema()
    payload['eligible_evidence_refs']=lock['eligible_evidence_refs']
    payload['deep_work_instruction']='Investigate native Decision4 WorkRequest without prior evaluator solutions.'
    if completed:
        payload['previous_deep_work_steps']=[{'step':i+1,'operation':r['output']['operation'],
            'research_delta':r['output']['research_delta'],
            'unverified_content':r['output']['provisional_work_content'],
            'next_question_or_test':r['output']['next_question_or_test'],
            'reason_for_choice':r['output']['reason_for_choice']} for i,r in enumerate(completed)]
    return {'model':native['model'],'temperature':native['temperature'],'max_tokens':native['max_tokens'],
            'role':'m19_e3_native_selected_deep_work','input_state_sha256':D4_SHA,
            'messages':[{'role':'system','content':native['messages'][0]['content']+'\n'+SYSTEM_APPEND},
                       {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}

def _parse(raw:str)->dict:
    return MicroOutput.model_validate(base._raw_obj(raw)).model_dump(mode='json')

def _step_record(out:Path,lock:dict,i:int,prior_sha:str|None,expected_req:dict)->tuple[dict,dict]:
    sd=step_dir(out,i)
    req=base.read(sd/'request.json');att=base.read(sd/'attempt.json')
    if req!=expected_req or att['request_sha256']!=base.sha(req) or att['previous_step_sha256']!=prior_sha:
        raise ValueError('STEP_REQUEST_OR_PREVIOUS_SHA_MISMATCH')
    raw=base.read(sd/'raw_response.json');output=_parse(raw['raw_text'])
    normalized=sd/'normalized.json';recpath=sd/'record.json'
    if not normalized.exists() or base.read(normalized)!=output:raise ValueError('STEP_NORMALIZED_MISSING_OR_CHANGED')
    invalid=sorted(set(output['source_or_prior_work_refs'])-set(lock['eligible_evidence_refs']))
    note={'invalid_ref_ids_not_accepted_as_evidence':invalid,
          'evidence_claims_still_unverified':True,'model_sources_unverified':True}
    record={'step':i,'previous_step_sha256':prior_sha,
       'request_file_sha256':base.file_sha(sd/'request.json'),
       'attempt_file_sha256':base.file_sha(sd/'attempt.json'),
       'raw_file_sha256':base.file_sha(sd/'raw_response.json'),
       'normalized_file_sha256':base.file_sha(normalized),
       'response_id':raw.get('response_id'),'usage':raw.get('usage'),
       'operation':output['operation'],'next_step':output['next_step'],
       'source_reference_audit':note,'scientific_validation':False}
    if not recpath.exists() or base.read(recpath)!=record:raise ValueError('STEP_RECORD_MISSING_OR_CHANGED')
    return {'output':output,'record':record,'sha':base.sha(record)},note

def verify(out:Path)->dict:
    out=guard(out)
    lock=base.read(out/'lock.json')
    if lock.get('schema')!=SCHEMA or lock['selected_d4_state_sha256']!=D4_SHA or lock['max_steps']!=MAX_STEPS:
        raise ValueError('BAD_DEEP_WORK_LOCK')
    for k,mod in [('base_source_sha256',base),('v18_source_sha256',v18),('e3_source_sha256',entry)]:
        if base.file_sha(Path(mod.__file__))!=lock[k]:raise ValueError('SOURCE_DEPENDENCY_CHANGED_'+k)
    if lock['work_runner_source_sha256']!=base.file_sha(Path(__file__)):
        raise ValueError('DEEP_WORK_RUNNER_SOURCE_CHANGED')
    e3=Path(lock['e3_path']);c,z=_pin_e3(e3)
    if tree_shas(e3/entry.FORK_SUBDIR)!=lock['source_e3_file_shas']:
        raise ValueError('E3_SOURCE_FORK_CHANGED')
    nativestatus=v18.verify(out/NATIVE)
    if base.sha(nativestatus['state'])!=D4_SHA:raise ValueError('NATIVE_RESEARCH_WORK_EPISODE_WAS_MUTATED')
    curreq=v18._make_call_request(nativestatus,'work')
    if curreq!=lock['native_work_request'] or base.sha(curreq)!=lock['native_work_request_sha256']:
        raise ValueError('NATIVE_WORK_PROMPT_CHANGED')
    if lock['work_id']!=c['state']['pending_work']['work_id'] or lock['work_request_sha256']!=base.sha(c['state']['pending_work']['request']):
        raise ValueError('NATIVE_WORK_REQUEST_CHANGED')
    dirs=sorted((out/'micro_steps').glob('step_*')) if (out/'micro_steps').exists() else []
    if len(dirs)>MAX_STEPS:raise ValueError('EXCEEDED_MICRO_BUDGET')
    completed=[];incomplete=None;parent=None
    for i,sd in enumerate(dirs,1):
        if sd.name!=f'step_{i:04d}':raise ValueError('NONCONTIGUOUS_MICRO_STEPS')
        if completed and completed[-1]['output']['next_step']=='RETURN':raise ValueError('CONTINUED_AFTER_RETURN')
        req=_request(lock,completed,i)
        if base.read(sd/'request.json')!=req:raise ValueError('MISMATCHED_REPLAYED_STEP_REQUEST')
        att=base.read(sd/'attempt.json')
        if att['request_sha256']!=base.sha(req) or att['previous_step_sha256']!=parent:
            raise ValueError('BAD_ATTEMPT_METADATA')
        if not (sd/'raw_response.json').is_file():
            incomplete={'step':i,'status':'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY'}
            break
        if not (sd/'record.json').is_file():
            incomplete={'step':i,'status':'RAW_SAVED_OFFLINE_RECOVER_REQUIRED'}
            break
        rr,note=_step_record(out,lock,i,parent,req)
        completed.append(rr);parent=rr['sha']
    if incomplete and dirs[-1]!=step_dir(out,incomplete['step']):
        raise ValueError('INCOMPLETE_IS_NOT_LAST_ATTEMPT')
    finalpath=out/'final_product.json';tracepath=out/'deep_work_trace.json'
    if finalpath.exists()!=tracepath.exists():raise ValueError('HALF_FINALIZED')
    if finalpath.exists():
        if incomplete:raise ValueError('FINALIZED_WITH_INCOMPLETE_ATTEMPT')
        product,trace=_final_artifacts(out,lock,completed)
        if base.read(finalpath)!=product or base.read(tracepath)!=trace:
            raise ValueError('FINAL_PRODUCT_TRACE_MISMATCH')
    return {'status':'VERIFIED_NO_CALL','work_id':lock['work_id'],
        'decision4_state_sha256':D4_SHA,'completed_count':len(completed),
        'attempt_count':len(dirs), 'incomplete':incomplete,
        'next_choice':(completed[-1]['output']['next_step'] if completed else None),
        'finalized':finalpath.exists(), 'scientific_validation':False,'steps':completed,
        'model':lock['model'],'native_episode_modified':False}

def preview(out:Path)->dict:
    out=guard(out);v=verify(out)
    if v['incomplete']:raise RuntimeError('INCOMPLETE_CALL_MUST_BE_RESOLVED_NO_RETRY')
    if v['finalized']:raise RuntimeError('FINALIZED_NO_NEW_STEPS')
    if v['completed_count']>=MAX_STEPS:raise RuntimeError('MAX_STEPS_REACHED')
    if v['next_choice']=='RETURN':raise RuntimeError('AUTONOMOUS_RETURN_SELECTED')
    lock=base.read(out/'lock.json')
    req=_request(lock,v['steps'],v['completed_count']+1)
    if 'json' not in ' '.join(m['content'] for m in req['messages']).lower():
        raise ValueError('JSON_OBJECT_PROVIDER_PREFLIGHT_FAILED')
    return {'status':'PREVIEW_NO_CALL','work_id':lock['work_id'],
        'next_step':v['completed_count']+1,'model':lock['model'],
        'request_sha256':base.sha(req), 'json_mode_ready':True,'paid_calls':0}

def _save_step(out:Path,i:int,lock:dict)->dict:
    # Existing raw is immutable; parser or citation failure never initiates another call.
    v=verify(out)   # returns incomplete with saved raw until normalized/recorded
    if not v['incomplete'] or v['incomplete']['step']!=i or v['incomplete']['status']!='RAW_SAVED_OFFLINE_RECOVER_REQUIRED':
        raise RuntimeError('NO_SAVED_RAW_TO_COMMIT')
    sd=step_dir(out,i)
    req=base.read(sd/'request.json');raw=base.read(sd/'raw_response.json')
    output=_parse(raw['raw_text'])
    normfile=sd/'normalized.json'
    if normfile.exists():
        if base.read(normfile)!=output:raise ValueError('EXISTING_NORMALIZED_DIFFERS')
    else:base.write_new(normfile,output)
    parent=v['steps'][-1]['sha'] if v['steps'] else None
    invalid=sorted(set(output['source_or_prior_work_refs'])-set(lock['eligible_evidence_refs']))
    record={'step':i,'previous_step_sha256':parent,
       'request_file_sha256':base.file_sha(sd/'request.json'),
       'attempt_file_sha256':base.file_sha(sd/'attempt.json'),
       'raw_file_sha256':base.file_sha(sd/'raw_response.json'),
       'normalized_file_sha256':base.file_sha(sd/'normalized.json'),
       'response_id':raw.get('response_id'),'usage':raw.get('usage'),
       'operation':output['operation'],'next_step':output['next_step'],
       'source_reference_audit':{
           'invalid_ref_ids_not_accepted_as_evidence':invalid,
           'evidence_claims_still_unverified':True,'model_sources_unverified':True},
       'scientific_validation':False}
    base.write_new(sd/'record.json',record)
    verify(out)
    return {'status':'SAVED_RAW_COMMITTED_NO_NEW_CALL','step':i,
       'next_choice':output['next_step'],'invalid_ref_ids':invalid}

def next_step(out:Path,*,allow_paid:bool=False,max_total_paid_attempts:int=3,backend=None)->dict:
    out=guard(out)
    if not 1<=max_total_paid_attempts<=MAX_STEPS:raise ValueError('INVALID_PAID_CEILING')
    pv=preview(out);v=verify(out)
    if v['attempt_count']>=max_total_paid_attempts:raise RuntimeError('PAID_CEILING_REACHED')
    base._require_paid(backend,allow_paid,1)
    lock=base.read(out/'lock.json');req=_request(lock,v['steps'],pv['next_step'])
    sd=step_dir(out,pv['next_step'])
    if sd.exists():raise FileExistsError('EXISTING_STEP_NEVER_REPEAT_PAID_CALL')
    sd.mkdir(parents=True,exist_ok=False)
    base.write_new(sd/'request.json',req)
    base.write_new(sd/'attempt.json',{'request_sha256':base.sha(req),
        'previous_step_sha256':v['steps'][-1]['sha'] if v['steps'] else None,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'STARTED_UNKNOWN_BILLING_UNTIL_SAVED_RESPONSE'})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(sd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        r=_save_step(out,pv['next_step'],lock)
    except Exception as exc:
        raise RuntimeError('RAW_SAVED_OFFLINE_RECOVER_REQUIRED: '+str(sd)) from exc
    return {'status':'STEP_COMMITTED','work_id':lock['work_id'],
       'step':pv['next_step'],'next_choice':r['next_choice'],
       'invalid_ref_ids':r['invalid_ref_ids'],
       'new_paid_calls_this_command':int(backend is None)}

def recover(out:Path)->dict:
    out=guard(out);v=verify(out)
    if not v['incomplete']:return {'status':'NO_SAVED_RAW_TO_RECOVER_NO_CALL'}
    if v['incomplete']['status']!='RAW_SAVED_OFFLINE_RECOVER_REQUIRED':
        raise RuntimeError('UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
    return _save_step(out,v['incomplete']['step'],base.read(out/'lock.json'))

def _final_artifacts(out:Path,lock:dict,steps:list)->tuple[dict,dict]:
    last=steps[-1];content=last['output']['provisional_work_content'];i=len(steps)
    c=v18.verify(out/NATIVE);pending=c['state']['pending_work']
    input_sha=base.work_input_digest(pending['work_id'],pending['request'],c['manifest']['source_snapshot_sha256'])
    product=ResearchWorkProduct(work_id=lock['work_id'],work_kind='REASON',
        execution_status='COMPLETED',inputs_sha256=input_sha,
        output_artifact_sha256=base.sha(content),origin='LLM_REASONING_UNVERIFIED',
        content=content,raw_response_sha256=base.file_sha(step_dir(out,i)/'raw_response.json'),
        scientific_validation=False).model_dump(mode='json')
    trace={'schema':SCHEMA,'source_d4_state_sha256':D4_SHA,'work_id':lock['work_id'],
        'native_work_request_sha256':lock['native_work_request_sha256'],
        'micro_steps':[{'step':j,'record_sha256':s['sha'],'operation':s['output']['operation'],
         'research_delta':s['output']['research_delta'],
         'next_step':s['output']['next_step'],'why':s['output']['reason_for_choice'],
         'invalid_ref_ids':s['record']['source_reference_audit']['invalid_ref_ids_not_accepted_as_evidence']}
         for j,s in enumerate(steps,1)],
        'final_product_sha256':base.sha(product),
        'scientific_validation':False,
        'native_work_event_committed':False,
        'scope':'Candidate ResearchWorkProduct for scientific review, not automatically recorded in native Decision state.'}
    return product,trace

def finalize(out:Path)->dict:
    out=guard(out);v=verify(out)
    if v['incomplete'] or v['finalized']:raise RuntimeError('UNRESOLVED_OR_FINAL_EXISTS')
    if not v['completed_count']:raise RuntimeError('NO_RESEARCH_OUTPUT')
    if v['completed_count']<MAX_STEPS and v['next_choice']!='RETURN':
        raise RuntimeError('CONTINUATION_SELECTED_CANNOT_FINALIZE_EARLY')
    lock=base.read(out/'lock.json')
    product,trace=_final_artifacts(out,lock,v['steps'])
    base.write_new(out/'final_product.json',product)
    base.write_new(out/'deep_work_trace.json',trace)
    verify(out)
    return {'status':'CANDIDATE_WORK_PRODUCT_FINALIZED_NO_CALL','work_id':lock['work_id'],
        'steps':v['completed_count'],'native_work_event_committed':False,
        'scientific_validation':False}

def inspect(out:Path)->dict:
    v=verify(out)
    return {k:val for k,val in v.items() if k!='steps'}
