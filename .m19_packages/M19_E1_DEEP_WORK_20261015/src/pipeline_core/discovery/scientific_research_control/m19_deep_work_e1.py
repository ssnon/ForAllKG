"""M19-E1: independently frozen W08 Work3 B/C experiment. Add-only; never commits W08.

A recorded attempt is not retried. Persisted responses are recovered without network.
No validation asserts any discovery or empirical science.
"""
from __future__ import annotations
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import research_work_v12_stable_runner as base
from . import research_work_v18_feedback_runner as v18
from .research_work_v12_contracts import ReasonWorkContent, ResearchWorkProduct

EXPECTED_W08_SHA = 'e85eb24b89864401a36b27bd5a659afcf81c5239db65a4bce3c5b678dae29a3b'
EXPECTED_WORK_ID = 'work_0003_cd32f5d823b4'
MAX_PER_ARM = 3
ARMS = ('B','C')
SCHEMA = 'm19-scientific-deep-work-e1.0'
RESEARCH_INSTRUCTION = '''\nM19 SCIENTIFIC WORK: Act as an investigating scientist, not a scientific-claim auditor.
Produce a substantive provisional result NOW, not just a research plan. You may construct a new
causal account, derive a conditional law or prediction, explore alternatives, challenge a strong
claim, or develop/revise/abandon a proposal. Explicitly distinguish source observations,
model assumptions, derivations, counterexamples and unverified hypotheses. Use physical and
mathematical reasoning, not merely changes of wording. Serious conventional explanations
limit certification, not the freedom to explore. Do not fabricate calculations, references,
measurements or evidence. Do not claim scientific certification. Choose RETURN when research
has reached a useful provisional stopping point; otherwise CONTINUE with a substantive next
research question. Keep the completed provisional_work_content coherent as an ordinary
ReasonWorkContent that can already stand alone. The JSON response MUST be exactly an object:
{"operation":"CONSTRUCT|DERIVE|EXPLORE|CHALLENGE|DEVELOP|PIVOT_RECOMMEND|RETURN",
 "research_delta":"concrete scientific work accomplished",
 "provisional_work_content":{"attempted_question":"...","premises_used":["..."],
 "reasoning_or_operations":["..."],"candidate_consequence":null,
 "counterexample_or_obstruction":null,"limits_and_uncertainties":["..."],
 "work_conclusion":"DERIVATION_PROPOSED|COUNTEREXAMPLE_PROPOSED|INCONCLUSIVE"},
 "next_step":"CONTINUE|RETURN","next_question_or_test":null,
 "reason_for_choice":"...","source_or_prior_work_refs":[]}
Next question must be nonempty when CONTINUE. Do not respond with just a plan.
'''

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class StepOutput(Strict):
    operation: Literal['CONSTRUCT','DERIVE','EXPLORE','CHALLENGE','DEVELOP','PIVOT_RECOMMEND','RETURN']
    research_delta: str = Field(min_length=25)
    provisional_work_content: ReasonWorkContent
    next_step: Literal['CONTINUE','RETURN']
    next_question_or_test: str | None
    reason_for_choice: str = Field(min_length=15)
    source_or_prior_work_refs: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_next(self):
        if self.next_step=='CONTINUE' and (not self.next_question_or_test or len(self.next_question_or_test.strip())<15):
            raise ValueError('CONTINUE requires a substantial next research question')
        if len(set(self.source_or_prior_work_refs))!=len(self.source_or_prior_work_refs):
            raise ValueError('duplicate evidence references')
        return self


def _safe_repo_root(out:Path)->Path:
    # Requires user-selected experiment root within checkout; cannot write HOME/evaluation.
    repo=Path(__file__).resolve().parents[3]  # pipeline_core/discovery/scientific_research_control/file -> repo
    allowed=(repo/'experiments'/'m19_e1').resolve()
    o=out.expanduser().resolve()
    if o==allowed or allowed not in o.parents:raise ValueError('OUTPUT_MUST_BE_NEW_CHILD_UNDER_REPO_EXPERIMENTS_M19_E1')
    return repo


def _step_dir(out:Path,arm:str,index:int)->Path:
    return out/arm/'micro_steps'/f'step_{index:04d}'


def _run_lock(out:Path)->dict:
    return base.read(out/'experiment_lock.json')


def _source_references(frozen:dict)->set[str]:
    data=json.loads(frozen['original_v18_work_llm_request']['messages'][1]['content'])
    refs=set(data['work_request']['available_inputs'])|set(data['work_request']['source_ids'])
    refs|={x['id'] for x in data['read_scientific_sources']}
    refs|={x.get('work_id','')+':product' for x in data['read_work_products'] if x.get('work_id')}
    return refs


def _reconstruct_prework(c:dict,w08:Path)->dict:
    state=base.init_state(c['manifest']['question'])
    for i in range(1,6):
        event=base.read(w08/'events'/f'event_{i:04d}.json')
        if event['previous_state_sha256']!=base.sha(state):raise ValueError('W08_PREWORK_EVENT_CHAIN_MISMATCH')
        state=base.apply_event(state,event['kind'],event['payload'],manifest=c['manifest'],sources=c['sources'])
        if base.sha(state)!=event['resulting_state_sha256']:raise ValueError('W08_PREWORK_STATE_MISMATCH')
    event6=base.read(w08/'events'/'event_0006.json')
    if event6['kind']!='WORK_PRODUCT' or event6['previous_state_sha256']!=base.sha(state):
        raise ValueError('W08_WORK3_BOUNDARY_MISMATCH')
    pending=state['pending_work']
    if not pending or pending['status']!='REQUESTED' or pending['work_id']!=EXPECTED_WORK_ID:
        raise ValueError('NOT_W08_WORK3_PREEXECUTION')
    return state


def prepare(*,w08:Path,out:Path)->dict:
    repo=_safe_repo_root(out)
    if out.exists():raise FileExistsError('REFUSE_OVERWRITE_EXPERIMENT')
    c=v18.verify(w08)
    if base.sha(c['state'])!=EXPECTED_W08_SHA or c['state']['decision_count']!=3 or c['state']['event_count']!=6:
        raise ValueError('W08_FROZEN_BASELINE_NOT_VERIFIED')
    pre=_reconstruct_prework(c,w08)
    request=v18._make_call_request({**c,'state':pre},'work')
    persisted=base.read(w08/'work'/EXPECTED_WORK_ID/'llm_request.json')
    if request!=persisted:raise ValueError('W08_ORIGINAL_WORK_PROMPT_MISMATCH')
    original_req=pre['pending_work']['request']
    frozen={'schema':SCHEMA,'original_v18_work_llm_request':request,
            'original_w08_prework_state_sha256':base.sha(pre),
            'w08_reference_sha256':EXPECTED_W08_SHA,
            'original_llm_request_sha256':base.sha(request),
            'original_work_request_sha256':base.sha(original_req),
            'work_id':EXPECTED_WORK_ID,
            'work_input_digest':base.work_input_digest(EXPECTED_WORK_ID,original_req,c['manifest']['source_snapshot_sha256']),
            'source_snapshot_sha256':c['manifest']['source_snapshot_sha256'],
            'base_source_sha256':base.file_sha(Path(base.__file__)),
            'v18_source_sha256':base.file_sha(Path(v18.__file__)),
            'm19_source_sha256':base.file_sha(Path(__file__)),
            'original_source_work_id':'W08:work_0003',
            'no_posterior_work3_response_in_inputs':True}
    # Do not write any W08 source artifacts, read-only after verified reconstruction.
    out.mkdir(parents=True,exist_ok=False)
    base.write_new(out/'frozen_input.json',frozen)
    base.write_new(out/'experiment_lock.json',{'schema':SCHEMA,'frozen_input_sha256':base.sha(frozen),
        'max_steps_per_arm':MAX_PER_ARM,'model':request['model'],'temperature':request['temperature'],
        'max_tokens':request['max_tokens'],'cohort':'W08_WORK3_FROZEN_B_VS_C',
        'creation_note':'No paid call; frozen input predates W08 Work3 result'})
    verify(out)
    return {'status':'PREPARED_VERIFIED_NO_PAID','repo':str(repo),'out':str(out),
            'w08_original_prompt_sha256':base.sha(request),'model':request['model']}


def _request(out:Path,arm:str,index:int,verified:dict)->dict:
    frozen=verified['frozen']; orig=frozen['original_v18_work_llm_request']
    user_data=json.loads(orig['messages'][1]['content'])
    user_data['m19_model_output_schema']=StepOutput.model_json_schema()
    # B requests remain identical, and the first B/C requests are IDENTICAL.
    if arm=='C' and index>1:
        prev=verified['steps']['C']
        user_data['m19_continuation']={'previous_scientific_steps':[
            {'step_index':i,'operation':r['output']['operation'],'research_delta':r['output']['research_delta'],
             'provisional_work_content':r['output']['provisional_work_content'],
             'next_question_or_test':r['output']['next_question_or_test'],
             'reason_for_choice':r['output']['reason_for_choice']}
            for i,r in enumerate(prev,1)],
            'instruction':'Use these UNVERIFIED prior results as research history, NOT source-certified facts. '
                          'Actually pursue the scientific next question previously chosen; you may revise, expand or pivot.'}
    return {'model':orig['model'],'temperature':orig['temperature'],'max_tokens':orig['max_tokens'],
         'messages':[{'role':'system','content':orig['messages'][0]['content']+'\n'+RESEARCH_INSTRUCTION},
           {'role':'user','content':json.dumps(user_data,ensure_ascii=False,sort_keys=True)}],
         'role':'m19_e1_science_work'}


def _parse(out:Path,raw:str)->dict:
    v=StepOutput.model_validate(base._raw_obj(raw)).model_dump(mode='json')
    refs=_source_references(base.read(out/'frozen_input.json'))
    unknown=set(v['source_or_prior_work_refs'])-refs
    if unknown:raise ValueError('UNKNOWN_SOURCE_REF: '+str(sorted(unknown)))
    return v


def _record_for_step(out:Path,arm:str,index:int,prior_hash:str|None)->dict:
    sd=_step_dir(out,arm,index)
    raw=base.read(sd/'raw_response.json')
    normalized=_parse(out,raw['raw_text'])
    existing=sd/'normalized.json'
    if existing.exists():
        if base.read(existing)!=normalized:raise ValueError('NORMALIZED_ARTIFACT_CHANGED')
    else:base.write_new(existing,normalized)
    rec={'arm':arm,'step_index':index,'parent_step_sha256':prior_hash,
        'request_file_sha256':base.file_sha(sd/'request.json'),
        'attempt_file_sha256':base.file_sha(sd/'attempt.json'),
        'raw_response_file_sha256':base.file_sha(sd/'raw_response.json'),
        'normalized_file_sha256':base.file_sha(existing),
        'response_id':raw.get('response_id'),
        'usage':raw.get('usage'),'operation':normalized['operation'],
        'next_step':normalized['next_step'],'research_delta':normalized['research_delta'],
        'scientific_validation':False}
    rp=sd/'record.json'
    if rp.exists():
        if base.read(rp)!=rec:raise ValueError('RECORDED_STEP_CHANGED')
    else:base.write_new(rp,rec)
    return rec


def verify(out:Path)->dict:
    repo=_safe_repo_root(out)
    frozen=base.read(out/'frozen_input.json');lock=_run_lock(out)
    if lock.get('schema')!=SCHEMA or frozen.get('schema')!=SCHEMA or lock['frozen_input_sha256']!=base.sha(frozen):
        raise ValueError('FROZEN_E1_INPUT_TAMPERED')
    for key,module in [('base_source_sha256',base),('v18_source_sha256',v18)]:
        if frozen[key]!=base.file_sha(Path(module.__file__)):raise ValueError('DEPENDENCY_SOURCE_CHANGED: '+key)
    if frozen['m19_source_sha256']!=base.file_sha(Path(__file__)):
        raise ValueError('M19_RUNNER_SOURCE_CHANGED')
    orig=frozen['original_v18_work_llm_request']
    if (frozen['original_llm_request_sha256']!=base.sha(orig)
        or lock['model']!=orig['model'] or lock['temperature']!=orig['temperature']
        or lock['max_tokens']!=orig['max_tokens'] or lock['max_steps_per_arm']!=MAX_PER_ARM):
        raise ValueError('BASELINE_PROMPT_OR_MODEL_TAMPERED')
    if 'm19_continuation' in orig['messages'][1]['content']:
        raise ValueError('BASELINE_CONTAMINATED')
    complete={arm:[] for arm in ARMS};incomplete={};attempts=0
    for arm in ARMS:
        parent=None
        dirs=sorted((out/arm/'micro_steps').glob('step_*')) if (out/arm/'micro_steps').is_dir() else []
        if len(dirs)>MAX_PER_ARM:raise ValueError('ARM_CALL_CEILING_EXCEEDED')
        for index,sd in enumerate(dirs,1):
            if sd.name!=f'step_{index:04d}':raise ValueError('NONCONTIGUOUS_STEP_SEQUENCE')
            if arm=='C' and complete[arm] and complete[arm][-1]['output']['next_step']=='RETURN':
                raise ValueError('C_CONTINUED_AFTER_RETURN')
            reqp=sd/'request.json';ap=sd/'attempt.json'
            if not reqp.is_file() or not ap.is_file():raise ValueError('INCOMPLETE_ATTEMPT_METADATA')
            expected=_request(out,arm,index,{'frozen':frozen,'steps':complete})
            req=base.read(reqp)
            if req!=expected:raise ValueError('MODEL_REQUEST_REPLAY_MISMATCH')
            att=base.read(ap)
            if att['request_sha256']!=base.sha(req) or att['parent_step_sha256']!=parent:
                raise ValueError('ATTEMPT_REQUEST_OR_PARENT_HASH_MISMATCH')
            attempts+=1
            if not (sd/'raw_response.json').is_file():
                if (sd/'normalized.json').exists() or (sd/'record.json').exists():raise ValueError('RAW_MISSING_WITH_COMMITTED_STEP')
                incomplete[arm]={'step':index,'status':'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY'}
                if sd!=dirs[-1]:raise ValueError('INCOMPLETE_NOT_LAST_ATTEMPT')
                continue
            if not (sd/'record.json').is_file():
                incomplete[arm]={'step':index,'status':'SAVED_RAW_NEEDS_RECOVER_NO_CALL'}
                if sd!=dirs[-1]:raise ValueError('UNCOMMITTED_STEP_NOT_LAST')
                continue
            record=base.read(sd/'record.json');output=base.read(sd/'normalized.json')
            if _parse(out,base.read(sd/'raw_response.json')['raw_text'])!=output:
                raise ValueError('NORMALIZED_RESPONSE_MISMATCH')
            if record != {'arm':arm,'step_index':index,'parent_step_sha256':parent,
                'request_file_sha256':base.file_sha(reqp),'attempt_file_sha256':base.file_sha(ap),
                'raw_response_file_sha256':base.file_sha(sd/'raw_response.json'),
                'normalized_file_sha256':base.file_sha(sd/'normalized.json'),
                'response_id':base.read(sd/'raw_response.json').get('response_id'),
                'usage':base.read(sd/'raw_response.json').get('usage'),
                'operation':output['operation'],'next_step':output['next_step'],
                'research_delta':output['research_delta'],'scientific_validation':False}:
                raise ValueError('STEP_RECORD_MISMATCH')
            parent=base.sha(record)
            complete[arm].append({'record':record,'output':output,'step_sha256':parent})
    if attempts>2*MAX_PER_ARM:raise ValueError('TOTAL_MAX_ATTEMPTS_EXCEEDED')
    for arm in ARMS:
        fp=out/arm/'final_product.json'
        tp=out/arm/'deep_work_trace.json'
        if tp.exists()!=fp.exists():raise ValueError('FINAL_TRACE_PRODUCT_PAIR_MISSING')
        if fp.exists():
            if arm in incomplete:raise ValueError('FINAL_WITH_INCOMPLETE_STEP')
            products=_products(out,arm,complete[arm])
            if base.read(fp)!=products:raise ValueError('FINAL_PRODUCT_REPLAY_MISMATCH')
            if base.read(tp)!=_trace(out,arm,complete[arm],products):
                raise ValueError('DEEP_WORK_TRACE_REPLAY_MISMATCH')
    return {'status':'VERIFIED_NO_CALL','repo':str(repo),'model':lock['model'],
            'steps':complete,'incomplete':incomplete,'attempt_count':attempts,'frozen':frozen,
            'b_count':len(complete['B']),'c_count':len(complete['C'])}


def _products(out:Path,arm:str,completed:list[dict])->dict:
    frozen=base.read(out/'frozen_input.json')
    entries=[]
    for index,row in enumerate(completed,1):
        content=row['output']['provisional_work_content']
        product=ResearchWorkProduct(work_id=frozen['work_id'],work_kind='REASON',
            execution_status='COMPLETED',inputs_sha256=frozen['work_input_digest'],
            output_artifact_sha256=base.sha(content),origin='LLM_REASONING_UNVERIFIED',
            content=content,raw_response_sha256=base.file_sha(_step_dir(out,arm,index)/'raw_response.json'),
            scientific_validation=False).model_dump(mode='json')
        entries.append({'step':index,'parent_step_sha256':row['record']['parent_step_sha256'],
                        'product':product,'research_delta':row['output']['research_delta'],
                        'operation':row['output']['operation']})
    return {'schema':SCHEMA,'arm':arm,'scientific_validation':False,'entries':entries,
      'final_candidate_step':len(entries) if arm=='C' else None,
      'independent_candidates_not_ranked':arm=='B',
      'original_source_work_id':frozen['work_id']}


def _trace(out:Path,arm:str,completed:list[dict],products:dict)->dict:
    return {'schema':SCHEMA,'arm':arm,
      'frozen_input_sha256':base.sha(base.read(out/'frozen_input.json')),
      'steps':[{'step':i,'step_record_sha256':r['step_sha256'],
         'operation':r['output']['operation'],'research_delta':r['output']['research_delta'],
         'next_choice':r['output']['next_step'],
         'reason_for_choice':r['output']['reason_for_choice'],
         'response_usage':r['record']['usage']} for i,r in enumerate(completed,1)],
      'final_product_sha256':base.sha(products),'scientific_validation':False,
      'comparison_note':'B independent candidates; C sequential research. No scientific superiority automatically inferred.'}


def preview(out:Path,arm:str)->dict:
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    v=verify(out)
    if v['incomplete'] or (out/arm/'final_product.json').exists():
        raise RuntimeError('INCOMPLETE_OR_FINALIZED_STEPS_REQUIRE_REVIEW')
    index=len(v['steps'][arm])+1
    if index>MAX_PER_ARM:raise RuntimeError('ARM_BUDGET_EXHAUSTED')
    if arm=='C' and v['steps'][arm] and v['steps'][arm][-1]['output']['next_step']=='RETURN':
        raise RuntimeError('C_SELECTED_RETURN_NO_MORE_CALLS')
    return {'arm':arm,'next_step':index,'request':_request(out,arm,index,v),
            'max_calls_per_arm':MAX_PER_ARM,'new_paid_calls':0}


def next_step(out:Path,arm:str,*,allow_paid:bool=False,max_total_paid_attempts:int=6,backend=None)->dict:
    if max_total_paid_attempts<1 or max_total_paid_attempts>6:raise ValueError('BAD_TOTAL_PAID_CEILING')
    pv=preview(out,arm)
    v=verify(out)
    if v['attempt_count']>=max_total_paid_attempts:raise RuntimeError('PAID_TOTAL_CEILING_REACHED')
    base._require_paid(backend,allow_paid,1)
    index=pv['next_step'];sd=_step_dir(out,arm,index)
    if sd.exists():raise FileExistsError('STEP_ATTEMPT_EXISTS_DO_NOT_REPEAT')
    parent=v['steps'][arm][-1]['step_sha256'] if v['steps'][arm] else None
    sd.mkdir(parents=True,exist_ok=False)
    base.write_new(sd/'request.json',pv['request'])
    base.write_new(sd/'attempt.json',{'status':'STARTED_UNKNOWN_UNTIL_RECEIPT',
        'request_sha256':base.sha(pv['request']), 'parent_step_sha256':parent,
        'started_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=base._invoke(pv['request'],backend)
    base.write_new(sd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        rec=_record_for_step(out,arm,index,parent)
        verify(out)
    except Exception as exc:
        raise RuntimeError('RAW_SAVED_OFFLINE_RECOVER_REQUIRED: '+str(sd)) from exc
    return {'status':'STEP_COMMITTED','arm':arm,'step':index,
            'next_choice':rec['next_step'],'new_paid_calls_this_command':int(backend is None)}


def recover(out:Path)->dict:
    v=verify(out)
    if not v['incomplete']:return {'status':'NOTHING_TO_RECOVER_NO_CALL'}
    if len(v['incomplete'])!=1:raise RuntimeError('AMBIGUOUS_INCOMPLETE_ARMS_REVIEW_MANUALLY')
    arm,info=next(iter(v['incomplete'].items()))
    if info['status']!='SAVED_RAW_NEEDS_RECOVER_NO_CALL':
        raise RuntimeError('UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
    index=info['step'];parent=v['steps'][arm][-1]['step_sha256'] if v['steps'][arm] else None
    rec=_record_for_step(out,arm,index,parent)
    verify(out)
    return {'status':'RECOVERED_SAVED_RAW_NO_CALL','arm':arm,'step':index,'next_choice':rec['next_step']}


def finalize(out:Path,arm:str)->dict:
    v=verify(out)
    if arm not in ARMS or v['incomplete']:raise RuntimeError('INVALID_ARM_OR_UNRESOLVED_ATTEMPT')
    if (out/arm/'final_product.json').exists():raise FileExistsError('FINAL_ALREADY_EXISTS')
    rows=v['steps'][arm]
    if not rows:raise RuntimeError('NO_SUCCESSFUL_SCIENCE_STEPS')
    if arm=='B' and len(rows)!=3:raise RuntimeError('B_NEEDS_3_INDEPENDENT_SAMPLES')
    if arm=='C' and len(rows)<3 and rows[-1]['output']['next_step']!='RETURN':
        raise RuntimeError('C_STILL_WANTS_CONTINUATION')
    product=_products(out,arm,rows)
    base.write_new(out/arm/'final_product.json',product)
    base.write_new(out/arm/'deep_work_trace.json',_trace(out,arm,rows,product))
    verify(out)
    return {'status':'FINALIZED_NO_PAID_CALL','arm':arm,'candidates':len(product['entries']),
            'scientific_validation':False}


def inspect(out:Path)->dict:
    v=verify(out)
    return {'status':v['status'],'experiment':str(out),
        'model':v['model'],'attempt_count_total':v['attempt_count'],
        'b_independent_completed':v['b_count'],'c_serial_completed':v['c_count'],
        'incomplete':v['incomplete'],
        'c_next_choice':v['steps']['C'][-1]['output']['next_step'] if v['steps']['C'] else None,
        'finalized_arms':[a for a in ARMS if (out/a/'final_product.json').exists()],
        'paid_calls_this_command':0}
