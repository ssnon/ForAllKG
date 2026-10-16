"""M19-E4: audit an E3-RETURN without rewriting or retroactively extending it.

Two *parallel* readouts are deliberately independent:
 SELF: original unverified work only. No auditor answer.
 EVIDENCE: identical original work plus a deterministic external countermodel.
Neither is an autonomously chosen continuation, nor certified science.
"""
from __future__ import annotations
import json, shutil
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from . import m19_e3_deep_work as source
from . import research_work_v12_stable_runner as base

SCHEMA='m19-e4-independent-audit-and-feedback-v1'
ARMS=('self','evidence')
TARGET_SENTENCE='This produces a rate crossover from approximately linear to threshold- and overlap-dependent saturation.'
EVIDENCE_ORIGIN='EXTERNAL_DETERMINISTIC_MATHEMATICAL_AUDIT_NOT_FAKG'
MAX_PAID=2

class ReviewOutput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    original_claim_under_review:str=Field(min_length=15)
    disposition:Literal['RETAIN_WITH_PROOF','REVISE','RETRACT','UNRESOLVED']
    mathematical_analysis:list[str]=Field(min_length=1)
    explicit_assumptions:list[str]=Field(min_length=1)
    counterexample_or_proof:str|None=None
    corrected_conditional_claim:str=Field(min_length=20)
    earlier_return_was_scientifically_sufficient:bool
    return_assessment_reason:str=Field(min_length=20)
    new_testable_consequence:str|None=None
    next_research_action:str|None=None
    limitations:list[str]=Field(min_length=1)
    scientific_validation:Literal[False]=False

SYSTEM='''You are continuing the scientific *assessment* of a completed, unverified FAKG WorkProduct.
Return precisely one JSON object matching output_schema. The word JSON explicitly requests JSON object mode.
You may retain, revise, retract, or leave unresolved the claim, but justify the choice mathematically.
Try to derive a valid restricted theorem, countermodel, or new falsifiable conditional consequence.
Do not assume the input WorkProduct is right. Do not pretend data from Ye were measured or model verified.
This is NOT the original autonomous research decision and must not be described as self-chosen CONTINUE.\nIf no specific claim is singled out, independently choose the most consequential underived or potentially mistaken claim (or defend the original return).
If independent external audit evidence is supplied, clearly distinguish facts learned from it from derivations you actually perform.
The scientific_validation field MUST be false. A plausible argument is not scientific validation.
'''

def repo_root()->Path:return Path(__file__).resolve().parents[3]

def guard(out:Path)->Path:
    root=(repo_root()/'experiments'/'m19_e4'/'runs').resolve()
    p=out.expanduser().resolve()
    if p==root or root not in p.parents:raise ValueError('OUTSIDE_REPO_M19_E4_RUNS')
    return p

def _source_check(work:Path)->tuple[dict,dict,dict]:
    w=source.verify(work)
    if not w['finalized'] or w['completed_count']!=1 or w['next_choice']!='RETURN' or w['incomplete']:
        raise ValueError('EXPECTED_FINALIZED_E3_RETURN_STEP1')
    product=base.read(work/'final_product.json');trace=base.read(work/'deep_work_trace.json')
    if product['origin']!='LLM_REASONING_UNVERIFIED' or product['scientific_validation'] is not False:
        raise ValueError('SOURCE_IS_NOT_UNVERIFIED_RESEARCH')
    if trace['scientific_validation'] is not False or trace['native_work_event_committed'] is not False:
        raise ValueError('SOURCE_NATIVE_WORK_WAS_MODIFIED_OR_VALIDATED')
    if product['work_id']!=trace['work_id']:raise ValueError('SOURCE_WORK_ID_MISMATCH')
    if trace['final_product_sha256']!=base.sha(product):raise ValueError('TRACE_PRODUCT_SHA_MISMATCH')
    if not any(TARGET_SENTENCE in s for s in product['content']['reasoning_or_operations']):
        raise ValueError('CLAIM_TO_AUDIT_NOT_FOUND_IN_SOURCE')
    return product,trace,w

def prepare(*,e3_work:Path,out:Path)->dict:
    dest=guard(out)
    if dest.exists() or dest.is_symlink():raise FileExistsError('E4_ALREADY_EXISTS_NO_OVERWRITE')
    work=e3_work.expanduser().resolve(strict=True)
    product,trace,_=_source_check(work)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.preparing')
    if tmp.exists():raise FileExistsError('STALE_PREPARATION_REQUIRES_REVIEW')
    tmp.mkdir(mode=0o700)
    try:
        shutil.copyfile(work/'final_product.json',tmp/'frozen_final_product.json')
        shutil.copyfile(work/'deep_work_trace.json',tmp/'frozen_deep_work_trace.json')
        lock={'schema':SCHEMA,'source_path':str(work),
              'source_file_shas':{'final_product.json':base.file_sha(work/'final_product.json'),
                                  'deep_work_trace.json':base.file_sha(work/'deep_work_trace.json')},
              'source_work_id':product['work_id'],
              'source_d4_sha256':trace['source_d4_state_sha256'],
              'source_return_record_sha256':trace['micro_steps'][0]['record_sha256'],
              'source_runner_sha256':base.file_sha(Path(source.__file__)),
              'base_runner_sha256':base.file_sha(Path(base.__file__)),
              'e4_runner_sha256':base.file_sha(Path(__file__)),
              'model':base.read(work/'lock.json')['model'],
              'max_new_paid_attempts':MAX_PAID,
              'analysis_scope':'SECONDARY_AUDIT_NOT_AUTONOMOUS_CONTINUE',
              'external_findings_are_not_FAKG_discoveries':True,
              'scientific_validation':False}
        base.write_new(tmp/'lock.json',lock)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():shutil.rmtree(tmp)
        raise
    v=verify(dest)
    return {'status':'PREPARED_PINNED_E3_RETURN_NO_CALL','work_id':product['work_id'],
            'out':str(dest),'source_unchanged':True,'audit_exists':False,
            'paid_calls':0,'scientific_validation':False}

def _enumerated_entry_rate(n:int,a:F,b:F)->F:
    # Independent check: enumerate stationary dark states for threshold >=1.
    # The only dark state is all-OFF; there are n independent OFF->ON channels.
    total=F(0)
    for mask in range(1<<n):
        if mask==0:
            prob=(b/(a+b))**n
            for bit in range(n):
                if (mask>>bit)&1==0:total+=prob*a
    return total

def expected_audit(product:dict)->dict:
    if not any(TARGET_SENTENCE in t for t in product['content']['reasoning_or_operations']):
        raise ValueError('SOURCE_CLAIM_SELECTION_CHANGED')
    a,b=F(1,5),F(4,5)
    p=a/(a+b)
    rows=[]
    for n in (1,2,4,5,10,20):
        occupancy=1-(1-p)**n
        rate=n*a*(1-p)**n
        if n<=5 and rate!=_enumerated_entry_rate(n,a,b):raise AssertionError('INDEPENDENT_ENUMERATION_FAILED')
        rows.append({'N':n,'bright_occupancy_exact':str(occupancy),
                     'bright_occupancy':float(occupancy),
                     'bright_entry_rate_exact':str(rate),
                     'bright_entry_rate':float(rate)})
    return {'schema':'m19-e4-ctmc-audit-v1',
            'origin':EVIDENCE_ORIGIN,
            'source_claim_exact':TARGET_SENTENCE,
            'source_claim_area':'independent rare-event regime / concentration-scaling transition',
            'status':'CONDITIONAL_COUNTERMODEL_FOUND',
            'hypothesis_under_test':'The wording about rate saturation must not be interpreted as a general monotone approach to a positive plateau merely because bright occupancy saturates.',
            'countermodel':{'kind':'N independent identical reversible binary CTMC sites',
                'off_to_on_rate_exact':str(a),'on_to_off_rate_exact':str(b),
                'single_site_on_probability_exact':str(p),
                'intensity':'I_N(t)=sum_i X_i(t); one unit per ON site',
                'bright_rule':'I_N >= 1; instantaneous detection, stationarity, no bleaching, no measurement noise',
                'fixed_site_coefficients':True},
            'derived_exact_relations':{
                'bright_fraction':'F_N=1-(1-p)^N',
                'entry_rate':'R_N=N*a*(1-p)^N',
                'proof':'Only the all-OFF state is dark. Its stationary probability is (1-p)^N. From this state any of N independent OFF->ON transitions at rate a begins a bright episode. Hence R_N=(1-p)^N * N*a.',
                'asymptotic':'For fixed 0<p<1 and a>0, F_N tends to 1 but R_N tends to 0 as N tends to infinity.',
                'enumeration_check':'For N=1..5, explicit finite-state dark-to-bright boundary enumeration agrees exactly.'},
            'computed_examples':rows,
            'audit_interpretation':'This falsifies a universal nondecreasing plateau interpretation for entry-event rates; it does not refute low-occupancy linear scaling, occupancy saturation, or all rate behavior under different detection thresholds and site heterogeneity.',
            'researcher_credit_policy':'This independent countermodel was constructed by the external audit after the original E3 RETURN. It must not be attributed to autonomous FAKG discovery.',
            'scientific_validation':False}

def audit(out:Path)->dict:
    v=verify(out)
    out=guard(out)
    p=out/'external_audit.json'
    if p.exists():return {'status':'EXTERNAL_AUDIT_ALREADY_EXISTS_NO_CALL','audit_sha256':base.file_sha(p),'paid_calls':0}
    if v['attempts_total']:raise RuntimeError('CANNOT_ADD_AUDIT_AFTER_MODEL_CALLS')
    evidence=expected_audit(base.read(out/'frozen_final_product.json'))
    base.write_new(p,evidence)
    verify(out)
    return {'status':'EXTERNAL_DETERMINISTIC_AUDIT_COMMITTED_NO_CALL',
            'audit_sha256':base.file_sha(p),
            'audit_finding':evidence['status'],
            'scientific_validation':False,'paid_calls':0}

def _request(out:Path,arm:str)->dict:
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    lock=base.read(out/'lock.json');product=base.read(out/'frozen_final_product.json')
    trace=base.read(out/'frozen_deep_work_trace.json')
    payload={'original_scientific_question':product['content']['attempted_question'],
       'original_work_product_unverified':product,
       'original_return_choice':trace['micro_steps'][0]['next_step'],
       'original_return_reason':trace['micro_steps'][0]['why'],
       'review_kind': 'INDEPENDENT_SELF_CHECK_NO_EXTERNAL_EVIDENCE' if arm=='self' else 'EXTERNAL_EVIDENCE_TRIGGERED_REVIEW',
       'output_schema':ReviewOutput.model_json_schema(),
       'attribution':'Any externally supplied audit findings are not autonomous research discoveries.'}
    if arm=='self':
        payload['self_selection_rule']='No issue or target is preselected. Inspect your original entire result and its RETURN reason; independently select an important unfinished claim or explain why no material problem remains.'
    if arm=='evidence':
        payload['specific_claim_for_review']=TARGET_SENTENCE
        evidence_file=out/'external_audit.json'
        if not evidence_file.is_file():raise RuntimeError('NO_EXTERNAL_AUDIT')
        payload['externally_supplied_deterministic_audit_not_self_discovered']=base.read(evidence_file)
    # Crucial: self arm must never depend on audit content or presence.
    return {'model':lock['model'],'temperature':0,'max_tokens':6000,
       'role':f'm19_e4_{arm}_review','input_state_sha256':lock['source_d4_sha256'],
       'messages':[{'role':'system','content':SYSTEM},
                   {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}

def _parse(raw:str)->dict:
    return ReviewOutput.model_validate(base._raw_obj(raw)).model_dump(mode='json')

def _arm_record(out:Path,arm:str)->dict:
    ad=out/'arms'/arm
    req=base.read(ad/'request.json')
    if req!=_request(out,arm):raise ValueError('ARM_REQUEST_REPLAY_MISMATCH')
    att=base.read(ad/'attempt.json')
    if att['request_sha256']!=base.sha(req) or att['arm']!=arm:raise ValueError('ARM_ATTEMPT_BAD')
    raw=base.read(ad/'raw_response.json')
    norm=_parse(raw['raw_text'])
    if base.read(ad/'normalized.json')!=norm:raise ValueError('ARM_NORMALIZED_MISMATCH')
    rec={'arm':arm,'request_sha256':base.file_sha(ad/'request.json'),
         'attempt_sha256':base.file_sha(ad/'attempt.json'),
         'raw_sha256':base.file_sha(ad/'raw_response.json'),
         'normalized_sha256':base.file_sha(ad/'normalized.json'),
         'response_id':raw.get('response_id'),'usage':raw.get('usage'),
         'review_kind':'BLIND_SELF_REVIEW' if arm=='self' else 'EXTERNAL_EVIDENCE_TRIGGERED_REVIEW',
         'autonomous_continuation':False,
         'external_audit_used':arm=='evidence',
         'scientific_validation':False}
    if base.read(ad/'record.json')!=rec:raise ValueError('ARM_RECORD_MISMATCH')
    return rec

def verify(out:Path)->dict:
    out=guard(out)
    lock=base.read(out/'lock.json')
    if lock.get('schema')!=SCHEMA or lock['max_new_paid_attempts']!=MAX_PAID or lock.get('scientific_validation') is not False:
        raise ValueError('BAD_E4_LOCK')
    for k,mod in [('e4_runner_sha256',__import__(__name__,fromlist=['*'])),('source_runner_sha256',source),('base_runner_sha256',base)]:
        if base.file_sha(Path(mod.__file__))!=lock[k]:raise ValueError('PINNED_SOURCE_CHANGED_'+k)
    work=Path(lock['source_path']);p,t,_=_source_check(work)
    if p['work_id']!=lock['source_work_id'] or t['source_d4_state_sha256']!=lock['source_d4_sha256'] or t['micro_steps'][0]['record_sha256']!=lock['source_return_record_sha256']:
        raise ValueError('PINNED_E3_STATE_CHANGED')
    for rel,h in lock['source_file_shas'].items():
        if base.file_sha(work/rel)!=h:raise ValueError('E3_SOURCE_FILE_CHANGED_'+rel)
    if base.read(out/'frozen_final_product.json')!=p or base.read(out/'frozen_deep_work_trace.json')!=t:
        raise ValueError('FROZEN_SOURCE_SNAPSHOT_CHANGED')
    ev=out/'external_audit.json'
    if ev.exists() and base.read(ev)!=expected_audit(p):raise ValueError('EXTERNAL_AUDIT_CHANGED')
    attempts={};complete=[]
    armroot=out/'arms'
    if armroot.exists():
        for ad in armroot.iterdir():
            if ad.is_dir() and ad.name not in ARMS:raise ValueError('UNKNOWN_ARM_DIR')
    for arm in ARMS:
        ad=armroot/arm
        if not ad.exists():continue
        if not ev.exists():raise ValueError('MODEL_ATTEMPT_BEFORE_AUDIT')
        if base.read(ad/'request.json')!=_request(out,arm):raise ValueError('ARM_PROMPT_TAMPERED_'+arm)
        att=base.read(ad/'attempt.json')
        if att['arm']!=arm or att['request_sha256']!=base.sha(base.read(ad/'request.json')):
            raise ValueError('ARM_ATTEMPT_BAD_'+arm)
        if not (ad/'raw_response.json').is_file():
            attempts[arm]='UNKNOWN_BILLING_NO_RAW_NEVER_RETRY';continue
        if not (ad/'record.json').is_file():
            attempts[arm]='RAW_SAVED_OFFLINE_RECOVER_REQUIRED';continue
        _arm_record(out,arm)
        attempts[arm]='COMPLETE';complete.append(arm)
    fin=out/'final_report.json'
    if fin.exists():
        if set(complete)!=set(ARMS):raise ValueError('FINAL_REPORT_WITHOUT_BOTH_ARMS')
        if base.read(fin)!=_report(out):raise ValueError('FINAL_REPORT_TAMPERED')
    return {'status':'VERIFIED_NO_CALL', 'source_work_id':p['work_id'],
        'audit_status':'CONDITIONAL_COUNTERMODEL_FOUND' if ev.exists() else 'NOT_RUN',
        'audit_sha256':base.file_sha(ev) if ev.exists() else None,
        'attempts_total':len(attempts),'arms':attempts,'finalized':fin.exists(),
        'scientific_validation':False,'original_autonomous_return_unchanged':True,
        'source_original_pinned':True}

def preview(out:Path,arm:str)->dict:
    out=guard(out);v=verify(out)
    if v['audit_status']=='NOT_RUN':raise RuntimeError('AUDIT_MUST_BE_RUN_BEFORE_MODEL_CALLS')
    if v['finalized']:raise RuntimeError('FINALIZED_NO_NEW_CALLS')
    if arm not in ARMS:raise ValueError('UNKNOWN_ARM')
    if arm in v['arms']:raise RuntimeError('ARM_ALREADY_ATTEMPTED_NEVER_REPEAT')
    req=_request(out,arm)
    if 'json' not in ' '.join(x['content'].lower() for x in req['messages']):raise ValueError('JSON_MODE_NOT_READY')
    return {'status':'PREVIEW_NO_CALL','arm':arm,'model':req['model'],
            'request_sha256':base.sha(req),'paid_calls':0,
            'external_audit_in_prompt':arm=='evidence','json_mode_ready':True}

def _commit_raw(out:Path,arm:str)->dict:
    v=verify(out)
    if v['arms'].get(arm)!='RAW_SAVED_OFFLINE_RECOVER_REQUIRED':raise RuntimeError('NO_SAVED_RAW_TO_RECOVER')
    ad=out/'arms'/arm;raw=base.read(ad/'raw_response.json')
    norm=_parse(raw['raw_text'])
    normalized=ad/'normalized.json'
    if normalized.exists():
        if base.read(normalized)!=norm:raise ValueError('EXISTING_NORMALIZED_MISMATCH')
    else:base.write_new(normalized,norm)
    req=base.read(ad/'request.json');att=base.read(ad/'attempt.json')
    rec={'arm':arm,'request_sha256':base.file_sha(ad/'request.json'),
         'attempt_sha256':base.file_sha(ad/'attempt.json'),
         'raw_sha256':base.file_sha(ad/'raw_response.json'),
         'normalized_sha256':base.file_sha(normalized),
         'response_id':raw.get('response_id'),'usage':raw.get('usage'),
         'review_kind':'BLIND_SELF_REVIEW' if arm=='self' else 'EXTERNAL_EVIDENCE_TRIGGERED_REVIEW',
         'autonomous_continuation':False,
         'external_audit_used':arm=='evidence',
         'scientific_validation':False}
    base.write_new(ad/'record.json',rec)
    verify(out)
    return {'status':'SAVED_RAW_RECOVERED_OFFLINE_NO_CALL','arm':arm,'disposition':norm['disposition'],
            'scientific_validation':False}

def recover(out:Path,arm:str)->dict:
    v=verify(out)
    status=v['arms'].get(arm)
    if status=='UNKNOWN_BILLING_NO_RAW_NEVER_RETRY':raise RuntimeError(status)
    if status!='RAW_SAVED_OFFLINE_RECOVER_REQUIRED':raise RuntimeError('NO_RECOVERABLE_SAVED_RAW')
    return _commit_raw(guard(out),arm)

def next_step(out:Path,arm:str,*,allow_paid:bool=False,max_new_paid_attempts:int=2,backend=None)->dict:
    out=guard(out)
    if max_new_paid_attempts not in (1,2):raise ValueError('INVALID_PAID_CEILING')
    info=preview(out,arm)
    v=verify(out)
    if v['attempts_total']>=max_new_paid_attempts:raise RuntimeError('PAID_ATTEMPT_CEILING_REACHED')
    # Do not spend if a previous request has unknown billing or saved raw.
    if any(s!='COMPLETE' for s in v['arms'].values()):raise RuntimeError('UNRESOLVED_ATTEMPT_NO_NEW_CALLS')
    base._require_paid(backend,allow_paid,1)
    req=_request(out,arm)
    ad=out/'arms'/arm
    if ad.exists():raise FileExistsError('ARM_ATTEMPT_EXISTS_NO_REPEAT')
    ad.mkdir(parents=True,exist_ok=False)
    base.write_new(ad/'request.json',req)
    base.write_new(ad/'attempt.json',{'arm':arm,'request_sha256':base.sha(req),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'STARTED_UNKNOWN_BILLING_UNTIL_RESPONSE_SAVED'})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(ad/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:result=_commit_raw(out,arm)
    except Exception as exc:raise RuntimeError('RAW_SAVED_OFFLINE_RECOVER_REQUIRED: '+str(ad)) from exc
    return {'status':'ARM_COMMITTED','arm':arm,'review_disposition':result['disposition'],
            'new_paid_calls_this_command':int(backend is None),
            'external_audit_in_prompt':info['external_audit_in_prompt'],
            'scientific_validation':False}

def _report(out:Path)->dict:
    reads={arm:base.read(out/'arms'/arm/'normalized.json') for arm in ARMS}
    rec={arm:base.file_sha(out/'arms'/arm/'record.json') for arm in ARMS}
    return {'schema':SCHEMA,'source_work_id':base.read(out/'lock.json')['source_work_id'],
        'source_original_return_unchanged':True,
        'external_audit_sha256':base.file_sha(out/'external_audit.json'),
        'audit_credit':'EXTERNAL_NOT_FAKG',
        'arm_results':{arm:{'disposition':r['disposition'],
             'return_justified':r['earlier_return_was_scientifically_sufficient'],
             'has_new_testable_consequence':bool(r.get('new_testable_consequence')),
             'has_next_action':bool(r.get('next_research_action')),
             'record_sha256':rec[arm],'claim_text_unverified':r['corrected_conditional_claim']}
               for arm,r in reads.items()},
        'auto_scientific_judgment':'NOT_EVALUATED_REQUIRES_INDEPENDENT_MATHEMATICAL_REVIEW',
        'scientific_validation':False,
        'native_research_episode_modified':False,
        'no_autonomous_continuation_credit':True}

def finalize(out:Path)->dict:
    out=guard(out);v=verify(out)
    if v['finalized']:raise RuntimeError('ALREADY_FINALIZED')
    if set(k for k,s in v['arms'].items() if s=='COMPLETE')!=set(ARMS):
        raise RuntimeError('BOTH_ARMS_NOT_COMPLETE')
    base.write_new(out/'final_report.json',_report(out))
    verify(out)
    return {'status':'E4_FINALIZED_NO_CALL','scientific_validation':False,
            'independent_audit_not_attributed_to_FAKG':True}

def inspect(out:Path)->dict:return verify(out)
