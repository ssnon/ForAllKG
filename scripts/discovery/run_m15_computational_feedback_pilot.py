#!/usr/bin/env python3
"""M15: matched computational-feedback pilot; additive, no baseline source changes.

Compare same starting SERS questions, with/without actual Meep scattering spectra.
Simulation results are computational observations of a restricted model, not a new
scientific fact, nor mechanism support, nor prior-art novelty certification.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics

VERSION='m15-computational-feedback-pilot-v1'
INVENTORY='M15_MEEP_INVENTORY.json'
PLAN='M15_FROZEN_PLAN.json'
PROMPTS='M15_FROZEN_PROMPTS.json'
PUBLIC='M15_BLINDED_REVIEW.json'
PRIVATE='M15_PRIVATE_ARM_MAP.json'
REVIEW='M15_REVIEW_TEMPLATE.csv'
REPORT='M15_PILOT_REPORT.json'
CASE_FIELDS={'case_id','research_question','initial_hypothesis','strong_null','control_request_id','intervention_request_id','variable_under_test'}
VARIABLES={'nanorod_length_nm','nanorod_diameter_nm','shell_thickness_nm','core_material','shell_material','polarization','surrounding_medium'}
CANDIDATE_KEYS={'research_question','conditional_prediction','strongest_null','discriminating_measurement','falsification','prior_art_risk','next_action'}
REVIEW_FIELDS=['blind_id','reviewer_id','prior_art_status','conceptual_delta','null_discrimination','independent_testability','scientific_value','rationale']
SCORABLE={'conceptual_delta','null_discrimination','independent_testability','scientific_value'}


def canonical(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def hashfile(path):return sha(Path(path).read_bytes())
def load(path):return json.loads(Path(path).read_bytes())
def freeze(path,obj):
    path=Path(path);b=canonical(obj)
    if path.exists():
        if path.read_bytes()!=b:raise ValueError('FROZEN_ARTIFACT_CHANGED '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(b)
def fail(msg):raise ValueError(msg)

def find_execution(path):
    p=Path(path).resolve()
    record_path=p/'sers_fdtd.meep_execution_records.shadow.json'
    request_path=p/'sers_fdtd.meep_execution_requests.shadow.json'
    manifest_path=p/'sers_fdtd.meep_execution_shadow.manifest.json'
    if not all(x.is_file() for x in [record_path,request_path,manifest_path]):fail('MISSING_EXECUTION_BUNDLE '+str(p))
    records=load(record_path);requests=load(request_path);manifest=load(manifest_path)
    if records.get('schema_version')!='sers-meep-execution-record-bundle-v0' or requests.get('schema_version')!='sers-meep-execution-request-bundle-v0':fail('UNEXPECTED_BUNDLE_SCHEMA')
    if manifest.get('execution_records_sha256')!=hashfile(record_path) or manifest.get('request_bundle_sha256')!=hashfile(request_path):fail('UPSTREAM_MANIFEST_SHA_MISMATCH')
    if records['record_count']!=len(records['records']) or requests['request_count']!=len(requests['requests']):fail('UPSTREAM_BUNDLE_COUNT_MISMATCH')
    if records['source_request_bundle_id']!=requests['bundle_id']:fail('UPSTREAM_RECORD_REQUEST_LINEAGE_MISMATCH')
    reqs={a['request_id']:a for a in requests['requests']}
    if len(reqs)!=len(requests['requests']):fail('DUPLICATE_REQUEST')
    results=[];excluded=[]
    for rec in records['records']:
        rid=rec['request_id'];req=reqs.get(rid)
        if not req:fail('UNKNOWN_EXECUTION_REQUEST '+rid)
        if rec['execution_status']!='completed':
            excluded.append({'request_id':rid,'reason':'NOT_COMPLETED_'+rec['execution_status']});continue
        if rec.get('return_code') not in (0,None) or not rec.get('worker_result_path') or not rec.get('worker_result_sha256'):
            fail('INCOMPLETE_COMPLETED_RECORD '+rid)
        worker=Path(rec['worker_result_path']).resolve()
        if not worker.is_file() or hashfile(worker)!=rec['worker_result_sha256']:fail('WORKER_RESULT_HASH_MISMATCH '+rid)
        obj=load(worker)
        if obj.get('schema_version')!='sers-meep-worker-result-v0' or obj.get('worker_status')!='completed' or obj.get('backend')!='meep' or obj.get('request_id')!=rid:fail('NOT_GENUINE_COMPLETED_MEEP_SCHEMA '+rid)
        if not obj.get('backend_version'):fail('MISSING_MEEP_VERSION '+rid)
        if obj.get('shadow_only') is not True or obj.get('scientific_interpretation_permitted') is not False:fail('MEEP_SHADOW_FLAGS_INCONSISTENT '+rid)
        if obj.get('evidence_eligibility')!=rec['evidence_eligibility'] or obj.get('numerical_resolution_status')!=rec['numerical_resolution_status']:
            fail('MEEP_RESULT_RECORD_STATUS_MISMATCH '+rid)
        if obj.get('scientific_interpretation_permitted') is not False or obj.get('hypothesis_rejection_authority') is not False:fail('MEEP_WORKER_IMPROPER_AUTHORITY '+rid)
        if obj.get('field_component')!=req.get('field_component') or obj.get('material_model')!=req.get('material_model') or obj.get('resolution_px_per_um')!=req.get('resolution_px_per_um'):
            fail('MEEP_RESULT_REQUEST_METADATA_MISMATCH '+rid)
        if len(obj.get('spectrum',[]))<8:fail('MEEP_SPECTRUM_INSUFFICIENT '+rid)
        vals=[]; wavelengths=[]
        for sp in obj['spectrum']:
            v=sp.get('scattering_cross_section_um2');w=sp.get('wavelength_nm')
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or not isinstance(w,(int,float)) or not math.isfinite(w):fail('INVALID_WORKER_SPECTRUM '+rid)
            vals.append(float(v));wavelengths.append(float(w))
        if wavelengths!=sorted(wavelengths) or len(set(wavelengths))!=len(wavelengths):fail('WORKER_SPECTRUM_AXIS_INVALID '+rid)
        if max(vals)<=0:fail('NO_POSITIVE_WORKER_SIGNAL '+rid)
        results.append({'request_id':rid,'hypothesis_id':req['hypothesis_id'],
             'request_path':str(request_path),'request_sha256':hashfile(request_path),
             'request_fields':req,'worker_path':str(worker),'worker_sha256':hashfile(worker),
             'worker':obj,'execution_dir':str(p),'record_manifest_sha256':hashfile(manifest_path)})
    return results,excluded


def inventory(executions,out):
    if not executions:fail('NEED_EXECUTION_DIR')
    all_results=[];all_excluded=[];seen=set()
    for path in executions:
        rows,exc=find_execution(path)
        for row in rows:
            if row['request_id'] in seen:fail('DUPLICATE_REQUEST_ACROSS_EXECUTIONS '+row['request_id'])
            seen.add(row['request_id']);all_results.append(row)
        all_excluded.extend(exc)
    inv={'schema':VERSION,'status':'REAL_MEEP_RECORDS_SCREENED_SCHEMA_AND_SHA_ONLY',
         'source_execution_dirs':[str(Path(p).resolve()) for p in executions],
         'completed_request_ids':[row['request_id'] for row in all_results],
         'excluded':all_excluded,'request_count':len(all_results),
         'records':[{'request_id':row['request_id'],'hypothesis_id':row['hypothesis_id'],
             'execution_dir':row['execution_dir'],'worker_path':row['worker_path'],
             'worker_sha256':row['worker_sha256'],
             'numerical_resolution_status':row['worker']['numerical_resolution_status'],
             'evidence_eligibility':row['worker']['evidence_eligibility'],
             'core_material':row['request_fields']['core_material'],
             'shell_material':row['request_fields']['shell_material'],
             'nanorod_length_nm':row['request_fields']['nanorod_length_nm'],
             'nanorod_diameter_nm':row['request_fields']['nanorod_diameter_nm'],
             'shell_thickness_nm':row['request_fields']['shell_thickness_nm'],
             'polarization':row['request_fields']['polarization']}
                    for row in sorted(all_results,key=lambda x:x['request_id'])],
         'numerical_convergence_certified':False,'scientific_novelty_certified':False}
    freeze(out,inv)
    return {'status':inv['status'],'completed_requests':len(all_results),'excluded':len(all_excluded),
            'scientific_verdicts':0,'new_model_calls':0}



def suggest(inventory_path,output):
    by_id=_read_sources(inventory_path)
    vals=sorted(by_id.values(),key=lambda x:x['request_id'])
    pairs=[]
    for i,a in enumerate(vals):
        for b in vals[i+1:]:
            if a['hypothesis_id']!=b['hypothesis_id']:continue
            changes=[k for k in VARIABLES if a['request_fields'].get(k)!=b['request_fields'].get(k)]
            if len(changes)!=1:continue
            try:summary=compare(a,b,changes[0])
            except (ValueError,KeyError):continue
            pairs.append({'case_id':'EDIT_CASE_'+str(len(pairs)+1),
                'research_question':'REPLACE_WITH_AN_AUTHENTIC_SCIENTIFIC_RESEARCH_QUESTION',
                'initial_hypothesis':'REPLACE_WITH_THE_EXISTING_FIXED_HYPOTHESIS_NOT_THE_SIMULATION_RESULT',
                'strong_null':'REPLACE_WITH_THE_STRONG_KNOWN_CLASSICAL_MAXWELL_NULL',
                'control_request_id':a['request_id'],
                'intervention_request_id':b['request_id'],
                'variable_under_test':changes[0],
                'selection_notes':{'hypothesis_id':a['hypothesis_id'],
                    'reference_value':summary['reference_value'],
                    'intervention_value':summary['intervention_value'],
                    'quality':summary['numerical_gate']}})
    record={'schema':VERSION,'status':'PAIR_SUGGESTIONS_NOT_READY_CASEBOOK',
            'inventory_path':str(Path(inventory_path).resolve()),'inventory_sha256':hashfile(inventory_path),
            'candidate_pair_count':len(pairs),'pair_suggestions':pairs,
            'important':'Review and select nonoverlapping request IDs; put only CASE_FIELDS keys in final casebook. Fill genuine scientific hypotheses and strong null. Never count these suggestions as discovered science.',
            'no_auto_hypothesis_generation':True}
    freeze(output,record)
    return {'status':record['status'],'candidate_pair_count':len(pairs),
            'recommended_next_action':'CURATE_NONOVERLAPPING_REAL_SCIENCE_CASEBOOK' if pairs else 'GENERATE_REAL_MEEP_CONDITION_PAIRS',
            'new_paid_calls':0}

def _read_sources(inv_path):
    inv=load(inv_path)
    if inv.get('schema')!=VERSION:fail('INVALID_INVENTORY')
    parsed=[]
    for p in inv['source_execution_dirs']:
        a,_=find_execution(p);parsed.extend(a)
    if sorted(inv['completed_request_ids'])!=sorted(x['request_id'] for x in parsed):fail('INVENTORY_REPLAY_REQUEST_DIFFERENCE')
    by_id={x['request_id']:x for x in parsed}
    for row in inv['records']:
        if by_id[row['request_id']]['worker_sha256']!=row['worker_sha256']:fail('INVENTORY_WORKER_HASH_DIFFERENCE')
    return by_id


def compare(a,b,var):
    if var not in VARIABLES:fail('UNSUPPORTED_CONTROLLED_VARIABLE '+str(var))
    x=a['request_fields'];y=b['request_fields'];xa=a['worker'];yb=b['worker']
    allowed=set(VARIABLES)|{'geometry_candidate_id','job_id','request_id','hypothesis_id','source_solver_job_id','source_numerical_plan_id','source_numerical_plan_bundle_id','source_solver_job_bundle_id','design_id','source_spec_id'}
    if x[var]==y[var]:fail('CONTROLLED_VARIABLE_DID_NOT_CHANGE')
    changed=[k for k in VARIABLES if x[k]!=y[k]]
    if changed!=[var]:fail('MULTIPLE_PHYSICAL_VARIABLES_CHANGED '+str(changed))
    if x['hypothesis_id']!=y['hypothesis_id']:fail('NOT_SAME_INITIAL_HYPOTHESIS')
    # Field polarization must change consistently with the single controlled variable.
    if var=='polarization' and x.get('field_component')==y.get('field_component'):fail('POLARIZATION_CHANGE_NOT_REFLECTED_IN_FIELD_COMPONENT')
    if var!='polarization' and x.get('field_component')!=y.get('field_component'):fail('UNDECLARED_FIELD_POLARIZATION_CHANGE')
    for k in ('resolution_px_per_um','pml_thickness_um','padding_um','wavelength_min_nm','wavelength_max_nm',
              'spectrum_frequency_points','courant','run_until_after_sources','material_model','background_index','background_model',
              'source_profile','coordinate_convention','incident_propagation'):
        if x.get(k)!=y.get(k):fail('NUMERICAL_OR_OPTICAL_CONFOUNDER '+k)
    # Core material/shell etc may change, but geometry cell dimensions are solver engineering dependent.
    if x['scientific_wavelength_min_nm']!=y['scientific_wavelength_min_nm'] or x['scientific_wavelength_max_nm']!=y['scientific_wavelength_max_nm']:fail('DIFFERENT_SCIENTIFIC_WINDOW')
    wa=[row['wavelength_nm'] for row in xa['spectrum']];wb=[row['wavelength_nm'] for row in yb['spectrum']]
    if len(wa)!=len(wb) or any(abs(u-v)>0.002 for u,v in zip(wa,wb)):fail('SPECTRAL_AXIS_DIFFERENCE')
    ca=[row['scattering_cross_section_um2'] for row in xa['spectrum']];cb=[row['scattering_cross_section_um2'] for row in yb['spectrum']]
    ia=max(range(len(ca)),key=lambda i:ca[i]);ib=max(range(len(cb)),key=lambda i:cb[i])
    denom=max(max(ca),max(cb),1e-30)
    delta=sum(abs(u-v) for u,v in zip(ca,cb))/len(ca)/denom
    near=[{'wavelength_nm':round(wa[i],5),'reference_um2':round(ca[i],9),'intervention_um2':round(cb[i],9)} for i in sorted(set([0,len(wa)//4,len(wa)//2,3*len(wa)//4,len(wa)-1]))]
    gate='SCREENING_ONLY'
    if (xa['evidence_eligibility'] in {'calibration_candidate','quantitative_candidate'} and yb['evidence_eligibility'] in {'calibration_candidate','quantitative_candidate'} and xa['numerical_resolution_status']=='target_resolved' and yb['numerical_resolution_status']=='target_resolved'):
        gate='RESOLUTION_CANDIDATE_NOT_CONVERGENCE_CERTIFIED'
    return {'controlled_variable':var,'reference_value':x[var],'intervention_value':y[var],
       'observable':'scattering_cross_section_um2', 'peak_reference_nm':wa[ia],
       'peak_intervention_nm':wb[ib],'sampled_peak_shift_nm':wb[ib]-wa[ia],
       'peak_at_boundary_reference':ia in (0,len(wa)-1),
       'peak_at_boundary_intervention':ib in (0,len(wb)-1),
       'normalized_mean_absolute_spectral_delta':delta,
       'sampled_spectral_points':near,'n_spectral_samples':len(wa),
       'numerical_gate':gate,'independent_convergence_verified':False,
       'positive_control_verified':False,'hypothesis_supported_or_falsified':False,
       'scientific_novelty_certified':False}


def _prompt(case,summary,arm):
    shared=("You are proposing ONE *new testable follow-up scientific research question* in SERS. "
      "This is hypothesis exploration, not scientific novelty certification. "
      "Keep the question in the scope of the supplied initial SERS research problem, "
      "but genuinely propose a distinct follow-up or discriminating condition. "
      "DO NOT confuse classical Maxwell scattering cross sections with molecular SERS band ratios, "
      "chemical enhancement, adsorption kinetics, or molecular reaction rates. "
      "Use a strong known-physics null and an independently measurable discriminator. "
      "Your answer MUST be one JSON object with exactly keys: " + ', '.join(sorted(CANDIDATE_KEYS)) + '. '
      "All values must be informative plain-text strings. No claims of novelty certification or simulation proof. "
      "Do not mention the presence or absence of any tool/data feed in the final research question or prediction.\n")
    header={'case_id':case['case_id'],'research_question':case['research_question'],
            'initial_hypothesis':case['initial_hypothesis'],'strong_null':case['strong_null'],
            'controlled_variable_under_study':case['variable_under_test'],
            'target_observable':'single-particle scattering_cross_section_um2'}
    if arm=='B':
        feed={k:summary[k] for k in ('controlled_variable','reference_value','intervention_value','observable','peak_reference_nm','peak_intervention_nm',
                                     'sampled_peak_shift_nm','peak_at_boundary_reference','peak_at_boundary_intervention',
                                     'normalized_mean_absolute_spectral_delta','sampled_spectral_points','numerical_gate','independent_convergence_verified','positive_control_verified')}
        feed['warning']='Actual simulation data of restricted classical Maxwell model; no molecular mechanism inference or converged quantitative physics is certified.'
    else:
        feed={'controlled_variable':summary['controlled_variable'],
           'reference_value':summary['reference_value'], 'intervention_value':summary['intervention_value'],
           'observable':summary['observable'],
           'warning':'Numerical solver outcomes deliberately unavailable for this baseline arm. Propose a discriminating question using only the stated known physics and initial research context; do not invent numeric outcomes.'}
    return {'system':shared,'user':json.dumps({'initial_case':header,'scientific_feedback':feed},ensure_ascii=False,sort_keys=True)}


def prepare(inventory_path,casebook_path,out,seed):
    if out.exists():fail('OUTPUT_DIR_EXISTS_USE_NEW_DIRECTORY')
    inventory_path=Path(inventory_path).resolve();casebook_path=Path(casebook_path).resolve()
    book=load(casebook_path)
    if not isinstance(book,dict) or not isinstance(book.get('cases'),list):fail('INVALID_CASEBOOK')
    cases=book['cases']
    if not 2<=len(cases)<=8:fail('REQUIRE_2_TO_8_INDEPENDENT_CASES')
    ids=set();used=set();rows=[];by_id=_read_sources(inventory_path)
    for c in cases:
        if not isinstance(c,dict) or set(c)!=CASE_FIELDS or any(not isinstance(c[k],str) or not c[k].strip() for k in CASE_FIELDS):fail('INVALID_CASE_SCHEMA')
        if len(c['research_question'])<25 or len(c['strong_null'])<25 or len(c['initial_hypothesis'])<25:fail('INSUFFICIENT_SCIENTIFIC_QUESTION')
        ident=c['case_id'];a_id=c['control_request_id'];b_id=c['intervention_request_id'];var=c['variable_under_test']
        if ident in ids or a_id in used or b_id in used or a_id==b_id:fail('DUPLICATED_CASE_OR_SOLVER_REQUEST')
        if a_id not in by_id or b_id not in by_id:fail('UNLISTED_MEEP_REQUEST')
        ids.add(ident);used.update((a_id,b_id))
        row={'case':c,'control_sha256':by_id[a_id]['worker_sha256'],'intervention_sha256':by_id[b_id]['worker_sha256'],
             'observation':compare(by_id[a_id],by_id[b_id],var)}
        rows.append(row)
    # Two matched arms for each initial case. Simple seed-controlled order; not statistical randomization inference.
    units=[{'case_id':r['case']['case_id'],'arm':arm} for r in rows for arm in ('A','B')]
    random.Random(seed).shuffle(units)
    prompts=[]
    for row in rows:
        for arm in ('A','B'):
            p=_prompt(row['case'],row['observation'],arm)
            prompts.append({'unit_id':sha(canonical([row['case']['case_id'],arm]))[:20],
                            'case_id':row['case']['case_id'],'arm':arm,'prompt':p,'prompt_sha256':sha(canonical(p))})
    plan={'schema':VERSION,'status':'FROZEN_EXPLORATORY_MATCHED_COMPUTATIONAL_FEEDBACK_PILOT',
       'inventory_path':str(inventory_path),'inventory_sha256':hashfile(inventory_path),
       'casebook_path':str(casebook_path),'casebook_sha256':hashfile(casebook_path),
       'runner_sha256':hashfile(Path(__file__)), 'seed':seed,
       'matched_cases':rows,'randomized_execution_order':units,
       'arm_A':'no FDTD observation but same initial question and experimental variables',
       'arm_B':'actual FDTD numerical observation of single-particle scattering',
       'model_calls_required':len(units),'paid_calls_completed':0,
       'priors':['Same base model and max output tokens per arm','Novelty not inferred from Maxwell calculations',
       'No stochastic statistical claims for fewer than 6 independent science cases',
       'At least two independent blinded reviewers needed for scientific-quality descriptive scoring',
       'Literature overlap reviewed separately, no automatic novelty certificate'],
       'no_authoritative_physics_conclusions':True,'scientific_novelty_certified':False}
    out.mkdir(parents=True)
    freeze(out/PLAN,plan);freeze(out/PROMPTS,{'schema':VERSION,'prompts':prompts})
    return {'status':plan['status'],'case_count':len(rows),'arm_calls':len(units),
       'numerically_candidate_cases':sum(r['observation']['numerical_gate'].startswith('RESOLUTION_') for r in rows),
       'numerical_convergence_certified':False,'new_paid_calls':0,'novelty_certified':False}


def preflight(out):
    plan=load(out/PLAN)
    if plan.get('schema')!=VERSION:fail('PLAN_SCHEMA_MISMATCH')
    if plan['runner_sha256']!=hashfile(Path(__file__)):fail('M15_RUNNER_CHANGED_AFTER_FREEZE')
    for name in ('inventory','casebook'):
        if hashfile(plan[name+'_path'])!=plan[name+'_sha256']:fail('FROZEN_INPUT_MUTATED '+name)
    source=_read_sources(plan['inventory_path'])
    book=load(plan['casebook_path'])
    if book['cases']!=[row['case'] for row in plan['matched_cases']]:fail('CASEBOOK_PLAN_DIFFERENCE')
    for row in plan['matched_cases']:
        c=row['case'];a=source[c['control_request_id']];b=source[c['intervention_request_id']]
        if a['worker_sha256']!=row['control_sha256'] or b['worker_sha256']!=row['intervention_sha256'] or compare(a,b,c['variable_under_test'])!=row['observation']:fail('UPSTREAM_REPLAY_DIFFERENCE')
    spec=load(out/PROMPTS)
    if len(spec['prompts'])!=len(plan['randomized_execution_order']):fail('PROMPT_COUNT_MISMATCH')
    actual={ (p['case_id'],p['arm']):p for p in spec['prompts']}
    for row in plan['matched_cases']:
        for arm in ('A','B'):
            p=actual[(row['case']['case_id'],arm)]
            expected=_prompt(row['case'],row['observation'],arm)
            if p['prompt']!=expected or p['prompt_sha256']!=sha(canonical(expected)):fail('PROMPT_DIFFERENCE')
    return plan,spec


def run_paid(out,model,api_env,base_url,max_tokens,max_new_calls,allow_paid,authorize):
    plan,spec=preflight(out)
    if not allow_paid or not authorize:fail('EXPLICIT_TWO_FLAGS_REQUIRED_FOR_PAID_CALLS')
    if not model or max_new_calls<1 or max_tokens<200:fail('MODEL_AND_BUDGET_REQUIRED')
    key=os.getenv(api_env)
    if not key:fail('MISSING_API_KEY_ENV '+api_env)
    try:from openai import OpenAI
    except ImportError as e:raise RuntimeError('openai package required') from e
    by={(p['case_id'],p['arm']):p for p in spec['prompts']}
    # freeze model and generation budget BEFORE any paid call; changing it later is forbidden.
    config={'model':model,'base_url':base_url,'max_output_tokens':max_tokens,'temperature':0.6,
            'prompt_file_sha256':hashfile(out/PROMPTS),'number_of_total_units':len(plan['randomized_execution_order'])}
    freeze(out/'M15_PAID_CALL_CONFIG.json',config)
    client=OpenAI(api_key=key,base_url=base_url,max_retries=0,timeout=300)
    new=0
    for u in plan['randomized_execution_order']:
        p=by[(u['case_id'],u['arm'])];i=p['unit_id']
        intent=out/'paid'/(i+'.intent.json');receipt=out/'paid'/(i+'.receipt.json')
        if receipt.exists():
            r=load(receipt)
            if r.get('prompt_sha256')!=p['prompt_sha256'] or r.get('model')!=model:fail('PAID_RECEIPT_MISMATCH '+i)
            continue
        if intent.exists():
            return {'status':'BLOCKED_UNKNOWN_PAID_CALL_OUTCOME_NO_AUTOMATIC_RETRY',
                    'unit_id':i,'new_paid_calls':new,'next_action':'inspect intent and provider billing'}
        if new>=max_new_calls:break
        freeze(intent,{'schema':VERSION,'unit_id':i,'case_id':u['case_id'],'arm':u['arm'],
                       'prompt_sha256':p['prompt_sha256'],'model':model,'call_started':True})
        try:
            rsp=client.chat.completions.create(model=model,
                messages=[{'role':'system','content':p['prompt']['system']},
                          {'role':'user','content':p['prompt']['user']}],temperature=0.6,max_tokens=max_tokens)
        except Exception as exc:
            return {'status':'PROVIDER_EXCEPTION_MAY_HAVE_CHARGED_NO_RETRY',
                    'unit_id':i,'error':str(exc),'new_paid_calls':new,'intent_preserved':True}
        freeze(receipt,{'schema':VERSION,'unit_id':i,'prompt_sha256':p['prompt_sha256'],
           'model':model,'raw':rsp.choices[0].message.content or '',
           'finish_reason':rsp.choices[0].finish_reason,
           'usage':rsp.usage.model_dump(mode='json') if rsp.usage else None})
        new+=1
        if rsp.choices[0].finish_reason!='stop':
            return {'status':'PAID_RECEIPT_TRUNCATED_NO_AUTOMATIC_RETRY','unit_id':i,'new_paid_calls':new}
    total=sum((out/'paid'/(p['unit_id']+'.receipt.json')).exists() for p in spec['prompts'])
    return {'status':'ALL_SAVED_FOR_BLIND_REVIEW' if total==len(spec['prompts']) else 'PARTIAL_SAVED_MODEL_RECEIPTS',
            'new_paid_calls':new,'completed_receipts':total,'total_required':len(spec['prompts'])}


def _valid_candidate(raw):
    def no_duplicates(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:fail('DUPLICATE_JSON_KEY '+k)
            obj[k]=v
        return obj
    try:value=json.loads(raw,object_pairs_hook=no_duplicates,parse_constant=lambda x:fail('INVALID_JSON_CONSTANT'))
    except json.JSONDecodeError as e:fail('INVALID_MODEL_JSON '+str(e))
    if not isinstance(value,dict) or set(value)!=CANDIDATE_KEYS:fail('CANDIDATE_SCHEMA_KEYS_MISMATCH')
    if any(not isinstance(v,str) or len(v.strip())<35 for v in value.values()):fail('CANDIDATE_FIELDS_TOO_SHORT')
    return value


def materialize(out):
    plan,spec=preflight(out)
    by={(p['case_id'],p['arm']):p for p in spec['prompts']}
    rows=[];assignment=[]
    for case in [x['case']['case_id'] for x in plan['matched_cases']]:
        for arm in ('A','B'):
            p=by[(case,arm)];receipt_file=out/'paid'/(p['unit_id']+'.receipt.json')
            if not receipt_file.is_file():fail('MISSING_SAVED_MODEL_RECEIPT '+p['unit_id'])
            receipt=load(receipt_file)
            if receipt.get('schema')!=VERSION or receipt.get('prompt_sha256')!=p['prompt_sha256'] or receipt.get('finish_reason')!='stop':fail('RAW_RECEIPT_ID_OR_FINISH_INVALID')
            candidate=_valid_candidate(receipt['raw'])
            blind_id='b_'+sha(canonical([plan['seed'],p['unit_id'],hashfile(receipt_file)]))[:16]
            rows.append({'blind_id':blind_id,'candidate':candidate,'seed_research_question':next(x['case']['research_question'] for x in plan['matched_cases'] if x['case']['case_id']==case)})
            assignment.append({'blind_id':blind_id,'case_id':case,'arm':arm,'unit_id':p['unit_id'],
                               'receipt_sha256':hashfile(receipt_file)})
    rows=sorted(rows,key=lambda x:x['blind_id'])
    freeze(out/PUBLIC,{'schema':VERSION,'status':'BLINDED_CANDIDATE_CONTENT_MAY_REVEAL_SIMULATION_ARM',
          'reviewers_should_not_see_PRIVATE_ARM_MAP':True,'entries':rows})
    freeze(out/PRIVATE,{'schema':VERSION,'entries':assignment,
          'warning':'Do not share this with reviewers until blind scores are frozen'})
    # CSV is a starter template, and editable: do not gate its initial creation via freeze on subsequent analyses.
    path=out/REVIEW
    if not path.exists():
        path.write_text(','.join(REVIEW_FIELDS)+'\n'+''.join(f"{e['blind_id']},,,,,,,\n" for e in rows),encoding='utf-8')
    return {'status':'M15_BLINDED_PACKET_PREPARED','blind_candidates':len(rows),
            'new_model_calls':0,'novelty_certified':False}


def _read_reviews(out):
    with (out/REVIEW).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if not rows or any(set(r)!=set(REVIEW_FIELDS) for r in rows):fail('REVIEW_CSV_SCHEMA_INVALID')
    keys=set()
    for r in rows:
        if (r['blind_id'],r['reviewer_id']) in keys:fail('DUPLICATE_REVIEW')
        keys.add((r['blind_id'],r['reviewer_id']))
        if len(r['reviewer_id'])<2 or len(r['rationale'])<30 or r['prior_art_status'] not in {'known','uncertain','potentially_distinct'}:fail('REVIEW_INCOMPLETE')
        for field in SCORABLE:
            if r[field] not in {'0','1','2','3','4'}:fail('REVIEW_SCORE_INVALID')
    return rows


def score(out):
    plan,_=preflight(out)
    public=load(out/PUBLIC);private=load(out/PRIVATE)
    blinded={e['blind_id'] for e in public['entries']}
    mapping={e['blind_id']:e for e in private['entries']}
    if set(mapping)!=blinded:fail('PRIVATE_PUBLIC_MAPPING_INVALID')
    # Verify candidate content and original paid receipts every time.
    materialize(out)
    rows=_read_reviews(out)
    if {r['blind_id'] for r in rows}!=blinded:fail('INCOMPLETE_REVIEW_COVERAGE')
    reviewers={bid:{r['reviewer_id'] for r in rows if r['blind_id']==bid} for bid in blinded}
    if any(len(x)<2 for x in reviewers.values()):fail('REQUIRE_TWO_DISTINCT_BLIND_REVIEWERS_PER_CANDIDATE')
    # All candidates must be rated by the same pair of reviewers to avoid arm-dependent missingness.
    if len({tuple(sorted(x)) for x in reviewers.values()})!=1:fail('REVIEWER_COVERAGE_IMBALANCE')
    aggregate={}
    for bid in blinded:
        q=[r for r in rows if r['blind_id']==bid]
        aggregate[bid]={field:statistics.mean(int(r[field]) for r in q) for field in SCORABLE}
        aggregate[bid]['prior_art_status_counts']={k:sum(r['prior_art_status']==k for r in q) for k in ('known','uncertain','potentially_distinct')}
    table=[]
    for item in plan['matched_cases']:
        case=item['case']['case_id'];ids={mapping[bid]['arm']:bid for bid in blinded if mapping[bid]['case_id']==case}
        if set(ids)!={'A','B'}:fail('MISSING_MATCHED_ARM')
        a=aggregate[ids['A']];b=aggregate[ids['B']]
        table.append({'case_id':case,'arm_A_no_FDTD':a,'arm_B_FDTD':b,
             'paired_B_minus_A':{field:b[field]-a[field] for field in SCORABLE},
             'outcome_is_human_judgment_not_novelty_verdict':True})
    mean={field:statistics.mean(r['paired_B_minus_A'][field] for r in table) for field in SCORABLE}
    result={'schema':VERSION,'status':'EXPLORATORY_COMPUTATIONAL_FEEDBACK_COMPARISON',
       'matched_cases':len(table),'comparisons':table,'mean_paired_B_minus_A':mean,
       'model_call_count_per_arm':len(table),'preregistered_plan_sha256':hashfile(out/PLAN),
       'blinded_packet_sha256':hashfile(out/PUBLIC),'review_csv_sha256':hashfile(out/REVIEW),
       'causal_limitation':'Single model sampling per arm, small N; feedback exposure causal assignment but novelty is human-estimated; not randomized statistical inference',
       'mechanism_discovery_certified':False,'independent_experiment_executed':False,
       'new_scientific_fact_certified':False,'scientific_novelty_certified':False}
    freeze(out/REPORT,result)
    return {'status':result['status'],'cases':len(table),'mean_paired_B_minus_A':mean,
            'new_paid_calls':0,'novelty_certified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inventory');a.add_argument('--execution-dir',action='append',required=True,type=Path);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('suggest');a.add_argument('--inventory',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('prepare');a.add_argument('--inventory',type=Path,required=True);a.add_argument('--casebook',type=Path,required=True);a.add_argument('--out-dir',type=Path,required=True);a.add_argument('--seed',type=int,default=1501)
    for name in ('preflight','generate-paid','materialize','score'):
        a=sub.add_parser(name);a.add_argument('--out-dir',type=Path,required=True)
        if name=='generate-paid':
            a.add_argument('--model',default=os.getenv('OPENROUTER_AGENT_MODEL'));a.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
            a.add_argument('--base-url',default='https://openrouter.ai/api/v1');a.add_argument('--max-output-tokens',type=int,default=1800)
            a.add_argument('--max-new-calls',type=int,default=2);a.add_argument('--allow-paid',action='store_true');a.add_argument('--i-authorize-model-calls',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='inventory':res=inventory(args.execution_dir,args.output)
    elif args.command=='suggest':res=suggest(args.inventory,args.output)
    elif args.command=='prepare':res=prepare(args.inventory,args.casebook,args.out_dir,args.seed)
    elif args.command=='preflight':
        plan,_=preflight(args.out_dir);res={'status':'M15_SOURCES_AND_PROMPTS_VERIFIED','cases':len(plan['matched_cases']),
                                        'new_paid_calls':0,'novelty_certified':False}
    elif args.command=='generate-paid':res=run_paid(args.out_dir,args.model,args.api_key_env,args.base_url,args.max_output_tokens,args.max_new_calls,args.allow_paid,args.i_authorize_model_calls)
    elif args.command=='materialize':res=materialize(args.out_dir)
    else:res=score(args.out_dir)
    print(json.dumps(res,ensure_ascii=False,indent=2))
    return res
if __name__=='__main__':main()
