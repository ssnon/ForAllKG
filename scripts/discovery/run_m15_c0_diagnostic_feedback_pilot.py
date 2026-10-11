#!/usr/bin/env python3
"""M15-C0: computational *uncertainty* feedback experiment, NOT novelty validation.

Reads the frozen M15-B evidence gate and actual Meep worker request/results. Creates
matched A/B model prompts, permits optional explicit paid model calls, and makes a
blind review packet. Never executes Meep or mutates upstream science artifacts.
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
import statistics

VERSION = 'm15-c0-diagnostic-feedback-pilot-v1'
FIELDS = (
    'research_direction', 'competing_explanations', 'discriminating_test',
    'independent_observables', 'falsification', 'uncertainty_handling',
    'next_action', 'prior_art_qualification',
)
SCORE_FIELDS = (
    'null_discrimination', 'independent_measurement', 'uncertainty_calibration',
    'physical_scope_awareness', 'falsifiability',
)
CSV_FIELDS = ('blind_id', 'reviewer_id', *SCORE_FIELDS, 'unsupported_quantitative_claim', 'rationale')
SOURCE_DIRS = (
    'diagnostic_short_v1', 'diagnostic_short_trans_v1',
    'diagnostic_time16_ex_v1', 'diagnostic_time16_ey_v1',
    'diagnostic_resolution90_ey_v1',
)


def fail(message):
    raise ValueError(message)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hashfile(path):
    return digest(Path(path).read_bytes())


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(path, obj):
    path = Path(path)
    data = canonical(obj)
    if path.exists():
        if path.read_bytes() != data:
            fail('FROZEN_ARTIFACT_CHANGED '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)


def require(bool_expr, msg):
    if not bool_expr:
        fail(msg)


def spectrum_comparison(first, second):
    x, y = first['spectrum'], second['spectrum']
    require(len(x)==len(y) and len(x)>=8, 'SPECTRUM_LENGTH_MISMATCH')
    w = []
    vx = []
    vy = []
    for row_a, row_b in zip(x, y):
        a, b = float(row_a['wavelength_nm']), float(row_b['wavelength_nm'])
        require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-6, 'SPECTRUM_GRID_MISMATCH')
        m = float(row_a['scattering_cross_section_um2'])
        n = float(row_b['scattering_cross_section_um2'])
        require(math.isfinite(m) and math.isfinite(n) and m>=0 and n>=0, 'INVALID_SPECTRUM')
        w.append(a); vx.append(m); vy.append(n)
    require(all(a < b for a,b in zip(w,w[1:])), 'INVALID_WAVELENGTH_ORDER')
    require(sum(vy)>0, 'ZERO_SPECTRUM')
    peak_a = max(range(len(vx)), key=lambda i:vx[i])
    peak_b = max(range(len(vy)), key=lambda i:vy[i])
    return {
        'relative_l1':sum(abs(a-b) for a,b in zip(vx,vy))/sum(vy),
        'peak_a_nm':w[peak_a], 'peak_b_nm':w[peak_b],
        'peak_a_boundary':peak_a in (0,len(w)-1),
        'peak_b_boundary':peak_b in (0,len(w)-1),
        'points':len(w),
    }


def validate_gate(gate_path):
    gate_path = Path(gate_path).resolve()
    gate = load(gate_path)
    require(gate.get('schema_version')=='m15b-evidence-gate-v1', 'GATE_SCHEMA_MISMATCH')
    for field, value in (
        ('engineering_execution','PASS'),('temporal_stability','PRELIMINARY_PASS'),
        ('spatial_stability','FAIL'),('quantitative_physics_feedback','BLOCKED'),
        ('scientific_novelty','NOT_TESTED'),
        ('permitted_use','numerical_diagnostic_feedback_only'),
    ):
        require(gate.get(field)==value,'GATE_STATUS_MISMATCH_'+field)
    raw_sources = gate.get('sources')
    require(isinstance(raw_sources,list) and len(raw_sources)==10, 'GATE_SOURCE_COUNT')
    by_rel = {}
    base = gate_path.parent.resolve()
    for row in raw_sources:
        p = Path(row['path']).resolve()
        require(p.is_file(), 'MISSING_SOURCE '+str(p))
        require(p.is_relative_to(base), 'SOURCE_OUTSIDE_FROZEN_BASE '+str(p))
        relative = p.relative_to(base).as_posix()
        require(relative not in by_rel, 'DUPLICATE_SOURCE '+relative)
        require(hashfile(p)==row['sha256'], 'SOURCE_SHA_MISMATCH '+relative)
        by_rel[relative] = p
    expected = {f'{folder}/{name}' for folder in SOURCE_DIRS for name in ('request.json','result.json')}
    require(set(by_rel)==expected, 'GATE_SOURCE_PATHS_MISMATCH')
    cases = {}
    for folder in SOURCE_DIRS:
        req=load(by_rel[f'{folder}/request.json'])
        result=load(by_rel[f'{folder}/result.json'])
        require(result.get('schema_version')=='sers-meep-worker-result-v0', 'WORKER_SCHEMA '+folder)
        require(result.get('worker_status')=='completed' and result.get('backend')=='meep', 'NOT_COMPLETED_MEEP '+folder)
        require(req['request_id']==result['request_id'], 'REQUEST_ID_MISMATCH '+folder)
        require(req['field_component']==result['field_component'], 'POLARIZATION_MISMATCH '+folder)
        require(float(req['resolution_px_per_um'])==float(result['resolution_px_per_um']), 'RESOLUTION_MISMATCH '+folder)
        require(float(req['run_until_after_sources'])==float(result['run_until_after_sources']), 'TIME_MISMATCH '+folder)
        require(result.get('scientific_interpretation_permitted') is False and result.get('shadow_only') is True, 'UNSAFE_WORKER_AUTHORITY '+folder)
        require(result.get('evidence_eligibility')=='execution_sanity_only', 'NON_SANITY_EVIDENCE '+folder)
        require(result.get('numerical_resolution_status')=='underresolved', 'NOT_UNDERRESOLVED '+folder)
        cases[folder]=(req,result)
    ex8,ex16=cases['diagnostic_short_v1'],cases['diagnostic_time16_ex_v1']
    ey8,ey16=cases['diagnostic_short_trans_v1'],cases['diagnostic_time16_ey_v1']
    ey90=cases['diagnostic_resolution90_ey_v1']
    # Any changes other than request_id and specifically varied numerical settings are forbidden.
    def match(a,b,allowed):
        x,y=a[0],b[0]
        require(set(x)==set(y), 'REQUEST_FIELD_SET_MISMATCH')
        changes={k for k in x if x[k]!=y[k]}
        require(changes <= allowed|{'request_id'}, 'UNCONTROLLED_REQUEST_DIFFERENCE '+str(sorted(changes)))
    match(ex8,ex16,{'run_until_after_sources'})
    match(ey8,ey16,{'run_until_after_sources'})
    match(ey16,ey90,{'resolution_px_per_um'})
    match(ex8,ey8,{'field_component','polarization','source_solver_job_id','source_numerical_plan_id','wavelength_case_bindings'})
    require(ex8[0]['field_component']=='Ex' and ey8[0]['field_component']=='Ey', 'POLARIZATION_CASE_ASSIGNMENT')
    require(gate.get('ex_peak_location')=='UPPER_BOUNDARY_CENSORED', 'EX_PEAK_GATE_MISMATCH')
    require(float(ex8[0]['run_until_after_sources'])==8 and float(ex16[0]['run_until_after_sources'])==16,'EX_TIME_CASE')
    require(float(ey8[0]['run_until_after_sources'])==8 and float(ey16[0]['run_until_after_sources'])==16,'EY_TIME_CASE')
    require(float(ey16[0]['resolution_px_per_um'])==60 and float(ey90[0]['resolution_px_per_um'])==90,'SPATIAL_CASE')
    ex_time=spectrum_comparison(ex8[1],ex16[1]);ey_time=spectrum_comparison(ey8[1],ey16[1]);spatial=spectrum_comparison(ey16[1],ey90[1])
    require(abs(spatial['relative_l1']-float(gate['ey_spatial_relative_l1_change']))<0.000005,'SPATIAL_METRIC_GATE_MISMATCH')
    require(abs((spatial['peak_b_nm']-spatial['peak_a_nm'])-float(gate['ey_peak_shift_nm']))<1e-6,'PEAK_SHIFT_GATE_MISMATCH')
    require(abs(ex_time['peak_b_nm']-float(ex16[1]['spectrum'][-1]['wavelength_nm']))<1e-6, 'EX_PEAK_NOT_UPPER_CENSORED')
    metrics={
        'ex_time_relative_l1':round(ex_time['relative_l1'],8),
        'ey_time_relative_l1':round(ey_time['relative_l1'],8),
        'ey_spatial_relative_l1':round(spatial['relative_l1'],8),
        'ey_peak_60_nm':round(spatial['peak_a_nm'],4),
        'ey_peak_90_nm':round(spatial['peak_b_nm'],4),
        'ey_peak_shift_nm':round(spatial['peak_b_nm']-spatial['peak_a_nm'],4),
        'ex_peak_upper_boundary_censored':True,
        'spatial_shell_cells_60':round(float(ey16[0]['shell_thickness_nm'])*60/1000,3),
        'spatial_shell_cells_90':round(float(ey90[0]['shell_thickness_nm'])*90/1000,3),
    }
    return gate, metrics, gate_path, hashfile(gate_path)


SYSTEM_PROMPT = '''You are a research-methodology assistant designing an independent, falsifiable scientific investigation. Return exactly one JSON object with string fields: research_direction, competing_explanations, discriminating_test, independent_observables, falsification, uncertainty_handling, next_action, prior_art_qualification. Each field must be detailed (at least 40 characters). No markdown fences. Never claim novelty, mechanism confirmation, or verified SERS band ratios without independent experiments or prior-art review. Be specific about a next experiment and its controls; refuse to treat unvalidated solver output as established physics.'''
COMMON_USER = '''Research-design task (M15-C0 exploratory; no novelty certification):
A hypothetical isolated Au-core/Ag-shell spherocylindrical nanorod is excited under parallel/longitudinal (Ex) versus transverse (Ey) polarization. Distinguish a purely classical Maxwell/polarization-based explanation for an optical spectral response from an additional molecular orientation / adsorption explanation for any hypothetical SERS band-ratio effect. Design one decisive, feasible next research investigation, with controlled geometry, a competing null, independent optical and molecular observables, explicit failure criteria, and a prior-art search strategy.
Known limitation, identical for both arms: the Meep worker computes isolated-particle scattering cross-section only. It does NOT calculate near-field SERS enhancement, Raman band ratios, molecular orientation, chemical enhancement, or adsorption kinetics. The previously attempted numerical calculations are not scientifically qualified; no band-ratio predictions can be inferred from them. Avoid presenting the well-known existence of polarization-dependent nanorod resonances as a new finding.
Please prioritize which specific measurement or calculation would distinguish the competing explanations and what evidence would change the scientific research direction.'''
ARM_A = '''Information available for this condition: generic numerical-quality guidance only. A 3D FDTD computation of a thin-shell metallic nanorod requires independent time-window and spatial-resolution convergence tests, wavelength-window controls, and external physical benchmarks. No measured convergence metrics or trustworthy quantitative spectra are provided. Formulate a research plan without assuming any numerical difference is established.'''


def make_prompts(metrics, seed):
    arm_b = f'''Information available for this condition: audited numerical diagnostic, NOT validated physics. Same single-particle Meep model, hash-matched Meep worker results, same matched experimental setup. Doubling the post-source runtime from 8 to 16 changed the Ex spectrum by {100*metrics['ex_time_relative_l1']:.3f}% (relative L1) and Ey spectrum by {100*metrics['ey_time_relative_l1']:.3f}%. In contrast, increasing the Ey spatial grid from 60 to 90 px/um changed its spectrum by {100*metrics['ey_spatial_relative_l1']:.3f}% (relative L1), even though its sampled peak stayed at {metrics['ey_peak_60_nm']:.2f} nm. The Ex spectral maximum lies at the 850 nm upper window edge and is censored. This finite-resolution diagnostic has a thin shell sampled at only about {metrics['spatial_shell_cells_60']:.1f} to {metrics['spatial_shell_cells_90']:.1f} grid cells and explicitly FAILED the spatial-stability gate. DO NOT use these values as physical estimates, Raman predictions, novelty evidence, or evidence favoring either chemical or Maxwell mechanism. Instead interpret what the numerical failure teaches about the next discriminating research design.'''
    prompts={arm:{'system':SYSTEM_PROMPT,'user':COMMON_USER+'\n\n'+(ARM_A if arm=='A' else arm_b)} for arm in ('A','B')}
    order=['A','B'];random.Random(seed).shuffle(order)
    return prompts,order


def prepare(gate_path, out, seed):
    gate, metrics, gate_path, gate_sha = validate_gate(gate_path)
    prompts, order = make_prompts(metrics, seed)
    plan={'schema':VERSION,'gate_path':str(gate_path),'gate_sha256':gate_sha,
          'source_count':len(gate['sources']),'source_validation':'RECOMPUTED_FROM_HASH_MATCHED_MEEP_WORKER_JSON_NO_SIGNED_EXECUTION_ATTESTATION',
          'metrics':metrics,'seed':seed,'execution_order':order,
          'research_task_scope':'RESEARCH_DESIGN_FROM_NUMERICAL_UNCERTAINTY_NOT_NOVELTY',
          'quantitative_physics_feedback':'BLOCKED',
          'new_fdtd_runs':0,'new_model_calls':0,'novelty_certified':False}
    freeze(out/'M15C0_FROZEN_PLAN.json',plan)
    freeze(out/'M15C0_FROZEN_PROMPTS.json',{'schema':VERSION,'prompts':prompts,
               'prompt_sha256_by_arm':{arm:digest(canonical(p)) for arm,p in prompts.items()}})
    return {'status':'M15C0_PREPARED','verified_sources':10,'new_fdtd_runs':0,'new_paid_calls':0,
            'review_scope':'research_design_only','out_dir':str(out.resolve())}


def preflight(out):
    plan=load(out/'M15C0_FROZEN_PLAN.json')
    spec=load(out/'M15C0_FROZEN_PROMPTS.json')
    require(plan.get('schema')==VERSION and spec.get('schema')==VERSION,'PLAN_SCHEMA')
    require(plan['quantitative_physics_feedback']=='BLOCKED','QUANTITATIVE_GATE_NOT_BLOCKED')
    gate, metrics, _, sha=validate_gate(plan['gate_path'])
    require(sha==plan['gate_sha256'],'GATE_CHANGED_AFTER_FREEZE')
    require(metrics==plan['metrics'],'METRICS_CHANGED_AFTER_FREEZE')
    prompts,order=make_prompts(metrics,plan['seed'])
    require(order==plan['execution_order'] and prompts==spec['prompts'],'FROZEN_PROMPTS_CHANGED')
    for arm in ('A','B'):
        require(spec['prompt_sha256_by_arm'][arm]==digest(canonical(prompts[arm])),'PROMPT_HASH_INVALID_'+arm)
    return plan,spec


def generate_paid(out, model, api_key_env, base_url, max_new_calls, max_tokens, allow_paid, authorize):
    plan, spec=preflight(out)
    require(allow_paid and authorize,'PAID_CALL_EXPLICIT_AUTHORIZATION_REQUIRED')
    require(model and max_tokens>=500 and 0<=max_new_calls<=2,'MODEL_OR_BUDGET_INVALID')
    api_key=os.environ.get(api_key_env)
    require(bool(api_key),'MODEL_API_KEY_MISSING')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('openai package required only for explicitly authorized model calls') from exc
    freeze(out/'M15C0_PAID_CONFIG.json',{'schema':VERSION,'model':model,'base_url':base_url,'max_tokens':max_tokens,'temperature':0.5})
    client=OpenAI(base_url=base_url,api_key=api_key,timeout=120,max_retries=0)
    receipts=out/'paid';receipts.mkdir(exist_ok=True)
    count=0
    for arm in plan['execution_order']:
        p=spec['prompts'][arm];ph=spec['prompt_sha256_by_arm'][arm]
        receipt=receipts/f'{arm}.receipt.json';intent=receipts/f'{arm}.intent.json'
        if receipt.exists():
            require(load(receipt).get('prompt_sha256')==ph,'EXISTING_RECEIPT_PROMPT_MISMATCH')
            continue
        if intent.exists():
            return {'status':'BLOCKED_UNKNOWN_PAID_CALL_OUTCOME_NO_RETRY','arm':arm,'new_paid_calls':count}
        if count>=max_new_calls:break
        freeze(intent,{'schema':VERSION,'arm':arm,'prompt_sha256':ph,'model':model,'call_started':True})
        try:
            result=client.chat.completions.create(model=model,
                messages=[{'role':'system','content':p['system']},{'role':'user','content':p['user']}],
                temperature=0.5,max_tokens=max_tokens)
        except Exception as exc:
            return {'status':'MODEL_PROVIDER_EXCEPTION_NO_RETRY','arm':arm,'error':str(exc),'new_paid_calls':count}
        choice=result.choices[0]
        freeze(receipt,{'schema':VERSION,'arm':arm,'prompt_sha256':ph,'model':model,
               'finish_reason':choice.finish_reason,'raw':choice.message.content or '',
               'usage':result.usage.model_dump(mode='json') if result.usage else None})
        count+=1
        if choice.finish_reason!='stop':
            return {'status':'TRUNCATED_RECEIPT_NO_RETRY','arm':arm,'new_paid_calls':count}
    return {'status':'PAID_RECEIPTS_SAVED','new_paid_calls':count,
            'completed_receipts':sum((receipts/f'{arm}.receipt.json').exists() for arm in ('A','B'))}


def validate_candidate(obj):
    require(isinstance(obj,dict) and set(obj)==set(FIELDS),'CANDIDATE_SCHEMA_INVALID')
    require(all(isinstance(v,str) and len(v.strip())>=40 for v in obj.values()),'CANDIDATE_FIELDS_TOO_SHORT')
    return obj


def parse_strict_json(raw):
    def no_duplicate(pairs):
        obj={}
        for k,v in pairs:
            require(k not in obj,'DUPLICATE_CANDIDATE_KEY '+k)
            obj[k]=v
        return obj
    return json.loads(raw,object_pairs_hook=no_duplicate,
                      parse_constant=lambda s:fail('INVALID_JSON_CONSTANT'))


def materialize(out):
    plan,spec=preflight(out)
    candidates=[];mapping=[]
    for arm in ('A','B'):
        paid=out/'paid'/f'{arm}.receipt.json'
        manual=out/'manual'/f'{arm}.response.json'
        require(not (paid.exists() and manual.exists()),'AMBIGUOUS_RESPONSE_SOURCE_'+arm)
        require(paid.exists() or manual.exists(),'MISSING_RESPONSE_'+arm)
        if paid.exists():
            require((out/'M15C0_PAID_CONFIG.json').exists(), 'PAID_CONFIG_MISSING')
            config=load(out/'M15C0_PAID_CONFIG.json')
            payload=load(paid)
            require(payload.get('model')==config.get('model'), 'PAID_MODEL_NOT_FROZEN')
            require(payload['schema']==VERSION and payload['arm']==arm and payload['finish_reason']=='stop','PAID_RECEIPT_INVALID')
            require(payload['prompt_sha256']==spec['prompt_sha256_by_arm'][arm],'PAID_PROMPT_MISMATCH')
            obj=validate_candidate(parse_strict_json(payload['raw']))
            source='paid_provider_receipt'; h=hashfile(paid)
        else:
            obj=validate_candidate(parse_strict_json(manual.read_text(encoding='utf-8')))
            source='manual_response_unverified_provenance';h=hashfile(manual)
        bid='m15c0_'+digest(canonical([plan['seed'],arm,h]))[:14]
        candidates.append({'blind_id':bid,'response':obj})
        mapping.append({'blind_id':bid,'arm':arm,'source':source,'source_sha256':h})
    freeze(out/'M15C0_BLINDED_REVIEW.json',{'schema':VERSION,'entries':sorted(candidates,key=lambda a:a['blind_id']),
              'warning':'Responses may reveal arm through numerical language. No novelty claim.'})
    freeze(out/'M15C0_PRIVATE_ARM_MAP.json',{'schema':VERSION,'entries':mapping,
              'warning':'Keep private until both independent reviews are frozen'})
    template=out/'M15C0_REVIEW_TEMPLATE.csv'
    if not template.exists():
        with template.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=CSV_FIELDS);w.writeheader()
            for row in candidates:
                w.writerow({'blind_id':row['blind_id']})
    return {'status':'M15C0_BLIND_PACKET_READY','candidates':2,
            'new_paid_calls':0,'novelty_certified':False}


def score(out):
    plan,spec=preflight(out)
    pub=load(out/'M15C0_BLINDED_REVIEW.json')
    private=load(out/'M15C0_PRIVATE_ARM_MAP.json')
    require(pub.get('schema')==VERSION and private.get('schema')==VERSION,'BLIND_PACKET_SCHEMA')
    require(len(pub['entries'])==len(private['entries'])==2,'BLIND_PACKET_COUNT')
    # Check source candidates still match the blinded packet.
    materialize(out)
    with (out/'M15C0_REVIEW_TEMPLATE.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)>=4,'NEED_TWO_REVIEWERS_PER_CANDIDATE')
    ids={r['blind_id'] for r in pub['entries']}
    require(set(r['blind_id'] for r in rows)==ids,'REVIEW_COVERAGE_INVALID')
    require(all(set(r)==set(CSV_FIELDS) for r in rows),'REVIEW_COLUMNS_INVALID')
    reviewer_groups=[]
    summaries={}
    for bid in ids:
        group=[r for r in rows if r['blind_id']==bid]
        reviewers={r['reviewer_id'] for r in group}
        require(len(group)==len(reviewers)==2 and all(len(x)>=2 for x in reviewers),'REVIEWER_COUNT_INVALID')
        reviewer_groups.append(reviewers)
        for row in group:
            require(row['unsupported_quantitative_claim'] in {'yes','no'},'CLAIM_FLAG_INVALID')
            require(len(row['rationale'])>=30,'REVIEW_RATIONALE_TOO_SHORT')
            for k in SCORE_FIELDS:
                require(row[k] in {'0','1','2','3','4'},'SCORE_INVALID_'+k)
        summaries[bid]={k:statistics.mean(int(r[k]) for r in group) for k in SCORE_FIELDS}
        summaries[bid]['unsupported_claims_count']=sum(r['unsupported_quantitative_claim']=='yes' for r in group)
    require(reviewer_groups[0]==reviewer_groups[1],'REVIEWER_PAIRS_UNBALANCED')
    by_arm={r['arm']:summaries[r['blind_id']] for r in private['entries']}
    require(set(by_arm)=={'A','B'},'ARM_MAP_INVALID')
    deltas={k:by_arm['B'][k]-by_arm['A'][k] for k in SCORE_FIELDS}
    report={'schema':VERSION,'status':'ONE_CASE_RESEARCH_DESIGN_DIAGNOSTIC_FEEDBACK_ONLY',
            'response_provenance_by_arm':{r['arm']:r['source'] for r in private['entries']},
            'model_identity_verified':all(r['source']=='paid_provider_receipt' for r in private['entries']),
            'score_A':by_arm['A'],'score_B':by_arm['B'],
            'paired_B_minus_A':deltas,
            'unverified_quantitative_claim_delta':by_arm['B']['unsupported_claims_count']-by_arm['A']['unsupported_claims_count'],
            'review_sha256':hashfile(out/'M15C0_REVIEW_TEMPLATE.csv'),
            'frozen_plan_sha256':hashfile(out/'M15C0_FROZEN_PLAN.json'),
            'one_case_only':True,'scientific_novelty_certified':False,
            'scientific_knowledge_created':False,'quantitative_physics_feedback':'BLOCKED',
            'claim_limitation':'No paired replicates or prior-art audit; may test design discipline, NOT scientific novelty.'}
    freeze(out/'M15C0_REPORT.json',report)
    return {'status':report['status'],'paired_B_minus_A':deltas,
            'scientific_novelty_certified':False,'new_paid_calls':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--gate',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path);p.add_argument('--seed',type=int,default=1501)
    for name in ('preflight','generate-paid','materialize','score'):
        p=sub.add_parser(name);p.add_argument('--out-dir',required=True,type=Path)
        if name=='generate-paid':
            p.add_argument('--model',default=os.getenv('OPENROUTER_AGENT_MODEL'))
            p.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
            p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
            p.add_argument('--max-new-calls',type=int,default=2)
            p.add_argument('--max-output-tokens',type=int,default=2000)
            p.add_argument('--allow-paid',action='store_true')
            p.add_argument('--i-authorize-model-calls',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare':res=prepare(args.gate,args.out_dir,args.seed)
    elif args.command=='preflight':
        plan,_=preflight(args.out_dir)
        res={'status':'M15C0_PREFLIGHT_OK','metrics':plan['metrics'],'new_paid_calls':0}
    elif args.command=='generate-paid':
        res=generate_paid(args.out_dir,args.model,args.api_key_env,args.base_url,
                          args.max_new_calls,args.max_output_tokens,args.allow_paid,args.i_authorize_model_calls)
    elif args.command=='materialize':res=materialize(args.out_dir)
    else:res=score(args.out_dir)
    print(json.dumps(res,ensure_ascii=False,indent=2))
    return res

if __name__=='__main__':
    main()
