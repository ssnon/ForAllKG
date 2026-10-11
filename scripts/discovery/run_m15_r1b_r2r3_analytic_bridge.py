#!/usr/bin/env python3
"""M15-R1B/R2/R3: genuine analytic-scattering computation feedback workflow.

IMPORTANT: This is a homogeneous dielectric SPHERE analytical Mie benchmark, NOT a
validated Au/Ag nanorod/SERS backend, not a novelty engine, and never edits the KG.

Only conditionally user-authorized OpenRouter API calls may be made. The default
workflow is fully offline and uses no FDTD; no x1 oracle is computed before the
G1 responses are frozen, and no x2 oracle before the G2 responses are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

SCHEMA = 'm15-r1b-r2r3-analytic-sphere-feedback-v1'
SPHERE = {'radius_nm':95.0,'particle_index_real':2.2,'medium_index_real':1.0,
          'nonabsorbing':True,'geometry':'homogeneous_dielectric_sphere'}
X0 = 475.0
X1 = 635.0
X2 = 720.0
G1_KEYS = {'prediction_um2','prediction_interval_um2','preferred_model','conditional_claim','falsification','reasoning'}
G2_KEYS = {'prediction_x2_um2','prediction_interval_um2','preferred_model','next_test_wavelength_nm',
           'decision','conditional_claim','falsification','reason_for_change'}


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha(Path(path).read_bytes())


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('INVALID_NUMBER')))


def freeze(path, value):
    path=Path(path)
    value_bytes=canonical(value)
    if path.exists():
        require(path.read_bytes()==value_bytes,'FROZEN_FILE_CONFLICT '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:
            f.write(value_bytes)
    return path


def mie_sphere_cross_section_um2(wavelength_nm, *, radius_nm=95.0, n_particle=2.2, n_medium=1.0):
    """Full-wave Lorenz-Mie plane-wave total scattering cross-section.

    Nonmagnetic, homogeneous, nonabsorbing dielectric sphere in lossless medium;
    nm input, um^2 output. scipy spherical Bessel routines used for stability.
    This is not a nanorod or a core-shell metal model.
    """
    try:
        from scipy.special import spherical_jn, spherical_yn
    except ImportError as exc:
        raise RuntimeError('scipy is required for the optional analytic benchmark; no solver attempted') from exc
    for value in (wavelength_nm,radius_nm,n_particle,n_medium):
        require(isinstance(value,(float,int)) and not isinstance(value,bool)
                and math.isfinite(value) and value>0,'INVALID_MIE_PARAMETER')
    x = (2.0*math.pi*n_medium*radius_nm)/wavelength_nm
    m = n_particle/n_medium
    k = 2.0*math.pi*n_medium/wavelength_nm  # nm^-1
    nmax = max(3, int(math.ceil(x+4.0*x**(1.0/3.0)+8)))
    total=0.0
    for order in range(1,nmax+1):
        def psi(z):
            return z*spherical_jn(order,z)
        def dpsi(z):
            return spherical_jn(order,z)+z*spherical_jn(order,z,derivative=True)
        def xi(z):
            return z*(spherical_jn(order,z)+1j*spherical_yn(order,z))
        def dxi(z):
            return (spherical_jn(order,z)+1j*spherical_yn(order,z)
                  + z*(spherical_jn(order,z,derivative=True)+1j*spherical_yn(order,z,derivative=True)))
        u,v=psi(x),psi(m*x)
        du,dv=dpsi(x),dpsi(m*x)
        z,dz=xi(x),dxi(x)
        a=(m*v*du-u*dv)/(m*v*dz-z*dv)
        b=(v*du-m*u*dv)/(v*dz-m*z*dv)
        total += (2*order+1)*(abs(a)**2+abs(b)**2)
    cross_nm2 = (2*math.pi/(k*k))*total
    result=float(cross_nm2/1_000_000.0)
    require(math.isfinite(result) and result>=0,'MIE_NUMERICAL_FAILURE')
    return result


def rayleigh_cross_section_um2(wavelength_nm, *, radius_nm=95.0, n_particle=2.2, n_medium=1.0):
    """Electric-dipole Rayleigh scattering approximation (not exact at finite x)."""
    for value in (wavelength_nm,radius_nm,n_particle,n_medium):
        require(isinstance(value,(int,float)) and math.isfinite(value) and value>0,'INVALID_RAYLEIGH_PARAMETER')
    k=2*math.pi*n_medium/wavelength_nm
    ratio=(n_particle**2-n_medium**2)/(n_particle**2+2*n_medium**2)
    return (8*math.pi/3)*(k**4)*(radius_nm**6)*(ratio**2)/1_000_000.0


def check_decision(decision_path):
    decision_path=Path(decision_path).resolve()
    decision=load(decision_path)
    require(decision.get('schema_version')=='m15r-backend-decision-v1','UNSUPPORTED_DECISION')
    require(decision.get('science_backend_choice')=='NO_QUALIFIED_BACKEND','UNEXPECTED_SCIENCE_BACKEND')
    require(decision.get('science_x0_x1_status')=='BLOCKED','EXPECTED_BENCHMARK_ONLY_GATE')
    require(decision.get('scientific_interpretation_permitted') is False,'SCIENCE_GATE_NOT_BLOCKED')
    require(decision.get('physics_claim_authority') is False,'PHYSICS_AUTHORITY_UNEXPECTED')
    return decision_path,sha_file(decision_path)


def _prompts(plan, x0):
    system=(
        'You are testing a computational-research feedback workflow on an established '
        'Maxwell scattering benchmark, NOT discovering new SERS physics. Return exactly '
        'one JSON object with the fields specified in the user prompt, no markdown. '
        'Do not invent simulator results or claim any Au/Ag nanorod, Raman, chemical or '
        'scientific novelty evidence. All numeric predictions must be in um^2.'
    )
    common=(
        'BENCHMARK ONLY: a homogeneous, nonabsorbing dielectric sphere in air '
        '(radius 95 nm, refractive index 2.2, air index 1.0), illuminated by a plane wave. '
        'The observable is TOTAL scattering cross-section in um^2. '
        'Two established computational explanations are (a) full-wave Lorenz-Mie '
        'scattering and (b) the small-particle Rayleigh approximation. '
        'For a held-out wavelength of 635 nm, predict the total scattering cross-section '
        'using your selected model and report why. The numerical value at 635 nm is '
        'NOT provided. You may derive physical scaling, but you must not execute code or '
        'call tools to compute exact Mie coefficients. Both arms have the same task, '
        'materials, geometry and reference library information. '
        'Return precisely JSON keys: prediction_um2 (positive number), '
        'prediction_interval_um2 ([positive low, positive high] including prediction), '
        'preferred_model ("full_mie" or "rayleigh" or "uncertain"), '
        'conditional_claim (detailed string), falsification (detailed string), '
        'reasoning (detailed string). No additional keys.'
    )
    control=(
        'A condition: no measured or computed reference cross-section at 475 nm '
        'is supplied. Give a cautious, independent prospective prediction for 635 nm.'
    )
    treatment=(
        'B condition: a full-wave exact Mie-scattering reference at 475 nm is provided '
        f'with total C_sca={x0["mie_um2"]:.12g} um^2 and benchmark Rayleigh '
        f'C_sca={x0["rayleigh_um2"]:.12g} um^2, both for exactly the geometry above. '
        'These are established-model results, NOT newly discovered effects. '
        'Use this scoped observation to update your 635 nm prediction.'
    )
    return {arm:{'system':system,'user':common+'\n\n'+(control if arm=='A' else treatment)} for arm in ('A','B')}


def prepare(out_dir, decision_path, seed):
    out=Path(out_dir).resolve()
    dp,dh=check_decision(decision_path)
    require(isinstance(seed,int) and seed>=0,'INVALID_SEED')
    x0_mie=mie_sphere_cross_section_um2(X0,**_mie_parameters())
    x0_ray=rayleigh_cross_section_um2(X0,**_mie_parameters())
    x0={'wavelength_nm':X0,'mie_um2':x0_mie,'rayleigh_um2':x0_ray}
    order=['A','B'];random.Random(seed).shuffle(order)
    plan={'schema':SCHEMA,'decision_path':str(dp),'decision_sha256':dh,'seed':seed,
          'execution_order':order,'analytic_benchmark':SPHERE,'x0_wavelength_nm':X0,
          'held_out_x1_wavelength_nm':X1,'held_out_x2_wavelength_nm':X2,
          'quantitative_nanorod_evidence':'BLOCKED','benchmark_scope':'homogeneous_dielectric_sphere_only',
          'backend_choice':'ANALYTIC_BENCHMARK_ONLY','scientific_novelty_authority':False,
          'physics_claim_authority':False,'new_meep_executions':0,'new_paid_calls':0}
    prompts=_prompts(plan,x0)
    bundle={'schema':SCHEMA,'prompts':prompts,'sha256_by_arm':{a:sha(canonical(p)) for a,p in prompts.items()}}
    freeze(out/'M15R_BENCHMARK_PLAN.json',plan)
    freeze(out/'M15R_X0_REFERENCE.json',{'schema':SCHEMA,'x0':x0,'model':'Lorenz-Mie exact sphere vs Rayleigh approximation',
            'not_nanoparticle_model_validation':True,'nanorod_claim_authority':False})
    freeze(out/'M15R_G1_PROMPTS.json',bundle)
    return {'status':'M15R_BENCHMARK_PREPARED','x0_mie_um2':x0_mie,'x0_rayleigh_um2':x0_ray,
            'x1_x2_values_revealed':False,'nanorod_science_gate':'BLOCKED','meep_executions':0,'paid_calls':0}


def _mie_parameters():
    return {'radius_nm':SPHERE['radius_nm'],'n_particle':SPHERE['particle_index_real'],
            'n_medium':SPHERE['medium_index_real']}


def preflight(out):
    out=Path(out).resolve()
    plan=load(out/'M15R_BENCHMARK_PLAN.json')
    require(plan.get('schema')==SCHEMA,'INVALID_FROZEN_PLAN')
    dp,dh=check_decision(plan['decision_path'])
    require(str(dp)==plan['decision_path'] and dh==plan['decision_sha256'],'DECISION_TAMPERED')
    require(plan['analytic_benchmark']==SPHERE and plan['x0_wavelength_nm']==X0
            and plan['held_out_x1_wavelength_nm']==X1
            and plan['held_out_x2_wavelength_nm']==X2,'PLAN_TAMPERED')
    require(plan['quantitative_nanorod_evidence']=='BLOCKED' and plan['backend_choice']=='ANALYTIC_BENCHMARK_ONLY'
            and plan['scientific_novelty_authority'] is False,'SCIENTIFIC_AUTHORITY_MISMATCH')
    x0=load(out/'M15R_X0_REFERENCE.json')['x0']
    require(x0=={'wavelength_nm':X0,'mie_um2':mie_sphere_cross_section_um2(X0,**_mie_parameters()),
                 'rayleigh_um2':rayleigh_cross_section_um2(X0,**_mie_parameters())}, 'X0_EVIDENCE_TAMPERED')
    prompts=load(out/'M15R_G1_PROMPTS.json')
    expected=_prompts(plan,x0)
    require(prompts['prompts']==expected and prompts['sha256_by_arm']=={a:sha(canonical(p)) for a,p in expected.items()},
            'G1_PROMPTS_TAMPERED')
    require(sorted(plan['execution_order'])==['A','B'],'INVALID_ARM_ORDER')
    return plan,prompts


def _strict_dict(raw):
    def pairs(p):
        out={}
        for k,v in p:
            require(k not in out,'DUPLICATE_JSON_KEY '+k)
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('INVALID_NUMBER')))


def _positive_float(value, name):
    require(type(value) in (float,int) and math.isfinite(value) and value>0,'INVALID_POSITIVE_NUMBER '+name)
    return float(value)


def validate_g1(obj):
    require(isinstance(obj,dict) and set(obj)==G1_KEYS,'G1_KEYS_INVALID')
    p=_positive_float(obj['prediction_um2'],'prediction_um2')
    iv=obj['prediction_interval_um2']
    require(isinstance(iv,list) and len(iv)==2,'G1_INTERVAL_INVALID')
    low,high=(_positive_float(iv[0],'interval_low'),_positive_float(iv[1],'interval_high'))
    require(low<=p<=high,'G1_PREDICTION_NOT_IN_INTERVAL')
    require(obj['preferred_model'] in {'full_mie','rayleigh','uncertain'},'G1_MODEL_INVALID')
    for k in ('conditional_claim','falsification','reasoning'):
        require(isinstance(obj[k],str) and len(obj[k].strip())>=25,'G1_TEXT_INVALID '+k)
    return obj


def validate_g2(obj):
    require(isinstance(obj,dict) and set(obj)==G2_KEYS,'G2_KEYS_INVALID')
    p=_positive_float(obj['prediction_x2_um2'],'prediction_x2_um2')
    iv=obj['prediction_interval_um2']
    require(isinstance(iv,list) and len(iv)==2,'G2_INTERVAL_INVALID')
    low,high=(_positive_float(iv[0],'interval_low'),_positive_float(iv[1],'interval_high'))
    require(low<=p<=high,'G2_PREDICTION_NOT_IN_INTERVAL')
    require(obj['preferred_model'] in {'full_mie','rayleigh','uncertain'},'G2_MODEL_INVALID')
    next_w=_positive_float(obj['next_test_wavelength_nm'],'next_test_wavelength_nm')
    require(420<=next_w<=800 and next_w not in {X0,X1,X2},'G2_NEXT_WAVELENGTH_INVALID')
    require(obj['decision'] in {'REFINE','BRANCH','ESCAPE','SUSPEND'},'G2_DECISION_INVALID')
    for k in ('conditional_claim','falsification','reason_for_change'):
        require(isinstance(obj[k],str) and len(obj[k].strip())>=25,'G2_TEXT_INVALID '+k)
    return obj


def _paid_config_path(out):
    return out/'M15R_PAID_CONFIG.json'


def paid_generate(out, phase, model, max_tokens, max_new_calls, allow_paid, authorize, api_key_env, base_url):
    out=Path(out).resolve()
    plan,spec=preflight(out)
    require(phase in {'g1','g2'},'INVALID_PAID_PHASE')
    require(allow_paid and authorize,'EXPLICIT_PAID_AUTHORIZATION_REQUIRED')
    require(model and 500<=max_tokens<=16000 and 0<=max_new_calls<=2,'INVALID_PAID_BUDGET')
    key=os.getenv(api_key_env)
    require(key,'API_KEY_MISSING')
    if phase=='g1':
        require(not (out/'M15R_G1_FROZEN.json').exists() and not (out/'M15R_X1_FEEDBACK.json').exists(),
                'G1_ALREADY_FROZEN_OR_X1_REVEALED')
    if phase=='g2':
        _verify_x1_feedback(out)
        require(not (out/'M15R_G2_FROZEN.json').exists() and not (out/'M15R_X2_FEEDBACK.json').exists(),
                'G2_ALREADY_FROZEN_OR_X2_REVEALED')
        spec=load(out/'M15R_G2_PROMPTS.json')
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('openai package must be installed for paid calls') from exc
    config={'schema':SCHEMA,'model':model,'max_tokens':max_tokens,'temperature':0.5,'base_url':base_url}
    freeze(_paid_config_path(out),config)
    client=OpenAI(api_key=key,base_url=base_url,timeout=120,max_retries=0)
    count=0
    for arm in plan['execution_order']:
        p=spec['prompts'][arm]
        ph=spec['sha256_by_arm'][arm]
        receipt=out/'paid'/f'{phase}_{arm}.receipt.json'
        intent=out/'paid'/f'{phase}_{arm}.intent.json'
        if receipt.exists():
            old=load(receipt)
            require(old['prompt_sha256']==ph and old['model']==model,'PAID_RECEIPT_MISMATCH')
            continue
        if intent.exists():
            return {'status':'UNKNOWN_CALL_OUTCOME_NO_AUTORETRY','phase':phase,'arm':arm,'new_calls':count}
        if count>=max_new_calls:
            break
        freeze(intent,{'schema':SCHEMA,'phase':phase,'arm':arm,'prompt_sha256':ph,'model':model,'begun':True})
        try:
            result=client.chat.completions.create(model=model,
                messages=[{'role':'system','content':p['system']},{'role':'user','content':p['user']}],
                temperature=0.5,max_tokens=max_tokens)
        except Exception as exc:
            return {'status':'PROVIDER_ERROR_NO_AUTORETRY','phase':phase,'arm':arm,'error':str(exc),'new_calls':count}
        chosen=result.choices[0]
        freeze(receipt,{'schema':SCHEMA,'phase':phase,'arm':arm,'prompt_sha256':ph,'model':model,
                        'finish_reason':chosen.finish_reason,'raw':chosen.message.content or '',
                        'usage':result.usage.model_dump(mode='json') if result.usage else None})
        count+=1
        if chosen.finish_reason!='stop':
            return {'status':'TRUNCATED_NO_AUTORETRY','phase':phase,'arm':arm,'new_calls':count}
    return {'status':'PAID_RESPONSES_SAVED','phase':phase,'new_calls':count,
            'receipts_saved':sum((out/'paid'/f'{phase}_{arm}.receipt.json').exists() for arm in ('A','B'))}


def _read_candidates(out, phase, spec, control_path=None, treatment_path=None):
    manual_paths={'A':control_path,'B':treatment_path}
    manual=control_path is not None or treatment_path is not None
    require(not manual or (control_path is not None and treatment_path is not None),'BOTH_MANUAL_PATHS_REQUIRED')
    candidates={}
    for arm in ('A','B'):
        if manual:
            path=Path(manual_paths[arm]).resolve()
            require(path.exists(),'MANUAL_RESPONSE_MISSING')
            require(not path.is_relative_to(out) or 'paid' not in path.parts,'MANUAL_RECEIPT_PATH_NOT_ALLOWED')
            raw=path.read_text(encoding='utf-8')
            provenance='manual_unverified_model_identity'
        else:
            path=out/'paid'/f'{phase}_{arm}.receipt.json'
            require(path.exists(),'PAID_RECEIPT_MISSING '+arm)
            receipt=load(path)
            require(receipt['schema']==SCHEMA and receipt['phase']==phase and receipt['arm']==arm,
                    'INVALID_RECEIPT '+arm)
            require(receipt['prompt_sha256']==spec['sha256_by_arm'][arm],'PAID_PROMPT_HASH_MISMATCH '+arm)
            require(receipt['finish_reason']=='stop','INCOMPLETE_RECEIPT '+arm)
            config=load(_paid_config_path(out))
            require(receipt['model']==config['model'],'MODEL_MISMATCH '+arm)
            raw=receipt['raw']
            provenance='paid_provider_receipt'
        obj=_strict_dict(raw)
        if phase=='g1':
            validate_g1(obj)
        else:
            validate_g2(obj)
        candidates[arm]={'response':obj,'source_path':str(path),'source_sha256':sha_file(path),
                         'provenance':provenance,'prompt_sha256':spec['sha256_by_arm'][arm]}
    return candidates


def freeze_g1(out, control=None, treatment=None):
    out=Path(out).resolve()
    plan,spec=preflight(out)
    require(not (out/'M15R_X1_FEEDBACK.json').exists(),'X1_ALREADY_REVEALED_CANNOT_FREEZE_G1')
    c=_read_candidates(out,'g1',spec,control,treatment)
    freeze(out/'M15R_G1_FROZEN.json',{'schema':SCHEMA,'candidates':c,'x1_unrevealed_at_freeze':True,
           'frozen_plan_sha256':sha_file(out/'M15R_BENCHMARK_PLAN.json'),
           'scientific_novelty_authority':False})
    return {'status':'G1_FROZEN','arm_count':2,'heldout_x1_revealed':False,
            'provenance':{k:v['provenance'] for k,v in c.items()}}


def _verify_g1(out):
    plan,spec=preflight(out)
    frozen=load(out/'M15R_G1_FROZEN.json')
    require(frozen['schema']==SCHEMA and frozen.get('x1_unrevealed_at_freeze') is True,'BAD_G1_FROZEN')
    require(frozen['frozen_plan_sha256']==sha_file(out/'M15R_BENCHMARK_PLAN.json'),'PLAN_HASH_CHANGED')
    require(set(frozen['candidates'])=={'A','B'},'G1_ARMS_INVALID')
    for arm,c in frozen['candidates'].items():
        validate_g1(c['response'])
        require(c['prompt_sha256']==spec['sha256_by_arm'][arm],'G1_PROMPT_MISMATCH')
        require(sha_file(c['source_path'])==c['source_sha256'],'G1_SOURCE_TAMPERED')
    return frozen


def _g2_prompts(g1,x1):
    system=(
        'You are an AI researcher operating within an established dielectric-sphere '
        'scattering benchmark, not novel SERS physics. Return exactly the JSON fields '
        'specified with physically explicit falsification and revised prediction. '
        'Do not assert novelty, chemical enhancement or nanorod validity.'
    )
    prompts={}
    for arm,c in g1['candidates'].items():
        response=c['response']
        previous=f'Your frozen G1 prediction at {X1:g} nm was {response["prediction_um2"]:.12g} um^2; selected model: {response["preferred_model"]}.\n'
        revealed=(
            f'An independent exact Lorenz-Mie calculation at {X1:g} nm now gives '
            f'{x1["mie_um2"]:.12g} um^2 (Rayleigh approximation: '
            f'{x1["rayleigh_um2"]:.12g} um^2). This is an analytic '
            'benchmark data, not new nanorod/SERS evidence. Use this observation '
            'to revise your explanation, then predict the held-out total scattering '
            f'cross-section at {X2:g} nm for the same sphere. No x2 numerical '
            'result is provided.\n'
        )
        spec=(
            'Return precisely these JSON fields: prediction_x2_um2 (positive number), '
            'prediction_interval_um2 (array [low,high] including prediction), '
            'preferred_model ("full_mie", "rayleigh", "uncertain"), '
            'next_test_wavelength_nm (a number 420..800 distinct from 475,635,720), '
            'decision (REFINE, BRANCH, ESCAPE, SUSPEND), '
            'conditional_claim (detailed string), falsification (detailed string), '
            'reason_for_change (detailed string). No additional keys.'
        )
        prompts[arm]={'system':system,'user':previous+revealed+spec}
    return prompts


def reveal_x1(out):
    out=Path(out).resolve()
    g1=_verify_g1(out)
    x1={'wavelength_nm':X1,'mie_um2':mie_sphere_cross_section_um2(X1,**_mie_parameters()),
        'rayleigh_um2':rayleigh_cross_section_um2(X1,**_mie_parameters()),
        'evidence_class':'FULL_WAVE_ANALYTIC_MIE_SPHERE_ONLY'}
    comparisons={}
    for arm,c in g1['candidates'].items():
        p=c['response']['prediction_um2']
        lo,hi=c['response']['prediction_interval_um2']
        comparisons[arm]={'frozen_prediction_um2':p,
                          'absolute_error_um2':abs(p-x1['mie_um2']),
                          'relative_error':abs(p-x1['mie_um2'])/x1['mie_um2'],
                          'interval_contains_reference':lo<=x1['mie_um2']<=hi,
                          'model_choice':c['response']['preferred_model']}
    evidence={'schema':SCHEMA,'x1':x1,'g1_frozen_sha256':sha_file(out/'M15R_G1_FROZEN.json'),
              'comparisons':comparisons,'scientific_novelty_certified':False,
              'nanorod_quantitative_evidence':'BLOCKED',
              'interpretation':'ANALYTIC_SPHERE_WORKFLOW_ONLY'}
    freeze(out/'M15R_X1_FEEDBACK.json',evidence)
    prompts=_g2_prompts(g1,x1)
    freeze(out/'M15R_G2_PROMPTS.json',{'schema':SCHEMA,'prompts':prompts,
           'sha256_by_arm':{a:sha(canonical(p)) for a,p in prompts.items()},
           'x1_feedback_sha256':sha_file(out/'M15R_X1_FEEDBACK.json')})
    return {'status':'X1_REVEALED_G2_PROMPTS_READY','x1_mie_um2':x1['mie_um2'],
            'g1_relative_error_by_arm':{a:v['relative_error'] for a,v in comparisons.items()},
            'x2_revealed':False,'nanorod_science_gate':'BLOCKED'}


def _verify_x1_feedback(out):
    g1=_verify_g1(out)
    evidence=load(out/'M15R_X1_FEEDBACK.json')
    require(evidence['schema']==SCHEMA and evidence['g1_frozen_sha256']==sha_file(out/'M15R_G1_FROZEN.json'),
            'X1_LINEAGE_INVALID')
    require(evidence['x1']=={'wavelength_nm':X1,'mie_um2':mie_sphere_cross_section_um2(X1,**_mie_parameters()),
                  'rayleigh_um2':rayleigh_cross_section_um2(X1,**_mie_parameters()),
                  'evidence_class':'FULL_WAVE_ANALYTIC_MIE_SPHERE_ONLY'},'X1_ORACLE_TAMPERED')
    prompt_bundle=load(out/'M15R_G2_PROMPTS.json')
    expected=_g2_prompts(g1,evidence['x1'])
    require(prompt_bundle['prompts']==expected and
            prompt_bundle['sha256_by_arm']=={a:sha(canonical(p)) for a,p in expected.items()},'G2_PROMPTS_TAMPERED')
    require(prompt_bundle['x1_feedback_sha256']==sha_file(out/'M15R_X1_FEEDBACK.json'),'X1_HASH_MISMATCH')
    return evidence,prompt_bundle


def freeze_g2(out, control=None, treatment=None):
    out=Path(out).resolve()
    evidence,spec=_verify_x1_feedback(out)
    require(not (out/'M15R_X2_FEEDBACK.json').exists(),'X2_ALREADY_REVEALED_CANNOT_FREEZE_G2')
    c=_read_candidates(out,'g2',spec,control,treatment)
    g1=load(out/'M15R_G1_FROZEN.json')
    transitions={}
    for arm,row in c.items():
        older=g1['candidates'][arm]['response']
        newer=row['response']
        changed=(older['preferred_model']!=newer['preferred_model'] or
                 older['conditional_claim'].strip()!=newer['conditional_claim'].strip())
        transitions[arm]={'has_semantic_or_model_text_change':changed,
                          'note':'Textual/model change alone is not scientific novelty; G1 and G2 target different wavelengths.'}
    freeze(out/'M15R_G2_FROZEN.json',{'schema':SCHEMA,'candidates':c,
            'g1_sha256':sha_file(out/'M15R_G1_FROZEN.json'),
            'x1_feedback_sha256':sha_file(out/'M15R_X1_FEEDBACK.json'),
            'x2_unrevealed_at_freeze':True,'transitions':transitions,
            'scientific_novelty_certified':False})
    return {'status':'G2_FROZEN','arm_count':2,'x2_revealed':False,
            'change_report':{arm:t['has_semantic_or_model_text_change'] for arm,t in transitions.items()}}


def reveal_x2(out):
    out=Path(out).resolve()
    evidence,spec=_verify_x1_feedback(out)
    g2=load(out/'M15R_G2_FROZEN.json')
    require(g2['schema']==SCHEMA and g2['x2_unrevealed_at_freeze'] is True,'BAD_G2_FROZEN')
    require(g2['g1_sha256']==sha_file(out/'M15R_G1_FROZEN.json') and
            g2['x1_feedback_sha256']==sha_file(out/'M15R_X1_FEEDBACK.json'),'G2_LINEAGE_INVALID')
    require(set(g2['candidates'])=={'A','B'},'G2_ARMS_MISSING')
    x2={'wavelength_nm':X2,'mie_um2':mie_sphere_cross_section_um2(X2,**_mie_parameters()),
        'rayleigh_um2':rayleigh_cross_section_um2(X2,**_mie_parameters()),
        'evidence_class':'FULL_WAVE_ANALYTIC_MIE_SPHERE_ONLY'}
    scores={}
    for arm,c in g2['candidates'].items():
        validate_g2(c['response'])
        require(c['prompt_sha256']==spec['sha256_by_arm'][arm],'G2_PROMPT_MISMATCH')
        require(sha_file(c['source_path'])==c['source_sha256'],'G2_RECEIPT_TAMPERED')
        p=c['response']['prediction_x2_um2']
        lo,hi=c['response']['prediction_interval_um2']
        scores[arm]={'prediction_um2':p,'relative_error':abs(p-x2['mie_um2'])/x2['mie_um2'],
                     'interval_contains_reference':lo<=x2['mie_um2']<=hi,
                     'proposed_next_wavelength_nm':c['response']['next_test_wavelength_nm'],
                     'decision':c['response']['decision']}
    report={'schema':SCHEMA,'status':'BENCHMARK_COMPUTATIONAL_FEEDBACK_WORKFLOW_FUNCTIONAL',
            'benchmark_scope':'DIELECTRIC_SPHERE_NOT_AU_AG_NOT_SERS','x0_sha256':sha_file(out/'M15R_X0_REFERENCE.json'),
            'g1_sha256':sha_file(out/'M15R_G1_FROZEN.json'),
            'x1_feedback_sha256':sha_file(out/'M15R_X1_FEEDBACK.json'),
            'g2_sha256':sha_file(out/'M15R_G2_FROZEN.json'),
            'x2':x2,'g1_x1_errors':evidence['comparisons'],'g2_x2_errors':scores,
            'arm_model_provenance':{arm:c['provenance'] for arm,c in g2['candidates'].items()},
            'interpretation':'One paired instance is descriptive only; no demonstration of novelty or general benefit.',
            'scientific_novelty_certified':False,'nanorod_science_gate':'BLOCKED',
            'canonical_graph_mutated':False,'new_fdtd_runs':0}
    freeze(out/'M15R_X2_FEEDBACK.json',{'schema':SCHEMA,'x2':x2,
           'g2_frozen_sha256':sha_file(out/'M15R_G2_FROZEN.json'),'nanorod_evidence':'BLOCKED'})
    freeze(out/'M15R_BENCHMARK_REPORT.json',report)
    return {'status':report['status'],'g2_x2_relative_errors':{a:s['relative_error'] for a,s in scores.items()},
            'nanorod_science_gate':'BLOCKED','scientific_novelty_certified':False}


def probe_local():
    executables=['scuff-scatter','scuff-analyze','octave','matlab','gmsh']
    installed={e:bool(shutil.which(e)) for e in executables}
    libraries={}
    for name in ('scipy','bempp','pymiescatt','gmsh'):
        try:
            from importlib.util import find_spec
            libraries[name]=find_spec(name) is not None
        except (ImportError,ValueError,ModuleNotFoundError):
            libraries[name]=False
    return {'status':'AVAILABILITY_PROBE_ONLY','executables':installed,'python_modules':libraries,
            'matched_Au_Ag_nanorod_backend_qualified':False,
            'caveat':'Installed executable/library is not evidence of compatible geometry, optical constants or converged scattering.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    sub=subs.add_parser('probe')
    sub=subs.add_parser('prepare');sub.add_argument('--decision',required=True,type=Path)
    sub.add_argument('--out-dir',required=True,type=Path);sub.add_argument('--seed',type=int,default=1502)
    for name in ('preflight','freeze-g1','reveal-x1','freeze-g2','reveal-x2','generate-paid'):
        sub=subs.add_parser(name);sub.add_argument('--out-dir',required=True,type=Path)
        if name in ('freeze-g1','freeze-g2'):
            sub.add_argument('--control',type=Path);sub.add_argument('--treatment',type=Path)
        if name=='generate-paid':
            sub.add_argument('--phase',choices=['g1','g2'],required=True)
            sub.add_argument('--model',default=os.getenv('OPENROUTER_AGENT_MODEL'))
            sub.add_argument('--max-new-calls',type=int,default=2)
            sub.add_argument('--max-output-tokens',type=int,default=1700)
            sub.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
            sub.add_argument('--base-url',default='https://openrouter.ai/api/v1')
            sub.add_argument('--allow-paid',action='store_true')
            sub.add_argument('--i-authorize-model-calls',action='store_true')
    a=p.parse_args(argv)
    if a.command=='probe':value=probe_local()
    elif a.command=='prepare':value=prepare(a.out_dir,a.decision,a.seed)
    elif a.command=='preflight':
        plan,_=preflight(a.out_dir)
        value={'status':'PRE_FLIGHT_OK','scope':plan['benchmark_scope'],'x1_revealed':(a.out_dir/'M15R_X1_FEEDBACK.json').exists(),
               'x2_revealed':(a.out_dir/'M15R_X2_FEEDBACK.json').exists(),
               'science_gate':plan['quantitative_nanorod_evidence']}
    elif a.command=='freeze-g1':value=freeze_g1(a.out_dir,a.control,a.treatment)
    elif a.command=='reveal-x1':value=reveal_x1(a.out_dir)
    elif a.command=='freeze-g2':value=freeze_g2(a.out_dir,a.control,a.treatment)
    elif a.command=='reveal-x2':value=reveal_x2(a.out_dir)
    else:value=paid_generate(a.out_dir,a.phase,a.model,a.max_output_tokens,a.max_new_calls,
                             a.allow_paid,a.i_authorize_model_calls,a.api_key_env,a.base_url)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,KeyError) as e:
        print('M15R_ERROR:',str(e),file=sys.stderr)
        raise SystemExit(2)
