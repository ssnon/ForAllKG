"""M15-R10: genuine calculated phenomenon -> matched LLM scientific-question A/B.

The *only* quantitative physics source is independent exact Lorenz-Mie theory for
homogeneous, lossless dielectric spheres. It is a known textbook phenomenon,
NOT Au/Ag nanorod validation or a scientific novelty claim.

Workflow: prepare (offline train), run-paid (explicit authorization), blind,
reveal (validated completed blind review + fresh held-out calculations), check.
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
import secrets

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scripts.discovery import run_m15_r9_feedback_pilot as r9

SCHEMA='m15-r10-phenomenon-driven-ab-pilot-v1'
QUESTION=('We want to test whether observing an actual computed physical pattern, rather than a '
          'generic critique or solver error, changes a scientific research planner\'s next research '
          'question. Study scattering of a homogeneous, nonabsorbing dielectric sphere in vacuum '
          'using exact Maxwell theory. This is a methodological benchmark, not Au/Ag or SERS evidence. '
          'Propose one decisive next research question, two rival explanations, a directional '
          'quantitative prediction, an independent or withheld-condition check, and a concrete '
          'falsifier. State whether the idea is probably textbook physics and suggest what genuinely '
          'new data or condition would be required to motivate a research contribution.')
SYSTEM=('You are a scientist designing hypothesis-led studies. Treat the stated model and numbers '
        'strictly within their domain, do not invent measured data, and do not claim textbook Mie '
        'resonances or polarization features as newly discovered physics. Do not attribute any '
        'sphere result to Au/Ag nanorods or Raman band ratios. Give ONE genuinely discriminating '
        'research plan. Output ONLY one JSON object with nonempty fields: research_question, '
        'null_model, alternative_model, quantitative_prediction, independent_test, '
        'falsification_criterion, conceptual_change, prior_art_risk, limitations.')
RESPONSE_FIELDS=('research_question','null_model','alternative_model','quantitative_prediction',
                 'independent_test','falsification_criterion','conceptual_change','prior_art_risk','limitations')
SCORES=('conceptual_shift_0to2','model_separation_0to2','independent_test_0to2',
        'quantitative_grounding_0to2','falsifiability_0to2','novelty_caution_0to2')
HOLDOUT_RI=(1.48,1.52) # kept out of LLM prompts; disclosed only upon unblinding
BASE_M=1.5
GRID_START=5.0
GRID_STEP=0.05
GRID_STEPS=61


def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha(p:Path)->str:return sha_bytes(p.read_bytes())
def dump_new(p:Path,v:object)->None:
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')

def mie_qsca(x:float,relative_index:float)->float:
    """Exact nonmagnetic sphere Mie a_n,b_n; Qsca=2/x² sum (2n+1)(|a|²+|b|²)."""
    if not all(map(math.isfinite,(x,relative_index))) or not (0.01<=x<=12. and 1<=relative_index<=2.0):
        raise ValueError('UNSUPPORTED_MIE_PARAMETER_DOMAIN')
    if relative_index==1.0:return 0.0
    n=np.arange(1,int(x+4*x**(1/3)+8)+1)
    psi=lambda z:z*spherical_jn(n,z)
    dpsi=lambda z:spherical_jn(n,z)+z*spherical_jn(n,z,derivative=True)
    xi=x*(spherical_jn(n,x)+1j*spherical_yn(n,x))
    dxi=spherical_jn(n,x)+1j*spherical_yn(n,x)+x*(spherical_jn(n,x,derivative=True)+1j*spherical_yn(n,x,derivative=True))
    m=relative_index; mx=m*x
    a=(m*psi(mx)*dpsi(x)-psi(x)*dpsi(mx))/(m*psi(mx)*dxi-xi*dpsi(mx))
    b=(psi(mx)*dpsi(x)-m*psi(x)*dpsi(mx))/(psi(mx)*dxi-m*xi*dpsi(mx))
    q=float(2./x**2*np.sum((2*n+1)*(abs(a)**2+abs(b)**2)))
    if not math.isfinite(q) or q<0:raise ValueError('MIE_NONFINITE_OR_NEGATIVE_OUTPUT')
    return q


def rayleigh_qsca(x:float,m:float)->float:
    return (8./3.)*x**4*((m*m-1)/(m*m+2))**2


def discover_pattern()->dict:
    """Predefined, deterministic train-only local-max-to-next-min selection."""
    xs=[round(GRID_START+i*GRID_STEP,9) for i in range(GRID_STEPS)]
    vals=[mie_qsca(x,BASE_M) for x in xs]
    extrema=[]
    for i in range(1,len(xs)-1):
        if vals[i]>vals[i-1] and vals[i]>vals[i+1]:extrema.append(('MAX',i))
        if vals[i]<vals[i-1] and vals[i]<vals[i+1]:extrema.append(('MIN',i))
    candidates=[]
    for j,(tag,k) in enumerate(extrema[:-1]):
        nxt,idx=extrema[j+1]
        if tag!='MAX' or nxt!='MIN':continue
        rel=(vals[k]-vals[idx])/vals[k]
        if xs[idx]-xs[k]<=1.0 and rel>=0.05:
            candidates.append({'x_hi':xs[k],'x_lo':xs[idx],'q_hi':vals[k],
                               'q_lo':vals[idx],'fractional_drop':rel,
                               'peak_to_trough_delta':vals[k]-vals[idx]})
    if not candidates:raise ValueError('NO_PREDECLARED_NONMONOTONIC_PATTERN')
    winner=sorted(candidates,key=lambda z:(-z['fractional_drop'],z['x_hi']))[0]
    if not (winner['q_hi']>winner['q_lo']):raise ValueError('PATTERN_NOT_ORDERED')
    return {'method':'Exact Lorenz-Mie homogeneous lossless sphere; direct scipy.special spherical Bessel evaluations',
            'n_relative':BASE_M,'medium':'vacuum','material':'nondispersive dielectric, nonmetal',
            'independent_variable':'dimensionless size parameter x=2*pi*radius/lambda',
            'observable':'scattering efficiency Qsca= sigma_sca/(pi*radius^2)',
            'training_grid':{'start':GRID_START,'step':GRID_STEP,'steps':GRID_STEPS,
                             'minimum_x':xs[0],'maximum_x':xs[-1]},
            'selection_rule':'largest relative drop from a strict grid local maximum to its immediately subsequent grid local minimum; x separation <=1, drop>=5%',
            'selected_pattern':winner,'observed_local_extrema_count':len(extrema),
            'full_training_points':[{'x':x,'qsca':v} for x,v in zip(xs,vals)],
            'science_novelty_certified':False,'au_ag_science_gate':'BLOCKED'}


def treatment_card(train:dict)->str:
    v=train['selected_pattern']
    return (f'EXTERNAL EXACT-PHYSICS TRAINING OBSERVATION (ONLY TREATMENT SEES THIS):\n'
            f'Independent Lorenz-Mie exact-sphere computation; lossless nondispersive n=1.50 sphere '
            f'in vacuum. The dimensionless scattering efficiency Qsca is {v["q_hi"]:.7f} '
            f'at x={v["x_hi"]:.2f} and {v["q_lo"]:.7f} at x={v["x_lo"]:.2f}, '
            f'a {100*v["fractional_drop"]:.4f}% decline despite increasing size parameter. '
            'These points were selected by a prespecified peak-to-next-trough rule on x from '
            '5 to 8. A nearby, withheld refractive-index perturbation test has been fixed '
            'in advance, but its values and results are NOT available. This oscillatory pattern '
            'is expected to overlap textbook Mie scattering physics and IS NOT a novelty claim. '
            'Do not extrapolate to Au/Ag nanorods, molecular SERS, or surface chemistry. '
            'Explicitly identify what differentiating question this observation makes worth '
            'asking and how to falsify it.\n')


def ensure_plan(out:Path)->dict:
    out=out.resolve();plan=json.loads((out/'M15R10_PLAN_PRIVATE.json').read_text())
    if plan.get('schema_version')!=SCHEMA or plan.get('au_ag_science_gate')!='BLOCKED':
        raise ValueError('R10_PLAN_OR_SCIENCE_GATE_INVALID')
    physical=discover_pattern(); source=out/'M15R10_TRAIN_PHYSICS.json'
    if sha(source)!=plan['training_sha256'] or json.loads(source.read_text())!=physical:
        raise ValueError('TRAINING_PHYSICS_CHANGED')
    if plan['holdout_commitment_sha256']!=sha_bytes(json.dumps({
            'indices':HOLDOUT_RI,'x':[physical['selected_pattern']['x_hi'],physical['selected_pattern']['x_lo']]},
            sort_keys=True).encode()):
        raise ValueError('HOLDOUT_COMMITMENT_CHANGED')
    calls=plan['calls'];reps=plan['replicates']
    if len(calls)!=2*reps or len({v['call_id'] for v in calls})!=len(calls) or len({v['blind_id'] for v in calls})!=len(calls):
        raise ValueError('R10_IMBALANCED_OR_DUPLICATED_CALLS')
    if any(sum(c['arm']==arm and c['replicate']==rep for c in calls)!=1
           for arm in ('CONTROL','PHENOMENON_TREATMENT') for rep in range(1,reps+1)):
        raise ValueError('R10_IMBALANCED_PAIRING')
    for c in calls:
        expected=QUESTION if c['arm']=='CONTROL' else QUESTION+'\n\n'+treatment_card(physical)
        if c['user_prompt']!=expected or c['prompt_sha256']!=sha_bytes(expected.encode()):
            raise ValueError('R10_PROMPT_CHANGED')
        if any(str(x) in c['user_prompt'] for x in HOLDOUT_RI):
            raise ValueError('HOLDOUT_LEAK_IN_PROMPT')
    if plan['system_prompt']!=SYSTEM:raise ValueError('SYSTEM_PROMPT_CHANGED')
    return plan


def prepare(out:Path,model:str,reps:int,temp:float,tokens:int)->dict:
    out=out.resolve()
    if out.exists():raise FileExistsError('R10_OUT_DIR_EXISTS_NO_OVERWRITE')
    if not model.strip():raise ValueError('MODEL_ID_REQUIRED')
    if not 1<=reps<=3:raise ValueError('REPEATS_1_TO_3')
    if not 0<=temp<=1:raise ValueError('TEMPERATURE_OUT_OF_RANGE')
    if not 2000<=tokens<=12000:raise ValueError('TOKENS_OUT_OF_RANGE')
    train=discover_pattern();v=train['selected_pattern']
    calls=[]
    for rep in range(1,reps+1):
        for arm in ('CONTROL','PHENOMENON_TREATMENT'):
            prompt=QUESTION if arm=='CONTROL' else QUESTION+'\n\n'+treatment_card(train)
            calls.append({'call_id':secrets.token_hex(8),'blind_id':secrets.token_hex(8),
                          'arm':arm,'replicate':rep,'user_prompt':prompt,
                          'prompt_sha256':sha_bytes(prompt.encode())})
    random.SystemRandom().shuffle(calls)
    out.mkdir(parents=True)
    (out/'responses').mkdir()
    dump_new(out/'M15R10_TRAIN_PHYSICS.json',train)
    commit=sha_bytes(json.dumps({'indices':HOLDOUT_RI,'x':[v['x_hi'],v['x_lo']]},sort_keys=True).encode())
    plan={'schema_version':SCHEMA,'experiment_type':'PHENOMENON_DRIVEN_RESPONSIVENESS_PILOT',
          'model':model,'temperature':temp,'max_output_tokens':tokens,'replicates':reps,
          'calls':calls,'system_prompt':SYSTEM,'training_sha256':sha(out/'M15R10_TRAIN_PHYSICS.json'),
          'holdout_commitment_sha256':commit,
          'heldout_evaluated_before_response':False,'au_ag_science_gate':'BLOCKED',
          'scientific_novelty_certified':False,'physical_domain':'nonabsorbing homogeneous dielectric sphere in vacuum',
          'limitations':['Known Mie effect, not new science or SERS', 'Single contrastive training example',
                         'Context length and vocabulary may partially reveal treatment',
                         'Two matched repeats are underpowered',
                         'Held-out is a model parameter perturbation, not empirical validation',
                         'Blind rating is subjective, not direct test of discovery']}
    dump_new(out/'M15R10_PLAN_PRIVATE.json',plan)
    return {'status':'R10_PREPARED_TRAIN_ONLY','selected_peak_x':v['x_hi'],'selected_trough_x':v['x_lo'],
            'relative_drop_pct':v['fractional_drop']*100,'planned_paid_calls':len(calls),
            'heldout_values_evaluated':False,'new_solver_runs':0,'science_gate':'BLOCKED'}


def parsed_candidate(content)->tuple[dict|None,str]:
    if isinstance(content,list):content=''.join(z.get('text','') if isinstance(z,dict) else '' for z in content)
    if not isinstance(content,str) or not content.strip():return None,'EMPTY'
    s=content.strip()
    if s.startswith('```'):
        import re
        s=re.sub(r'^```(?:json)?\s*','',s,flags=re.I)
        s=re.sub(r'\s*```$','',s)
    try:obj=json.loads(s)
    except json.JSONDecodeError:return None,'NON_JSON'
    if not isinstance(obj,dict) or not all(k in obj and isinstance(obj[k],str) and obj[k].strip() for k in RESPONSE_FIELDS):
        return None,'MISSING_REQUIRED_FIELDS'
    return {key:obj[key] for key in RESPONSE_FIELDS},'VALID_UNREVIEWED'


def run_paid(out:Path,max_calls:int,allow:bool,authorize:bool)->dict:
    if not (allow and authorize):raise ValueError('EXPLICIT_PAID_AUTHORIZATION_REQUIRED')
    if not 1<=max_calls<=6:raise ValueError('MAX_NEW_CALLS_1_TO_6')
    out=out.resolve();plan=ensure_plan(out)
    if (out/'M15R10_BLIND_PACKET.json').exists():raise ValueError('RUN_CLOSED_AFTER_BLIND_PACKET')
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('OPENROUTER_API_KEY_MISSING')
    called=0
    for c in plan['calls']:
        path=out/'responses'/(c['call_id']+'.json')
        if path.exists():
            existing=json.loads(path.read_text())
            if existing.get('prompt_sha256')!=c['prompt_sha256'] or existing.get('blind_id')!=c['blind_id']:
                raise ValueError('SAVED_RECEIPT_MISMATCH')
            continue
        if called>=max_calls:break
        raw=r9.invoke_openrouter(c['user_prompt'],plan['system_prompt'],plan['model'],plan['temperature'],plan['max_output_tokens'])
        candidate,status=parsed_candidate(raw['choices'][0]['message'].get('content'))
        dump_new(path,{'schema_version':SCHEMA,'call_id':c['call_id'],'blind_id':c['blind_id'],
                      'prompt_sha256':c['prompt_sha256'],'candidate':candidate,'parse_status':status,
                      'raw_provider_response':raw,'scientific_novelty_certified':False})
        called+=1
    remaining=sum(not (out/'responses'/(c['call_id']+'.json')).exists() for c in plan['calls'])
    return {'status':'R10_RESPONSES_COMPLETE' if not remaining else 'R10_RESPONSES_PARTIAL',
            'new_paid_calls':called,'remaining':remaining,'heldout_values_evaluated':False}


def blind(out:Path)->dict:
    out=out.resolve();plan=ensure_plan(out)
    if (out/'M15R10_BLIND_PACKET.json').exists():raise FileExistsError('R10_BLIND_PACKET_ALREADY_EXISTS')
    specs=[];hashes={}
    for c in plan['calls']:
        p=out/'responses'/(c['call_id']+'.json')
        if not p.exists():raise ValueError('R10_MISSING_PAID_RESPONSE')
        obj=json.loads(p.read_text());hashes[p.name]=sha(p)
        if obj.get('prompt_sha256')!=c['prompt_sha256'] or obj.get('blind_id')!=c['blind_id']:
            raise ValueError('R10_RESPONSE_RECEIPT_CHANGED')
        specs.append({'blind_id':c['blind_id'],'candidate':obj['candidate'],
                      'parse_status':obj['parse_status']})
    # Shuffle to prevent call-order-based arm inference.
    random.SystemRandom().shuffle(specs)
    pkt={'schema_version':SCHEMA,'items':specs,'problem':QUESTION,
         'rubric':['conceptual shift relative to generic original task', 'rival-model discrimination',
                   'independent withheld test', 'grounded numerical inference',
                   'operational falsifier', 'recognizes known prior art and domain limits'],
         'reviewer_notice':'Do not seek or infer assignment. Technical cues can deblind. Model-generated thresholds are not predictions unless derived.',
         'scientific_novelty_certified':False}
    dump_new(out/'M15R10_BLIND_PACKET.json',pkt)
    dump_new(out/'M15R10_PRIVATE_ASSIGNMENT.json',{'schema_version':SCHEMA,
        'assignment':{c['blind_id']:{'arm':c['arm'],'replicate':c['replicate']} for c in plan['calls']},
        'receipt_sha256':hashes,'blind_packet_sha256':sha(out/'M15R10_BLIND_PACKET.json')})
    with (out/'M15R10_BLIND_RUBRIC.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['blind_id',*SCORES,'prior_art_risk_low_med_high','notes'])
        for v in specs:w.writerow([v['blind_id']]+['']*(len(SCORES)+2))
    return {'status':'R10_BLIND_REVIEW_READY','candidate_count':len(specs),
            'blind_packet':str(out/'M15R10_BLIND_PACKET.json'),
            'rubric':str(out/'M15R10_BLIND_RUBRIC.csv'),
            'heldout_values_evaluated':False}


def _review(scored:Path,assign:dict)->dict:
    scores={};expected=set(assign)
    with scored.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(['blind_id',*SCORES,'prior_art_risk_low_med_high']).issubset(reader.fieldnames or []):
            raise ValueError('REVIEW_MISSING_COLUMNS')
        for item in reader:
            bid=item['blind_id']
            if bid not in expected or bid in scores:raise ValueError('REVIEW_ID_INVALID_OR_REPEATED')
            vals={}
            for key in SCORES:
                try:val=int(item[key])
                except (ValueError,TypeError):raise ValueError('INVALID_OR_MISSING_SCORE_'+key)
                if val not in (0,1,2):raise ValueError('SCORE_OUT_OF_RANGE_'+key)
                vals[key]=val
            risk=item['prior_art_risk_low_med_high'].strip().lower()
            if risk not in ('low','med','medium','high'):raise ValueError('RISK_RATING_REQUIRED')
            scores[bid]={'dimensions':vals,'total':sum(vals.values()),'prior_art_risk':risk,
                         'notes':item.get('notes','')}
    if set(scores)!=expected:raise ValueError('REVIEW_INCOMPLETE')
    return scores


def reveal(out:Path,scored:Path)->dict:
    out=out.resolve();scored=scored.resolve();plan=ensure_plan(out)
    if (out/'M15R10_POST_UNBLIND.json').exists():raise FileExistsError('ALREADY_UNBLINDED')
    private=json.loads((out/'M15R10_PRIVATE_ASSIGNMENT.json').read_text());pkt=out/'M15R10_BLIND_PACKET.json'
    if sha(pkt)!=private['blind_packet_sha256']:raise ValueError('BLIND_PACKET_CHANGED')
    for fn,digest in private['receipt_sha256'].items():
        if sha(out/'responses'/fn)!=digest:raise ValueError('SOURCE_RECEIPT_CHANGED')
    assignment=private['assignment'];scores=_review(scored,assignment)
    pairs=[]
    for rep in range(1,plan['replicates']+1):
        rows={data['arm']:(bid,scores[bid]) for bid,data in assignment.items() if data['replicate']==rep}
        if set(rows)!= {'CONTROL','PHENOMENON_TREATMENT'}:raise ValueError('UNPAIRED_REVIEW')
        c=rows['CONTROL'][1]['total'];t=rows['PHENOMENON_TREATMENT'][1]['total']
        pairs.append({'replicate':rep,'control_blind_id':rows['CONTROL'][0],
                      'treatment_blind_id':rows['PHENOMENON_TREATMENT'][0],
                      'control_total':c,'treatment_total':t,'delta':t-c})
    # Genuine computation of the previously committed *unseen* refractive-index cases.
    physical=json.loads((out/'M15R10_TRAIN_PHYSICS.json').read_text())
    v=physical['selected_pattern'];p=v['x_hi'];q=v['x_lo']
    held=[]
    for m in HOLDOUT_RI:
        q_hi=mie_qsca(p,m);q_lo=mie_qsca(q,m)
        held.append({'index':m,'x_hi':p,'x_lo':q,'q_hi':q_hi,'q_lo':q_lo,
                     'direction_preserved':bool(q_hi>q_lo),
                     'fractional_drop':(q_hi-q_lo)/q_hi})
    result={'schema_version':SCHEMA,'status':'R10_POST_UNBLIND_WITH_HELDOUT_MIE',
            'scored_review_sha256':sha(scored),'blind_packet_sha256':sha(pkt),
            'training_sha256':plan['training_sha256'],
            'holdout_commitment_sha256':plan['holdout_commitment_sha256'],
            'review_scores':scores,'paired':pairs,'mean_delta':sum(x['delta'] for x in pairs)/len(pairs),
            'heldout':held,'heldout_direction_pass':all(v['direction_preserved'] for v in held),
            'au_ag_science_gate':'BLOCKED','scientific_novelty_certified':False,
            'claim':'TESTS_RESPONSIVENESS_TO_A_KNOWN_ANALYTICAL_MIE_PATTERN_NOT_SCIENTIFIC_NOVELTY',
            'limitations':plan['limitations']}
    dump_new(out/'M15R10_POST_UNBLIND.json',result)
    lines=['# M15-R10 — Phenomenon-driven A/B pilot','','| Pair | Control /12 | Treatment /12 | Difference |',
           '|---|---:|---:|---:|']
    lines += [f'| {v["replicate"]} | {v["control_total"]} | {v["treatment_total"]} | {v["delta"]:+} |' for v in pairs]
    lines += ['',f'- Mean reviewer difference: {result["mean_delta"]:+.3f}/12',
              '- PRE-REGISTERED withheld changes in refractive index (model-matched analytic calculation):',
              *(f'  - n={v["index"]:.2f}: Q_hi={v["q_hi"]:.6g}, Q_lo={v["q_lo"]:.6g}, reversed={not v["direction_preserved"]}' for v in held),
              '- All heldout directions preserved: '+str(result['heldout_direction_pass']),
              '- **Known Mie scattering phenomenon; no new-science certification.**',
              '- **Au/Ag nanorod and SERS science gates remain BLOCKED.**',
              '- Paired model behavior does not prove scientific novelty, nor physical transfer to a metal nanorod.']
    (out/'M15R10_POST_UNBLIND.md').write_text('\n'.join(lines)+'\n')
    return {'status':result['status'],'mean_delta':result['mean_delta'],
            'heldout_direction_pass':result['heldout_direction_pass'],
            'novelty_certified':False}


def check(out:Path)->dict:
    plan=ensure_plan(out.resolve())
    completed=0
    for c in plan['calls']:
        f=out/'responses'/(c['call_id']+'.json')
        if f.is_file():
            data=json.loads(f.read_text())
            if data.get('prompt_sha256')!=c['prompt_sha256'] or data.get('blind_id')!=c['blind_id']:
                raise ValueError('SAVED_RECEIPT_CHANGED')
            completed+=1
    if (out/'M15R10_PRIVATE_ASSIGNMENT.json').exists():
        private=json.loads((out/'M15R10_PRIVATE_ASSIGNMENT.json').read_text())
        if sha(out/'M15R10_BLIND_PACKET.json')!=private['blind_packet_sha256']:raise ValueError('BLIND_PACKET_MODIFIED')
        for fn,digest in private['receipt_sha256'].items():
            if sha(out/'responses'/fn)!=digest:raise ValueError('RESPONSE_CHANGED_AFTER_BLIND')
    if (out/'M15R10_POST_UNBLIND.json').exists():
        result=json.loads((out/'M15R10_POST_UNBLIND.json').read_text())
        if result['au_ag_science_gate']!='BLOCKED' or result['scientific_novelty_certified'] is not False:
            raise ValueError('ILLEGAL_SCIENCE_AUTHORITY')
        for v in result['heldout']:
            if not math.isclose(v['q_hi'],mie_qsca(v['x_hi'],v['index']),rel_tol=1e-12):
                raise ValueError('HELDOUT_CHANGED')
            if not math.isclose(v['q_lo'],mie_qsca(v['x_lo'],v['index']),rel_tol=1e-12):
                raise ValueError('HELDOUT_CHANGED')
    return {'status':'R10_CHECK_OK','planned_model_calls':len(plan['calls']),
            'saved_receipts':completed,'blind_ready':(out/'M15R10_BLIND_PACKET.json').exists(),
            'revealed':(out/'M15R10_POST_UNBLIND.json').exists(),
            'au_ag_science_gate':'BLOCKED','scientific_novelty_certified':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__);s=ap.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare');a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--model',required=True);a.add_argument('--replicates',type=int,default=2)
    a.add_argument('--temperature',type=float,default=0.3);a.add_argument('--max-output-tokens',type=int,default=6000)
    a=s.add_parser('run-paid');a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--max-new-calls',type=int,default=4);a.add_argument('--allow-paid',action='store_true')
    a.add_argument('--i-authorize-model-calls',action='store_true')
    a=s.add_parser('blind');a.add_argument('--out-dir',type=Path,required=True)
    a=s.add_parser('reveal');a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--scored-csv',type=Path,required=True)
    a=s.add_parser('check');a.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.cmd=='prepare':resp=prepare(args.out_dir,args.model,args.replicates,args.temperature,args.max_output_tokens)
        elif args.cmd=='run-paid':resp=run_paid(args.out_dir,args.max_new_calls,args.allow_paid,args.i_authorize_model_calls)
        elif args.cmd=='blind':resp=blind(args.out_dir)
        elif args.cmd=='reveal':resp=reveal(args.out_dir,args.scored_csv)
        else:resp=check(args.out_dir)
        print(json.dumps(resp,indent=2,ensure_ascii=False))
    except (ValueError,OSError,TypeError,KeyError,IndexError,json.JSONDecodeError) as e:
        ap.exit(2,'M15R10_ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
