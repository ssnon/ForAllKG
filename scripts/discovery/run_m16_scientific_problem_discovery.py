"""M16: three-arm SERS scientific problem discovery experiment.

Gates: R10 is known Lorenz-Mie physics, NOT Au/Ag or SERS evidence.
Shared prior-art card for all arms; CONTROL, KNOWN_PHYSICS, NUMERIC_PHENOMENON.
Paid calls are separately authorized. Reviewer sees only candidate text; unblind only
with complete scored CSV. No novel finding / physical validity can be certified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets

from scripts.discovery import run_m15_r9_feedback_pilot as r9

SCHEMA='m16-scientific-problem-discovery-three-arm-v1'
ARMS=('BASELINE','KNOWN_PHYSICS','NUMERIC_PHENOMENON')
REQUIRED=('decision','research_problem','prior_art_boundary','missing_evidence',
          'null_model','alternative_model','discriminating_observable',
          'conditional_prediction','orthogonal_measurements','minimal_decisive_test',
          'heldout_protocol','falsification_and_abstention','value_of_information',
          'novelty_search_queries','limitations')
SCORE_COLS=('scientific_problem_specificity_0to2',
            'beyond_known_pattern_0to2','rival_model_separation_0to2',
            'orthogonal_measurability_0to2','heldout_falsifiability_0to2',
            'epistemic_caution_0to2')
CSV_TAIL=('prior_art_risk_low_med_high','misuses_mie_as_sers_evidence_yes_no','notes')
REFS=(
 {'doi':'10.1021/acsnano.4c01474','title':'Enhancement Factors: A Central Concept during 50 Years of Surface-Enhanced Raman Spectroscopy',
  'year':2024,'scope':'Review of SERS EF, spatial hot spots, orientation/tensorial and nonradiative effects'},
 {'doi':'10.1021/acsnano.4c02670','title':'Surface-Enhanced Raman Spectroscopy: Current Understanding, Challenges, and Opportunities',
  'year':2024,'scope':'Review of SERS EM and chemical enhancement, spectral-structure interpretation and reliability'}
)
COMMON_PROBLEM=(
 'FORALLKG RESEARCH GOAL — SCIENTIFIC PROBLEM DISCOVERY, NOT KNOWN-EFFECT REFINEMENT.\n'
 'In polarization-resolved SERS on Au-core/Ag-shell nanorods, determine whether and under which '
 'controlled circumstances relative Raman-band intensities can separately identify molecular '
 'orientation, tensor projections, excitation/Stokes local electromagnetic fields, adsorption '
 'populations and chemical enhancement. Existing work has discussed these mechanisms; merely '
 'proposing polarization-dependent enhancement or more meticulous controls is not a novel problem. '
 'Find at most ONE defensible research problem at a still-unresolved information boundary. '
 'Include specific rival explanations, a measurement in which they make different predictions, '
 'an independently sourced variable, a held-out intervention and a falsifier. If no genuinely '
 'distinct problem is supported, choose ABSTAIN and say what information would be required. '
 'No unverified molecular constants, invented empirical observations, or arbitrarily precise '
 'effect-size thresholds. Known-scattering simulations alone cannot predict Raman ratios.\n'
 'COMMON LITERATURE BOUNDARY FOR ALL ARMS (two review abstracts, NOT comprehensive prior-art clearance):\n'
 '1) Le Ru & Auguie (2024), DOI 10.1021/acsnano.4c01474: SERS enhancement-factor '
 'definition, hotspot localization, molecular capture, orientation/tensor and nonradiative issues.\n'
 '2) Ma et al. (2024), DOI 10.1021/acsnano.4c02670: electromagnetic and chemical mechanisms, '
 'spectrum-structure correlations, sensitivity/selectivity/reliability.\n'
 'The above sources cover categories, not a proven missing problem; verify claim-specific literature later.\n'
)
SYSTEM=(
 'You are a scientific research problem-discovery scientist. Output ONLY one JSON object with '
 'string fields: '+', '.join(REQUIRED)+'. '
 'decision must be PROPOSE or ABSTAIN. A PROPOSE is a research-program proposal, not a scientific '
 'finding. Prefer a sharp observable/condition that separates mechanistic rivals to ornamental '
 'complexity. Acknowledge established science and never manufacture evidence. A conservative '
 'ABSTAIN is acceptable. Every other field must be nonempty, even for abstention. '
 'Treat all Mie sphere calculations as irrelevant to Au/Ag molecular Raman ratios unless a '
 'separate physical correspondence is experimentally established.\n'
)
BASE_CARD=(
 'ADDITIONAL CONTEXT — STANDARD SCIENTIFIC-PLANNING NOTE: Separate observational variables '
 'from latent mechanisms. Investigate whether an experiment changes only the readout while '
 'leaving the proposed mechanism unidentifiable. Describe missing calibrations, held-out '
 'conditions, independent measurement, and potential confounders. It is acceptable to '
 'conclude that the current literature background and uncertainty prevent a defensible '
 'new scientific question. No external numerical physics observation is provided in this card.'
)
KNOWN_CARD=(
 'ADDITIONAL CONTEXT — ESTABLISHED PHYSICS, NO NEW NUMBERS: In exact Lorenz–Mie scattering '
 'by an ideal lossless homogeneous dielectric sphere, partial-wave interference can produce '
 'nonmonotonic scattering efficiency as the size parameter changes. Changes in index may '
 'shift oscillatory features. This is textbook Maxwell physics and is neither a metal '
 'nanorod measurement nor a prediction about molecular Raman ratios. It serves only as '
 'an analogy for seeking condition-specific discriminating experiments, not as scientific '
 'evidence about the requested Au/Ag SERS system.'
)


def hbytes(v:bytes)->str:return hashlib.sha256(v).hexdigest()
def sha(p:Path)->str:return hbytes(p.read_bytes())
def canonical(v)->str:return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def new_json(p:Path,v)->None:
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')


def r10_source(post_path:Path,train_path:Path)->dict:
    post=json.loads(post_path.read_text(encoding='utf-8'))
    train=json.loads(train_path.read_text(encoding='utf-8'))
    if post.get('schema_version')!='m15-r10-phenomenon-driven-ab-pilot-v1' or post.get('status')!='R10_POST_UNBLIND_WITH_HELDOUT_MIE':
        raise ValueError('R10_POST_STATUS_INVALID')
    if post.get('scientific_novelty_certified') is not False or post.get('au_ag_science_gate')!='BLOCKED':
        raise ValueError('R10_AUTHORITY_INVALID')
    if post.get('training_sha256')!=sha(train_path):raise ValueError('R10_TRAINING_SHA_MISMATCH')
    sel=train.get('selected_pattern',{})
    if train.get('material')!='nondispersive dielectric, nonmetal' or train.get('science_novelty_certified') is not False:
        raise ValueError('R10_NOT_KNOWN_DIELECTRIC_SPHERE')
    if not (train.get('n_relative')==1.5 and sel.get('x_hi')==6.6 and sel.get('x_lo')==7.15):
        raise ValueError('R10_DIFFERENT_PHYSICAL_CASE')
    if not (2.4<sel['q_hi']<2.5 and 1.7<sel['q_lo']<1.8 and 0.28<sel['fractional_drop']<0.30):
        raise ValueError('R10_UNEXPECTED_TRAINING_RESULTS')
    heldout=post.get('heldout',[])
    if len(heldout)!=2 or sorted(x['index'] for x in heldout)!=[1.48,1.52] or post.get('heldout_direction_pass') is not True:
        raise ValueError('R10_HOLDOUT_SCOPE_UNEXPECTED')
    if any(x.get('direction_preserved') is not True for x in heldout):
        raise ValueError('R10_HOLDOUT_NOT_PRESERVED')
    return {'r10_post_sha256':sha(post_path),'r10_training_sha256':sha(train_path),
            'r10_post_path':str(post_path.resolve()),'r10_training_path':str(train_path.resolve()),
            'r10_status':'KNOWN_MIE_PHYSICS_ONLY','pattern':sel,
            'r10_holdout_index_conditions_verified':[1.48,1.52],
            'scientific_novelty_certified':False,'au_ag_science_gate':'BLOCKED'}


def numeric_card(source:dict)->str:
    p=source['pattern']
    return (
        'ADDITIONAL CONTEXT — ACTUALLY COMPUTED BUT KNOWN NON-SERS PHENOMENON: Exact '
        f'lossless dielectric-sphere Lorenz–Mie calculation in vacuum at n=1.50 gives Qsca={p["q_hi"]:.7f} '
        f'at x={p["x_hi"]:.2f} and Qsca={p["q_lo"]:.7f} at x={p["x_lo"]:.2f}, '
        f'a {100*p["fractional_drop"]:.3f}% decrease over increasing size parameter. Nearby index '
        'perturbations preserve this fixed-point decrease. This is textbook sphere interference, '
        'not a new effect; it is NOT measured or validated Au/Ag nanorod optics, Raman spectra, '
        'chemical enhancement, or molecular orientation evidence. Use only as a methodological '
        'inspiration about seeking differentiating condition-dependent phenomena. Do not assert '
        'a sphere-to-SERS transfer.'
    )


def arm_prompt(arm:str,source:dict)->str:
    cards={'BASELINE':BASE_CARD,'KNOWN_PHYSICS':KNOWN_CARD,
           'NUMERIC_PHENOMENON':numeric_card(source)}
    if arm not in cards:raise ValueError('UNKNOWN_ARM')
    return COMMON_PROBLEM+'\n'+cards[arm]


def load_plan(out:Path)->dict:
    out=out.resolve();plan=json.loads((out/'M16_PLAN_PRIVATE.json').read_text(encoding='utf-8'))
    if plan.get('schema_version')!=SCHEMA or plan.get('au_ag_science_gate')!='BLOCKED' or plan.get('scientific_novelty_certified') is not False:
        raise ValueError('M16_PLAN_AUTHORITY_INVALID')
    source=r10_source(Path(plan['source']['r10_post_path']),Path(plan['source']['r10_training_path']))
    if source!=plan['source']:raise ValueError('R10_SOURCE_OR_COMMITMENT_CHANGED')
    if plan.get('system_prompt')!=SYSTEM or plan.get('common_problem')!=COMMON_PROBLEM or plan.get('literature_anchors')!=list(REFS):
        raise ValueError('M16_COMMON_SOURCE_CHANGED')
    if len(plan['calls'])!=len(ARMS)*plan['replicates']:raise ValueError('M16_ARMS_NOT_BALANCED')
    if not 1<=plan['replicates']<=4:raise ValueError('M16_REPLICATES_OUT_OF_RANGE')
    if len({c['call_id'] for c in plan['calls']})!=len(plan['calls']) or len({c['blind_id'] for c in plan['calls']})!=len(plan['calls']):
        raise ValueError('M16_CALL_ID_REPEATED')
    for rep in range(1,plan['replicates']+1):
        for arm in ARMS:
            if sum(c['replicate']==rep and c['arm']==arm for c in plan['calls'])!=1:
                raise ValueError('M16_ARM_REP_MISSING')
    for c in plan['calls']:
        expected=arm_prompt(c['arm'],source)
        if expected!=c['prompt'] or c['prompt_sha256']!=hbytes(expected.encode()):
            raise ValueError('M16_PROMPT_CHANGED')
    return plan


def prepare(post:Path,train:Path,out:Path,model:str,reps:int,temp:float,tokens:int)->dict:
    post,train,out=(p.resolve() for p in (post,train,out))
    if out.exists():raise FileExistsError('M16_OUT_EXISTS_NO_OVERWRITE')
    if not model or not model.strip():raise ValueError('MODEL_ID_MISSING')
    if not 1<=reps<=4:raise ValueError('REPLICATES_1_TO_4')
    if not 0<=temp<=1:raise ValueError('TEMPERATURE_0_TO_1')
    if not 3000<=tokens<=14000:raise ValueError('MAX_TOKENS_3000_TO_14000')
    source=r10_source(post,train)
    calls=[]
    for rep in range(1,reps+1):
        for arm in ARMS:
            text=arm_prompt(arm,source)
            calls.append({'call_id':secrets.token_hex(8),'blind_id':secrets.token_hex(8),
                          'arm':arm,'replicate':rep,'prompt':text,
                          'prompt_sha256':hbytes(text.encode())})
    random.SystemRandom().shuffle(calls)
    card_lengths={a:len(arm_prompt(a,source)) for a in ARMS}
    plan={'schema_version':SCHEMA,'status':'FROZEN_AWAITING_MODEL',
          'source':source,'common_problem':COMMON_PROBLEM,'literature_anchors':list(REFS),
          'calls':calls,'system_prompt':SYSTEM,'replicates':reps,'model':model,
          'temperature':temp,'max_output_tokens':tokens,'prompt_character_counts':card_lengths,
          'experimental_comparison':'three arms compare no numerical feedback, known science text and actual numerical but non-SERS phenomenon',
          'literature_search_performed_by_pipeline':False,'independent_science_review_required':True,
          'evidence_authority':'Only known dielectric Mie physics; shared SERS review abstracts are NOT exhaustive prior-art review',
          'au_ag_science_gate':'BLOCKED','scientific_novelty_certified':False,
          'limitations':['Cross-domain Mie to SERS: explicitly no claim of physical transfer',
                         'Prompt content differs by arm; information quantity not fully equalized',
                         'Reviewer may infer arm from text','2-4 repeats/arm are not statistical confirmation',
                         'No SERS experiment, independent manuscript-level novelty search, or new Au/Ag solver run',
                         'Human blind science rating measures proposal merit, not validated scientific discovery']}
    out.mkdir(parents=True);(out/'responses').mkdir()
    new_json(out/'M16_PLAN_PRIVATE.json',plan)
    return {'status':'M16_THREE_ARM_PREPARED','replicates':reps,'planned_paid_calls':len(calls),
            'prompt_character_counts':card_lengths,'novelty_certified':False,'au_ag_science_gate':'BLOCKED','new_solver_runs':0}


def parse_candidate(content)->tuple[dict|None,str]:
    if isinstance(content,list):content=''.join(v.get('text','') if isinstance(v,dict) else '' for v in content)
    if not isinstance(content,str):return None,'CONTENT_NOT_STRING'
    text=content.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text,flags=re.I).strip()
        text=re.sub(r'\s*```$','',text).strip()
    try:data=json.loads(text)
    except json.JSONDecodeError:return None,'NON_JSON'
    if not isinstance(data,dict) or not all(isinstance(data.get(k),str) and data[k].strip() for k in REQUIRED):
        return None,'REQUIRED_FIELDS_MISSING'
    if data['decision'].strip().upper() not in ('PROPOSE','ABSTAIN'):
        return None,'INVALID_DECISION'
    return {k:data[k] for k in REQUIRED},'VALID_UNREVIEWED'


def run_paid(out:Path,max_calls:int,allow:bool,authorize:bool)->dict:
    if not (allow and authorize):raise ValueError('EXPLICIT_PAID_AUTHORIZATION_REQUIRED')
    if not 1<=max_calls<=12:raise ValueError('MAX_NEW_CALLS_1_TO_12')
    out=out.resolve();plan=load_plan(out)
    if (out/'M16_BLIND_PACKET.json').exists():raise ValueError('NO_MODEL_CALLS_AFTER_BLIND')
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('OPENROUTER_API_KEY_MISSING')
    calls=0
    for c in plan['calls']:
        path=out/'responses'/(c['call_id']+'.json')
        if path.exists():
            v=json.loads(path.read_text())
            if v.get('blind_id')!=c['blind_id'] or v.get('prompt_sha256')!=c['prompt_sha256']:
                raise ValueError('RECEIPT_IDENTITY_CHANGED')
            continue
        if calls>=max_calls:break
        raw=r9.invoke_openrouter(c['prompt'],SYSTEM,plan['model'],plan['temperature'],plan['max_output_tokens'])
        if not raw.get('choices'):raise ValueError('NO_PROVIDER_CHOICES')
        candidate,status=parse_candidate(raw['choices'][0]['message'].get('content'))
        receipt={'schema_version':SCHEMA,'call_id':c['call_id'],'blind_id':c['blind_id'],
                 'prompt_sha256':c['prompt_sha256'],'parse_status':status,
                 'candidate':candidate,'raw_provider_response':raw,
                 'scientific_novelty_certified':False,'source_science_gate':'BLOCKED'}
        new_json(path,receipt)
        calls+=1
    remaining=sum(not (out/'responses'/(c['call_id']+'.json')).exists() for c in plan['calls'])
    invalid=sum(json.loads((out/'responses'/(c['call_id']+'.json')).read_text())['parse_status']!='VALID_UNREVIEWED'
                for c in plan['calls'] if (out/'responses'/(c['call_id']+'.json')).exists())
    return {'status':'M16_RESPONSE_RECEIPTS_COMPLETE' if remaining==0 else 'M16_RESPONSE_RECEIPTS_PARTIAL',
            'new_paid_calls':calls,'remaining_calls':remaining,'invalid_structured_responses':invalid,
            'scientific_novelty_certified':False}


def blind(out:Path)->dict:
    out=out.resolve();plan=load_plan(out)
    if (out/'M16_BLIND_PACKET.json').exists():raise FileExistsError('BLIND_PACKET_EXISTS_NO_OVERWRITE')
    items=[];receipts={}
    for c in plan['calls']:
        p=out/'responses'/(c['call_id']+'.json')
        if not p.is_file():raise ValueError('SOME_RESPONSES_NOT_SAVED')
        row=json.loads(p.read_text())
        if row.get('blind_id')!=c['blind_id'] or row.get('prompt_sha256')!=c['prompt_sha256']:
            raise ValueError('RECEIPT_NOT_MATCHING_PLAN')
        if row['parse_status']!='VALID_UNREVIEWED' or not row['candidate']:
            raise ValueError('MODEL_OUTPUT_INVALID_NO_SILENT_RETRY')
        receipts[p.name]=sha(p)
        items.append({'blind_id':c['blind_id'],'candidate':row['candidate'],
                      'parse_status':row['parse_status']})
    random.SystemRandom().shuffle(items)
    packet={'schema_version':SCHEMA,'problem':COMMON_PROBLEM,
            'prior_art_scope':list(REFS),'items':items,
            'review_dimensions':list(SCORE_COLS),'reviewer_notice':
            'Independent reviewer: do not seek or infer arm. A proposal is not a proven novel discovery. '
            'Flag Mie-sphere-to-SERS misinterpretation and made-up quantitative thresholds. '
            'Shared references are reviews, not comprehensive novelty certification.',
            'scientific_novelty_certified':False}
    new_json(out/'M16_BLIND_PACKET.json',packet)
    private={'schema_version':SCHEMA,
             'assignment':{c['blind_id']:{'arm':c['arm'],'replicate':c['replicate']} for c in plan['calls']},
             'receipt_sha256':receipts,'blind_packet_sha256':sha(out/'M16_BLIND_PACKET.json'),
             'plan_sha256':sha(out/'M16_PLAN_PRIVATE.json')}
    new_json(out/'M16_PRIVATE_ASSIGNMENT.json',private)
    with (out/'M16_BLIND_RUBRIC.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['blind_id',*SCORE_COLS,*CSV_TAIL])
        for item in items:w.writerow([item['blind_id'],*(['']*(len(SCORE_COLS)+len(CSV_TAIL)))])
    return {'status':'M16_BLIND_PACKET_READY','candidate_count':len(items),
            'blind_packet':'M16_BLIND_PACKET.json','rubric':'M16_BLIND_RUBRIC.csv',
            'assignment_private':True}


def score_rows(path:Path,expected:set[str])->dict:
    scores={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        expected_columns={'blind_id',*SCORE_COLS,*CSV_TAIL}
        if not expected_columns.issubset(reader.fieldnames or []):raise ValueError('REVIEW_COLUMNS_MISSING')
        for row in reader:
            bid=row['blind_id']
            if bid not in expected or bid in scores:raise ValueError('DUPLICATE_OR_UNKNOWN_BLIND_ID')
            nums={}
            for col in SCORE_COLS:
                if row[col] not in ('0','1','2'):raise ValueError('INCOMPLETE_0_TO_2_REVIEW_'+bid+'_'+col)
                nums[col]=int(row[col])
            risk=row['prior_art_risk_low_med_high'].strip().lower()
            flag=row['misuses_mie_as_sers_evidence_yes_no'].strip().lower()
            if risk not in ('low','med','medium','high'):raise ValueError('PRIOR_ART_RISK_MISSING')
            if flag not in ('yes','no'):raise ValueError('INVALID_DOMAIN_MISUSE_FLAG')
            if len(row.get('notes','').strip())<15:raise ValueError('SUBSTANTIVE_REVIEW_NOTES_REQUIRED')
            scores[bid]={'scores':nums,'total':sum(nums.values()),'risk':risk,
                         'domain_misuse':flag=='yes','notes':row['notes']}
    if set(scores)!=expected:raise ValueError('REVIEW_INCOMPLETE')
    return scores


def unblind(out:Path,scored:Path)->dict:
    out=out.resolve();plan=load_plan(out)
    result=out/'M16_POST_UNBLIND.json'
    if result.exists():raise FileExistsError('M16_POST_UNBLIND_EXISTS')
    packet_path=out/'M16_BLIND_PACKET.json'
    if not packet_path.exists():raise ValueError('BLIND_PACKET_MISSING')
    private=json.loads((out/'M16_PRIVATE_ASSIGNMENT.json').read_text())
    if private['blind_packet_sha256']!=sha(packet_path) or private['plan_sha256']!=sha(out/'M16_PLAN_PRIVATE.json'):
        raise ValueError('UNBLIND_FREEZE_HASH_MISMATCH')
    for name,digest in private['receipt_sha256'].items():
        if sha(out/'responses'/name)!=digest:raise ValueError('RECEIPT_CHANGED_AFTER_BLIND')
    packet=json.loads(packet_path.read_text())
    assignment=private['assignment']
    if {i['blind_id'] for i in packet['items']}!=set(assignment):raise ValueError('ASSIGNMENT_INCONSISTENT')
    scores=score_rows(scored,set(assignment))
    paired=[]
    for rep in range(1,plan['replicates']+1):
        row={'replicate':rep}
        for arm in ARMS:
            bids=[bid for bid,info in assignment.items() if info['replicate']==rep and info['arm']==arm]
            if len(bids)!=1:raise ValueError('PAIR_MISSING_ARM')
            row[arm]={'blind_id':bids[0],**scores[bids[0]]}
        row['numeric_minus_baseline']=row['NUMERIC_PHENOMENON']['total']-row['BASELINE']['total']
        row['numeric_minus_known_physics']=row['NUMERIC_PHENOMENON']['total']-row['KNOWN_PHYSICS']['total']
        row['known_physics_minus_baseline']=row['KNOWN_PHYSICS']['total']-row['BASELINE']['total']
        paired.append(row)
    def avg(key):return sum(r[key] for r in paired)/len(paired)
    means={arm:sum(row[arm]['total'] for row in paired)/len(paired) for arm in ARMS}
    summary={'schema_version':SCHEMA,'status':'M16_BLIND_RATINGS_COMPARED_NO_NOVELTY_AUTHORITY',
             'source_plan_sha256':private['plan_sha256'],
             'blind_packet_sha256':private['blind_packet_sha256'],
             'review_sha256':sha(scored),'means_out_of_12':means,
             'mean_numeric_minus_baseline':avg('numeric_minus_baseline'),
             'mean_numeric_minus_known_physics':avg('numeric_minus_known_physics'),
             'mean_known_physics_minus_baseline':avg('known_physics_minus_baseline'),
             'paired':paired,
             'interpretation':'descriptive three-arm contrast, not powered causal confirmation; scores cannot certify science novelty',
             'au_ag_science_gate':'BLOCKED','scientific_novelty_certified':False,
             'independent_literature_novelty_search_performed':False,
             'actual_new_scientific_observation_obtained':False}
    lines=['# M16 — Post-blind independent review: three-arm scientific-problem-discovery pilot','',
           '| Repeat | Baseline /12 | Known-physics /12 | Numeric-phenomenon /12 | Numeric − Known |',
           '|---:|---:|---:|---:|---:|']
    for r in paired:lines.append(f'| {r["replicate"]} | {r["BASELINE"]["total"]} | {r["KNOWN_PHYSICS"]["total"]} | {r["NUMERIC_PHENOMENON"]["total"]} | {r["numeric_minus_known_physics"]:+} |')
    lines+=['',f'- Baseline mean: {means["BASELINE"]:.2f}/12',
            f'- Known-physics mean: {means["KNOWN_PHYSICS"]:.2f}/12',
            f'- Numeric-phenomenon mean: {means["NUMERIC_PHENOMENON"]:.2f}/12',
            f'- Numeric − known-physics mean: {summary["mean_numeric_minus_known_physics"]:+.2f}/12','',
            '**Scores are not discoveries.** New Au/Ag optical simulations, independent SERS measurements and claim-specific prior art searches have NOT occurred.',
            '**Au/Ag science gate: BLOCKED.**']
    with (out/'M16_POST_UNBLIND_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    summary['report_sha256']=sha(out/'M16_POST_UNBLIND_REPORT.md')
    new_json(result,summary)
    return {'status':summary['status'],'means_out_of_12':means,
            'numeric_minus_known_physics':summary['mean_numeric_minus_known_physics'],
            'au_ag_science_gate':'BLOCKED','novelty_certified':False}


def check(out:Path)->dict:
    out=out.resolve();plan=load_plan(out)
    saved=0
    for c in plan['calls']:
        p=out/'responses'/(c['call_id']+'.json')
        if p.exists():
            row=json.loads(p.read_text())
            if row.get('blind_id')!=c['blind_id'] or row.get('prompt_sha256')!=c['prompt_sha256']:
                raise ValueError('RECEIPT_CONTENT_MISMATCH')
            saved+=1
    pkt=out/'M16_BLIND_PACKET.json'
    if pkt.exists():
        a=json.loads((out/'M16_PRIVATE_ASSIGNMENT.json').read_text())
        if a['blind_packet_sha256']!=sha(pkt) or a['plan_sha256']!=sha(out/'M16_PLAN_PRIVATE.json'):
            raise ValueError('BLIND_ARTIFACT_HASH_MISMATCH')
        for n,digest in a['receipt_sha256'].items():
            if sha(out/'responses'/n)!=digest:raise ValueError('RECEIPT_CHANGED_AFTER_BLIND')
        if len(a['assignment'])!=len(plan['calls']):raise ValueError('INVALID_ASSIGNMENT_SIZE')
    if (out/'M16_POST_UNBLIND.json').exists():
        report=json.loads((out/'M16_POST_UNBLIND.json').read_text())
        if report['report_sha256']!=sha(out/'M16_POST_UNBLIND_REPORT.md'):
            raise ValueError('POST_REPORT_HASH_MISMATCH')
        if report['source_plan_sha256']!=sha(out/'M16_PLAN_PRIVATE.json') or report['blind_packet_sha256']!=sha(pkt):
            raise ValueError('POST_SOURCE_HASH_MISMATCH')
    return {'status':'M16_CHECK_OK','responses_saved':saved,'planned':len(plan['calls']),
            'blind_ready':pkt.exists(),'post_unblind':(out/'M16_POST_UNBLIND.json').exists(),
            'au_ag_science_gate':'BLOCKED','novelty_certified':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--r10-post',type=Path,required=True);p.add_argument('--r10-training',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--replicates',type=int,default=2);p.add_argument('--temperature',type=float,default=0.3)
    p.add_argument('--max-output-tokens',type=int,default=7000)
    p=sub.add_parser('run-paid');p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-new-calls',type=int,default=6)
    p.add_argument('--allow-paid',action='store_true');p.add_argument('--i-authorize-model-calls',action='store_true')
    p=sub.add_parser('blind');p.add_argument('--out-dir',type=Path,required=True)
    p=sub.add_parser('unblind');p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--scored-csv',type=Path,required=True)
    p=sub.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    try:
        if a.command=='prepare':result=prepare(a.r10_post,a.r10_training,a.out_dir,a.model,a.replicates,a.temperature,a.max_output_tokens)
        elif a.command=='run-paid':result=run_paid(a.out_dir,a.max_new_calls,a.allow_paid,a.i_authorize_model_calls)
        elif a.command=='blind':result=blind(a.out_dir)
        elif a.command=='unblind':result=unblind(a.out_dir,a.scored_csv)
        else:result=check(a.out_dir)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError,FileExistsError,KeyError,OSError,TypeError,json.JSONDecodeError) as e:
        ap.exit(2,'M16_ERROR: '+str(e)+'\n')

if __name__=='__main__':main()
