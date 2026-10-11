"""M15-R9P: paired causal *responsiveness* pilot (unqualified Au/Ag diagnostics).

Not a test of scientific novelty or physical truth. Control gets the identical
research question without R8B diagnostic; treatment gets R8B diagnostic labeled
UNVALIDATED. Execution requires two explicit flags for paid calls.
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
import urllib.request
import urllib.error

from scripts.discovery import run_m15_r8b_minimal_gate as r8b

SCHEMA='m15-r9p-diagnostic-feedback-causal-pilot-v1'
BASE_SYSTEM='''You are a scientific research planner. Treat physical observations as uncertain unless validated. Your task is NOT to assert discovery. Propose a single precise new research question, two competing explanations, a falsifiable quantitative contrast, independent measurements and controls, and a held-out check. Avoid claims that Maxwell scattering directly predicts SERS molecular band ratios. Output ONLY JSON with fields: research_question, null_model, alternative_model, proposed_quantitative_prediction, independent_measurements, decisive_next_test, falsification_criteria, uncertainty_limits, prior_art_search_terms. Each field must be nonempty. No markdown.'''
BASE_USER='''SCIENTIFIC RESEARCH PROBLEM (identical in both arms):
In Au-core/Ag-shell nanorods, when can polarization-dependent relative SERS Raman band intensities distinguish molecular orientation from electromagnetic selection and adsorption-state effects? The structure is approximately 160 nm long Au core and 200 nm long Ag shell with 20 nm radial shell thickness. Distinguish far-field scattering, local near-field at excitation and Stokes frequencies, Raman tensor, and adsorption chemistry. The central aim is discovering an independently testable scientific research direction, not reproducing a known optical effect.

Common task: select the MOST informative next independent measurement or simulation and turn it into a genuinely discriminating scientific question. Give a conservative prior-art risk. Avoid stating novelty or physical mechanism as known. Provide one bounded falsifiable conditional prediction and a held-out test. Do not merely restate that polarization changes plasmonic enhancement.'''
KEYS=('research_question','null_model','alternative_model','proposed_quantitative_prediction',
      'independent_measurements','decisive_next_test','falsification_criteria','uncertainty_limits','prior_art_search_terms')


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path:Path,obj:object)->None:
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')


def load(r9dir:Path)->dict:
    plan=json.loads((r9dir/'M15R9P_PLAN_PRIVATE.json').read_text())
    if (plan.get('schema_version')!=SCHEMA or plan.get('experiment_type')!='DIAGNOSTIC_RESPONSIVENESS_NOT_NOVELTY'
        or plan.get('quantitative_physics_authority') is not False):
        raise ValueError('R9P_PLAN_AUTHORITY_INVALID')
    r8b_dir=Path(plan['r8b_dir']);r8b.check(r8b_dir)
    if sha(r8b_dir/'M15R8B_EVIDENCE_GATE.json')!=plan['r8b_evidence_sha256']:
        raise ValueError('R8B_BASIS_CHANGED')
    for spec in plan['calls']:
        if spec['arm'] not in ('CONTROL','DIAGNOSTIC_TREATMENT'):
            raise ValueError('R9P_UNKNOWN_ARM')
        if spec['prompt_sha256']!=hashlib.sha256(spec['user_prompt'].encode()).hexdigest():
            raise ValueError('R9P_PROMPT_CHANGED')
    if len(plan['calls'])!=2*plan['replicates']:
        raise ValueError('R9P_IMBALANCED_CALLS')
    if sorted([c['arm'] for c in plan['calls']]).count('CONTROL')!=plan['replicates']:
        raise ValueError('R9P_IMBALANCED_ARMS')
    return plan


def diagnostic_card(evidence:dict)->str:
    if evidence['numerical_science_feedback_permitted'] is not False or evidence['science_gate']!='BLOCKED':
        raise ValueError('R8B_AUTHORITY_MISREPRESENTED')
    outer=evidence['surface_pft_rows']['OuterShell'];inner=evidence['surface_pft_rows']['InnerCore']
    return ('''ADDITIONAL EXTERNAL COMPUTATIONAL DIAGNOSTIC (TREATMENT ARM ONLY):
SCUFF-EM ran a single-mesh Au/Ag core-shell model at 633 nm with incident Ex polarization and a water-like exterior. The SCUFF geometry checker identified nested Au and Ag regions and closed interfaces. The surface PFT output (NOT independently validated or interpreted as physical particle observables) reported:
'''
            f'- Ag exterior-interface PFT: absorbed power {outer["absorbed_w"]:.8g} W; scattered-power column {outer["scattered_w"]:.8g} W.\n'
            f'- Au interior-interface PFT: absorbed-power column {inner["absorbed_w"]:.8g} W; scattered-power column {inner["scattered_w"]:.8g} W.\n'
            '''The negative inner-interface absorbed-power entry is unresolved and could reflect PFT interface conventions or numerical/model errors; it is NOT a demonstration of physical negative absorption. These rows CANNOT be naively summed into a total cross section. There is no Au/Ag mesh convergence, independent near-field data, Ey comparison, Raman measurement, or chemical-state evidence. Treat these numbers ONLY as an uncertain diagnostic revealing what to check next. Explicitly say what would resolve the uncertainty; DO NOT interpret magnitudes as trusted scattering physics or infer scientific novelty from them.
''')


def prepare(r8b_dir:Path,out_dir:Path,model:str,repeats:int,temp:float,max_tokens:int)->dict:
    if out_dir.exists():raise FileExistsError('R9P_OUTPUT_EXISTS_NO_OVERWRITE')
    if not (1<=repeats<=3):raise ValueError('R9P_REPEATS_MUST_BE_1_TO_3')
    if not (0<=temp<=1):raise ValueError('R9P_TEMPERATURE_RANGE')
    if not (4000<=max_tokens<=16000):raise ValueError('R9P_OUTPUT_TOKEN_RANGE')
    if not isinstance(model,str) or not model.strip():raise ValueError('R9P_MODEL_REQUIRED')
    r8b_dir=r8b_dir.resolve();r8b.check(r8b_dir)
    source=json.loads((r8b_dir/'M15R8B_EVIDENCE_GATE.json').read_text())
    common=BASE_USER
    additional=diagnostic_card(source['evidence'])
    calls=[]
    for rep in range(1,repeats+1):
        for arm in ('CONTROL','DIAGNOSTIC_TREATMENT'):
            prompt=common if arm=='CONTROL' else common+'\n\n'+additional
            calls.append({'call_id':secrets.token_hex(8),'blind_id':secrets.token_hex(8),
                          'replicate':rep,'arm':arm,'user_prompt':prompt,
                          'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    random.SystemRandom().shuffle(calls)
    out_dir.mkdir(parents=True)
    (out_dir/'responses').mkdir()
    plan={'schema_version':SCHEMA,'experiment_type':'DIAGNOSTIC_RESPONSIVENESS_NOT_NOVELTY',
          'research_question':source['frozen_science_problem'],'r8b_dir':str(r8b_dir),
          'r8b_evidence_sha256':sha(r8b_dir/'M15R8B_EVIDENCE_GATE.json'),
          'model':model,'temperature':temp,'max_output_tokens':max_tokens,
          'replicates':repeats,'calls':calls,'system_prompt':BASE_SYSTEM,
          'paid_calls_planned':len(calls),'quantitative_physics_authority':False,
          'scientific_novelty_certified':False,
          'causal_question':'Does disclosure of uncertain numerical solver diagnostic alter next scientific question and discriminatory experiment?',
          'limitations':['not an Au/Ag physics validation',
                         'diagnostic feedback may increase caution rather than discovery',
                         'model stochasticity and reviewer inference remain confounders',
                         'one problem does not establish general gains',
                         'independent prior art and measurement not run']}
    save(out_dir/'M15R9P_PLAN_PRIVATE.json',plan)
    return {'status':'R9P_FROZEN_AWAITING_EXPLICIT_PAID_RUN','model':model,
            'planned_paid_calls':len(calls),'physics_gate':'BLOCKED','out_dir':str(out_dir)}


def invoke_openrouter(prompt:str,system:str,model:str,temperature:float,max_tokens:int)->dict:
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise ValueError('OPENROUTER_API_KEY_MISSING')
    data={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
          'temperature':temperature,'max_tokens':max_tokens}
    request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(data).encode('utf-8'),method='POST',
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json',
                 'HTTP-Referer':'https://chatgpt.com','X-Title':'ForAllKG M15-R9P diagnostic-feedback pilot'})
    with urllib.request.urlopen(request,timeout=210) as response:
        payload=json.loads(response.read().decode('utf-8'))
    # No request secrets written to disk, only returned model response.
    if not isinstance(payload,dict) or 'choices' not in payload or not payload['choices']:
        raise ValueError('PROVIDER_RESPONSE_CHOICES_MISSING')
    return payload


def parse_candidate(content)->tuple[dict|None,str]:
    if isinstance(content,list):
        content=''.join(x.get('text','') if isinstance(x,dict) else '' for x in content)
    if not isinstance(content,str) or not content.strip():return None,'EMPTY_CONTENT'
    stripped=content.strip()
    if stripped.startswith('```'):
        stripped=re.sub(r'^```(?:json)?\s*','',stripped,flags=re.I)
        stripped=re.sub(r'\s*```$','',stripped)
    try:
        obj=json.loads(stripped)
    except json.JSONDecodeError:return None,'NOT_JSON'
    if not isinstance(obj,dict) or any(k not in obj or not obj[k] for k in KEYS):
        return None,'MISSING_REQUIRED_FIELDS'
    return obj,'PARSED_STRUCTURALLY_VALID_UNREVIEWED'


def run_paid(out_dir:Path,max_new_calls:int,allow_paid:bool,authorize:bool)->dict:
    if not (allow_paid and authorize):
        raise ValueError('PAID_EXECUTION_REQUIRES_BOTH_EXPLICIT_FLAGS')
    if not 1<=max_new_calls<=6:raise ValueError('MAX_CALLS_MUST_BE_1_TO_6')
    out_dir=out_dir.resolve();plan=load(out_dir)
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('OPENROUTER_API_KEY_MISSING')
    completed=0
    for spec in plan['calls']:
        path=out_dir/'responses'/(spec['call_id']+'.json')
        if path.exists():
            existing=json.loads(path.read_text())
            if existing.get('call_id')!=spec['call_id'] or existing.get('prompt_sha256')!=spec['prompt_sha256']:
                raise ValueError('EXISTING_CALL_RECEIPT_CHANGED')
            continue
        if completed>=max_new_calls:break
        raw=invoke_openrouter(spec['user_prompt'],plan['system_prompt'],plan['model'],plan['temperature'],plan['max_output_tokens'])
        message=raw['choices'][0].get('message',{})
        content=message.get('content')
        candidate,status=parse_candidate(content)
        receipt={'schema_version':SCHEMA,'call_id':spec['call_id'],
                 'prompt_sha256':spec['prompt_sha256'],'blind_id':spec['blind_id'],
                 'parse_status':status,'candidate':candidate,'raw_provider_response':raw,
                 'model':plan['model'],'arm_claims_not_revealed':True,
                 'science_validated':False,'novelty_certified':False}
        save(path,receipt)
        completed+=1
    pending=sum(not (out_dir/'responses'/(spec['call_id']+'.json')).exists() for spec in plan['calls'])
    return {'status':'R9P_PAID_RUN_COMPLETE' if pending==0 else 'R9P_PAID_RUN_PARTIAL',
            'new_paid_calls':completed,'pending_calls':pending,'science_gate':'BLOCKED',
            'next_step':'audit' if pending==0 else 'run-paid --max-new-calls N (explicit authorization)'}


def audit(out_dir:Path)->dict:
    out_dir=out_dir.resolve();plan=load(out_dir)
    blinded=[];assign={};sha_map={}
    for spec in plan['calls']:
        filename=spec['call_id']+'.json';fp=out_dir/'responses'/filename
        if not fp.is_file():raise ValueError('R9P_INCOMPLETE_' + spec['call_id'])
        r=json.loads(fp.read_text())
        if r['call_id']!=spec['call_id'] or r['prompt_sha256']!=spec['prompt_sha256']:
            raise ValueError('R9P_INVALID_RECEIPT')
        sha_map[filename]=sha(fp)
        blinded.append({'blind_id':spec['blind_id'],'candidate':r['candidate'],
                        'parse_status':r['parse_status']})
        assign[spec['blind_id']]={'arm':spec['arm'],'replicate':spec['replicate']}
    if (out_dir/'M15R9P_BLIND_REVIEW_PACKET.json').exists():
        raise FileExistsError('R9P_AUDIT_ALREADY_EXISTS_NO_OVERWRITE')
    save(out_dir/'M15R9P_BLIND_REVIEW_PACKET.json',{
        'schema_version':SCHEMA,'question':plan['research_question'],
        'items':blinded,'blindness_limit':'Technical cues may reveal treatment; never show this private assignment file to reviewer.',
        'rubric':['conceptual difference from baseline','prediction distinguishes rival models',
                 'independent control/measurement','held-out falsifiability',
                 'prior-art risk','calibrated epistemic caution'],
        'novelty_certified':False})
    save(out_dir/'M15R9P_PRIVATE_ASSIGNMENT.json',{'source_response_sha256':sha_map,'assignment':assign,
                                                    'no_scientific_claim_authority':True})
    rubric=out_dir/'M15R9P_BLIND_REVIEW_RUBRIC.csv'
    with rubric.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['blind_id','conceptual_delta_0to2','model_discrimination_0to2',
          'independent_controls_0to2','falsifiability_0to2','prior_art_risk_low_med_high',
          'epistemic_caution_0to2','notes'])
        for x in blinded:w.writerow([x['blind_id']]+['']*7)
    return {'status':'R9P_BLIND_REVIEW_READY','blind_candidates':len(blinded),
            'candidate_parse_valid':sum(v['parse_status']=='PARSED_STRUCTURALLY_VALID_UNREVIEWED' for v in blinded),
            'blind_review_path':str(out_dir/'M15R9P_BLIND_REVIEW_PACKET.json'),
            'rubric_path':str(rubric),
            'private_assignment_DO_NOT_SHARE':str(out_dir/'M15R9P_PRIVATE_ASSIGNMENT.json'),
            'novelty_certified':False}



def unblind(out_dir:Path,scored_csv:Path)->dict:
    """Reveal arms only after scores for every blind ID are frozen externally."""
    out_dir=out_dir.resolve();scored_csv=scored_csv.resolve()
    if (out_dir/'M15R9P_UNBLINDED_COMPARISON.json').exists():
        raise FileExistsError('UNBLIND_ALREADY_EXISTS_NO_OVERWRITE')
    plan=load(out_dir)
    packet=out_dir/'M15R9P_BLIND_REVIEW_PACKET.json'
    if not packet.exists():raise ValueError('BLIND_PACKET_NOT_YET_CREATED')
    pkt=json.loads(packet.read_text())
    priv=json.loads((out_dir/'M15R9P_PRIVATE_ASSIGNMENT.json').read_text())
    for fn,h in priv['source_response_sha256'].items():
        if sha(out_dir/'responses'/fn)!=h:raise ValueError('RESPONSE_MODIFIED_AFTER_FREEZE')
    ids={item['blind_id'] for item in pkt['items']}
    assignment=priv['assignment']
    if ids!=set(assignment):raise ValueError('BLIND_ASSIGNMENT_MISMATCH')
    cols=['conceptual_delta_0to2','model_discrimination_0to2','independent_controls_0to2',
          'falsifiability_0to2','epistemic_caution_0to2']
    risk_map={'low':0,'med':1,'medium':1,'high':2}
    scores={}
    with scored_csv.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            bid=row['blind_id']
            if bid not in ids or bid in scores:raise ValueError('UNKNOWN_OR_DUPLICATE_BLIND_ID')
            for k in cols:
                try: value=int(row[k])
                except (ValueError,TypeError,KeyError):raise ValueError('MISSING_VALID_REVIEW_SCORE_'+bid+'_'+k)
                if value not in (0,1,2):raise ValueError('SCORE_OUTSIDE_0_TO_2')
            risk=str(row['prior_art_risk_low_med_high']).strip().lower()
            if risk not in risk_map:raise ValueError('PRIOR_ART_RISK_NOT_REVIEWED')
            scores[bid]={'scores':{k:int(row[k]) for k in cols},'risk_label':risk,
                         'risk_code':risk_map[risk], 'notes':row.get('notes','')}
    if set(scores)!=ids:raise ValueError('BLIND_SCORES_INCOMPLETE')
    paired=[]
    for rep in range(1,plan['replicates']+1):
        m={v['arm']:scores[k] for k,v in assignment.items() if v['replicate']==rep}
        if set(m)!={'CONTROL','DIAGNOSTIC_TREATMENT'}:raise ValueError('BROKEN_PAIRING')
        totals={arm:sum(m[arm]['scores'].values()) for arm in m}
        paired.append({'replicate':rep,'control_total_0to10':totals['CONTROL'],
                       'treatment_total_0to10':totals['DIAGNOSTIC_TREATMENT'],
                       'treatment_minus_control':totals['DIAGNOSTIC_TREATMENT']-totals['CONTROL'],
                       'control_prior_art_risk':m['CONTROL']['risk_label'],
                       'treatment_prior_art_risk':m['DIAGNOSTIC_TREATMENT']['risk_label']})
    mean=sum(v['treatment_minus_control'] for v in paired)/len(paired)
    output={'schema_version':SCHEMA,'status':'R9P_UNBLINDED_RATINGS_ONLY',
            'paired_ratings':paired,'mean_treatment_minus_control':mean,
            'scored_csv_sha256':sha(scored_csv),'blind_packet_sha256':sha(packet),
            'provenance':'Subjective scores from user-supplied blinded scientific review; not validated discoveries.',
            'calibrated_physics':False,'scientific_novelty_certified':False,
            'limitations':['Very small matched study','reviewers may infer treatment from content',
                           'scored scientific merit is not measured novelty','not independent experimental replication']}
    save(out_dir/'M15R9P_UNBLINDED_COMPARISON.json',output)
    lines=['# M15-R9P — Blinded paired diagnostic-feedback pilot','',
           '| Replicate | Control 0–10 | Diagnostic-feedback 0–10 | Treatment − Control |',
           '|---|---:|---:|---:|']
    for x in paired:
        lines.append(f'| {x["replicate"]} | {x["control_total_0to10"]} | {x["treatment_total_0to10"]} | {x["treatment_minus_control"]:+} |')
    lines+=['',f'- Mean scored difference: {mean:+.3g}',
            '- Interpretation: suggestive feedback-responsiveness signal **only**, not physical validity or scientific novelty.',
            '- Au/Ag science gate: **BLOCKED**']
    (out_dir/'M15R9P_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return {'status':output['status'],'mean_treatment_minus_control':mean,
            'replicates':plan['replicates'],'scientific_novelty_certified':False}


def check(out_dir:Path)->dict:
    out_dir=out_dir.resolve();plan=load(out_dir)
    n=sum((out_dir/'responses'/(s['call_id']+'.json')).exists() for s in plan['calls'])
    result={'status':'R9P_CHECK_OK','planned_calls':len(plan['calls']),
            'saved_receipts':n,'science_gate':'BLOCKED','novelty_certified':False}
    assignment=out_dir/'M15R9P_PRIVATE_ASSIGNMENT.json'
    if assignment.exists():
        a=json.loads(assignment.read_text())
        for name,sh in a['source_response_sha256'].items():
            if sha(out_dir/'responses'/name)!=sh:raise ValueError('R9P_SOURCE_RECEIPT_MODIFIED')
        result['blind_packet_present']=(out_dir/'M15R9P_BLIND_REVIEW_PACKET.json').exists()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--r8b-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--replicates',type=int,default=2)
    p.add_argument('--temperature',type=float,default=0.3);p.add_argument('--max-output-tokens',type=int,default=8000)
    p=sub.add_parser('run-paid');p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--max-new-calls',type=int,default=4)
    p.add_argument('--allow-paid',action='store_true');p.add_argument('--i-authorize-model-calls',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--out-dir',type=Path,required=True)
    p=sub.add_parser('unblind');p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--scored-csv',type=Path,required=True)
    p=sub.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.cmd=='prepare':answer=prepare(args.r8b_dir,args.out_dir,args.model,args.replicates,args.temperature,args.max_output_tokens)
        elif args.cmd=='run-paid':answer=run_paid(args.out_dir,args.max_new_calls,args.allow_paid,args.i_authorize_model_calls)
        elif args.cmd=='audit':answer=audit(args.out_dir)
        elif args.cmd=='unblind':answer=unblind(args.out_dir,args.scored_csv)
        else:answer=check(args.out_dir)
        print(json.dumps(answer,indent=2,ensure_ascii=False))
    except (OSError,ValueError,KeyError,IndexError,TypeError,urllib.error.URLError,json.JSONDecodeError) as exc:
        ap.exit(2,'M15R9P_ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
