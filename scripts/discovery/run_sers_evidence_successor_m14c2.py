"""Evidence-conditioned *prospective* SERS successor, separate from frozen M14 A/B/C1.

prepare --out-dir ... --c1-dir ... --root ... (offline)
generate-paid (strict opt-in, no automatic retry) OR ingest --response-json ... (offline)
verify (offline, structural gate + mathematical forward calculation)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from pipeline_core.discovery.sers_evidence_successor_m14c2 import (SCHEMA, LANE_ESCAPE,LANE_SUSPEND,canon,sha,prompt_for,test_model_vectors)
from pipeline_core.discovery.sers_adaptive_policy_m14b import normalize,validate_transition,SCHEMA as POLICY_SCHEMA
from pipeline_core.discovery.sers_evidence_search_m14c import (execute_cases,canonical as c1canon)
from scripts.discovery.run_sers_adaptive_policy_m14b import science_rows,read_member
from scripts.discovery.run_sers_novelty_search_m14 import source_call,saved_model

C1FILES=('M14C_PREREGISTRATION.json','M14C_EXECUTED_OBSERVABLES.csv','M14C_NUMERICAL_SUMMARIES.json',
         'M14C_EXECUTION_RECEIPT.json','M14C_REPRODUCIBILITY_CERTIFICATE.json','M14C_EVIDENCE_CONDITIONED_POLICY.json')
CONTRACT='M14C2_FORWARD_CONTRACT.json'; RAW='M14C2_SUCCESSOR_SUBMISSION.json'; REPORT='M14C2_PROSPECTIVE_VALIDATION.json'

def read(path):return json.loads(path.read_bytes())
def once(path,raw):
    with path.open('xb') as f:f.write(raw)

def _verified_c1(c1:Path,root:Path)->tuple[dict,dict,dict]:
    if not all((c1/f).is_file() for f in C1FILES):raise ValueError('Missing one or more M14-C1 evidence artifacts')
    plan=read(c1/C1FILES[0]);receipt=read(c1/C1FILES[3]);certificate=read(c1/C1FILES[4]);policy=read(c1/C1FILES[5])
    if plan.get('replicate')!=3 or plan.get('generation')!=4 or plan.get('arm')!='ADAPTIVE':
        raise ValueError('M14-C2 v1 only supports preregistered r03/g04/ADAPTIVE')
    code=Path(__import__('pipeline_core.discovery.sers_evidence_search_m14c',fromlist=['execute_cases']).__file__).read_bytes()
    if plan['preregistered_code_sha256']!=sha(code) or receipt['numerical_code_sha256']!=sha(code):
        raise ValueError('M14-C1 code hash changed')
    if receipt['preregistration_sha256']!=sha((c1/C1FILES[0]).read_bytes()):raise ValueError('C1 plan hash mismatch')
    gen,gen_sha=read_member(root,plan['source_generation_relpath'])
    if gen_sha!=plan['source_generation_sha256'] or receipt['source_generation_sha256']!=gen_sha:
        raise ValueError('M14-C1 historical generation mismatch')
    if set(c['candidate_id'] for c in gen['candidates']) != set(v['candidate_id'] for v in plan['candidate_cases'].values()):
        raise ValueError('Generation candidate IDs changed')
    csv,summary=execute_cases()
    if csv!=(c1/C1FILES[1]).read_bytes() or c1canon(summary)!=(c1/C1FILES[2]).read_bytes():
        raise ValueError('M14-C1 numerical execution does not replay')
    if receipt['observables_csv_sha256']!=sha(csv) or receipt['summaries_sha256']!=sha(c1canon(summary)):
        raise ValueError('M14-C1 receipt hashes invalid')
    raw_receipt=(c1/C1FILES[3]).read_bytes()
    if certificate.get('receipt_sha256')!=sha(raw_receipt) or policy.get('receipt_sha256')!=sha(raw_receipt):
        raise ValueError('M14-C1 provenance broken')
    if certificate.get('status')!='M14C_NUMERICAL_RECEIPT_RECOMPUTED_AND_HASH_VERIFIED' or not policy.get('reproducibility_verified'):
        raise ValueError('Missing recomputation evidence')
    if policy.get('no_successor_generated') is not True or policy.get('novelty_certified') is not False:
        raise ValueError('Invalid C1 evidence state')
    if policy.get('policy_by_lane',{}).get(LANE_ESCAPE,{}).get('m14c_outcome_aware_action')!='ESCAPE' or policy.get('policy_by_lane',{}).get(LANE_SUSPEND,{}).get('m14c_outcome_aware_action')!='SUSPEND':
        raise ValueError('The preregistered ESCAPE/SUSPEND roles were changed')
    for lane,v in policy['policy_by_lane'].items():
        if v.get('candidate_id')!=plan['candidate_cases'][lane]['candidate_id'] or v.get('outcome')!=summary['case_results'][plan['candidate_cases'][lane]['case_id']]['outcome']:
            raise ValueError('M14-C1 policy not bound to exact candidate/model outcome')
    return plan,policy,gen


def derive_contract(c1:Path,root:Path)->dict:
    c1plan,policy,gen=_verified_c1(c1,root)
    study,cards,_,index=science_rows(root)
    if sha(c1canon(index))!=c1plan['source_study_artifact_index_digest']:
        raise ValueError('Historical study manifest differs from the C1 preregistration')
    prev=next(v for v in gen['candidates'] if v['lane']==LANE_ESCAPE)
    suspended_problem_id=policy['policy_by_lane'][LANE_SUSPEND]['problem_id']
    alternatives=[p for p in cards if p['problem_id'] not in (prev['problem_id'],suspended_problem_id)
                  and normalize(p['domain_family'])!=normalize(prev['family'])
                  and p.get('source_anchors') and p.get('gap_confirmed') is False]
    if not alternatives:raise ValueError('No independently anchored alternate problem cards: suspend instead of inventing gaps')
    by_lane={
      LANE_ESCAPE:{'decision':{'schema':POLICY_SCHEMA,'action':'ESCAPE','escape_alternative_problem_ids':sorted(p['problem_id'] for p in alternatives),'problem_expansion_required':False},
                   'previous':{'candidate_id':prev['candidate_id'],'problem_id':prev['problem_id'],'family':prev['family'],
                               'mechanism_signature':{'mechanism_family_id':prev['family'],'causal_driver':prev['mechanism'],
                                   'core_assumption':prev['claim_atom'],'predicted_observable':prev['conditional_predictions'][0],
                                   'independent_intervention':prev['interventions'][0],
                                   'strongest_null':prev['strongest_null'],'falsification':prev['falsification']}},
                   'alternative_cards':alternatives},
      LANE_SUSPEND:{'decision':{'schema':POLICY_SCHEMA,'action':'SUSPEND'},'previous_candidate_id':policy['policy_by_lane'][LANE_SUSPEND]['candidate_id'],
                    'next_evidence_required':policy['policy_by_lane'][LANE_SUSPEND]['next_evidence_required']}}
    return {'schema':SCHEMA,'status':'M14C2_VERIFIED_EVIDENCE_FORWARD_CONTRACT_PREPARED','root':str(root.resolve()),
         'source_generation_sha256':c1plan['source_generation_sha256'],
         'c1_input_sha256':{name:sha((c1/name).read_bytes()) for name in C1FILES},
         'lanes':by_lane,'verified_c1_findings':{lane:{'case':v['case_id'],'outcome':v['outcome'],'reason':v['m14c_reason'],
                                                 'next_evidence_required':v['next_evidence_required']} for lane,v in policy['policy_by_lane'].items()},
         'limitations':['C1 models are deliberately constructed examples, not experimental measurements.',
             'An anchored research question is not a confirmed literature gap.',
             'Nominal mechanism changes and polynomial separation do not establish scientific novelty.'],
         'generation_count':1,'active_lanes':[LANE_ESCAPE],'suspended_lanes':[LANE_SUSPEND],
         'new_model_calls':0,'real_scientific_discovery_certified':False}

def _load_contract(out):
    c=read(out/CONTRACT)
    if c.get('schema')!=SCHEMA:raise ValueError('Wrong schema')
    return c


def _validate_submission(contract:dict,submission:dict)->dict:
    if not isinstance(submission,dict) or set(submission)!={'successor','suspended_lane_successor'}:
        raise ValueError('Expected one successor and explicit suspended null')
    if submission['suspended_lane_successor'] is not None:
        raise ValueError('SUSPEND must not produce a candidate')
    successor=submission['successor']
    if not isinstance(successor,dict) or successor.get('novelty_certified') is not False:
        raise ValueError('Invalid successor/novelty certification')
    if set(successor)-{'problem_id','family','mechanism_signature','conditional_predictions','independent_observables','novelty_certified','test_proposal'}:
        raise ValueError('Unexpected unverified successor attributes')
    escape=contract['lanes'][LANE_ESCAPE]
    matched=[p for p in escape['alternative_cards'] if p['problem_id']==successor.get('problem_id')]
    if len(matched)!=1:raise ValueError('Escape must select one of the anchored cards')
    if normalize(matched[0]['domain_family'])!=normalize(successor.get('family')):
        raise ValueError('Escape family must match the selected anchored scientific card')
    for name in ('conditional_predictions','independent_observables'):
        xs=successor.get(name)
        if not isinstance(xs,list) or not xs or any(not isinstance(s,str) or not s.strip() for s in xs):
            raise ValueError('Specific scientific predictions and independent observables required')
    transition=validate_transition(escape['previous'],successor,escape['decision'])
    suspension=validate_transition(None,None,contract['lanes'][LANE_SUSPEND]['decision'])
    prospective=test_model_vectors(successor['test_proposal'])
    return {'schema':SCHEMA,'status':'M14C2_ONE_SUCCESSOR_STRUCTURALLY_VALIDATED',
            'novelty_certified':False,'real_world_discovery':False,'source_evidence':'SYNTHETIC_M14C1_ONLY',
            'successor_problem_id':successor['problem_id'],
            'source_anchor_works':matched[0]['source_anchors'],
            'transition':transition,'suspended_lane':suspension,'prospective_test':prospective,
            'limitations':['Structural escape is not proven conceptual novelty.',
                           'Prospective coefficients and nuisance bounds were proposed, not fitted to independent data.',
                           'A model separation under user/model-proposed equations is not empirical falsification.',
                           'Scientific and claim-level prior-art reviews remain necessary.']}

def cmd_prepare(args):
    if args.out_dir.exists():raise FileExistsError('No output overwrite')
    contract=derive_contract(args.c1_dir,args.root)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    once(args.out_dir/CONTRACT,canon(contract))
    once(args.out_dir/'M14C2_GENERATION_PROMPT.json',canon(prompt_for(contract)))
    print(json.dumps({'status':contract['status'],'out_dir':str(args.out_dir),'active_lanes':contract['active_lanes'],
                      'suspended_lanes':contract['suspended_lanes'],'anchored_alternatives':len(contract['lanes'][LANE_ESCAPE]['alternative_cards']),
                      'new_model_calls':0,'novelty_certified':False},indent=2))

def cmd_ingest(args):
    out=args.out_dir; contract=_load_contract(out)
    if (out/RAW).exists():raise FileExistsError('Existing candidate submission frozen')
    raw=args.response_json.read_bytes()
    sub=json.loads(raw)
    _validate_submission(contract,sub)  # fail closed, do not freeze invalid candidate
    once(out/RAW,canon(sub))
    print(json.dumps({'status':'M14C2_OFFLINE_SUCCESSOR_INGESTED','new_model_calls':0},indent=2))

def cmd_paid(args):
    if not (args.allow_paid and args.i_authorize_model_calls):
        raise RuntimeError('Explicit paid-model authorization required')
    out=args.out_dir;contract=_load_contract(out)
    if (out/RAW).exists():raise FileExistsError('Successor already present; refuse extra call')
    # Existing M14 receipt-safe semantics, including indeterminate intent freeze.
    fn=source_call(args)
    result,new=saved_model(out,'m14c2_successor',prompt_for(contract),args.model,fn)
    try:
        s=json.loads(result.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        _validate_submission(contract,s)
    except Exception as exc:
        err=out/'M14C2_INVALID_SUBMISSION.json'
        if not err.exists():once(err,canon({'error':str(exc),'raw_receipt_preserved':True,'automatic_recall_allowed':False}))
        raise ValueError('Saved raw receipt invalid; no automatic paid recall: '+str(exc)) from exc
    once(out/RAW,canon(s))
    print(json.dumps({'status':'M14C2_ONE_PAID_SUCCESSOR_SAVED','new_model_calls':int(new),'novelty_certified':False},indent=2))

def cmd_verify(args):
    out=args.out_dir;contract=_load_contract(out)
    submission=read(out/RAW)
    result=_validate_submission(contract,submission)
    result['contract_sha256']=sha((out/CONTRACT).read_bytes())
    result['submission_sha256']=sha((out/RAW).read_bytes())
    data=canon(result)
    if (out/REPORT).exists() and (out/REPORT).read_bytes()!=data:
        raise ValueError('Previously frozen report mismatch')
    if not (out/REPORT).exists():once(out/REPORT,data)
    print(json.dumps({'status':result['status'],'out_dir':str(out),
            'successor_problem_id':result['successor_problem_id'],
            'formal_outside_envelope':result['prospective_test']['formal_grid_points_outside_assumed_envelope'],
            'suspended_lane_has_successor':False,'empirical_novelty_certified':False},indent=2))

def main(argv=None):
    p=argparse.ArgumentParser(description='M14-C2 evidence-conditioned prospective successor')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,required=True);a.add_argument('--c1-dir',type=Path,required=True);a.add_argument('--out-dir',type=Path,required=True)
    a=sub.add_parser('ingest');a.add_argument('--out-dir',type=Path,required=True);a.add_argument('--response-json',type=Path,required=True)
    a=sub.add_parser('generate-paid');a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--model',default='openai/gpt-5.6-luna');a.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    a.add_argument('--api-key-env',default='OPENROUTER_API_KEY');a.add_argument('--max-output-tokens',type=int,default=9000)
    a.add_argument('--temperature',type=float,default=0.7);a.add_argument('--allow-paid',action='store_true');a.add_argument('--i-authorize-model-calls',action='store_true')
    a=sub.add_parser('verify');a.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args(argv)
    return {'prepare':cmd_prepare,'ingest':cmd_ingest,'generate-paid':cmd_paid,'verify':cmd_verify}[a.cmd](a)

if __name__=='__main__':main()
