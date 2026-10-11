"""M14-C: offline preregistration -> execute -> verify -> evidence-aware next policy.

No LLM calls, no network calls, no mutation of original M14/M14-B history.
Synthetic numerical evidence is NOT empirical evidence or scientific novelty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from pipeline_core.discovery.sers_evidence_search_m14c import (
    SCHEMA,CASE_PHASE,CASE_INJECTION,CASE_BY_LANE,canonical,sha,execute_cases,effective_action)
from pipeline_core.discovery.sers_adaptive_policy_m14b import choose, policy_for_next
from scripts.discovery.run_sers_adaptive_policy_m14b import read_member, science_rows

PLAN='M14C_PREREGISTRATION.json'
ROWS='M14C_EXECUTED_OBSERVABLES.csv'
SUMMARIES='M14C_NUMERICAL_SUMMARIES.json'
RECEIPT='M14C_EXECUTION_RECEIPT.json'
CERT='M14C_REPRODUCIBILITY_CERTIFICATE.json'
FEEDBACK='M14C_EVIDENCE_CONDITIONED_POLICY.json'


def _source_file():
    from pipeline_core.discovery import sers_evidence_search_m14c as core
    return Path(core.__file__)


def code_hash():
    return sha(_source_file().read_bytes())


def load(path:Path):
    return json.loads(path.read_bytes())


def write_once(path:Path,raw:bytes):
    with path.open('xb') as f:f.write(raw)


def _source_generation(root:Path,rep:int,g:int,arm:str):
    rel=f'r{rep:02d}/g{g:02d}/{arm}/M14_GENERATION_RESULT.json'
    generation,source_sha=read_member(root,rel)
    return generation,rel,source_sha


def make_plan(root:Path,rep:int,g:int,arm:str):
    all_plan,cards,histories,index=science_rows(root)
    if arm!='ADAPTIVE':
        raise ValueError('M14-C illustrative intervention is not a second experimental arm')
    if rep not in range(1,all_plan['design']['replicates']+1) or g not in range(1,all_plan['design']['generations']+1):
        raise ValueError('Invalid replicate/generation')
    sample,rel,source_sha=_source_generation(root,rep,g,arm)
    assert sample==histories[(rep,arm)][g-1]
    by_lane={}
    for c in sample['candidates']:
        lane=c['lane'];case=CASE_BY_LANE.get(lane)
        if lane in by_lane or not case:raise ValueError('Unrecognized or duplicated lane')
        claim=(c.get('claim_atom','')+' '+c.get('strongest_null','')).casefold()
        if case==CASE_PHASE and not ('phase' in claim and ('coheren' in claim or 'vibrational' in claim) and ('markov' in claim or 'hidden' in claim)):
            raise ValueError('Selected hypothesis does not support this linear phase countermodel: '+c.get('candidate_id',''))
        if case==CASE_INJECTION and not ('injection' in claim and 'thermal' in claim):
            raise ValueError('Selected hypothesis does not support the thermal injection challenge: '+c.get('candidate_id',''))
        if not re.fullmatch(r'm14:[0-9a-f]{20}',c['candidate_id']):raise ValueError('Bad candidate id')
        if c.get('novelty_certified', False) is not False:raise ValueError('Candidate novelty already claimed')
        by_lane[lane]={'candidate_id':c['candidate_id'],'problem_id':c['problem_id'],
                      'family':c['family'],'case_id':case,'claim_sha256':sha(canonical(c['claim_atom'])),
                      'case_alignment':'ILLUSTRATIVE_EXPLICIT_REGISTERED_EQUATIONS_NOT_FULL_CANDIDATE_VALIDATION'}
    if set(by_lane)!=set(CASE_BY_LANE):raise ValueError('Two expected lanes required')
    return {'schema':SCHEMA,'status':'M14C_PREREGISTERED_NOT_EXECUTED',
        'source_root':str(root.absolute()),'source_generation_relpath':rel,
        'source_generation_sha256':source_sha,'source_study_artifact_index_digest':sha(canonical(index)),
        'replicate':rep,'generation':g,'arm':arm,
        'candidate_cases':by_lane,'preregistered_code_sha256':code_hash(),
        'independent_empirical_data_supplied':False,
        'scientific_claims_covered':'Two constrained model-class examples only, NOT entire research ideas',
        'observational_scientific_novelty_certified':False,
        'interpretation_rule':'A positive restricted-null difference does not identify causal injection if an expanded thermal null reproduces it.',
        'evidence_absence_must_not_be_treated_as_novelty':True,
        'no_paid_calls':True}


def check_preregistration(folder:Path):
    p=load(folder/PLAN)
    if p['schema']!=SCHEMA or p.get('observational_scientific_novelty_certified') is not False:
        raise ValueError('Invalid preregistration')
    if p['preregistered_code_sha256']!=code_hash():
        raise ValueError('Numerical code changed since preregistration: refuse nonreproducible result')
    root=Path(p['source_root']);generation,rel,srcsha=_source_generation(root,p['replicate'],p['generation'],p['arm'])
    if rel!=p['source_generation_relpath'] or srcsha!=p['source_generation_sha256']:
        raise ValueError('Historical source changed after preregistration')
    if {x['lane']:x['candidate_id'] for x in generation['candidates']} != {k:v['candidate_id'] for k,v in p['candidate_cases'].items()}:
        raise ValueError('Candidate lineage changed after preregistration')
    if set(p['candidate_cases'])!=set(CASE_BY_LANE):raise ValueError('Missing preregistered lane')
    for lane,x in p['candidate_cases'].items():
        if x['case_id']!=CASE_BY_LANE[lane]:raise ValueError('Case map changed')
        c=next(z for z in generation['candidates'] if z['lane']==lane)
        if x['claim_sha256']!=sha(canonical(c['claim_atom'])):raise ValueError('Scientific claim changed')
    return p


def cmd_prepare(args):
    if args.out_dir.exists():raise FileExistsError('No overwrites: '+str(args.out_dir))
    plan=make_plan(args.root,args.replicate,args.generation,args.arm)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    write_once(args.out_dir/PLAN,canonical(plan))
    print(json.dumps({'status':plan['status'],'output':str(args.out_dir),
        'candidate_ids':{k:v['candidate_id'] for k,v in plan['candidate_cases'].items()},
        'new_model_calls':0,'empirical_evidence':False},indent=2))


def cmd_execute(args):
    plan=check_preregistration(args.out_dir)
    if any((args.out_dir/x).exists() for x in (ROWS,SUMMARIES,RECEIPT,CERT,FEEDBACK)):
        raise FileExistsError('Execution already started; never overwrite evidence')
    raw,summary=execute_cases()
    summary_raw=canonical(summary)
    receipt={'schema':SCHEMA,'status':'M14C_NUMERICAL_EQUATIONS_EXECUTED_NO_EMPIRICAL_DATA',
        'preregistration_sha256':sha((args.out_dir/PLAN).read_bytes()),
        'source_generation_sha256':plan['source_generation_sha256'],
        'numerical_code_sha256':code_hash(),
        'observables_csv_sha256':sha(raw),'summaries_sha256':sha(summary_raw),
        'case_bindings':plan['candidate_cases'],
        'source_data':'SYNTHETIC_AND_ANALYTICAL_ONLY',
        'no_empirical_intervention_performed':True,
        'new_model_calls':0,'network_calls':0,'novelty_certified':False}
    write_once(args.out_dir/ROWS,raw)
    write_once(args.out_dir/SUMMARIES,summary_raw)
    write_once(args.out_dir/RECEIPT,canonical(receipt))
    print(json.dumps({'status':receipt['status'],'output':str(args.out_dir),
        'observable_rows':summary['total_rows'],
        'outcomes':{k:v['outcome'] for k,v in summary['case_results'].items()},
        'new_model_calls':0,'empirical_evidence':False},indent=2))


def verified_material(folder:Path):
    plan=check_preregistration(folder)
    receipt=load(folder/RECEIPT)
    if receipt['schema']!=SCHEMA or receipt.get('novelty_certified') is not False:
        raise ValueError('Receipt integrity check failed')
    if receipt['preregistration_sha256']!=sha((folder/PLAN).read_bytes()) or \
       receipt['numerical_code_sha256']!=code_hash() or \
       receipt['source_generation_sha256']!=plan['source_generation_sha256'] or \
       receipt['case_bindings']!=plan['candidate_cases']:
        raise ValueError('Execution receipt does not match preregistration')
    expected_csv,expected_summary=execute_cases()
    if (folder/ROWS).read_bytes()!=expected_csv or (folder/SUMMARIES).read_bytes()!=canonical(expected_summary):
        raise ValueError('Stored numerical predictions do not replay exactly')
    if receipt['observables_csv_sha256']!=sha(expected_csv) or \
       receipt['summaries_sha256']!=sha(canonical(expected_summary)):
        raise ValueError('Numerical source hashes differ')
    return plan,receipt,expected_summary


def policy_update(folder:Path,plan:dict,receipt:dict,summary:dict):
    root=Path(plan['source_root'])
    _,cards,histories,_=science_rows(root)
    history=histories[(plan['replicate'],plan['arm'])][:plan['generation']]
    before=policy_for_next(history,problem_cards=cards)
    cert_hash=sha((folder/RECEIPT).read_bytes())
    evidence=[]
    for lane,ref in plan['candidate_cases'].items():
        evidence.append({'kind':'EXECUTED_NUMERICAL_MODEL','verified':True,
            'independent_of_generation':True,'candidate_id':ref['candidate_id'],
            'artifact_sha256':cert_hash,'artifact_id':'M14C:'+ref['case_id'],
            'scope':'SOURCE_CODE_RECOMPUTED_SYNTHETIC_MODEL_OUTPUT_ONLY',
            'not_real_experimental_data':True})
    with_refs=policy_for_next(history,evidence=evidence,problem_cards=cards)
    results={}
    for lane,ref in plan['candidate_cases'].items():
        case=ref['case_id']
        finding=summary['case_results'][case]
        resolution=effective_action(case,finding['outcome'],True)
        results[lane]={'candidate_id':ref['candidate_id'],'problem_id':ref['problem_id'],
             'case_id':case,'outcome':finding['outcome'],
             'baseline_m14b_action':before['lanes'][lane]['action'],
             'm14b_evidence_presence_only_action':with_refs['lanes'][lane]['action'],
             'm14c_outcome_aware_action':resolution['action'],
             'm14c_reason':resolution['reason'],'next_evidence_required':resolution['next_test'],
             'action_changed_from_baseline':resolution['action']!=before['lanes'][lane]['action'],
             'scientific_novelty_established':False,
             'real_world_causality_identified':False}
    return {'schema':SCHEMA,'status':'M14C_NUMERICAL_EVIDENCE_REPLAYED_AND_POLICY_UPDATED',
        'reproducibility_verified':True,'verified_scope':'CODE_EXECUTION_AND_ARTIFACT_INTEGRITY_ONLY',
        'independent_from_original_LLM_generation':'DETERMINISTIC_PREREGISTERED_EQUATIONS',
        'receipt_sha256':cert_hash,'evidence_references':evidence,'next_generation':plan['generation']+1,
        'policy_by_lane':results,'source_M14_history_is_frozen':True,
        'no_successor_generated':True,'external_empirical_data':False,
        'novelty_certified':False,'new_model_calls':0,'network_calls':0,
        'policy_change_is_not_scientific_discovery':True}


def cmd_verify(args):
    plan,receipt,summary=verified_material(args.out_dir)
    update=policy_update(args.out_dir,plan,receipt,summary)
    certificate={'schema':SCHEMA,'status':'M14C_NUMERICAL_RECEIPT_RECOMPUTED_AND_HASH_VERIFIED',
        'receipt_sha256':sha((args.out_dir/RECEIPT).read_bytes()),
        'observables_csv_sha256':receipt['observables_csv_sha256'],
        'numerical_code_sha256':code_hash(),
        'verification_scope':'Exact independent re-execution of deterministic mathematical code; NOT empirical or scientific truth certification',
        'new_model_calls':0,'external_empirical_data':False,'novelty_certified':False}
    outputs={CERT:canonical(certificate),FEEDBACK:canonical(update)}
    for rel,raw in outputs.items():
        f=args.out_dir/rel
        if f.exists() and f.read_bytes()!=raw:
            raise ValueError('Previously frozen verification differs: '+rel)
    for rel,raw in outputs.items():
        if not (args.out_dir/rel).exists():write_once(args.out_dir/rel,raw)
    print(json.dumps({'status':update['status'],'output':str(args.out_dir),
        'reproducibility_verified':True,
        'actions':{k:{'before':v['baseline_m14b_action'], 'after':v['m14c_outcome_aware_action']} for k,v in update['policy_by_lane'].items()},
        'empirical_data':False,'novelty_certified':False,'new_model_calls':0},indent=2))


def cmd_status(args):
    found={x:(args.out_dir/x).exists() for x in (PLAN,ROWS,SUMMARIES,RECEIPT,CERT,FEEDBACK)}
    print(json.dumps({'status':'M14C_ARTIFACT_STATUS','output':str(args.out_dir),
                      'artifacts':found,'new_model_calls':0},indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description='M14-C: offline executed model/null challenge, never real-world novelty certification')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--replicate',type=int,required=True);a.add_argument('--generation',type=int,required=True)
    a.add_argument('--arm',choices=('ADAPTIVE',),required=True);a.add_argument('--out-dir',type=Path,required=True)
    for cmd in ('execute','verify-and-update','status'):
        s=sub.add_parser(cmd);s.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args(argv)
    return {'prepare':cmd_prepare,'execute':cmd_execute,'verify-and-update':cmd_verify,'status':cmd_status}[a.command](a)

if __name__=='__main__':main()
