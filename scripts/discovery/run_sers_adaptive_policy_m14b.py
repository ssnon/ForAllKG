"""No-network, no-model-call M14-B replay and forward policy-task tool.

USAGE: python -m scripts.discovery.run_sers_adaptive_policy_m14b replay --root <M14_OUT> --out-dir <NEW_DIR>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from pipeline_core.discovery.sers_adaptive_policy_m14b import (
    SCHEMA, LANES, choose, normalize, policy_for_next, proposed_next_step,
)


def read_member(root: Path, relative: str) -> tuple[dict, str]:
    if '..' in Path(relative).parts or Path(relative).is_absolute():
        raise ValueError('Unsafe path')
    if root.is_dir():
        b = (root / relative).read_bytes()
    else:
        with zipfile.ZipFile(root) as z:
            b = z.read(relative)
    return json.loads(b), hashlib.sha256(b).hexdigest()


def science_rows(root: Path):
    plan, ph = read_member(root, 'M14_PLAN.json')
    problems, gh = read_member(root, 'M14_PROVISIONAL_PROBLEMS.json')
    summary, sh = read_member(root, 'M14_RUN_SUMMARY.json')
    design = plan['design']
    if summary.get('replicates') != design['replicates'] or summary.get('generations') != design['generations']:
        raise ValueError('This archive has incomplete or wrong plan/summary')
    if problems.get('novelty_certified') is not False:
        raise ValueError('Problem cards must not certify novelty')
    cards = problems['questions']
    results, sha_index = {}, {'M14_PLAN.json': ph, 'M14_PROVISIONAL_PROBLEMS.json': gh, 'M14_RUN_SUMMARY.json': sh}
    for rep in range(1, design['replicates']+1):
        for arm in ('ADAPTIVE','FIXED'):
            history=[]
            for g in range(1, design['generations']+1):
                rel = f'r{rep:02d}/g{g:02d}/{arm}/M14_GENERATION_RESULT.json'
                result, h = read_member(root, rel)
                sha_index[rel] = h
                if result['generation'] != g or result['replicate'] != rep:
                    raise ValueError('Generation lineage mismatch: '+rel)
                if result.get('novelty_certified') is not False:
                    raise ValueError('Historical novelty not certified')
                history.append(result)
            results[(rep, arm)] = history
    return plan, cards, results, sha_index


def replay(root: Path, out_dir: Path):
    if out_dir.exists():
        raise FileExistsError('Output directory exists; immutable replay: '+str(out_dir))
    plan, cards, results, sha_index = science_rows(root)
    rows=[]
    for (rep, arm), runs in sorted(results.items()):
        for g in range(2, len(runs)+1):
            previous = runs[g-2]
            current = runs[g-1]
            past = runs[:g-1]
            for lane in LANES:
                decision=choose(past,lane,problem_cards=cards)
                before=next(c for c in previous['candidates'] if c['lane']==lane)
                after=next(c for c in current['candidates'] if c['lane']==lane)
                observed=after.get('requested_action')
                rows.append({
                    'replicate':rep,'generation':g,'arm':arm,'lane':lane,
                    'observed_action':observed,'m14b_frozen_history_action':decision['action'],
                    'action_differs':int(observed!=decision['action']),
                    'normalized_family_visits':decision['normalized_family_visits'],
                    'same_lane_nonidentifiability_streak':decision['same_lane_nonidentifiability_streak'],
                    'prior_physics_status':next(c['physical_status'] for c in previous['candidate_reviews'] if c['lane']==lane),
                    'prior_literature_status':next(c['prior_art_status'] for c in previous['candidate_reviews'] if c['lane']==lane),
                    'previous_family':before['family'],
                    'next_family':after['family'],
                    'observed_family_changed':int(normalize(before['family'])!=normalize(after['family'])),
                    'observed_problem_changed':int(before['problem_id']!=after['problem_id']),
                    'new_problem_discovery_needed':int(decision['problem_expansion_required']),
                    'm14b_policy_reasons':'|'.join(decision['reasons']),
                    'historical_is_not_counterfactual_generation':True,
                })
    observations={}
    for arm in ('ADAPTIVE','FIXED'):
        a=[r for r in rows if r['arm']==arm]
        observations[arm]={
            'count_prior_to_next_transitions':len(a),
            'historical_actions':{k:sum(r['observed_action']==k for r in a) for k in ('REFINE','BRANCH','ESCAPE','SUSPEND')},
            'new_policy_frozen_history_actions':{k:sum(r['m14b_frozen_history_action']==k for r in a) for k in ('REFINE','BRANCH','ESCAPE','SUSPEND')},
            'observed_family_changed':sum(r['observed_family_changed'] for r in a),
            'observed_problem_changed':sum(r['observed_problem_changed'] for r in a),
            'counterfactual_replay_is_not_a_new_generation':True,
        }
    hash_index=hashlib.sha256(json.dumps(sha_index,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    summary={'schema':SCHEMA,'status':'M14B_OFFLINE_FROZEN_HISTORY_REPLAY_COMPLETE',
             'historic_study_unchanged':True,'source_path':str(root),
             'source_artifacts_sha256_manifest_digest':hash_index,
             'generations':plan['design']['generations'],'replicates':plan['design']['replicates'],
             'frozen_history_decision_rows':len(rows),'arms':observations,
             'limitation':'These are actions proposed on FIXED HISTORIES. Changing historical actions would have changed future generated hypotheses; replay is not a causal effectiveness experiment.',
             'semantic_novelty_certified':False,'model_calls':0,'network_calls':0,
             'next_milestone':'Independent evidence-producing candidate tests before matched paid-model comparison'}
    out_dir.mkdir(parents=True,exist_ok=False)
    with (out_dir/'M14B_POLICY_REPLAY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
    (out_dir/'M14B_POLICY_REPLAY_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out_dir/'M14B_SOURCE_SHA256.json').write_text(json.dumps(sha_index,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'output_directory':str(out_dir),**summary},ensure_ascii=False,indent=2))


def task(root:Path, out:Path, rep:int, g:int, arm:str):
    if out.exists():raise FileExistsError('Refusing overwrite: '+str(out))
    plan,cards,results,index=science_rows(root)
    if arm != 'ADAPTIVE' or rep<1 or g<1 or g>plan['design']['generations']+1:
        raise ValueError('Invalid replicate/generation/arm')
    hist=results[(rep,arm)][:g-1]
    # Explicit retrospective only; no mutation of old candidate receipts.
    contract=proposed_next_step(hist,cards)
    contract.update({'replicate':rep,'requested_generation':g,'arm':arm,
                     'source_data_is_historical':True,
                     'is_retroactive_diagnostic_not_part_of_m14_trial':True})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(contract,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'M14B_FORWARD_CONTRACT_DRAFTED_NO_PAID_CALLS',
                     'output':str(out),'by_lane':{k:v['action'] for k,v in contract['policy']['lanes'].items()},
                     'is_not_new_scientific_discovery':True},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description='M14B offline shadow policy; does not mutate M14 or call providers')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('replay',help='Replay policy on original frozen histories; no model calls')
    p.add_argument('--root',type=Path,required=True,help='M14 output directory OR full audit zip')
    p.add_argument('--out-dir',type=Path,required=True)
    q=sub.add_parser('draft-contract',help='Draft next-step structural policy on selected historical point')
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--replicate',type=int,required=True)
    q.add_argument('--generation',type=int,required=True)
    q.add_argument('--arm',choices=('ADAPTIVE',),required=True)
    args=parser.parse_args(argv)
    if args.command=='replay':return replay(args.root,args.out_dir)
    return task(args.root,args.out,args.replicate,args.generation,args.arm)


if __name__ == '__main__':
    main()
