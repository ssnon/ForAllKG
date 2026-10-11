"""Offline C2 successor identifiability challenge; no paid model calls or network.

prepare --c2-dir <verified C2 output> --out-dir <new C3 output>
execute --out-dir <prepared C3>
verify-and-adjudicate --out-dir <executed C3>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from pipeline_core.discovery import sers_successor_challenge_m14c3 as science
from scripts.discovery.run_sers_evidence_successor_m14c2 import (
    CONTRACT, RAW, REPORT, _load_contract, _validate_submission)
from scripts.discovery.recover_sers_evidence_successor_m14c2_promptaware import _ctx as original_paid_context

PLAN='M14C3_CHALLENGE_PREREGISTRATION.json'
OBS='M14C3_SYNTHETIC_OBSERVABLES.csv'
SWEEP='M14C3_CONDITIONAL_CALIBRATION_SWEEP.csv'
SUMMARY='M14C3_SYNTHETIC_SCIENCE_SUMMARY.json'
RECEIPT='M14C3_COMPUTATION_RECEIPT.json'
VERIFIED='M14C3_INDEPENDENT_REPLAY_CERTIFICATE.json'
POLICY='M14C3_EVIDENCE_CONDITIONED_POLICY.json'
AUDIT='M14C2_FAMILY_ALIGNMENT_RECOVERY_AUDIT.json'
ORIGINAL='m14c2_successor_RAW_RECEIPT.json'

def _read(p):return json.loads(p.read_bytes())
def _once(p,content):
    with p.open('xb') as f:f.write(content)
def _same(p,content):
    if not p.is_file() or p.read_bytes()!=content:
        raise ValueError('Frozen artifact changed or missing: '+str(p))

def _validate_c2(out:Path)->dict:
    if not out.is_dir():raise ValueError('C2 directory missing')
    context=original_paid_context(out)  # verifies original receipt, intent and equivalent frozen prompts
    contract=_load_contract(out); submission=_read(out/RAW);report=_read(out/REPORT);audit=_read(out/AUDIT)
    if (audit.get('original_raw_receipt_sha256')!=science.sha((out/ORIGINAL).read_bytes())
        or audit.get('submission_sha256')!=science.sha((out/RAW).read_bytes())
        or audit.get('frozen_contract_sha256')!=science.sha((out/CONTRACT).read_bytes())
        or audit.get('only_changed_scientific_field')!='successor.family'
        or audit.get('strict_original_c2_validation_replayed') is not True
        or audit.get('scientific_novelty_certified') is not False
        or context['patched']!=submission):
        raise ValueError('C2 family recovery provenance invalid')
    result=_validate_submission(contract,submission)
    if (report.get('status')!=result['status']
        or report.get('contract_sha256')!=science.sha((out/CONTRACT).read_bytes())
        or report.get('submission_sha256')!=science.sha((out/RAW).read_bytes())
        or report.get('successor_problem_id')!=submission['successor']['problem_id']):
        raise ValueError('Missing/mismatched C2 prospective validation')
    succ=submission['successor']
    # This is a narrowly scoped, preregistered scientific test, not an all-domain simulator.
    if succ['problem_id']!='m14gap:feceb611df04bd85':
        raise ValueError('This C3 v1 model applies only to the selected SERS-SEIRA adsorbate-population problem')
    card=context['selected']
    card_words=(str(card.get('scientific_question',''))+' '+str(card.get('title',''))).lower()
    if not ('sers' in card_words and 'seira' in card_words and 'adsor' in card_words):
        raise ValueError('Selected scientific card does not match the preregistered problem')
    if not (succ.get('novelty_certified') is False and
            'single' in succ['mechanism_signature']['strongest_null'].lower()):
        raise ValueError('Original C2 successor lacked a strong single-population null')
    return {'problem_id':succ['problem_id'],'candidate_scientific_null':succ['mechanism_signature']['strongest_null'],
      'candidate_prediction':succ['mechanism_signature']['predicted_observable'],
      'source_files_sha256':{p:science.sha((out/p).read_bytes()) for p in
          (CONTRACT,RAW,REPORT,AUDIT,ORIGINAL,'m14c2_successor_INTENT.json','M14C2_GENERATION_PROMPT.json')},
      'original_source_receipt_verified':True}

def code_sha():return science.sha(Path(science.__file__).read_bytes())

def cmd_prepare(args):
    c2=args.c2_dir.resolve();binding=_validate_c2(c2)
    if args.out_dir.exists():raise FileExistsError('C3 output directory must be new')
    plan={'schema':science.SCHEMA,'status':'M14C3_PREREGISTERED_WITH_VERIFIED_C2_SOURCE',
       'c2_dir':str(c2),'bound_c2':binding,
       'challenge_code_sha256':code_sha(),
       'potential_grid':science.POTENTIAL,'site_blocker_states':science.BLOCKER,
       'calibration_bounds':science.BOUNDS,
       'modalities':science.MODALITIES,
       'synthetic_model_parameters':science.RESPONSES,
       'challenge_scope':'Analytic observation equivalence under context-dependent single-population response and conditional calibration bounds',
       'NO_EMPIRICAL_EVIDENCE':True,'model_calls':0,'network_calls':0,
       'C2_proposed_polynomials_not_used_to_select_or_tune_null':True}
    args.out_dir.mkdir(parents=True,exist_ok=False)
    _once(args.out_dir/PLAN,science.canonical(plan))
    return {'status':plan['status'],'out_dir':str(args.out_dir),'original_C2_unchanged':True,'model_calls':0}

def _frozen(args):
    out=args.out_dir;plan=_read(out/PLAN)
    if plan.get('schema')!=science.SCHEMA:raise ValueError('Wrong C3 schema')
    if plan.get('challenge_code_sha256')!=code_sha():raise ValueError('C3 equation code changed since preregistration')
    source=_validate_c2(Path(plan['c2_dir']))
    if source!=plan['bound_c2']:raise ValueError('Original paid C2 science or provenance changed')
    return plan

def cmd_execute(args):
    plan=_frozen(args);out=args.out_dir
    rows,sweep,summary=science.calculate()
    files={OBS:science.rows_csv(rows),SWEEP:science.rows_csv(sweep),SUMMARY:science.canonical(summary)}
    # Fail before writing if a partial result exists; no silent overwrite.
    if any((out/n).exists() for n in (*files,RECEIPT,VERIFIED,POLICY)):
        raise FileExistsError('C3 execution artifacts already exist; verify/recover rather than overwrite')
    for name,raw in files.items():_once(out/name,raw)
    receipt={'schema':science.SCHEMA,'status':'M14C3_SYNTHETIC_CHALLENGE_EXECUTED',
      'preregistration_sha256':science.sha((out/PLAN).read_bytes()),
      'challenge_code_sha256':code_sha(),
      'outputs_sha256':{k:science.sha(v) for k,v in files.items()},
      'model_calls':0,'network_calls':0,'empirical_data':False,'novelty_certified':False}
    _once(out/RECEIPT,science.canonical(receipt))
    return {'status':receipt['status'],'signal_rows':len(rows),'calibration_scenarios':len(sweep),
      'flexible_null_max_abs_residual':summary['flexible_one_population_max_abs_residual'],
      'real_data_used':False,'model_calls':0}

def cmd_verify(args):
    plan=_frozen(args);out=args.out_dir;receipt=_read(out/RECEIPT)
    if (receipt.get('preregistration_sha256')!=science.sha((out/PLAN).read_bytes())
        or receipt.get('challenge_code_sha256')!=code_sha()
        or receipt.get('status')!='M14C3_SYNTHETIC_CHALLENGE_EXECUTED'):
        raise ValueError('C3 execution receipt invalid')
    rows,sweep,summary=science.calculate()
    outputs={OBS:science.rows_csv(rows),SWEEP:science.rows_csv(sweep),SUMMARY:science.canonical(summary)}
    if receipt.get('outputs_sha256')!={k:science.sha(v) for k,v in outputs.items()}:
        raise ValueError('Output hashes do not match receipt')
    for name,data in outputs.items():_same(out/name,data)
    if summary['flexible_one_population_null_excluded'] or not summary['synthetic_negative_control_passes_0_05_bound']:
        raise ValueError('Scientific inference gate failed; must not advance')
    certificate={'schema':science.SCHEMA,'status':'M14C3_NUMERICAL_COUNTERMODEL_REPRODUCED',
      'receipt_sha256':science.sha((out/RECEIPT).read_bytes()),
      'original_paid_source_c2_unchanged':True,'reproduced_output_hashes':receipt['outputs_sha256'],
      'external_empirical_observations':0,'novelty_certified':False}
    decision={'schema':science.SCHEMA,'status':'M14C3_PENDING_INDEPENDENT_RESPONSE_CALIBRATION',
      'source_receipt_sha256':certificate['receipt_sha256'],
      'action':'SUSPEND','action_scope':'THIS_SUCCESSOR_PROBLEM_ONLY',
      'basis':'Under unconstrained modality- and context-dependent optical/tensor response, a single population exactly reproduces the two-population synthetic observables; calibrated bounds are assumed, not measured.',
      'independent_evidence_required':summary['future_evidence_needed'],
      'does_not_falsify_two_population_mechanism':True,
      'does_not_prove_scientific_novelty':True,
      'not_a_general_SERS_identifiability_theorem':True,
      'external_empirical_observations':0,'model_calls':0}
    for name,value in ((VERIFIED,certificate),(POLICY,decision)):
        raw=science.canonical(value)
        if (out/name).exists():_same(out/name,raw)
        else:_once(out/name,raw)
    return {'status':certificate['status'],'successor_action':decision['action'],
      'conditional_strong_null_excluded':False,'empirical_novelty_certified':False,'new_model_calls':0}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('prepare','execute','verify-and-adjudicate'):
        sp=sub.add_parser(name);sp.add_argument('--out-dir',type=Path,required=True)
        if name=='prepare':sp.add_argument('--c2-dir',type=Path,required=True)
    args=p.parse_args(argv)
    result={'prepare':cmd_prepare,'execute':cmd_execute,'verify-and-adjudicate':cmd_verify}[args.cmd](args)
    print(json.dumps(result,indent=2,ensure_ascii=False));return result

if __name__=='__main__':main()
