"""Add-only scientific semantic recovery for E2 initial model (>= versus >).

This fork does NOT modify the existing trial, its tool_check, raw response, or
any W06-W08 episode. A source run with a completed initial response is pinned
read-only; the model's math is checked with an independently implemented CTMC.
The tool feedback is explicitly labelled external and never attributed to FAKG.
"""
from __future__ import annotations
import itertools
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from . import m19_e2_json_v2 as prior
from . import m19_e2_ctmc as calc
from . import research_work_v12_stable_runner as base

SCHEMA = 'm19-e2-semantic-fork-v1'
PHASES = ('control', 'tool')
REQUIRED_ALIASES = {
    'on_rates': 'on_rates',
    'off_rates': 'off_rates',
    'intensity_weights': 'weights',
    'total_intensity_threshold': 'threshold',
}


def guard(out: Path) -> Path:
    repo=prior.repo_root()
    root=(repo/'experiments'/'m19_e2'/'runs').resolve()
    out=out.expanduser().resolve()
    if root not in out.parents:raise ValueError('OUTPUT_OUTSIDE_REPO_M19_E2_RUNS')
    return repo


def _source(source:Path)->dict:
    guard(source)
    v=prior.verify(source)
    if v['phases'] != {'initial':'COMPLETE'} or v['attempt_count']!=1:
        raise ValueError('SOURCE_NOT_SINGLE_COMPLETED_INITIAL')
    if v['tool_check'] != 'UNSUPPORTED_MODEL_SPEC' or v['report_exists']:
        raise ValueError('SOURCE_NOT_ORIGINAL_UNSUPPORTED_STATE')
    files={x:base.file_sha(source/x) for x in (
        'lock.json','initial/request.json','initial/attempt.json',
        'initial/raw_response.json','initial/normalized.json','initial/receipt.json','tool_check.json')}
    return {'source_path':str(source.resolve()), 'files_sha256':files,
            'initial':base.read(source/'initial'/'normalized.json')}


def canonicalize_spec(initial:dict)->dict:
    spec=initial.get('proposed_calculator_spec')
    if not isinstance(spec,dict):raise ValueError('NO_CALCULATOR_SPEC')
    if set(spec) != set(REQUIRED_ALIASES):raise ValueError('UNEXPECTED_MODEL_SPEC_KEYS')
    assumptions=initial.get('model_and_assumptions') or []
    # No assumption inferred from the experiment code or human-written counterexample:
    # it must occur explicitly in the ORIGINAL model's saved text.
    if not any('bright state by I(x)≥T' in x or 'bright state by I(x)>=T' in x for x in assumptions):
        raise ValueError('NO_EXPLICIT_GE_BRIGHT_RULE_IN_ORIGINAL_MODEL')
    canonical={dst:spec[src] for src,dst in REQUIRED_ALIASES.items()}
    return calc.normalize_spec(canonical)


def checked_calculation(initial:dict)->dict:
    spec=canonicalize_spec(initial)
    weights=spec['weights'];threshold=spec['threshold']
    intensities=sorted(set(sum(weights[i]*bits[i] for i in range(len(weights)))
        for bits in itertools.product([0,1],repeat=len(weights))))
    dark=[v for v in intensities if v<threshold]
    bright=[v for v in intensities if v>=threshold]
    if not dark or not bright:raise ValueError('NO_BOTH_BRIGHT_AND_DARK_STATES_FOR_GE_RULE')
    # Existing math engine intentionally uses strict '>'. An equivalent strict
    # boundary is the midpoint of two neighboring attainable intensities.
    # Verify equivalence exhaustively over ALL finite state configurations.
    effective=(max(dark)+min(bright))/2
    if not 0<effective<sum(weights):raise ValueError('NONREPRESENTABLE_EFFECTIVE_THRESHOLD')
    if any((value>=threshold)!=(value>effective) for value in intensities):
        raise ValueError('THRESHOLD_REPRESENTATION_MISMATCH')
    check=calc.evaluate({**spec,'threshold':effective})
    result={
        'status':'EXACT_CTMC_RESULT_AFTER_EXPLICIT_ALIAS_AND_THRESHOLD_ADAPTER',
        'attribution':'EXTERNAL_DETERMINISTIC_CALCULATOR_NOT_FAKG_DISCOVERY',
        'initial_model_rule':'I(x) >= T',
        'initial_model_spec_original':initial['proposed_calculator_spec'],
        'canonical_model_spec':spec,
        'adapter':{
            'aliases':REQUIRED_ALIASES,
            'original_comparison':'>=',
            'calculator_comparison':'>',
            'effective_strict_threshold':effective,
            'explanation':'The finite attainable-intensity states are exhaustively checked: I>=T and I>effective_threshold select exactly the same configurations.',
        },
        'result':check,
        'scope':'Toy independent-site CTMC only; NOT validation of the Ye SERS mechanism.',
    }
    return result


def prepare(out:Path,source:Path)->dict:
    guard(out);guard(source)
    if out.exists():raise FileExistsError('FORK_ALREADY_EXISTS')
    s=_source(source)
    feedback=checked_calculation(s['initial'])
    files_sha=s.pop('files_sha256')
    s.pop('initial')
    lock={'schema':SCHEMA,'source_path':s['source_path'],
          'source_files_sha256':files_sha,
          'runner_sha256':base.file_sha(Path(__file__)),
          'previous_runner_sha256':base.file_sha(Path(prior.__file__)),
          'calculator_sha256':base.file_sha(Path(calc.__file__)),
          'base_sha256':base.file_sha(Path(base.__file__)),
          'max_new_paid_attempts':2,'scientific_validation':False,
          'origin_note':'Post-hoc, explicitly documented repair of field aliases and >= versus > model semantics; does not convert the original UNSUPPORTED result into historical success.'}
    out.mkdir(parents=True,exist_ok=False)
    base.write_new(out/'lock.json',lock)
    base.write_new(out/'external_calculation.json',feedback)
    verify(out)
    return {'status':'PREPARED_FROM_SAVED_INITIAL_NO_CALL','out':str(out),
      'source':str(source.resolve()),'calculation_status':feedback['status'],
      'initial_original_raw_sha256':files_sha['initial/raw_response.json']}


def _load(out:Path)->dict:return base.read(out/'lock.json')


def _request(out:Path,phase:str)->dict:
    if phase not in PHASES:raise ValueError('UNKNOWN_PHASE')
    lock=_load(out)
    source=Path(lock['source_path'])
    original=base.read(source/'initial'/'normalized.json')
    source_lock=base.read(source/'lock.json')
    feedback=base.read(out/'external_calculation.json')
    # The first research answer is byte-for-byte the same on both arms; only
    # the independently computed constraint is present in the tool arm.
    if phase=='control':
        tooldata={'status':'NO_EXTERNAL_MATH_FEEDBACK_IN_CONTROL_ARM',
                  'result':None}
    else:tooldata=feedback
    return prior._follow_request(source_lock,original,tooldata,phase)


def verify(out:Path)->dict:
    guard(out)
    lock=_load(out)
    if lock['schema']!=SCHEMA:raise ValueError('SCHEMA_MISMATCH')
    for key,p in (('runner_sha256',Path(__file__)),('previous_runner_sha256',Path(prior.__file__)),
                  ('calculator_sha256',Path(calc.__file__)),('base_sha256',Path(base.__file__))):
        if lock[key]!=base.file_sha(p):raise ValueError('LOCKED_SOURCE_CHANGED: '+key)
    source=Path(lock['source_path']);s=_source(source)
    if lock['source_files_sha256']!=s['files_sha256']:
        raise ValueError('ORIGINAL_SOURCE_EPISODE_CHANGED')
    if base.read(out/'external_calculation.json')!=checked_calculation(s['initial']):
        raise ValueError('EXTERNAL_CALCULATION_TAMPERED')
    statuses={};completed={}
    for phase in PHASES:
        d=out/phase
        if not d.exists():continue
        if not (d/'request.json').exists() or not (d/'attempt.json').exists():
            raise ValueError('INCOMPLETE_ATTEMPT_METADATA')
        req=_request(out,phase)
        if base.read(d/'request.json')!=req:raise ValueError('REQUEST_TAMPERED')
        if base.read(d/'attempt.json').get('request_sha256')!=base.sha(req):
            raise ValueError('ATTEMPT_TAMPERED')
        if not (d/'raw_response.json').exists():
            if (d/'normalized.json').exists() or (d/'receipt.json').exists():raise ValueError('NO_RAW_WITH_RESULT')
            statuses[phase]='UNKNOWN_BILLING_NO_RAW_NO_RETRY';continue
        if not (d/'normalized.json').exists() or not (d/'receipt.json').exists():
            statuses[phase]='SAVED_RAW_OFFLINE_RECOVER_REQUIRED';continue
        obj=prior._parse(base.read(d/'raw_response.json')['raw_text'],phase)
        if base.read(d/'normalized.json')!=obj:raise ValueError('NORMALIZED_TAMPERED')
        expected={'request_file_sha256':base.file_sha(d/'request.json'),
          'attempt_file_sha256':base.file_sha(d/'attempt.json'),
          'raw_file_sha256':base.file_sha(d/'raw_response.json'),
          'normalized_file_sha256':base.file_sha(d/'normalized.json'),
          'response_id':base.read(d/'raw_response.json').get('response_id')}
        if base.read(d/'receipt.json')!=expected:raise ValueError('RECEIPT_TAMPERED')
        statuses[phase]='COMPLETE';completed[phase]=obj
    if (out/'final_report.json').exists():
        if set(completed)!=set(PHASES):raise ValueError('REPORT_WITHOUT_COMPLETED_ARMS')
        if base.read(out/'final_report.json')!=_report(out):raise ValueError('REPORT_TAMPERED')
    return {'status':'VERIFIED_NO_CALL','source_initial':'PINNED_AND_UNCHANGED',
        'external_calculation_status':'EXACT_CTMC_RESULT',
        'phases':statuses,'new_attempts':len(statuses),
        'original_attempts':1,'total_attempts_across_trials':1+len(statuses),
        'scientific_validation':False,'report_exists':(out/'final_report.json').exists()}


def preview(out:Path,phase:str)->dict:
    v=verify(out)
    if phase not in PHASES:raise ValueError('UNKNOWN_PHASE')
    if phase in v['phases']:raise ValueError('PHASE_ALREADY_ATTEMPTED_NO_RETRY')
    if any(z!='COMPLETE' for z in v['phases'].values()):
        raise ValueError('UNRESOLVED_ATTEMPT_BLOCKS_NEW_CALLS')
    if v['report_exists']:raise ValueError('FINAL_REPORT_ALREADY_EXISTS')
    req=_request(out,phase)
    prior._assert_json_mode_ready(req)
    return {'status':'PREVIEW_NO_CALL','phase':phase,'request_sha256':base.sha(req),
            'json_mode_ready':True,'model':req['model']}


def _commit(out:Path,phase:str)->dict:
    d=out/phase;row=base.read(d/'raw_response.json')
    normalized=prior._parse(row['raw_text'],phase)
    if not (d/'normalized.json').exists():base.write_new(d/'normalized.json',normalized)
    elif base.read(d/'normalized.json')!=normalized:raise ValueError('NORMALIZED_DIFFERS')
    receipt={'request_file_sha256':base.file_sha(d/'request.json'),
          'attempt_file_sha256':base.file_sha(d/'attempt.json'),
          'raw_file_sha256':base.file_sha(d/'raw_response.json'),
          'normalized_file_sha256':base.file_sha(d/'normalized.json'),
          'response_id':row.get('response_id')}
    if not (d/'receipt.json').exists():base.write_new(d/'receipt.json',receipt)
    elif base.read(d/'receipt.json')!=receipt:raise ValueError('RECEIPT_DIFFERS')
    return normalized


def next_step(out:Path,phase:str,*,allow_paid:bool=False,max_new_paid_attempts:int=2,backend=None)->dict:
    if not 1<=max_new_paid_attempts<=2:raise ValueError('PAID_CEILING_1_TO_2_REQUIRED')
    pv=preview(out,phase)
    if verify(out)['new_attempts']>=max_new_paid_attempts:
        raise ValueError('PAID_CEILING_REACHED')
    base._require_paid(backend,allow_paid,1)
    req=_request(out,phase)
    d=out/phase;d.mkdir(parents=True,exist_ok=False)
    base.write_new(d/'request.json',req)
    base.write_new(d/'attempt.json',{'request_sha256':base.sha(req),
      'started_utc':datetime.now(timezone.utc).isoformat(),
      'status':'ATTEMPT_STARTED_BILLING_UNKNOWN_UNTIL_RAW'})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(d/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:_commit(out,phase)
    except Exception as e:raise RuntimeError('SAVED_RAW_RECOVER_OFFLINE_NO_RECALL: '+str(d)) from e
    return {'status':'COMMITTED','phase':phase,'new_paid_attempts_total':verify(out)['new_attempts']}


def recover(out:Path)->dict:
    v=verify(out);missing=[p for p,s in v['phases'].items() if s!='COMPLETE']
    if not missing:return {'status':'NOTHING_TO_RECOVER_NO_CALL'}
    if len(missing)!=1 or v['phases'][missing[0]]!='SAVED_RAW_OFFLINE_RECOVER_REQUIRED':
        raise ValueError('UNKNOWN_BILLING_OR_AMBIGUOUS_RECOVERY_REFUSED')
    _commit(out,missing[0]);verify(out)
    return {'status':'RECOVERED_SAVED_RAW_NO_CALL','phase':missing[0]}


def _report(out:Path)->dict:
    lock=_load(out)
    return {'schema':SCHEMA,'original_source':lock['source_path'],
        'original_raw_file_sha256':lock['source_files_sha256']['initial/raw_response.json'],
        'external_calculation':base.read(out/'external_calculation.json'),
        'responses':{p:{'response':base.read(out/p/'normalized.json'),
                   'raw_file_sha256':base.file_sha(out/p/'raw_response.json'),
                   'request_file_sha256':base.file_sha(out/p/'request.json')}
                   for p in PHASES},
        'scientific_validation':False,
        'interpretation':'Post-hoc semantic repair paired CONTROL/TOOL follow-up on one unchanged paid initial result. Evaluates external deterministic feedback effect only; not autonomous Level II discovery or preregistered confirmation.'}


def finalize(out:Path)->dict:
    v=verify(out)
    if v['report_exists']:raise FileExistsError('ALREADY_FINALIZED')
    if set(v['phases'])!=set(PHASES) or any(x!='COMPLETE' for x in v['phases'].values()):
        raise ValueError('BOTH_PAIRED_FOLLOWUPS_REQUIRED')
    base.write_new(out/'final_report.json',_report(out))
    verify(out)
    return {'status':'FINALIZED_NO_CALL','scientific_validation':False}
