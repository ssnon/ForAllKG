"""Explicit, append-only compatibility for one saved M21 Scientific Step 1.

A model's unsupported DERIVATION_PROPOSED label may be conservatively interpreted
as INCONCLUSIVE *without inventing or changing any scientific explanation*.
Only the exact saved Step 1 raw text is eligible. The paid raw and original run
remain frozen, and the reclassification is described in a provenance record.
All further model outputs use the unchanged strict parser.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

from pipeline_core.discovery.scientific_research_control import scientific_execution_work as work
from scripts.discovery import run_ctmc_group_replay as group

ORIGINAL_PARSE = work.parse

PROTOCOL = 'm21-step1-explicit-epistemic-downgrade-v1'
AUDIT_NAME = 'm21_step1_conservative_recovery.json'
RAW_STEP = 'step_0001'


def h(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def htext(raw: str) -> str:
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def strict_diagnosis(raw: str) -> dict:
    """No general JSON repair; require precisely one valid Step after conservative relabel."""
    try:
        ORIGINAL_PARSE(raw)
    except ValueError:
        pass
    else:
        raise ValueError('RAW_IS_STRICTLY_VALID_USE_NATIVE_RECOVER')
    try:
        obj=json.loads(raw)
    except (json.JSONDecodeError,TypeError) as exc:
        raise ValueError('NOT_A_SINGLE_JSON_OBJECT_DO_NOT_REWRITE') from exc
    if not isinstance(obj,dict):raise ValueError('NOT_A_SINGLE_JSON_OBJECT_DO_NOT_REWRITE')
    inner=obj.get('scientific_work_content')
    if not isinstance(inner,dict) or inner.get('work_conclusion')!='DERIVATION_PROPOSED':
        raise ValueError('UNSUPPORTED_REPAIR_NOT_DERIVATION_PROPOSED')
    if inner.get('candidate_consequence') and inner.get('reasoning_or_operations'):
        raise ValueError('DERIVATION_FIELDS_EXIST_DIFFERENT_SCHEMA_ERROR')
    conservative=json.loads(raw)
    conservative['scientific_work_content']['work_conclusion']='INCONCLUSIVE'
    # Every other strict schema check, including allowed scientific tool shapes, stays intact.
    valid=work.Step.model_validate(conservative).model_dump(mode='json')
    return valid


@contextmanager
def recovery_scope(raw_text_hash: str):
    original_parse=work.parse

    def scoped_parse(raw: str):
        if htext(raw)==raw_text_hash:
            return strict_diagnosis(raw)
        return original_parse(raw)

    work.parse=scoped_parse
    try:
        with group.compatible_scope():
            yield
    finally:
        work.parse=original_parse


def check_audit(out: Path) -> dict:
    out=Path(out).expanduser().resolve()
    audit=work.read(out/AUDIT_NAME)
    if audit.get('protocol')!=PROTOCOL or audit.get('scientific_validation') is not False:
        raise ValueError('RECOVERY_AUDIT_PROTOCOL_MISMATCH')
    src=Path(audit['source_run']).resolve()
    if src==out or src in out.parents or out in src.parents:
        raise ValueError('RECOVERY_FORK_PATH_INVALID')
    first=out/RAW_STEP
    for label,path in [('source_raw',src/RAW_STEP/'raw_response.json'),
                       ('fork_raw',first/'raw_response.json')]:
        if h(path)!=audit['saved_raw_file_sha256']:
            raise ValueError('RECOVERY_ORIGINAL_RAW_CHANGED:'+label)
    if h(first/'normalized.json')!=audit['conservatively_normalized_sha256']:
        raise ValueError('RECOVERY_NORMALIZATION_CHANGED')
    if h(out/'scientific_context.json')!=audit['frozen_context_sha256']:
        raise ValueError('RECOVERY_CONTEXT_CHANGED')
    if h(first/'request.json')!=audit['original_step_request_sha256']:
        raise ValueError('RECOVERY_FIRST_REQUEST_CHANGED')
    source_status=work.verify(src)
    if source_status['steps']!=0 or source_status['incomplete'] is None:
        raise ValueError('RECOVERY_ORIGINAL_IS_NOT_STILL_INCOMPLETE')
    raw=work.read(first/'raw_response.json')['raw_text']
    if htext(raw)!=audit['saved_raw_text_sha256']:
        raise ValueError('RECOVERY_RAW_TEXT_CHANGED')
    # Only semantic correction: unsupported derivation -> explicit inconclusive.
    derived=strict_diagnosis(raw)
    if work.read(first/'normalized.json')!=derived:
        raise ValueError('RECOVERY_SCIENTIFIC_FIELDS_CHANGED')
    return audit


def inspect(source: Path) -> dict:
    source=Path(source).expanduser().resolve()
    status=work.verify(source)
    if status['steps']!=0 or not status['incomplete']:
        raise ValueError('SOURCE_NOT_INCOMPLETE_FIRST_STEP')
    rawpath=source/RAW_STEP/'raw_response.json'
    if not rawpath.exists():raise ValueError('NO_SAVED_PAID_RAW_DO_NOT_RETRY')
    raw=work.read(rawpath)['raw_text']
    parsed=json.loads(raw)
    inner=parsed.get('scientific_work_content') or {}
    valid=strict_diagnosis(raw)
    return {'status':'M21_STEP1_RECOVERY_ELIGIBLE_NO_CALL',
      'source_untouched':True,'source_steps':0,
      'original_conclusion':inner.get('work_conclusion'),
      'has_candidate_consequence':bool(inner.get('candidate_consequence')),
      'has_reasoning_trace':bool(inner.get('reasoning_or_operations')),
      'conservative_conclusion':valid['scientific_work_content']['work_conclusion'],
      'raw_response_file_sha256':h(rawpath),
      'requires_new_paid_call':False}


def fork(source: Path, target: Path) -> dict:
    source=Path(source).expanduser().resolve()
    target=Path(target).expanduser().resolve()
    if target.exists():raise FileExistsError('NEW_FORK_TARGET_REQUIRED')
    if target==source or source in target.parents or target in source.parents:
        raise ValueError('FORK_PATHS_MUST_BE_SEPARATE')
    eligibility=inspect(source)
    src_raw=source/RAW_STEP/'raw_response.json'
    orig_hash=h(src_raw)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,target,symlinks=False)
    try:
        # No LLM invocation. A saved optional tool may be executed by native recover.
        with recovery_scope(htext(work.read(src_raw)['raw_text'])):
            result=work.recover(target)
            if result['step']!=1 or result['raw_response_unchanged'] is not True:
                raise RuntimeError('UNEXPECTED_RECOVERED_STATE')
        dst=target/RAW_STEP
        audit={
            'protocol':PROTOCOL,
            'source_run':str(source),
            'scientific_validation':False,
            'source_unchanged':True,
            'paid_calls_for_recovery':0,
            'reclassified_field':'scientific_work_content.work_conclusion',
            'original_label':'DERIVATION_PROPOSED',
            'conservative_label':'INCONCLUSIVE',
            'why':'The original model response omitted a candidate consequence and/or reasoning trace needed for DERIVATION_PROPOSED. This is a conservative classification only; no candidate, reasoning, or source claim is invented.',
            'saved_raw_file_sha256':orig_hash,
            'saved_raw_text_sha256':htext(work.read(src_raw)['raw_text']),
            'conservatively_normalized_sha256':h(dst/'normalized.json'),
            'frozen_context_sha256':h(target/'scientific_context.json'),
            'original_step_request_sha256':h(dst/'request.json'),
            'next_steps_require_this_compatibility_cli':True,
        }
        work.put(target/AUDIT_NAME,audit)
        check_audit(target)
        with recovery_scope(audit['saved_raw_text_sha256']):
            verified=work.verify(target)
        if h(src_raw)!=orig_hash:
            raise ValueError('PAID_RAW_CHANGED')
    except BaseException:
        shutil.rmtree(target)
        raise
    return {'status':'M21_STEP1_CONSERVATIVELY_RECOVERED_NO_PAID_CALL',
        'source_untouched':True,'steps':verified['steps'],
        'operation':result['operation'], 'next_action':result['next_action'],
        'tool_status':result['tool_status'], 'fork':str(target),
        'epistemic_downgrade':eligibility['original_conclusion']+' -> INCONCLUSIVE',
        'scientific_validation':False}


def run_compatible(command: str, out: Path, allow_paid: bool = False):
    out=Path(out).expanduser().resolve()
    audit=check_audit(out)
    with recovery_scope(audit['saved_raw_text_sha256']):
        if command=='verify':return work.verify(out)
        if command=='preview':return work.preview(out)
        if command=='trajectory':return work.trajectory(out)
        if command=='recover':return work.recover(out)
        if command=='next':return work.next_step(out,allow_paid=allow_paid)
        if command=='finish':return work.finish(out)
    raise ValueError('UNKNOWN_COMMAND')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','fork','verify','preview','next','recover','trajectory','finish'))
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--allow-paid',action='store_true')
    args=p.parse_args()
    if args.command=='inspect':
        if not args.source:p.error('--source required')
        result=inspect(args.source)
    elif args.command=='fork':
        if not args.source or not args.out:p.error('--source and --out required')
        result=fork(args.source,args.out)
    else:
        if not args.out:p.error('--out required')
        result=run_compatible(args.command,args.out,allow_paid=args.allow_paid)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=='__main__':main()
