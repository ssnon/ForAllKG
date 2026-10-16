"""M19-E3 entry: an isolated, audited W08 Decision-4 research-choice fork.

This is an autonomous QUESTION-SELECTION entry test, not Deep Work validation.
No E0-E2 answers, evaluator notes, or forced mechanism are injected.
Uses the native W08 Decision and its recover/replay logic unchanged.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from . import research_work_v12_stable_runner as base
from . import research_work_v18_feedback_runner as v18

SCHEMA = 'm19-e3-native-research-control-entry-v1'
EXPECTED_W08_SHA = 'e85eb24b89864401a36b27bd5a659afcf81c5239db65a4bce3c5b678dae29a3b'
EXPECTED_WORK_3 = 'work_0003_cd32f5d823b4'
FORK_SUBDIR = 'native_w08_episode'
LOCK = 'm19_e3_provenance.json'


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def guard(out: Path) -> Path:
    # Exact realpath containment; reject root, paths outside checkout and symlinked parents.
    root = (repo_root()/'experiments'/'m19_e3'/'runs').resolve()
    dest = out.expanduser().resolve()
    if dest == root or root not in dest.parents:
        raise ValueError('E3_OUTPUT_MUST_BE_A_CHILD_OF_REPO_EXPERIMENTS_M19_E3_RUNS')
    return dest


def _tree_files(folder: Path) -> dict[str,str]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError('REFERENCE_MUST_BE_A_REAL_DIRECTORY')
    records: dict[str,str] = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('SYMLINK_IN_REFERENCE_REFUSED: '+str(path))
        if path.is_file():
            name=path.relative_to(folder).as_posix()
            records[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def _check_current_w08(source: Path) -> dict[str,Any]:
    c=v18.verify(source)
    s=c['state']
    digest=base.sha(s)
    p=s.get('pending_work') or {}
    if (digest!=EXPECTED_W08_SHA or s['decision_count']!=3 or s['event_count']!=6
       or s['ended'] or p.get('work_id')!=EXPECTED_WORK_3
       or p.get('execution_status')!='COMPLETED' or p.get('status')!='EXECUTED'):
        raise ValueError('W08_REFERENCE_IS_NOT_PINNED_COMPLETED_WORK3_PRE_DECISION4: '+digest)
    return c


def prepare(*,source:Path,out:Path)->dict:
    target=guard(out)
    source=source.expanduser().resolve(strict=True)
    if source==target or source in target.parents:
        raise ValueError('SOURCE_NESTED_WITHIN_DESTINATION')
    if target.exists():
        raise FileExistsError('E3_EPISODE_ALREADY_EXISTS_NEVER_OVERWRITE')
    if target.parent.exists() and target.parent.is_symlink():
        raise ValueError('SYMLINK_OUTPUT_PARENT_REFUSED')
    original=_check_current_w08(source)
    records=_tree_files(source)
    if not records or 'state.json' not in records:
        raise ValueError('MISSING_W08_EPISODE_FILES')
    # Clone into staging below the repository. An interrupted clone is never resumed blindly.
    staging=target.with_name(target.name+'.copying')
    if staging.exists():
        raise FileExistsError('INCOMPLETE_COPYING_DIRECTORY_REQUIRES_MANUAL_REVIEW')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging.mkdir(mode=0o700)
    try:
        cloned=staging/FORK_SUBDIR
        shutil.copytree(source,cloned,symlinks=True)
        if _tree_files(cloned)!=records:
            raise ValueError('W08_EPISODE_COPY_MISMATCH')
        if base.sha(v18.verify(cloned)['state'])!=EXPECTED_W08_SHA:
            raise ValueError('W08_CLONE_VERIFY_MISMATCH')
        provenance={
            'schema':SCHEMA,
            'original_w08_path':str(source),
            'original_w08_state_sha256':EXPECTED_W08_SHA,
            'w08_initial_files_sha256':base.sha(records),
            'w08_initial_file_shas':records,
            'v18_source_sha256':base.file_sha(Path(v18.__file__)),
            'base_source_sha256':base.file_sha(Path(base.__file__)),
            'e3_source_sha256':base.file_sha(Path(__file__)),
            'copy_only_prior_to_decision4':True,
            'prior_w06_w07_are_unverified_historical_memory':True,
            'e1_e2_evaluator_information_injected':False,
            'scientific_validation':False,
            'maximum_new_paid_decision_calls':1,
            'next_stage':'observe self-selected native Decision 4; separate Work execution is not yet authorized by this harness',
        }
        base.write_new(staging/LOCK,provenance)
        staging.rename(target)
    except Exception:
        # Do not destroy an interrupted evidence folder; manual review is required.
        raise
    verified=verify(target)
    return {'status':'PREPARED_PINNED_W08_FORK_NO_PAID',
            'out':str(target),'original_sha256':EXPECTED_W08_SHA,
            'fork_decision_count':verified['decision_count'],
            'new_paid_calls':0}


def verify(out:Path)->dict:
    target=guard(out)
    lock=base.read(target/LOCK)
    if lock['schema']!=SCHEMA or lock['original_w08_state_sha256']!=EXPECTED_W08_SHA:
        raise ValueError('INVALID_E3_LOCK')
    if lock['v18_source_sha256']!=base.file_sha(Path(v18.__file__)) or lock['base_source_sha256']!=base.file_sha(Path(base.__file__)) or lock['e3_source_sha256']!=base.file_sha(Path(__file__)):
        raise ValueError('DEPENDENCY_OR_E3_CODE_CHANGED')
    initial=lock['w08_initial_file_shas']
    if lock['w08_initial_files_sha256']!=base.sha(initial):
        raise ValueError('ORIGINAL_W08_MANIFEST_TAMPERED')
    original=Path(lock['original_w08_path'])
    if _tree_files(original)!=initial:
        raise ValueError('SOURCE_W08_CHANGED_SINCE_E3_FORK')
    if base.sha(v18.verify(original)['state'])!=EXPECTED_W08_SHA:
        raise ValueError('ORIGINAL_W08_STATE_CHANGED')
    episode=target/FORK_SUBDIR
    current_files=_tree_files(episode)
    for name,sha in initial.items():
        if name!='state.json' and current_files.get(name)!=sha:
            raise ValueError('FORK_FROZEN_FILE_MUTATED: '+name)
    c=v18.verify(episode)
    s=c['state'];n=s['decision_count'];events=s['event_count']
    if n not in (3,4) or events not in (6,7) or events!=n+3:
        raise ValueError('E3_SCOPE_ONLY_PRE_D4_OR_POST_D4')
    if n==3 and base.sha(s)!=EXPECTED_W08_SHA:
        raise ValueError('FORK_INITIAL_STATE_CHANGED')
    if n==4 and s['decisions'][-1]['number']!=4:
        raise ValueError('MISSING_NATIVE_DECISION4')
    if n==4:
        # Exact original D1-D3/W1-W3 remain protected by pinned copied artifacts.
        if 'events/event_0007.json' not in current_files:
            raise ValueError('DECISION4_EVENT_MISSING')
    d4_dir=episode/'calls'/'decision_0004'
    if n==3 and d4_dir.exists():
        if not (d4_dir/'request.json').is_file() or not (d4_dir/'attempt.json').is_file():
            raise ValueError('DECISION4_INCOMPLETE_METADATA')
        status='SAVED_RAW_REQUIRES_OFFLINE_RECOVER' if (d4_dir/'raw_response.json').exists() else 'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY'
    elif n==4:
        status='DECISION4_COMMITTED_VERIFIED'
    else:
        status='PREPARED_FOR_AUTONOMOUS_DECISION4'
    return {'status':status,'decision_count':n,'event_count':events,
            'w08_original_sha256':EXPECTED_W08_SHA,
            'fork_state_sha256':base.sha(s),
            'last_action':s['decisions'][-1]['action'] if s['decisions'] else None,
            'work_request':(s.get('pending_work') or {}).get('request') if n==4 else None,
            'pending_work_kind':(s.get('pending_work') or {}).get('request',{}).get('work_kind') if n==4 else None,
            'scientific_validation':False,'paid_calls_this_command':0}


def preview(out:Path)->dict:
    z=verify(out)
    if z['status']!='PREPARED_FOR_AUTONOMOUS_DECISION4':
        raise RuntimeError('DECISION4_ATTEMPT_OR_RESULT_ALREADY_EXISTS')
    request=v18._make_call_request(v18.verify(guard(out)/FORK_SUBDIR),'decision')
    if 'json' not in ' '.join(m['content'] for m in request['messages']).lower():
        raise ValueError('JSON_MODE_PROVIDER_REQUIREMENT_NOT_SATISFIED')
    # Do not print the full research context and citations; it is unchanged and can be large.
    return {'status':'PREVIEW_NO_CALL','next_decision':4,
        'model':request['model'],'request_sha256':base.sha(request),
        'json_mode_ready':True,'paid_calls_this_command':0,
        'e1_e2_external_answers_added':False}


def decision(out:Path,*,allow_paid:bool=False,backend=None)->dict:
    z=preview(out)
    episode=guard(out)/FORK_SUBDIR
    result=v18.decision(episode,allow_paid=allow_paid,max_new_calls=1,backend=backend)
    updated=verify(out)
    return {'status':'E3_NATIVE_AUTONOMOUS_DECISION4_COMMITTED',
        'action':updated['last_action'],'requested_work_kind':updated['pending_work_kind'],
        'fork_state_sha256':updated['fork_state_sha256'],
        'new_paid_calls_this_command':result.get('new_paid_calls_this_command',0)}


def recover(out:Path)->dict:
    z=verify(out)
    if z['status']!='SAVED_RAW_REQUIRES_OFFLINE_RECOVER':
        raise RuntimeError('NO_SAVED_DECISION4_RAW_AVAILABLE_FOR_RECOVERY')
    r=v18.recover(guard(out)/FORK_SUBDIR)
    verify(out)
    return {'status':'RECOVERED_DECISION4_FROM_SAVED_RAW_NO_CALL',
            'underlying_status':r['status'],'paid_calls_this_command':0}
