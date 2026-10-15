"""Opt-in research-space policy atop verified stable v1.3 event contracts.

Only NEW decision prompts change. No replay, Work, science-contract, or state
mutations are performed outside the original stable runner's validated events.
"""
from __future__ import annotations
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from . import research_work_v12_stable_runner as base
from . import research_work_v14_space_policy as policy_mod
from .research_work_v14_space_policy import (POLICY_ID, POLICY_SYSTEM_APPEND,
                                             decorate_decision_payload, digest)

LOCK_FILE = 'RESEARCH_SPACE_POLICY_LOCK.json'
PROVENANCE_FILE = 'RESEARCH_SPACE_POLICY_PROVENANCE.json'


def _expected_lock(c: dict) -> dict:
    return {'policy_id':POLICY_ID, 'policy_append_sha256':digest(POLICY_SYSTEM_APPEND),
            'policy_source_file_sha256':base.file_sha(Path(policy_mod.__file__)),
            'policy_runner_file_sha256':base.file_sha(Path(__file__)),
            'stable_runner_file_sha256':base.file_sha(Path(base.__file__)),
            'manifest_sha256':base.sha(c['manifest']),
            'source_snapshot_sha256':base.sha(c['sources'])}


def verify(out: Path) -> dict:
    c=base.verify(out)
    if base.read(out/LOCK_FILE)!=_expected_lock(c):
        raise ValueError('POLICY_LOCK_MISMATCH: do not call models or mutate state')
    prov=base.read(out/PROVENANCE_FILE)
    if (prov.get('policy_id')!=POLICY_ID or
        prov.get('source_snapshot_sha256')!=base.sha(c['sources']) or
        prov.get('model')!=c['manifest']['model'] or
        prov.get('max_decisions')!=c['manifest']['max_decisions']):
        raise ValueError('POLICY_PROVENANCE_MISMATCH')
    # The base verifier checks complete event and artifact integrity, but it
    # intentionally does not know which *decision policy* generated the prompt.
    # Check each COMMITTED Decision against the EXACT policy-specific prompt
    # reconstructed from the state immediately before that event. This catches
    # accidentally executing v1.3 decisions inside a v1.4 policy episode.
    prior=base.init_state(c['manifest']['question'])
    for index in range(1,c['state']['event_count']+1):
        event=base.read(out/'events'/f'event_{index:04d}.json')
        if event['kind']=='DECISION':
            call=out/'calls'/f'decision_{prior["decision_count"]+1:04d}'
            request_file=call/'request.json'
            expected=_make_call_request({'state':prior, 'sources':c['sources'],
                                         'manifest':c['manifest']},'decision')
            if not request_file.is_file() or base.read(request_file)!=expected:
                raise ValueError('MIXED_OR_CHANGED_DECISION_POLICY: '+str(request_file))
        prior=base.apply_event(prior,event['kind'],event['payload'],
                               manifest=c['manifest'],sources=c['sources'])
    return c


def seed_fresh(*, reference: Path, out: Path) -> dict:
    """Fresh, ZERO imported Decision/Work/Idea, exact frozen source and settings."""
    source=base.verify(reference)
    m=source['manifest']; s=source['sources']
    if not source['state']['ended'] or source['state']['decision_count']!=m['max_decisions']:
        raise ValueError('REFERENCE_NOT_COMPLETED_BASELINE: use a frozen completed W03F')
    if out.exists():
        raise FileExistsError(f'episode already exists: {out}')
    state=base.init_state(m['question'])
    prov={'policy_id':POLICY_ID,'origin_reference':str(reference.resolve()),
          'reference_verified_state_sha256':base.sha(source['state']),
          'reference_decision_count':source['state']['decision_count'],
          'source_snapshot_sha256':base.sha(s),
          'manifest_sha256':base.sha(m),
          'model':m['model'],'max_decisions':m['max_decisions'],
          'comparison_note':('Same immutable Ye sources and model settings; fresh research history. '
                              'Policy prompt differs, so no claim of controlled scientific superiority.'),
          'imported_ideas':0,'imported_work_products':0,'new_paid_calls':0}
    # Creation is exclusive, so no existing episode can be overwritten.
    out.mkdir(parents=True, exist_ok=False)
    try:
        base.write_new(out/'manifest.json',copy.deepcopy(m))
        base.write_new(out/'manifest_lock.json',base.read(reference/'manifest_lock.json'))
        base.write_new(out/'sources.json',copy.deepcopy(s))
        base.write_new(out/'state.json',state)
        base.write_new(out/LOCK_FILE,_expected_lock(source))
        base.write_new(out/PROVENANCE_FILE,prov)
        verify(out)
    except BaseException:
        # Leave an explicitly detectable incomplete seed rather than wiping
        # potentially important files or unknown user modifications.
        raise
    return {'status':'NEW_DISCOVERY_SPACE_EPISODE_PREPARED_NO_PAID_CALLS',
            'out':str(out),'state_sha256':base.sha(state),
            'source_sha256':base.sha(s),'max_decisions':m['max_decisions'],
            'policy_id':POLICY_ID,'reference_unchanged':True}


def decision_payload(c: dict) -> dict:
    return decorate_decision_payload(base.decision_payload(c),c['state'],c['manifest']['max_decisions'])


def _make_call_request(c: dict, role: str) -> dict:
    request=base._make_call_request(c,role)
    if role=='decision':
        request['messages'][0]['content'] += '\n' + POLICY_SYSTEM_APPEND
        request['messages'][1]['content']=json.dumps(decision_payload(c),ensure_ascii=False,sort_keys=True)
    return request


def _commit_decision(out: Path, path: Path) -> dict:
    c=verify(out)
    saved_request=base.read(path/'request.json')
    if (saved_request!=_make_call_request(c,'decision') or
        base.read(path/'attempt.json')['request_sha256']!=base.sha(saved_request)):
        raise ValueError('POLICY_DECISION_REQUEST_REPLAY_MISMATCH: no model re-call')
    response=base.read(path/'raw_response.json')
    normalized=base._normalize_future_decision_references(
        base._raw_obj(response['raw_text']),c['state'])
    dec=base.ResearchDecision.model_validate(normalized)
    d=dec.model_dump(mode='json')
    # Validate FULL transition before creating an artifact/event.
    base.apply_event(c['state'],'DECISION',d,manifest=c['manifest'],sources=c['sources'])
    if dec.work_request is not None:
        wid=base.work_id_for(c['state'],dec.work_request)
        rp=out/'work'/wid/'request.json'
        if rp.exists():
            if base.read(rp)!=dec.work_request.model_dump(mode='json'):
                raise ValueError('persisted work request mismatch')
        else:
            base.write_new(rp,dec.work_request.model_dump(mode='json'))
    row=base._record_event(out,'DECISION',d,
        links={str((path/'request.json').relative_to(out)):base.file_sha(path/'request.json'),
               str((path/'raw_response.json').relative_to(out)):base.file_sha(path/'raw_response.json')})
    if not (path/'receipt.json').exists():
        base.write_new(path/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL',
                   'event':row['index'],'response_id':response.get('response_id')})
    return row


def decision(out: Path, *, allow_paid=False, max_new_calls=1, backend=None) -> dict:
    c=verify(out)
    req=_make_call_request(c,'decision')
    base._require_paid(backend,allow_paid,max_new_calls)
    call_dir=base._decision_call_dir(out,c['state'])
    if call_dir.exists():
        raise RuntimeError(f'existing decision attempt {call_dir}; recover, NEVER repeat paid call')
    call_dir.mkdir(parents=True,exist_ok=False)
    base.write_new(call_dir/'request.json',req)
    base.write_new(call_dir/'attempt.json',{'request_sha256':base.sha(req),
      'status':'STARTED_UNKNOWN_UNTIL_RECEIPT','created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(call_dir/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        event=_commit_decision(out,call_dir)
    except Exception as exc:
        raise RuntimeError(f'saved discovery-policy decision needs offline review: {call_dir}') from exc
    return {'status':'DECISION_COMMITTED','event':event['index'],
            'decision_count':verify(out)['state']['decision_count'],
            'new_paid_calls_this_command':1 if backend is None else 0}


def execute(out: Path, *, allow_paid=False, max_new_calls=1, backend=None) -> dict:
    verify(out)
    return base.execute(out,allow_paid=allow_paid,max_new_calls=max_new_calls,backend=backend)


def recover(out: Path) -> dict:
    # Never recover an unverified or wrong-policy episode as a policy episode.
    c=verify(out)
    p=c['state']['pending_work']
    if p and p['status']=='REQUESTED':
        return base.recover(out) # Work prompts intentionally unchanged.
    if c['state']['ended']:
        raise RuntimeError('episode already ended')
    path=base._decision_call_dir(out,c['state'])
    if not (path/'raw_response.json').is_file():
        raise RuntimeError('no saved response; do not repeat uncertain model call')
    event=_commit_decision(out,path)
    return {'status':'RECOVERED_POLICY_DECISION_NO_CALL','event':event['index']}


def inspect(out: Path) -> dict:
    verify(out)
    info=base.inspect(out)
    info['policy_id']=POLICY_ID
    return info
