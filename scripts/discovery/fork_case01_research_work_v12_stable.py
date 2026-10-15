"""Read-only source W03 -> new isolated Decision-7 branch, no paid calls.

The rejected Decision 7 remains in the original W03. The fork carries only the
committed event prefix through Decision 6 / completed Work 6 and records exact
provenance of the rejected response. It does not reinterpret or rewrite science.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as rw
from pipeline_core.discovery.scientific_research_control import research_work_v12_runner as legacy


def _raw_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fork(source: Path, target: Path) -> dict:
    if not source.is_dir() or target.exists():
        raise ValueError('source must exist and fork destination must NOT already exist')
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError('fork destination must not be inside the source episode')
    # Never copy external content through symlinks.
    for p in source.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'symlink in source episode: {p}')
    checked=rw.verify(source)
    state=checked['state']; pending=state['pending_work']
    if (state['decision_count'] != 6 or state['event_count'] != 12 or
            state['ended'] or pending is None or pending['status'] != 'EXECUTED' or
            pending['request']['work_kind'] != 'REASON'):
        raise ValueError('expected a verified post-Work6/pre-Decision7 state; no fork created')
    saved=source/'calls'/'decision_0007'/'raw_response.json'
    if not saved.is_file():
        raise ValueError('saved rejected Decision7 response missing; no fork created')
    raw=json.loads(saved.read_text(encoding='utf-8'))
    raw_obj=json.loads(raw['raw_text'])
    if raw_obj.get('interpretation', '__ABSENT__') is not None:
        raise ValueError('this fork tool only handles verified missing-interpretation Decision7')
    if raw_obj.get('action') not in ('DEEPEN', 'READ_EVIDENCE','CHALLENGE','EXPLORE','PIVOT','END'):
        raise ValueError('unexpected rejected response action')
    original=rw._decision_call_dir(source,state)
    if original != source/'calls'/'decision_0007':
        raise ValueError('unexpected raw response location')
    # Attest that the rejected raw was genuinely generated for the replayed
    # pre-Decision7 state under the *original* prompt, not an arbitrary file.
    old_request=rw.read(original/'request.json')
    old_attempt=rw.read(original/'attempt.json')
    if old_request != legacy._make_call_request(legacy.verify(source), 'decision'):
        raise ValueError('rejected response request does not match the original verified prompt')
    if old_attempt.get('request_sha256') != rw.sha(old_request):
        raise ValueError('rejected response attempt digest mismatch')
    fail_sha=_raw_sha(saved)
    # The source episode is never modified. The branch's new call slot must be
    # absent, or the original runner would treat it as an already-attempted call.
    def ignore(folder, children):
        if Path(folder) == source/'calls':
            return {'decision_0007'}
        return set()
    try:
        shutil.copytree(source, target, ignore=ignore, symlinks=False)
        if (target/'calls'/'decision_0007').exists():
            raise ValueError('rejected attempt accidentally copied into branch')
        branch=rw.verify(target)
        if rw.sha(branch['state']) != rw.sha(state):
            raise ValueError('fork state differs from verified source')
        key_paths=['manifest.json','manifest_lock.json','sources.json','state.json',
                   'events/event_0012.json',
                   f'work/{pending["work_id"]}/product.json',
                   f'work/{pending["work_id"]}/raw_response.json']
        for rel in key_paths:
            if _raw_sha(source/rel)!=_raw_sha(target/rel):
                raise ValueError(f'fork changed a protected historical file: {rel}')
        provenance={
            'lineage':'W03_DECISION_7_CONTRACT_STABILIZATION_FORK',
            'scientific_hypotheses_from_original': 'NONE_ADDED',
            'fork_boundary':'AFTER_COMPLETED_WORK6_BEFORE_REJECTED_DECISION7',
            'original_episode':str(source.resolve()),
            'original_verified_event_count':state['event_count'],
            'original_decision_count':state['decision_count'],
            'original_verified_state_sha256':rw.sha(state),
            'rejected_original_decision_7_raw_sha256':fail_sha,
            'rejected_original_decision_7_location':str(saved.resolve()),
            'branch_has_rejected_raw_response':False,
            'branch_uses_new_conditional_prompt_contract':True,
            'new_paid_calls':0,
            'non_scientific_provenance_note': 'No physical mechanism, WorkInterpretation or ScientificDelta was fabricated to repair the invalid original response.',
        }
        rw.write_new(target/'CONTRACT_STABILIZATION_PROVENANCE.json',provenance)
        return {'status':'VERIFIED_W03_FORK_CREATED_NO_PAID_CALLS',
                'fork':str(target), 'decision_count':state['decision_count'],
                'work_id':pending['work_id'], 'source_state_sha256':rw.sha(state),
                'rejected_decision_7_raw_sha256':fail_sha,
                'next_command':'Use run_case01_research_work_v12_stable decision ONLY in fork after reviewing preview'}
    except BaseException:
        if target.is_dir():
            shutil.rmtree(target)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True,help='original W03; never mutated')
    p.add_argument('--fork-out',type=Path,required=True,help='new episode directory')
    a=p.parse_args()
    print(json.dumps(fork(a.out,a.fork_out),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
