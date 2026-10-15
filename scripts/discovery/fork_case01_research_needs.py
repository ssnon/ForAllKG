"""Fork a verified, quiescent Case01 V2 episode with opt-in research-need context.

No paid calls; original episode and its frozen scientific history remain untouched.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import runner as r

MODE = 'open_gap_resolution_v1'


def fork(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and destination must be distinct non-nested research directories')
    if destination.exists():
        raise FileExistsError('Destination exists; will not overwrite an episode')
    check = r.verify(source)
    manifest = check['manifest']
    state = check['state']
    if manifest['schema_version'] != r.V2_MANIFEST:
        raise ValueError('this fork supports only a verified mechanistic (V2) episode')
    if manifest.get('research_decision_mode'):
        raise ValueError('source already carries an experimental research decision mode')
    if state['ended'] or state['step'] >= manifest['max_steps']:
        raise ValueError('source has no remaining research actions')
    next_call = source / 'calls' / f"step_{state['step']+1:02d}"
    if next_call.exists():
        raise RuntimeError('source has an unresolved next-step call; recover/examine before forking')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fakg_gap_fork_', dir=str(destination.parent)) as scratch:
        stage = Path(scratch)/'episode'
        shutil.copytree(source, stage, symlinks=False)
        clone_manifest = r.load(stage/'manifest.json')
        clone_manifest['research_decision_mode'] = MODE
        clone_manifest['fork_provenance'] = {
            'kind': 'FORK_FROM_VERIFIED_EPISODE_NOT_INDEPENDENT_DISCOVERY',
            'source_step': state['step'],
            'source_state_sha256': r.digest(state),
            'source_manifest_sha256': r.digest(manifest),
            'source_evidence_sha256': manifest['source_snapshot_sha256'],
            'original_frozen_research_steps_preserved': True,
            'prompt_intervention_only_for_subsequent_calls': True,
        }
        (stage/'manifest.json').write_text(json.dumps(clone_manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        clone = r.verify(stage)
        if r.digest(clone['state']) != r.digest(state):
            raise AssertionError('fork unexpectedly changed research memory')
        need = r.prompt_payload(clone['state'],clone['source_index'],clone_manifest)
        if 'open_gap_resolution' not in need['research_decision_context']:
            raise AssertionError('fork does not activate intended decision context')
        if destination.exists():
            raise FileExistsError('destination created concurrently; refusing overwrite')
        os.rename(stage, destination)
    # Validate both the untouched original and the fork after atomic publication.
    if r.digest(r.verify(source)['state']) != r.digest(state):
        raise AssertionError('source episode changed during fork operation')
    r.verify(destination)
    return {'status':'FORKED_NO_PAID_CALL','source_step':state['step'],
            'next_step':state['step']+1,'destination':str(destination),
            'decision_mode':MODE,'source_state_sha256':r.digest(state)}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-episode',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(fork(args.from_episode,args.out),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
