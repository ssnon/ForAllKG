"""Fork verified GAP Step 8 research with opt-in replay-safe learning projection.

Never alters the source episode. No paid calls. Existing completed action rows and
state hashes are copied without modification; only the fork's prompt is enhanced.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import runner as r

MODE = r.MEMORY_PROJECTION_MODE


def fork(source: Path, destination: Path) -> dict:
    source=source.resolve(strict=True)
    destination=destination.resolve(strict=False)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and destination must be distinct non-nested episodes')
    if destination.exists():
        raise FileExistsError('refusing to overwrite research episode')
    checked=r.verify(source)
    manifest=checked['manifest']
    state=checked['state']
    if manifest['schema_version'] != r.V2_MANIFEST:
        raise ValueError('only the V2 mechanistic episode is supported')
    if manifest.get('research_decision_mode') != 'open_gap_resolution_v1':
        raise ValueError('source must be the verified R02 GAP Fork decision mode')
    if manifest.get('research_memory_projection_mode'):
        raise ValueError('source already has a memory projection mode')
    if state['ended'] or state['step'] >= manifest['max_steps']:
        raise ValueError('no remaining research actions')
    if (source / 'calls' / f"step_{state['step']+1:02d}").exists():
        raise ValueError('source has an unresolved call; inspect and recover before forking')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.case01_memory_fork_',dir=str(destination.parent)) as work:
        stage=Path(work)/'episode'
        shutil.copytree(source,stage,symlinks=False)
        new_manifest=r.load(stage/'manifest.json')
        new_manifest['research_memory_projection_mode']=MODE
        new_manifest['research_memory_projection_provenance']={
            'scope':'UNVERIFIED_MODEL_RESEARCH_HISTORY_NOT_ADDITIONAL_SCIENTIFIC_EVIDENCE',
            'source_step':state['step'],
            'source_state_sha256':r.digest(state),
            'source_manifest_sha256':r.digest(manifest),
            'past_actions_unchanged':True,
            'new_calls_only':True,
        }
        (stage/'manifest.json').write_text(json.dumps(new_manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        clone=r.verify(stage)
        if r.digest(clone['state']) != r.digest(state):
            raise AssertionError('fork has altered verified research state')
        prompt=r.prompt_payload(clone['state'],clone['source_index'],clone['manifest'],episode_dir=stage)
        ledger=prompt['research_decision_context']['verified_prior_action_learning']
        if not ledger['entries']:
            raise ValueError('no previous scientific learning found in the supplied episode')
        # The target GAP Step 8 must include its contested scientific argument.
        if state['step']==8 and not any(entry['step']==8 and entry['original_action']=='CHALLENGE'
            for entry in ledger['entries']):
            raise AssertionError('missing GAP Step 8 scientific challenge learning')
        if destination.exists():
            raise FileExistsError('destination created concurrently')
        os.rename(stage,destination)
    if r.digest(r.verify(source)['state']) != r.digest(state):
        raise AssertionError('original research episode changed')
    r.verify(destination)
    return {'status':'FORKED_NO_PAID_CALL','from_step':state['step'],
            'next_step':state['step']+1,'source_state_sha256':r.digest(state),
            'destination':str(destination),'mode':MODE,
            'previous_insight_entries':len(ledger['entries'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-episode',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(fork(args.from_episode,args.out),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
