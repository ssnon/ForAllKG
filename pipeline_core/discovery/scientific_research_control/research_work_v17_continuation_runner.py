"""Add-only W07 scientific-continuation overlay. Only new episodes are affected.

Native v1.2 Decision/Work event contracts and W06 Seed transitions are reused.
Unverified W06 memory is imported as a LOCKED read-only previous-run advisory,
never as live current-episode ResearchIdea or Seed, and never as certification.
"""
from __future__ import annotations
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from . import research_work_v12_stable_runner as base
from . import research_work_v14_space_policy as p14
from . import research_work_v15_formation_policy as p15
from . import research_work_v16_seed_policy as p16
from . import research_work_v16_seed_runner as v16
from . import research_work_v17_continuation_policy as policy

POLICY_ID=policy.POLICY_ID
LOCK_FILE='SCIENTIFIC_CONTINUATION_POLICY_LOCK.json'
PROVENANCE_FILE='SCIENTIFIC_CONTINUATION_PROVENANCE.json'
PRIOR_FILE='PRIOR_W06_UNVERIFIED_RESEARCH.json'
SEED_SIDECAR='seed_decision.json'
FOCUS_SIDECAR='focus_decision.json'


def _lock(c:dict)->dict:
    from . import research_work_v12_contracts as contracts
    from . import research_work_v12_stable_tools as tools
    files=[base.__file__,contracts.__file__,tools.__file__,
           p14.__file__,p15.__file__,p16.__file__,v16.__file__,policy.__file__,__file__]
    return {'policy_id':POLICY_ID,'dependency_source_shas':{
        Path(x).name:base.file_sha(Path(x)) for x in files},
        'policy_text_sha256':base.sha(policy.POLICY_SYSTEM_APPEND),
        'manifest_sha256':base.sha(c['manifest']),
        'sources_sha256':base.sha(c['sources']),
        'prior_memory_sha256':base.sha(c['prior'])}


def _initial_focus(question:str)->dict:
    return {'decision_count':0,'current_question':question,'observation_refs':[],
            'causal_path':None,'blocking_unknown':None,'target_hypothesis_id':None,
            'latest_work_interpretation':None,'history':[],
            'scientific_validation':False}


def _focus_transition(before:dict, focus:dict, decision:dict, focus_update, after:dict, seeds:dict)->dict:
    f=copy.deepcopy(focus)
    n=before['decision_count']+1
    explicit=policy.ResearchFocus.model_validate(focus_update) if focus_update is not None else None
    u=explicit.model_dump(mode='json') if explicit else None
    if u:
        unknown_refs=set(u['observation_refs'])-set(before['read_ids'])
        if unknown_refs:raise ValueError('FOCUS_UNREAD_OBSERVATION_REF: '+str(sorted(unknown_refs)))
        target=u['target_hypothesis_id']
        if target and target not in before['idea_registry'] and target not in seeds['registry']:
            raise ValueError('FOCUS_NOT_A_CURRENT_EPISODE_HYPOTHESIS: '+target)
        f['observation_refs']=u['observation_refs']
        f['causal_path']=u['causal_path']
        f['blocking_unknown']=u['blocking_unknown']
        f['target_hypothesis_id']=target
    elif decision.get('work_request'):
        req=decision['work_request']
        # No inferred scientific claim. All text comes verbatim from the model's
        # validated WorkRequest. Never auto-link historical W06 identities.
        f['observation_refs']=[x for x in req['source_ids'] if x in before['read_ids']]
        f['blocking_unknown']=req['gap_or_assumption']
        f['causal_path']=None
        f['target_hypothesis_id']=req['target_idea_id']
    f['current_question']=u['question'] if u else decision['research_question_now']
    interp=decision.get('interpretation')
    if interp:
        f['latest_work_interpretation']={
            'work_id':interp['work_id'],'assessment':interp['assessment'],
            'what_changed':interp['what_changed'],'remaining_gap':interp['remaining_gap'],
            'scientific_delta_kind':(interp.get('scientific_delta') or {}).get('kind'),
            'unverified_model_interpretation':True}
    f['history'].append({'decision':n,'question':f['current_question'],
        'observation_refs':f['observation_refs'],'causal_path':f['causal_path'],
        'blocking_unknown':f['blocking_unknown'],'hypothesis_id':f['target_hypothesis_id'],
        'work_id':(after.get('pending_work') or {}).get('work_id'),
        'work_interpreted':(interp or {}).get('work_id'),
        'explicit_update':u is not None})
    f['decision_count']=n
    return f


def _summarize_prior(previous:dict)->dict:
    state=previous['state'];seed=previous['seed_state']
    interpretations=[]
    for x in state['interpretations']:
        interpretations.append({'work_id':x['work_id'],'assessment':x['assessment'],
            'what_changed':x['what_changed'],'remaining_gap':x['remaining_gap'],
            'scientific_delta':x.get('scientific_delta')})
    return {'source':'PREVIOUS_W06_COMPLETED_EPISODE_FAKG_GENERATED_ONLY',
      'epistemic_status':'ALL_SCIENTIFIC_IDEAS_UNVERIFIED',
      'not_imported_as_current_objects':True,
      'previous_state_sha256':base.sha(state),
      'decision_count':state['decision_count'],
      'final_disposition':state['final_disposition'],
      'research_end_reason':state['research_end_reason'],
      'previous_research_ideas':copy.deepcopy(state['idea_registry']),
      'previous_idea_history':copy.deepcopy(state['idea_history']),
      'previous_hypothesis_seeds':copy.deepcopy(seed['registry']),
      'previous_seed_history':copy.deepcopy(seed['history']),
      'previous_work_interpretations':interpretations,
      'previous_decisions':copy.deepcopy(state['decisions'])}


def prepare(*,reference:Path,out:Path)->dict:
    old=v16.verify(reference)
    state=old['state'];manifest=old['manifest'];sources=old['sources']
    if not state['ended'] or state['decision_count']!=manifest['max_decisions']:
        raise ValueError('W06_REFERENCE_MUST_BE_COMPLETED')
    if out.exists():raise FileExistsError('never overwrite old episode: '+str(out))
    previous=_summarize_prior(old)
    new_state=base.init_state(manifest['question'])
    out.mkdir(parents=True,exist_ok=False)
    base.write_new(out/'manifest.json',copy.deepcopy(manifest))
    base.write_new(out/'manifest_lock.json',base.read(reference/'manifest_lock.json'))
    base.write_new(out/'sources.json',copy.deepcopy(sources))
    base.write_new(out/'state.json',new_state)
    base.write_new(out/PRIOR_FILE,previous)
    base.write_new(out/LOCK_FILE,_lock({'manifest':manifest,'sources':sources,'prior':previous}))
    base.write_new(out/PROVENANCE_FILE,{'policy_id':POLICY_ID,
       'reference_episode_path':str(reference.resolve()),
       'verified_reference_state_sha256':base.sha(state),
       'prior_memory_sha256':base.sha(previous),
       'source_snapshot_sha256':base.sha(sources),
       'model':manifest['model'],'max_decisions':manifest['max_decisions'],
       'prior_ideas_imported_as_current':0,'prior_seeds_imported_as_current':0,
       'paid_calls_during_prepare':0,
       'comparison_note':'WARM START with FAKG-generated prior unverified findings. NOT a controlled improvement comparison.'})
    verify(out)
    return {'status':'NEW_SCIENTIFIC_CONTINUATION_PREPARED_NO_PAID_CALLS',
       'episode':str(out),'reference_unchanged':True,
       'prior_state_sha256':base.sha(state),'state_sha256':base.sha(new_state)}


def decision_payload(c:dict)->dict:
    payload=v16.decision_payload(c)
    payload['decision_schema']=policy.focus_extension_schema(payload['decision_schema'])
    payload['prior_unverified_research']=copy.deepcopy(c['prior'])
    payload['current_scientific_focus']=copy.deepcopy(c['focus_state'])
    payload['focus_contract']={'research_focus_update':'OPTIONAL, null or fully typed fields',
      'observation_refs':'Only previously READ source IDs in current episode; never invented refs',
      'historical_ids':'W06 IDs are NOT current registered IDs or allowable target_idea_id.',
      'hypothesis_registration':'Current episode ideas/seeds remain empty until independently proposed.',
      'purpose':'Choose a physically important causal unknown; Work result can change next question.',
      'mechanism_development_not_level_ii_certification':True,
      'do_not_require_computation_or_new_experiments':True}
    return payload


def _make_call_request(c:dict,role:str)->dict:
    req=base._make_call_request(c,role)
    if role=='decision':
        req['messages'][0]['content']+='\n'+policy.POLICY_SYSTEM_APPEND+('\nSTRUCTURED SEED FIELDS: Output seed_update and seed_work_target_id in EVERY Decision '
            '(either may be null). FORM only after an interpreted completed REASON; use exact supporting_work_id '
            'and supporting_product_sha256. To target an already registered current Seed, pass its ID in '
            'seed_work_target_id, cite the origin Work in available_inputs, include its ID in work_instructions, '
            'and set work_request.target_idea_id to null. DEVELOP/CHALLENGE/DISCARD/PROMOTE requires the '
            'preceding REASON Work to have been explicitly targeted to that Seed. ResearchIdea is separate; '
            'new IDs do not exist until commit. No required mutations or scientific certification.')
        req['messages'][1]['content']=json.dumps(decision_payload(c),ensure_ascii=False,sort_keys=True)
    return req


def verify(out:Path)->dict:
    c=base.verify(out)
    prior=base.read(out/PRIOR_FILE)
    if base.read(out/LOCK_FILE)!=_lock({'manifest':c['manifest'],'sources':c['sources'],'prior':prior}):
        raise ValueError('CONTINUATION_POLICY_LOCK_MISMATCH')
    prov=base.read(out/PROVENANCE_FILE)
    if (prov['policy_id']!=POLICY_ID or prov['prior_memory_sha256']!=base.sha(prior)
        or prov['verified_reference_state_sha256']!=prior['previous_state_sha256']
        or prov['source_snapshot_sha256']!=base.sha(c['sources'])
        or prov['model']!=c['manifest']['model'] or prov['max_decisions']!=c['manifest']['max_decisions']
        or prov['prior_ideas_imported_as_current']!=0 or prov['prior_seeds_imported_as_current']!=0):
        raise ValueError('CONTINUATION_PROVENANCE_MISMATCH')
    before=base.init_state(c['manifest']['question'])
    seeds=v16._blank_seeds();focus=_initial_focus(c['manifest']['question'])
    for index in range(1,c['state']['event_count']+1):
        event=base.read(out/'events'/f'event_{index:04d}.json')
        after=base.apply_event(before,event['kind'],event['payload'],manifest=c['manifest'],sources=c['sources'])
        if event['kind']=='DECISION':
            n=before['decision_count']+1
            cd=out/'calls'/f'decision_{n:04d}'
            expected=_make_call_request({'state':before,'seed_state':seeds,'focus_state':focus,
                            'prior':prior,'manifest':c['manifest'],'sources':c['sources']},'decision')
            if not (cd/'request.json').is_file() or base.read(cd/'request.json')!=expected:
                raise ValueError('CONTINUATION_DECISION_REQUEST_CHANGED')
            raw=base._raw_obj(base.read(cd/'raw_response.json')['raw_text'])
            for name in ('seed_update','seed_work_target_id'):
                if name not in raw:raise ValueError('SEED_FIELDS_MISSING_RAW_RESPONSE')
            side=base.read(cd/SEED_SIDECAR)
            if side['decision_number']!=n or side['seed_fields']!={
                       'seed_update':raw['seed_update'], 'seed_work_target_id':raw['seed_work_target_id'] }:
                raise ValueError('SEED_SIDECAR_RAW_MISMATCH')
            if side['previous_seed_state_sha256']!=base.sha(seeds):
                raise ValueError('SEED_STATE_PREVIOUS_MISMATCH')
            prior_seeds=copy.deepcopy(seeds)
            seeds=v16._transition(before,seeds,event['payload'],side['seed_fields'],after)
            if side['resulting_seed_state_sha256']!=base.sha(seeds):
                raise ValueError('SEED_STATE_NEXT_MISMATCH')
            fside=base.read(cd/FOCUS_SIDECAR)
            if fside['decision_number']!=n or fside['focus_update']!=raw.get('research_focus_update'):
                raise ValueError('FOCUS_SIDECAR_RAW_MISMATCH')
            if fside['previous_focus_sha256']!=base.sha(focus):
                raise ValueError('FOCUS_STATE_PREVIOUS_MISMATCH')
            # Focus can refer only to IDs registered in an earlier decision.
            focus=_focus_transition(before,focus,event['payload'],fside['focus_update'],after,prior_seeds)
            if fside['resulting_focus_sha256']!=base.sha(focus):
                raise ValueError('FOCUS_STATE_NEXT_MISMATCH')
            links=event.get('linked_artifact_shas',{})
            for path in [cd/SEED_SIDECAR,cd/FOCUS_SIDECAR]:
                if str(path.relative_to(out)) not in links:raise ValueError('UNLINKED_CONTINUATION_SIDECAR')
        before=after
    if base.sha(before)!=base.sha(c['state']):
        raise ValueError('CONTINUATION_STATE_REPLAY_MISMATCH')
    c['prior']=prior;c['seed_state']=seeds;c['focus_state']=focus
    return c


def _commit_decision(out:Path,path:Path)->dict:
    c=verify(out)
    req=base.read(path/'request.json');attempt=base.read(path/'attempt.json')
    if req!=_make_call_request(c,'decision') or attempt['request_sha256']!=base.sha(req):
        raise ValueError('CONTINUATION_REQUEST_REPLAY_MISMATCH')
    response=base.read(path/'raw_response.json')
    parsed=base._raw_obj(response['raw_text'])
    if 'seed_update' not in parsed or 'seed_work_target_id' not in parsed:
        raise ValueError('REQUIRED_SEED_FIELDS_MISSING')
    side={'seed_update':parsed.pop('seed_update'),
          'seed_work_target_id':parsed.pop('seed_work_target_id')}
    focus_update=parsed.pop('research_focus_update',None)
    dec=base.ResearchDecision.model_validate(base._normalize_future_decision_references(parsed,c['state']))
    d=dec.model_dump(mode='json')
    after=base.apply_event(c['state'],'DECISION',d,manifest=c['manifest'],sources=c['sources'])
    next_seed=v16._transition(c['state'],c['seed_state'],d,side,after)
    next_focus=_focus_transition(c['state'],c['focus_state'],d,focus_update,after,c['seed_state'])
    rseed={'decision_number':after['decision_count'],
         'previous_seed_state_sha256':base.sha(c['seed_state']),
         'resulting_seed_state_sha256':base.sha(next_seed),'seed_fields':side}
    rfocus={'decision_number':after['decision_count'],
         'previous_focus_sha256':base.sha(c['focus_state']),
         'resulting_focus_sha256':base.sha(next_focus),'focus_update':focus_update}
    for name,value in ((SEED_SIDECAR,rseed),(FOCUS_SIDECAR,rfocus)):
        file=path/name
        if file.exists():
            if base.read(file)!=value:raise ValueError('EXISTING_SIDECAR_DIFFERS: '+str(file))
        else:base.write_new(file,value)
    if dec.work_request is not None:
        wid=base.work_id_for(c['state'],dec.work_request)
        work_path=out/'work'/wid/'request.json'
        wd=dec.work_request.model_dump(mode='json')
        if work_path.exists():
            if base.read(work_path)!=wd:raise ValueError('WORK_REQUEST_MISMATCH')
        else:base.write_new(work_path,wd)
    links={str((path/name).relative_to(out)):base.file_sha(path/name)
           for name in ('request.json','raw_response.json',SEED_SIDECAR,FOCUS_SIDECAR)}
    event=base._record_event(out,'DECISION',d,links=links)
    if not (path/'receipt.json').exists():
        base.write_new(path/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL',
                             'event':event['index'],'response_id':response.get('response_id')})
    return event


def decision(out:Path,*,allow_paid=False,max_new_calls=1,backend=None)->dict:
    c=verify(out);req=_make_call_request(c,'decision')
    base._require_paid(backend,allow_paid,max_new_calls)
    cd=base._decision_call_dir(out,c['state'])
    if cd.exists():raise RuntimeError('existing Decision attempt; NEVER recall: '+str(cd))
    cd.mkdir(parents=True,exist_ok=False)
    base.write_new(cd/'request.json',req)
    base.write_new(cd/'attempt.json',{'request_sha256':base.sha(req),
        'status':'STARTED_UNKNOWN_UNTIL_RECEIPT','created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(cd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:event=_commit_decision(out,cd)
    except Exception as exc:
        raise RuntimeError('saved W07 Decision requires offline review: '+str(cd)) from exc
    return {'status':'DECISION_COMMITTED','event':event['index'],
       'decision_count':verify(out)['state']['decision_count'],
       'new_paid_calls_this_command':1 if backend is None else 0}


def execute(out:Path,*,allow_paid=False,max_new_calls=1,backend=None)->dict:
    verify(out)
    return base.execute(out,allow_paid=allow_paid,max_new_calls=max_new_calls,backend=backend)


def inspect(out:Path)->dict:
    c=verify(out);info=base.inspect(out)
    info.update({'policy_id':POLICY_ID,'seed_count':len(c['seed_state']['registry']),
      'seed_registry':c['seed_state']['registry'],
      'seed_history':c['seed_state']['history'],
      'focus_state':c['focus_state'],
      'prior_reference_sha256':c['prior']['previous_state_sha256']})
    return info


def recover(out:Path)->dict:
    c=verify(out);p=c['state']['pending_work']
    if p and p['status']=='REQUESTED':return base.recover(out)
    if c['state']['ended']:raise RuntimeError('episode already ended')
    cd=base._decision_call_dir(out,c['state'])
    if not (cd/'raw_response.json').is_file():
        raise RuntimeError('no saved response; do not repeat an uncertain paid call')
    event=_commit_decision(out,cd)
    return {'status':'RECOVERED_W07_DECISION_NO_CALL','event':event['index']}
