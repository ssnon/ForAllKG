"""Add-only W08 historical-work-feedback overlay. Only new episodes are affected.

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
from . import research_work_v17_continuation_policy as p17
from . import research_work_v17_continuation_runner as v17
from . import research_work_v18_feedback_policy as policy

POLICY_ID=policy.POLICY_ID
LOCK_FILE='SCIENTIFIC_FEEDBACK_V18_POLICY_LOCK.json'
PROVENANCE_FILE='SCIENTIFIC_FEEDBACK_V18_PROVENANCE.json'
PRIOR_FILE='PRIOR_W06_UNVERIFIED_RESEARCH.json'
W07_PRIOR_FILE='PRIOR_W07_UNVERIFIED_RESEARCH.json'
SEED_SIDECAR='seed_decision.json'
FOCUS_SIDECAR='focus_decision.json'


def _lock(c:dict)->dict:
    from . import research_work_v12_contracts as contracts
    from . import research_work_v12_stable_tools as tools
    files=[base.__file__,contracts.__file__,tools.__file__,
           p14.__file__,p15.__file__,p16.__file__,v16.__file__,p17.__file__,v17.__file__,policy.__file__,__file__]
    return {'policy_id':POLICY_ID,'dependency_source_shas':{
        Path(x).name:base.file_sha(Path(x)) for x in files},
        'policy_text_sha256':base.sha(policy.POLICY_SYSTEM_APPEND),
        'manifest_sha256':base.sha(c['manifest']),
        'sources_sha256':base.sha(c['sources']),
        'prior_memory_sha256':base.sha(c['prior']),
        'prior_w07_sha256':base.sha(c['prior_w07'])}


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


def _summarize_w07(episode:dict)->dict:
    """Factual, source-preserving snapshot. No model-generated judgement is promoted to truth."""
    state=episode['state']
    return {
      'source':'PREVIOUS_W07_UNVERIFIED_RESEARCH_EPISODE',
      'epistemic_status':'UNVERIFIED_MODEL_REASONING_AND_SOURCE_EXCERPTS',
      'not_imported_as_current_objects':True,
      'previous_state_sha256':base.sha(state),
      'decision_count':state['decision_count'],
      'ended':state['ended'],
      'work_results':[{'work_id':w['work_id'],'work_kind':w['work_kind'],
          'execution_status':w['execution_status'],
          'content':copy.deepcopy(w['content'])} for w in state['work_history']
          if w['work_kind']=='REASON'],
      'decisions':[{'number':d['number'],'action':d['action'],
          'question':d['question'],'reason':d['reason'],
          'work_id':d['work_id']} for d in state['decisions']],
      'work_interpretations':copy.deepcopy(state['interpretations']),
      'idea_registry':copy.deepcopy(state['idea_registry']),
      'focus_state':copy.deepcopy(episode['focus_state']),
    }


def prepare(*,reference:Path,out:Path,expected_w07_state_sha256:str)->dict:
    old=v17.verify(reference)
    state=old['state']
    exact=base.sha(state)
    if exact!=expected_w07_state_sha256:
        raise ValueError('W07_REFERENCE_STATE_SHA_MISMATCH: '+exact)
    if state['pending_work'] and state['pending_work']['status']=='REQUESTED':
        raise ValueError('W07_REFERENCE_HAS_PENDING_UNEXECUTED_WORK')
    if state['event_count']<1:
        raise ValueError('W07_REFERENCE_HAS_NO_RESEARCH')
    if out.exists():raise FileExistsError('never overwrite episode: '+str(out))
    w07=_summarize_w07(old)
    prior=copy.deepcopy(old['prior'])
    manifest=copy.deepcopy(old['manifest']);sources=copy.deepcopy(old['sources'])
    if prior['previous_state_sha256']!=old['prior']['previous_state_sha256']:
        raise ValueError('W06_PRIOR_REFERENCE_MISMATCH')
    if base.sha(sources)!=manifest['source_snapshot_sha256']:
        raise ValueError('FROZEN_SOURCES_MISMATCH')
    new_state=base.init_state(manifest['question'])
    out.mkdir(parents=True,exist_ok=False)
    base.write_new(out/'manifest.json',manifest)
    base.write_new(out/'manifest_lock.json',base.read(reference/'manifest_lock.json'))
    base.write_new(out/'sources.json',sources)
    base.write_new(out/'state.json',new_state)
    base.write_new(out/PRIOR_FILE,prior)
    base.write_new(out/W07_PRIOR_FILE,w07)
    base.write_new(out/LOCK_FILE,_lock({'manifest':manifest,'sources':sources,'prior':prior,'prior_w07':w07}))
    base.write_new(out/PROVENANCE_FILE,{'policy_id':POLICY_ID,
       'reference_w07_path':str(reference.resolve()),
       'verified_w07_state_sha256':exact,
       'prior_w06_memory_sha256':base.sha(prior),
       'prior_w07_memory_sha256':base.sha(w07),
       'source_snapshot_sha256':base.sha(sources),
       'model':manifest['model'],'max_decisions':manifest['max_decisions'],
       'prior_ideas_imported_as_current':0,'prior_seeds_imported_as_current':0,
       'paid_calls_during_prepare':0,
       'comparison_note':'W08 WARM START with prior W06 and W07 research. NOT an independent improvement comparison.'})
    verify(out)
    return {'status':'NEW_W08_FORK_PREPARED_NO_PAID_CALLS',
       'reference_unchanged':True, 'w07_state_sha256':exact,
       'state_sha256':base.sha(new_state),'episode':str(out)}


def decision_payload(c:dict)->dict:
    payload=v16.decision_payload(c)
    payload['decision_schema']=policy.focus_extension_schema(payload['decision_schema'])
    payload['prior_unverified_research']=copy.deepcopy(c['prior'])
    payload['current_scientific_focus']=copy.deepcopy(c['focus_state'])
    payload['prior_w07_unverified_work']=copy.deepcopy(c['prior_w07'])
    payload['scientific_research_choice']=policy.CHOICE_GUIDANCE
    payload['focus_contract']={'research_focus_update':'OPTIONAL, null or fully typed fields',
      'observation_refs':'Only previously READ source IDs in current episode; never invented refs',
      'historical_ids':'W06 IDs are NOT current registered IDs or allowable target_idea_id.',
      'hypothesis_registration':'Current episode ideas/seeds remain empty until independently proposed.',
      'purpose':'Choose a physically important causal unknown; Work result can change next question.',
      'mechanism_development_not_level_ii_certification':True,
      'avoid_unproductive_repetition_without_forcing_pivot':True,
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
    elif role=='work':
        # A Work is independently called. Give it the same epistemic history as
        # Decision so past dead-ends are not lost at the research-execution step.
        work_data=json.loads(req['messages'][1]['content'])
        work_data['prior_W06_unverified_research_compact']={
            'prior_ideas':copy.deepcopy(c['prior']['previous_research_ideas']),
            'prior_seeds':copy.deepcopy(c['prior']['previous_hypothesis_seeds']),
            'prior_work_interpretations':copy.deepcopy(c['prior']['previous_work_interpretations']),
            'epistemic_status':'UNVERIFIED_PRIOR_EPISODE'}
        work_data['prior_W07_unverified_work']=copy.deepcopy(c['prior_w07'])
        work_data['current_scientific_focus']=copy.deepcopy(c['focus_state'])
        work_data['scientific_work_choice_guidance']=policy.WORK_GUIDANCE
        req['messages'][0]['content']+='\n'+policy.WORK_SYSTEM_APPEND
        req['messages'][1]['content']=json.dumps(work_data,ensure_ascii=False,sort_keys=True)
    return req


def verify(out:Path)->dict:
    c=base.verify(out)
    prior=base.read(out/PRIOR_FILE)
    prior_w07=base.read(out/W07_PRIOR_FILE)
    if base.read(out/LOCK_FILE)!=_lock({'manifest':c['manifest'],'sources':c['sources'],'prior':prior,'prior_w07':prior_w07}):
        raise ValueError('CONTINUATION_POLICY_LOCK_MISMATCH')
    prov=base.read(out/PROVENANCE_FILE)
    if (prov['policy_id']!=POLICY_ID or prov['prior_w06_memory_sha256']!=base.sha(prior)
        or prov['prior_w07_memory_sha256']!=base.sha(prior_w07)
        or prov['verified_w07_state_sha256']!=prior_w07['previous_state_sha256']
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
                            'prior':prior,'prior_w07':prior_w07,'manifest':c['manifest'],'sources':c['sources']},'decision')
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
        if event['kind']=='WORK_PRODUCT' and event['payload'].get('work_kind')=='REASON' and event['payload'].get('execution_status')=='COMPLETED':
            wid=event['payload']['work_id']
            reqp=out/'work'/wid/'llm_request.json'
            ctx={'state':before,'manifest':c['manifest'],'sources':c['sources'],
                 'prior':prior,'prior_w07':prior_w07,'focus_state':focus,'seed_state':seeds}
            if not reqp.is_file() or base.read(reqp)!=_make_call_request(ctx,'work'):
                raise ValueError('V18_WORK_INPUT_REPLAY_MISMATCH')
            attempt=base.read(out/'work'/wid/'attempt.json')
            if attempt['request_sha256']!=base.sha(base.read(reqp)):
                raise ValueError('V18_WORK_ATTEMPT_SHA_MISMATCH')
        before=after
    if base.sha(before)!=base.sha(c['state']):
        raise ValueError('CONTINUATION_STATE_REPLAY_MISMATCH')
    c['prior']=prior;c['prior_w07']=prior_w07;c['seed_state']=seeds;c['focus_state']=focus
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
        raise RuntimeError('saved W08 Decision requires offline review: '+str(cd)) from exc
    return {'status':'DECISION_COMMITTED','event':event['index'],
       'decision_count':verify(out)['state']['decision_count'],
       'new_paid_calls_this_command':1 if backend is None else 0}


def _work_commit(out:Path,*,raw_path:Path|None=None,administrative_product:dict|None=None)->dict:
    c=verify(out)
    p=c['state']['pending_work']
    if p is None or p['status']!='REQUESTED':raise ValueError('no unexecuted pending work')
    wd=out/'work'/p['work_id'];rp=wd/'request.json'
    if not rp.is_file() or base.read(rp)!=p['request']:
        raise ValueError('work request missing/tampered')
    product=administrative_product if administrative_product is not None else base._make_product(c,raw_path=raw_path)
    fp=wd/'product.json'
    if fp.exists():
        if base.read(fp)!=product:raise ValueError('persisted product mismatch; never overwrite')
    else:base.write_new(fp,product)
    links={str(rp.relative_to(out)):base.file_sha(rp),str(fp.relative_to(out)):base.file_sha(fp)}
    if raw_path is not None:
        links[str(raw_path.relative_to(out))]=base.file_sha(raw_path)
        request_file=wd/'llm_request.json'
        if not request_file.is_file() or base.read(request_file)!=_make_call_request(c,'work'):
            raise ValueError('v18 work model request mismatch')
        links[str(request_file.relative_to(out))]=base.file_sha(request_file)
    event=base._record_event(out,'WORK_PRODUCT',product,links=links)
    if not (wd/'receipt.json').exists():
        base.write_new(wd/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL',
            'event':event['index'],'work_id':p['work_id'],'work_kind':p['request']['work_kind']})
    return event


def execute(out:Path,*,allow_paid=False,max_new_calls=1,backend=None)->dict:
    c=verify(out);p=c['state']['pending_work']
    if not p or p['status']!='REQUESTED':raise ValueError('no unexecuted work')
    wd=out/'work'/p['work_id']
    if p['request']['work_kind']=='READ_FROZEN_SOURCE':
        event=_work_commit(out)
        return {'status':'WORK_EXECUTED','event':event['index'],
                'kind':'READ_FROZEN_SOURCE','new_paid_calls_this_command':0}
    req=_make_call_request(c,'work')
    base._require_paid(backend,allow_paid,max_new_calls)
    if any((wd/f).exists() for f in ('attempt.json','raw_response.json','product.json','llm_request.json')):
        raise RuntimeError('existing work attempt; recover saved response, NEVER rerun: '+str(wd))
    base.write_new(wd/'llm_request.json',req)
    base.write_new(wd/'attempt.json',{'request_sha256':base.sha(req),
        'status':'STARTED_UNKNOWN_UNTIL_RECEIPT','created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(wd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:event=_work_commit(out,raw_path=wd/'raw_response.json')
    except Exception as exc:
        raise RuntimeError('saved W08 REASON response requires offline review: '+str(wd)) from exc
    return {'status':'WORK_EXECUTED','event':event['index'],'kind':'REASON',
            'new_paid_calls_this_command':1 if backend is None else 0}


def inspect(out:Path)->dict:
    c=verify(out);info=base.inspect(out)
    info.update({'policy_id':POLICY_ID,'seed_count':len(c['seed_state']['registry']),
      'seed_registry':c['seed_state']['registry'],
      'seed_history':c['seed_state']['history'],
      'focus_state':c['focus_state'],
      'prior_reference_sha256':c['prior']['previous_state_sha256'],
      'prior_w07_reference_sha256':c['prior_w07']['previous_state_sha256']})
    return info


def recover(out:Path)->dict:
    c=verify(out)
    state=c['state'];p=state['pending_work']
    if p and p['status']=='REQUESTED':
        wd=out/'work'/p['work_id']
        if p['request']['work_kind']=='READ_FROZEN_SOURCE':
            e=_work_commit(out)
            return {'status':'RECOVERED_FROZEN_SOURCE_NO_CALL','event':e['index']}
        if not (wd/'raw_response.json').is_file():
            raise RuntimeError('No persisted REASON response; uncertain billing. Never recall.')
        req=base.read(wd/'llm_request.json')
        if req!=_make_call_request(c,'work') or base.read(wd/'attempt.json')['request_sha256']!=base.sha(req):
            raise ValueError('V18_SAVED_REASON_REQUEST_MISMATCH')
        e=_work_commit(out,raw_path=wd/'raw_response.json')
        return {'status':'RECOVERED_WORK_NO_CALL','event':e['index']}
    if state['ended']:raise RuntimeError('episode already ended')
    cd=base._decision_call_dir(out,state)
    if not (cd/'raw_response.json').is_file():
        raise RuntimeError('No persisted Decision response; uncertain billing. Never recall.')
    e=_commit_decision(out,cd)
    return {'status':'RECOVERED_W08_DECISION_NO_CALL','event':e['index']}
