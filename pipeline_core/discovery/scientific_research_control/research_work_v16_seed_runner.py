"""W06 add-only causal-seed overlay on frozen v1.2 event+artifact contracts.

All seed mutations are linked as immutable Decision sidecars in the native event
chain. No new base event kinds, no rewrite of v1.2/v1.4/v1.5 modules.
"""
from __future__ import annotations
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from . import research_work_v12_stable_runner as base
from . import research_work_v14_space_policy as v14_policy
from . import research_work_v15_formation_runner as v15
from . import research_work_v15_formation_policy as v15_policy
from . import research_work_v16_seed_policy as policy

POLICY_ID=policy.POLICY_ID
LOCK_FILE='MECHANISM_SEED_POLICY_LOCK.json'
PROVENANCE_FILE='MECHANISM_SEED_POLICY_PROVENANCE.json'
SIDECAR_NAME='seed_decision.json'


def _lock(c: dict)->dict:
    return {'policy_id':POLICY_ID,'policy_sha256':base.sha(policy.POLICY_SYSTEM_APPEND),
            'policy_source_sha256':base.file_sha(Path(policy.__file__)),
            'runner_source_sha256':base.file_sha(Path(__file__)),
            'v15_runner_sha256':base.file_sha(Path(v15.__file__)),
            'v15_policy_sha256':base.file_sha(Path(v15_policy.__file__)),
            'v14_policy_sha256':base.file_sha(Path(v14_policy.__file__)),
            'stable_runner_sha256':base.file_sha(Path(base.__file__)),
            'manifest_sha256':base.sha(c['manifest']),
            'source_snapshot_sha256':base.sha(c['sources'])}


def _blank_seeds()->dict:
    return {'registry':{},'history':[],'last_work_target':None,'decision_count':0,
            'science_certified':False}


def _transition(before: dict, seeds: dict, decision:dict, payload:dict, after:dict)->dict:
    """Pure, strict seed-state transition. All data derived from event-linked files."""
    s=copy.deepcopy(seeds)
    if not isinstance(payload,dict) or set(payload)!={'seed_update','seed_work_target_id'}:
        raise ValueError('SEED_FIELDS_MISSING_OR_EXTRA')
    update_raw=payload['seed_update']
    update=policy.SeedMutation.model_validate(update_raw) if update_raw is not None else None
    target=payload['seed_work_target_id']
    prior_target=s['last_work_target']
    n=before['decision_count']+1
    p=before['pending_work']
    if update:
        if p is None or p['status']!='EXECUTED' or p['execution_status']!='COMPLETED':
            raise ValueError('SEED_MUTATION_REQUIRES_COMPLETED_WORK')
        works=[w for w in before['work_history'] if w['work_id']==p['work_id']]
        if len(works)!=1 or works[0]['work_kind']!='REASON':
            raise ValueError('SEED_MUTATION_REQUIRES_REASON_WORK')
        if (update.supporting_work_id!=p['work_id'] or
                update.supporting_product_sha256!=p['product_sha256']):
            raise ValueError('SEED_MUTATION_WRONG_WORK_PROVENANCE')
        if not decision.get('interpretation'):
            raise ValueError('SEED_MUTATION_REQUIRES_INTERPRETATION')
        d=update.model_dump(mode='json')
        if update.kind=='FORM':
            seed_id='hypothesis_seed:'+base.sha({'decision':n,'work_id':p['work_id'],
                        'draft':d['draft']})[:20]
            if seed_id in s['registry']:
                raise ValueError('duplicate seed identity')
            s['registry'][seed_id]={'seed_id':seed_id,'status':'EXPLORATORY_UNVERIFIED',
                'versions':[{'version':1,'draft':d['draft'],'decision':n,
                             'supporting_work_id':p['work_id']}],
                'native_idea_id':None}
        else:
            seed_id=update.target_seed_id
            if seed_id not in s['registry']:
                raise ValueError('SEED_TARGET_DOES_NOT_EXIST')
            if prior_target!=seed_id:
                raise ValueError('SEED_UPDATE_REQUIRES_PRIOR_TARGETED_REASON_WORK')
            r=s['registry'][seed_id]
            if r['status'] in ('DROPPED','PROMOTED_UNVERIFIED'):
                raise ValueError('cannot change closed seed')
            if update.kind=='DEVELOP':
                r['versions'].append({'version':len(r['versions'])+1,'draft':d['draft'],
                            'decision':n,'supporting_work_id':p['work_id']})
                r['status']='EXPLORATORY_UNVERIFIED'
            elif update.kind=='CHALLENGE':
                r['status']='CHALLENGED_UNVERIFIED'
            elif update.kind=='DISCARD':
                r['status']='DROPPED'
            elif update.kind=='PROMOTE':
                native=decision.get('idea_update')
                if not isinstance(native,dict) or native.get('kind')!='PROPOSE':
                    raise ValueError('PROMOTE_REQUIRES_NATIVE_PROPOSE_IN_SAME_DECISION')
                ids=set(after['idea_registry'])-set(before['idea_registry'])
                if len(ids)!=1:
                    raise ValueError('PROMOTE_REQUIRES_ONE_NEW_NATIVE_IDEA')
                r['native_idea_id']=next(iter(ids))
                r['status']='PROMOTED_UNVERIFIED'
        s['history'].append({'decision':n,'seed_id':seed_id,'kind':update.kind,
                  'target_seed_id':update.target_seed_id,
                  'supporting_work_id':p['work_id'],
                  'supporting_product_sha256':p['product_sha256'],
                  'rationale':update.rationale,
                  'challenge_or_change':update.challenge_or_change,
                  'native_idea_id':s['registry'][seed_id]['native_idea_id']})
    if target is not None:
        if not isinstance(target,str) or target not in seeds['registry']:
            raise ValueError('SEED_WORK_TARGET_MUST_ALREADY_EXIST')
        if s['registry'][target]['status'] in ('DROPPED','PROMOTED_UNVERIFIED'):
            raise ValueError('cannot target discarded/promoted seed')
        req=decision.get('work_request')
        if not req or req.get('work_kind')!='REASON' or req.get('target_idea_id') is not None:
            raise ValueError('SEED_TARGET_REQUIRES_REASON_WITH_NULL_NATIVE_TARGET')
        origin=seeds['registry'][target]['versions'][0]['supporting_work_id']
        if origin+':product' not in req.get('available_inputs',[]):
            raise ValueError('SEED_TARGET_WORK_MUST_READ_ORIGIN_WORK_PRODUCT')
        if target not in req['work_instructions']:
            raise ValueError('SEED_TARGET_ID_MISSING_FROM_WORK_INSTRUCTIONS')
    s['last_work_target']=target
    s['decision_count']=n
    if decision.get('action')=='END' and target is not None:
        raise ValueError('END_CANNOT_REQUEST_SEED_WORK')
    return s


def verify(out:Path)->dict:
    c=base.verify(out)
    if base.read(out/LOCK_FILE)!=_lock(c):
        raise ValueError('MECHANISM_SEED_POLICY_LOCK_MISMATCH')
    prov=base.read(out/PROVENANCE_FILE)
    if (prov.get('policy_id')!=POLICY_ID or
            prov.get('source_snapshot_sha256')!=base.sha(c['sources']) or
            prov.get('model')!=c['manifest']['model'] or
            prov.get('max_decisions')!=c['manifest']['max_decisions']):
        raise ValueError('MECHANISM_SEED_PROVENANCE_MISMATCH')
    before=base.init_state(c['manifest']['question'])
    seeds=_blank_seeds()
    for index in range(1,c['state']['event_count']+1):
        event=base.read(out/'events'/f'event_{index:04d}.json')
        after=base.apply_event(before,event['kind'],event['payload'],
                manifest=c['manifest'],sources=c['sources'])
        if event['kind']=='DECISION':
            number=before['decision_count']+1
            call_dir=out/'calls'/f'decision_{number:04d}'
            expected=_make_call_request({'state':before,'manifest':c['manifest'],
                   'sources':c['sources'],'seed_state':seeds},'decision')
            if not (call_dir/'request.json').is_file() or base.read(call_dir/'request.json')!=expected:
                raise ValueError('MIXED_OR_CHANGED_W06_DECISION_POLICY')
            path=call_dir/SIDECAR_NAME
            key=str(path.relative_to(out))
            if key not in event.get('linked_artifact_shas',{}) or not path.is_file():
                raise ValueError('MISSING_EVENT_LINKED_SEED_RECORD')
            side=base.read(path)
            raw=base._raw_obj(base.read(call_dir/'raw_response.json')['raw_text'])
            if 'seed_update' not in raw or 'seed_work_target_id' not in raw:
                raise ValueError('RAW_RESPONSE_OMITS_W06_SEED_FIELDS')
            if side.get('decision_number')!=number or side.get('seed_fields')!={
                    'seed_update':raw['seed_update'],
                    'seed_work_target_id':raw['seed_work_target_id']}:
                raise ValueError('SEED_RECORD_RAW_RESPONSE_MISMATCH')
            if side.get('previous_seed_state_sha256')!=base.sha(seeds):
                raise ValueError('SEED_STATE_CHAIN_PREVIOUS_MISMATCH')
            seeds=_transition(before,seeds,event['payload'],side['seed_fields'],after)
            if side.get('resulting_seed_state_sha256')!=base.sha(seeds):
                raise ValueError('SEED_STATE_CHAIN_NEXT_MISMATCH')
        before=after
    if base.sha(before)!=base.sha(c['state']):
        raise ValueError('W06_LOCAL_REPLAY_DIFFERENCE')
    c['seed_state']=seeds
    return c


def seed_fresh(*,reference:Path,out:Path)->dict:
    reference_c=v15.verify(reference)
    m=reference_c['manifest'];src=reference_c['sources']
    if not reference_c['state']['ended'] or reference_c['state']['decision_count']!=m['max_decisions']:
        raise ValueError('W05_REFERENCE_NOT_COMPLETED')
    if out.exists():
        raise FileExistsError(f'episode exists: {out}')
    out.mkdir(parents=True,exist_ok=False)
    state=base.init_state(m['question'])
    base.write_new(out/'manifest.json',copy.deepcopy(m))
    base.write_new(out/'manifest_lock.json',base.read(reference/'manifest_lock.json'))
    base.write_new(out/'sources.json',copy.deepcopy(src))
    base.write_new(out/'state.json',state)
    base.write_new(out/LOCK_FILE,_lock({'manifest':m,'sources':src}))
    base.write_new(out/PROVENANCE_FILE,{
      'policy_id':POLICY_ID,'origin_reference':str(reference.resolve()),
      'reference_verified_state_sha256':base.sha(reference_c['state']),
      'reference_decision_count':reference_c['state']['decision_count'],
      'source_snapshot_sha256':base.sha(src),'manifest_sha256':base.sha(m),
      'model':m['model'],'max_decisions':m['max_decisions'],
      'new_paid_calls':0,'imported_ideas':0,'imported_seeds':0,'imported_work_products':0,
      'comparison_note':'Fresh history, same frozen sources and model settings; W06 Seed overlay changes the Decision output contract. Not a controlled proof of superiority.'})
    verify(out)
    return {'status':'NEW_MECHANISM_SEED_EPISODE_PREPARED_NO_PAID_CALLS',
        'out':str(out),'policy_id':POLICY_ID,'max_decisions':m['max_decisions'],
        'state_sha256':base.sha(state),'source_sha256':base.sha(src),
        'reference_unchanged':True}


def decision_payload(c:dict)->dict:
    state=c['state'];s=c['seed_state']
    payload=v15.decision_payload(c)
    payload['decision_schema']=policy.seed_extension_schema(payload['decision_schema'])
    payload['exploratory_seed_registry']={'registry':copy.deepcopy(s['registry']),
        'history':copy.deepcopy(s['history']),
        'previous_targeted_seed_work':s['last_work_target'],
        'registered_seed_ids':sorted(s['registry']),
        'source':'W06_CURRENT_EPISODE_ONLY',
        'advisory_not_certification':True,
        'native_research_idea_is_separate':True,
        'supporting_reason_only':True}
    return payload


def _make_call_request(c:dict,role:str)->dict:
    req=base._make_call_request(c,role)
    if role=='decision':
        req['messages'][0]['content']+='\n'+v14_policy.POLICY_SYSTEM_APPEND+'\n'+v15_policy.POLICY_SYSTEM_APPEND+'\n'+policy.POLICY_SYSTEM_APPEND
        req['messages'][1]['content']=json.dumps(decision_payload(c),ensure_ascii=False,sort_keys=True)
    return req


def _commit_decision(out:Path,path:Path)->dict:
    c=verify(out)
    req=base.read(path/'request.json')
    attempt=base.read(path/'attempt.json')
    if req!=_make_call_request(c,'decision') or attempt['request_sha256']!=base.sha(req):
        raise ValueError('W06_SAVED_REQUEST_REPLAY_MISMATCH_NO_RECALL')
    response=base.read(path/'raw_response.json')
    parsed=base._raw_obj(response['raw_text'])
    if 'seed_update' not in parsed or 'seed_work_target_id' not in parsed:
        raise ValueError('W06_SEED_FIELDS_MISSING_NO_RECALL')
    side={'seed_update':parsed.pop('seed_update'),
          'seed_work_target_id':parsed.pop('seed_work_target_id')}
    dec=base.ResearchDecision.model_validate(
        base._normalize_future_decision_references(parsed,c['state']))
    d=dec.model_dump(mode='json')
    next_base=base.apply_event(c['state'],'DECISION',d,manifest=c['manifest'],sources=c['sources'])
    next_seed=_transition(c['state'],c['seed_state'],d,side,next_base)
    side_path=path/SIDECAR_NAME
    rec={'decision_number':next_base['decision_count'],
         'previous_seed_state_sha256':base.sha(c['seed_state']),
         'resulting_seed_state_sha256':base.sha(next_seed),'seed_fields':side}
    if side_path.exists():
        if base.read(side_path)!=rec:
            raise ValueError('SEED_SIDECAR_EXISTS_WITH_DIFFERENT_CONTENT')
    else:
        base.write_new(side_path,rec)
    if dec.work_request is not None:
        wid=base.work_id_for(c['state'],dec.work_request)
        work_path=out/'work'/wid/'request.json'
        wd=dec.work_request.model_dump(mode='json')
        if work_path.exists():
            if base.read(work_path)!=wd:
                raise ValueError('SAVED_WORK_REQUEST_MISMATCH')
        else:base.write_new(work_path,wd)
    event=base._record_event(out,'DECISION',d,links={
         str((path/'request.json').relative_to(out)):base.file_sha(path/'request.json'),
         str((path/'raw_response.json').relative_to(out)):base.file_sha(path/'raw_response.json'),
         str(side_path.relative_to(out)):base.file_sha(side_path)})
    if not (path/'receipt.json').exists():
        base.write_new(path/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL',
            'event':event['index'],'response_id':response.get('response_id')})
    return event


def decision(out:Path,*,allow_paid=False,max_new_calls=1,backend=None)->dict:
    c=verify(out)
    req=_make_call_request(c,'decision')
    base._require_paid(backend,allow_paid,max_new_calls)
    cd=base._decision_call_dir(out,c['state'])
    if cd.exists():
        raise RuntimeError('existing Decision attempt; recover offline, NEVER repeat paid call: '+str(cd))
    cd.mkdir(parents=True,exist_ok=False)
    base.write_new(cd/'request.json',req)
    base.write_new(cd/'attempt.json',{'request_sha256':base.sha(req),
          'status':'STARTED_UNKNOWN_UNTIL_RECEIPT','created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=base._invoke(req,backend)
    base.write_new(cd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:event=_commit_decision(out,cd)
    except Exception as exc:
        raise RuntimeError('saved W06 Decision requires offline review: '+str(cd)) from exc
    return {'status':'DECISION_COMMITTED','event':event['index'],
        'decision_count':verify(out)['state']['decision_count'],
        'new_paid_calls_this_command':1 if backend is None else 0}


def execute(out:Path,*,allow_paid=False,max_new_calls=1,backend=None)->dict:
    verify(out)
    return base.execute(out,allow_paid=allow_paid,max_new_calls=max_new_calls,backend=backend)


def recover(out:Path)->dict:
    c=verify(out)
    p=c['state']['pending_work']
    if p and p['status']=='REQUESTED':
        return base.recover(out)
    if c['state']['ended']:
        raise RuntimeError('episode ended')
    cd=base._decision_call_dir(out,c['state'])
    if not (cd/'raw_response.json').is_file():
        raise RuntimeError('no saved decision response: do not repeat uncertain call')
    event=_commit_decision(out,cd)
    return {'status':'RECOVERED_W06_DECISION_NO_CALL','event':event['index']}


def inspect(out:Path)->dict:
    c=verify(out)
    info=base.inspect(out)
    info.update({'policy_id':POLICY_ID,'seed_count':len(c['seed_state']['registry']),
       'seed_mutation_count':len(c['seed_state']['history']),
       'seed_work_target':c['seed_state']['last_work_target'],
       'seed_registry':c['seed_state']['registry'],
       'seed_history':c['seed_state']['history']})
    return info
