"""FAKG Research Work v1: opt-in decision -> work -> interpretation.

Separately versioned from runner.py. No import-time network access, arbitrary code
execution, scientific-truth certification, or mutation of existing episodes.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .research_work_v11_contracts import ResearchDecision, ResearchWorkProduct, ResearchWorkRequest, ReasonWorkContent
from .research_work_v11_tools import DECISION_SYSTEM, REASON_SYSTEM, frozen_excerpt, require_reason_content

PROTOCOL = 'fakg-research-work-v1.1'
QUESTION = ('Which physically specified causal process can account for the reported intermittent, '
            'mode-selective SERS blinking at an individual gold nanorod, and which distinctive '
            'prediction or limitation follows after comparison with strong conventional explanations?')


def canon(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value: Any) -> str:
    return hashlib.sha256(canon(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def refresh_state(out: Path, state: dict) -> None:
    temp = out/'state.next.json'
    if temp.exists():
        if sha(read(temp)) != sha(state):
            raise ValueError('existing state.next.json differs from replay; inspect before continuing')
        temp.unlink()
    write_new(temp, state)
    temp.replace(out/'state.json')


def init_state(question: str) -> dict:
    return {'schema_version': PROTOCOL, 'event_count': 0, 'decision_count': 0,
            'research_question': question, 'read_ids': ['YE:P1'], 'active_question': question,
            'decisions': [], 'work_history': [], 'interpretations': [], 'pending_work': None,
            'ended': False, 'final_disposition': None, 'final_justification': None,
            'science_certified': False, 'provisional_candidate': None}


def _check_sources(sources: dict) -> None:
    pages = sources.get('pages')
    works = sources.get('works')
    if not isinstance(pages, list) or [p.get('id') for p in pages] != [f'YE:P{i}' for i in range(1, 9)]:
        raise ValueError('requires all eight ordered Ye original-page IDs')
    if not isinstance(works, list) or not works or len({w.get('id') for w in works}) != len(works):
        raise ValueError('requires nonempty uniquely identified FAKG selected-work metadata')
    if any(not w.get('title') or not str(w['id']).startswith('WORK:') for w in works):
        raise ValueError('malformed FAKG selected works')
    if ('10.1103/4yg7-1f85' not in pages[0].get('text','') or
            'Blinking in surface-enhanced Raman spectrum' not in pages[0].get('text','')):
        raise ValueError('not the frozen original Ye 2025 source')
    if 'prior_research' in sources or set(sources) != {'pages', 'works'}:
        raise ValueError('source snapshot must not include prior research or reviewer-side hypotheses')


def prepare(*, out: Path, ye_pages: Path, selected_works: Path, native_context: Path,
            model: str = 'openai/gpt-5.6-luna', max_decisions: int = 12,
            temperature: float = 0.5, max_tokens: int = 4500) -> dict:
    if out.exists():
        raise FileExistsError('new work episode required; never overwrite a previous episode')
    if not 2 <= max_decisions <= 30 or not 1000 <= max_tokens <= 12000 or not 0 <= temperature <= 1.5:
        raise ValueError('invalid budget')
    # Reuse the existing rigorously typed native-context validation, not assistant sidecars.
    from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
    native = HypothesisContext.model_validate_json(native_context.read_text(encoding='utf-8'))
    pages = read(ye_pages)
    works = read(selected_works)
    if not isinstance(pages,list) or not isinstance(works,list):
        raise ValueError('invalid frozen source files')
    source_snapshot = {'pages': [{'id':f'YE:P{p["page"]}', 'page':p['page'], 'text':p['text']} for p in pages],
                       'works': [{'id':f'WORK:{i+1:02d}', 'title':w['work']['title'],
                                  'doi':w['work'].get('doi',''), 'abstract':w['work'].get('abstract') or '',
                                  'origin':'FAKG_OPEN_WORLD_SELECTED_WORK',
                                  'source_support':'METADATA_ABSTRACT_ONLY'}
                                 for i,w in enumerate(works) if w.get('work',{}).get('title')]}
    _check_sources(source_snapshot)
    manifest = {'schema_version':PROTOCOL,'status':'PREPARED_NO_PAID_CALLS','question':QUESTION,
                'model':model,'max_decisions':max_decisions,'temperature':temperature,'max_tokens':max_tokens,
                'source_snapshot_sha256':sha(source_snapshot),
                'input_file_hashes': {'ye_pages':file_sha(ye_pages), 'selected_works':file_sha(selected_works),
                                      'native_context':file_sha(native_context)},
                'native_context_id':native.context_id,'native_context_sha256':native.context_sha256,
                'native_source_packet_id':native.source_packet_id,
                'source_science_certified':False,'research_lineage':'FRESH_NO_PRIOR_R01_R02_IDEAS',
                'allowed_work_kinds':['REASON','READ_FROZEN_SOURCE']}
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'manifest.json',manifest)
    write_new(out/'manifest_lock.json', {'manifest_sha256':sha(manifest), 'protocol':PROTOCOL,
                                          'scope':'EPISODE_SETTINGS_AT_PREPARE_NOT_EXTERNAL_ATTESTATION'})
    write_new(out/'sources.json',source_snapshot)
    write_new(out/'state.json',init_state(QUESTION))
    return manifest


def work_id_for(state: dict, request: ResearchWorkRequest) -> str:
    return f'work_{state["decision_count"]+1:04d}_{sha(request.model_dump(mode="json"))[:12]}'


def work_input_digest(work_id: str, request: dict, sources_sha: str) -> str:
    return sha({'work_id':work_id,'request':request,'source_snapshot_sha256':sources_sha})


def _source_ids(source: dict) -> set[str]:
    return {row['id'] for row in [*source['pages'], *source['works']]}


def _resolved_reason_inputs(state: dict, sources: dict, request: dict) -> tuple[list[str], list[dict]]:
    """Only verified, previously completed WorkProducts can be used as artifact inputs.

    Distinguishes frozen evidence IDs from artifact references. The content of
    an artifact is copied from the event-replayed research state, never fetched
    by treating its pseudo-ID as a scientific source. No automatic reading.
    """
    all_ids = list(dict.fromkeys([*request['available_inputs'], *request['source_ids']]))
    source_ids = _source_ids(sources)
    products = {f"{w['work_id']}:product": w for w in state['work_history']
                if w['execution_status'] == 'COMPLETED'}
    frozen = []
    artifacts = []
    for ref in all_ids:
        if ref in source_ids:
            if ref not in state['read_ids']:
                raise ValueError(f'REASON cannot use unread sources {[ref]}; READ_EVIDENCE first. '
                                 'Saved paid response retained; do not automatically retry')
            frozen.append(ref)
        elif ref in products:
            row = products[ref]
            artifacts.append({'artifact_ref':ref, 'work_id':row['work_id'],
                              'product_sha256':row['product_sha256'],
                              'work_kind':row['work_kind'],
                              'origin':row['origin'],
                              'science_certified':False,
                              'content':copy.deepcopy(row['content'])})
        else:
            raise ValueError(f'work request cites source outside frozen source snapshot or '
                             f'unknown/incomplete WorkProduct: {ref}')
    return frozen, artifacts


def apply_event(state: dict, event_kind: str, payload: dict, *, manifest: dict, sources: dict) -> dict:
    s = copy.deepcopy(state)
    if s['ended']:
        raise ValueError('cannot add events after END')
    if event_kind == 'DECISION':
        dec = ResearchDecision.model_validate(payload)
        prev = s['pending_work']
        if prev is not None:
            if prev['status'] != 'EXECUTED':
                raise ValueError('cannot make next decision while prior work is unexecuted')
            interpretation = dec.interpretation
            if interpretation is None:
                raise ValueError('cannot ignore prior WorkProduct: interpretation required')
            if (interpretation.work_id != prev['work_id'] or
                    interpretation.product_sha256 != prev['product_sha256']):
                raise ValueError('interpretation work ID or product digest does not match')
            if f'{prev["work_id"]}:product' not in interpretation.supported_by_artifact_refs:
                raise ValueError('interpretation must cite the exact preceding work artifact')
            # A source page can be cited using its inventory ID (YE:P2) or
            # the typed alias (source:YE:P2). Both MUST have been read already.
            # This only broadens representation, never what is trusted.
            prior_artifact_ref = f'{prev["work_id"]}:product'
            valid_refs = {prior_artifact_ref}
            valid_refs.update(s['read_ids'])
            valid_refs.update(f'source:{x}' for x in s['read_ids'])
            if any(x not in valid_refs for x in interpretation.supported_by_artifact_refs):
                raise ValueError('interpretation cited an unknown or unread source/artifact')
            s['interpretations'].append(interpretation.model_dump(mode='json'))
            s['pending_work'] = None
        elif dec.interpretation is not None:
            raise ValueError('interpretation without executed prior work')
        if s['decision_count'] >= manifest['max_decisions']:
            raise ValueError('decision budget exceeded')
        if dec.work_request is not None:
            if s['decision_count'] + 1 >= manifest['max_decisions']:
                raise ValueError('reserve the final decision for interpreting the last WorkProduct or END')
            req = dec.work_request
            if req.work_kind == 'REASON':
                _resolved_reason_inputs(s, sources, req.model_dump(mode='json'))
            elif any(x not in _source_ids(sources) for x in [*req.available_inputs, *req.source_ids]):
                # Frozen reads must never accept pseudo-ID work artifacts.
                raise ValueError('frozen read request cites source outside frozen source snapshot')
            # The initial vertical slice contains no imported/human-proposed ResearchIdea nodes.
            # Do not silently accept an arbitrary model-generated ID as a native ResearchIdea.
            if req.target_idea_id is not None:
                raise ValueError('fresh work v1 has no native ResearchIdea registry; use null target and a research question')
            wid = work_id_for(s,req)
            s['pending_work'] = {'work_id':wid, 'status':'REQUESTED','request':req.model_dump(mode='json'),
                                  'request_sha256':sha(req.model_dump(mode='json')), 'product_sha256':None,
                                  'execution_status':None}
        s['decision_count'] += 1
        s['active_question'] = dec.research_question_now
        s['decisions'].append({'number':s['decision_count'],'action':dec.action,
                               'reason':dec.scientific_reason,'question':dec.research_question_now,
                               'work_id':s['pending_work']['work_id'] if s['pending_work'] else None,
                               'interpreted_prior':prev['work_id'] if prev else None})
        if dec.action == 'END':
            if dec.final_disposition == 'PROVISIONAL_RESEARCH_CANDIDATE':
                snapshot=dec.provisional_candidate
                if snapshot is None:
                    raise ValueError('candidate must include a concrete unverified causal account')
                # Scientific result is not certified. However candidate nomination must
                # correspond to a genuine completed/INTERPRETED REASON work product.
                interpreted={i['work_id'] for i in s['interpretations']}
                supported={w['work_id'] for w in s['work_history'] if
                           w['work_kind']=='REASON' and w['execution_status']=='COMPLETED'}
                if not set(snapshot.supporting_work_ids) <= (interpreted & supported):
                    raise ValueError('candidate must cite completed, interpreted REASON works')
                if len(set(snapshot.supporting_work_ids)) != len(snapshot.supporting_work_ids):
                    raise ValueError('duplicate supporting work IDs')
                s['provisional_candidate']=snapshot.model_dump(mode='json')
            s['ended'] = True
            s['final_disposition'] = dec.final_disposition
            s['final_justification'] = dec.final_justification
    elif event_kind == 'WORK_PRODUCT':
        product=ResearchWorkProduct.model_validate(payload)
        p=s['pending_work']
        if p is None or p['status'] != 'REQUESTED':
            raise ValueError('no unexecuted work request exists')
        if product.work_id != p['work_id'] or product.work_kind != p['request']['work_kind']:
            raise ValueError('wrong work identity or kind')
        if product.inputs_sha256 != work_input_digest(p['work_id'],p['request'],manifest['source_snapshot_sha256']):
            raise ValueError('wrong request/source binding')
        if product.output_artifact_sha256 != sha(product.content):
            raise ValueError('work content digest mismatch')
        if product.work_kind == 'READ_FROZEN_SOURCE':
            if product.origin != 'FROZEN_SOURCE_EXCERPT' or product.raw_response_sha256:
                raise ValueError('frozen read must not masquerade as LLM reasoning')
            if product.content != {'source_material':frozen_excerpt(sources,p['request']['source_ids']),
                                   'work_conclusion':'SOURCE_MATERIAL_OBTAINED'}:
                raise ValueError('frozen source content not identical to snapshot')
            if product.execution_status != 'COMPLETED':
                raise ValueError('validated frozen-source extraction must complete')
            s['read_ids'] = list(dict.fromkeys([*s['read_ids'],*p['request']['source_ids']]))
        elif product.work_kind == 'REASON':
            if product.execution_status == 'COMPLETED':
                if product.origin != 'LLM_REASONING_UNVERIFIED' or not product.raw_response_sha256 or product.failure_reason:
                    raise ValueError('completed reason product must link actual preserved LLM response')
            elif (product.origin != 'EXECUTION_STATUS_RECEIPT' or product.raw_response_sha256 or
                  not product.failure_reason or len(product.failure_reason) < 12):
                raise ValueError('failed/blocked reason needs a transparent execution failure receipt')
            require_reason_content(product.content)
        p['status'] = 'EXECUTED'
        p['product_sha256'] = sha(payload)
        p['execution_status'] = product.execution_status
        s['work_history'].append({'work_id':p['work_id'],'work_kind':product.work_kind,
                                  'execution_status':product.execution_status,
                                  'origin':product.origin,'product_sha256':sha(payload),
                                  'work_conclusion':product.content.get('work_conclusion'),
                                  'content':product.content, 'science_certified':False})
    else:
        raise ValueError('unknown work event')
    s['event_count'] += 1
    return s


def _replay(out: Path) -> dict:
    m=read(out/'manifest.json')
    sources=read(out/'sources.json')
    if m.get('schema_version') != PROTOCOL or sha(sources) != m['source_snapshot_sha256']:
        raise ValueError('manifest or frozen source content tampered')
    lock=read(out/'manifest_lock.json')
    if lock != {'manifest_sha256':sha(m), 'protocol':PROTOCOL,
                'scope':'EPISODE_SETTINGS_AT_PREPARE_NOT_EXTERNAL_ATTESTATION'}:
        raise ValueError('manifest changed from its initial prepared settings; inspect before new calls')
    if m.get('research_lineage') != 'FRESH_NO_PRIOR_R01_R02_IDEAS':
        raise ValueError('unexpected prior research lineage')
    if (not isinstance(m.get('max_decisions'),int) or not 2 <= m['max_decisions'] <= 30 or
        not isinstance(m.get('max_tokens'),int) or not 1000 <= m['max_tokens'] <= 12000 or
        not isinstance(m.get('temperature'),(int,float)) or not 0 <= m['temperature'] <= 1.5):
        raise ValueError('invalid manifest run parameters')
    _check_sources(sources)
    state=init_state(m['question'])
    event_paths=sorted((out/'events').glob('event_*.json')) if (out/'events').exists() else []
    for n,path in enumerate(event_paths,1):
        if path.name != f'event_{n:04d}.json':
            raise ValueError('noncontiguous event numbering')
        row=read(path)
        if row.get('schema_version') != PROTOCOL or row.get('index') != n:
            raise ValueError('invalid event header')
        if row.get('previous_state_sha256') != sha(state):
            raise ValueError('event state chain broken')
        for rel,expected in row.get('linked_artifact_shas',{}).items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or not (out/rel).is_file():
                raise ValueError('linked artifact missing/unsafe path')
            if file_sha(out/rel) != expected:
                raise ValueError(f'linked artifact changed: {rel}')
        if row['kind'] == 'DECISION':
            dec=ResearchDecision.model_validate(row['payload'])
            if dec.work_request is not None:
                wid=work_id_for(state,dec.work_request)
                req_path=out/'work'/wid/'request.json'
                if not req_path.is_file() or read(req_path) != dec.work_request.model_dump(mode='json'):
                    raise ValueError('WorkRequest file differs from decision event')
        if row['kind'] == 'WORK_PRODUCT':
            product=ResearchWorkProduct.model_validate(row['payload'])
            fp=out/'work'/product.work_id/'product.json'
            if not fp.is_file() or read(fp) != row['payload']:
                raise ValueError('WorkProduct not byte-consistent with recorded event')
            if product.work_kind == 'REASON' and product.execution_status == 'COMPLETED':
                rf=out/'work'/product.work_id/'raw_response.json'
                if not rf.is_file() or file_sha(rf) != product.raw_response_sha256:
                    raise ValueError('reason raw response is missing/tampered')
        state=apply_event(state,row['kind'],row['payload'],manifest=m,sources=sources)
        if row.get('resulting_state_sha256') != sha(state):
            raise ValueError('event result state digest mismatch')
    return {'manifest':m,'sources':sources,'state':state,'event_count':len(event_paths)}


def verify(out: Path, *, strict_state: bool = True) -> dict:
    verified=_replay(out)
    disk=read(out/'state.json')
    if strict_state and sha(disk) != sha(verified['state']):
        raise ValueError('state.json differs from event replay; recover without new calls')
    return verified


def _record_event(out: Path, kind: str, payload: dict, *, links: dict[str,str] | None = None) -> dict:
    c=verify(out)
    state=c['state']
    fresh=apply_event(state,kind,payload,manifest=c['manifest'],sources=c['sources'])
    n=state['event_count']+1
    row={'schema_version':PROTOCOL,'index':n,'kind':kind,'payload':payload,
         'previous_state_sha256':sha(state),'resulting_state_sha256':sha(fresh),
         'linked_artifact_shas':links or {}}
    write_new(out/'events'/f'event_{n:04d}.json',row)
    refresh_state(out,fresh)
    return row


def decision_payload(c: dict) -> dict:
    s=c['state']
    pending=s['pending_work']
    if pending is not None and pending['status'] != 'EXECUTED':
        raise ValueError('work must be performed before next decision')
    if s['ended'] or s['decision_count'] >= c['manifest']['max_decisions']:
        raise ValueError('decision unavailable: ended or budget exhausted')
    return {'research_question':c['manifest']['question'],
            'candidate_disposition_rule': ('A PROVISIONAL_RESEARCH_CANDIDATE requires a full, unverified '
                'candidate snapshot citing at least one completed and interpreted REASON work; '
                'otherwise END with ABSTAIN. This does not certify Level II.'),
            'source_inventory':[{'id':x['id'],'title':x.get('title'),'source_type':'PAPER_PAGE' if x['id'].startswith('YE:') else 'WORK_ABSTRACT'}
                                for x in [*c['sources']['pages'],*c['sources']['works']]],
            'read_evidence':frozen_excerpt(c['sources'],s['read_ids']),
            'research_memory':{'decisions':s['decisions'],'work_history':s['work_history'],
                               'interpretations':s['interpretations'],'active_question':s['active_question'],
                               'read_ids':s['read_ids'],'science_certified':False},
            'pending_interpretation':({'work_id':pending['work_id'],
                                       'product_sha256':pending['product_sha256'],
                                       'artifact_ref':f'{pending["work_id"]}:product',
                                       'work_result':s['work_history'][-1]} if pending else None),
            'remaining_decisions':c['manifest']['max_decisions']-s['decision_count'],
            'budget_note':'The final decision slot may only END, not open a work item that cannot be interpreted.',
            'work_available':['REASON','READ_FROZEN_SOURCE'],
            'decision_schema':ResearchDecision.model_json_schema(),
            'epistemic_rule':'WorkProduct and self-interpretation are proposals, not verified science. No external experiments required.'}


def work_payload(c: dict) -> dict:
    s=c['state']; p=s['pending_work']
    if p is None or p['status'] != 'REQUESTED':
        raise ValueError('no unexecuted work request')
    req=p['request']
    if req['work_kind'] != 'REASON':
        raise ValueError('READ_FROZEN_SOURCE is deterministic and needs no model payload')
    source_ids, artifacts = _resolved_reason_inputs(s, c['sources'], req)
    return {'research_question':c['manifest']['question'],'work_id':p['work_id'],
            'work_request':req,
            'read_scientific_sources':frozen_excerpt(c['sources'],source_ids),
            'read_work_products':artifacts,
            'known_history_unverified':s['work_history'],
            'work_content_schema': ReasonWorkContent.model_json_schema(),
            'science_certification':False}


def _require_paid(backend, allow_paid: bool, max_new_calls: int) -> None:
    if max_new_calls != 1:
        raise ValueError('exactly one new model call per user authorization (max_new_calls=1)')
    if backend is None and (not allow_paid or os.getenv('FAKG_RESEARCH_ALLOW_PAID') != 'YES'):
        raise PermissionError('requires --allow-paid AND FAKG_RESEARCH_ALLOW_PAID=YES')
    if backend is None and not os.getenv('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY is missing; no call attempted')


def _invoke(request: dict, backend) -> tuple[str, Any, str | None]:
    if backend is not None:
        return backend(request)
    from openai import OpenAI
    client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                  base_url=os.getenv('OPENAI_BASE_URL','https://openrouter.ai/api/v1'),
                  timeout=300,max_retries=0)
    resp=client.chat.completions.create(model=request['model'],temperature=request['temperature'],
                                        max_tokens=request['max_tokens'],messages=request['messages'],
                                        response_format={'type':'json_object'})
    return (resp.choices[0].message.content or '',resp.usage.model_dump() if resp.usage else None,resp.id)


def _make_call_request(c: dict, role: str) -> dict:
    state=c['state']; m=c['manifest']
    data=decision_payload(c) if role=='decision' else work_payload(c)
    msg=[{'role':'system','content':DECISION_SYSTEM if role=='decision' else REASON_SYSTEM},
         {'role':'user','content':json.dumps(data,ensure_ascii=False,sort_keys=True)}]
    return {'model':m['model'],'temperature':m['temperature'],'max_tokens':m['max_tokens'],
            'messages':msg,'role':role,'input_state_sha256':sha(state)}


def _raw_obj(raw: str) -> dict:
    obj=json.loads(raw)
    if not isinstance(obj,dict):
        raise ValueError('response is not one JSON object')
    return obj


def _decision_call_dir(out: Path, s: dict) -> Path:
    return out/'calls'/f'decision_{s["decision_count"]+1:04d}'


def _decision_commit(out: Path, *, path: Path) -> dict:
    c=verify(out)
    req=read(path/'request.json')
    if req != _make_call_request(c,'decision') or read(path/'attempt.json')['request_sha256'] != sha(req):
        raise ValueError('decision request differs from replayed state/prompt; no new call permitted')
    response=read(path/'raw_response.json')
    dec=ResearchDecision.model_validate(_raw_obj(response['raw_text']))
    d=dec.model_dump(mode='json')
    # Validate the FULL scientific decision against the verified current state
    # before creating any request artifact. A failed saved response must leave
    # neither an event nor a half-created work request behind.
    apply_event(c['state'], 'DECISION', d, manifest=c['manifest'], sources=c['sources'])
    if dec.work_request is not None:
        wid=work_id_for(c['state'],dec.work_request)
        rp=out/'work'/wid/'request.json'
        if rp.exists():
            if read(rp) != dec.work_request.model_dump(mode='json'):
                raise ValueError('existing work request is inconsistent')
        else:
            write_new(rp,dec.work_request.model_dump(mode='json'))
    row=_record_event(out,'DECISION',d,links={str((path/'request.json').relative_to(out)):file_sha(path/'request.json'),
                                                   str((path/'raw_response.json').relative_to(out)):file_sha(path/'raw_response.json')})
    if not (path/'receipt.json').exists():
        write_new(path/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL','event':row['index'],
                                       'response_id':response.get('response_id')})
    return row


def decision(out: Path, *, allow_paid: bool=False, max_new_calls: int=1, backend=None) -> dict:
    c=verify(out)
    req=_make_call_request(c,'decision')
    _require_paid(backend,allow_paid,max_new_calls)
    call_dir=_decision_call_dir(out,c['state'])
    if call_dir.exists():
        raise RuntimeError(f'existing decision attempt {call_dir}; use recover, NEVER repeat paid call')
    call_dir.mkdir(parents=True,exist_ok=False)
    write_new(call_dir/'request.json',req)
    write_new(call_dir/'attempt.json',{'request_sha256':sha(req),'status':'STARTED_UNKNOWN_UNTIL_RECEIPT',
                                       'created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=_invoke(req,backend)
    write_new(call_dir/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        event=_decision_commit(out,path=call_dir)
    except Exception as e:
        raise RuntimeError(f'saved decision response needs offline recovery: {call_dir}') from e
    return {'status':'DECISION_COMMITTED','event':event['index'], 'decision_count':verify(out)['state']['decision_count'],
            'new_paid_calls_this_command':1 if backend is None else 0}


def _make_product(c: dict, *, raw_path: Path | None = None) -> dict:
    p=c['state']['pending_work']; req=p['request']; wid=p['work_id']
    if req['work_kind']=='READ_FROZEN_SOURCE':
        content={'source_material':frozen_excerpt(c['sources'],req['source_ids']),
                 'work_conclusion':'SOURCE_MATERIAL_OBTAINED'}
        origin='FROZEN_SOURCE_EXCERPT';raw_hash=None
    else:
        if raw_path is None or not raw_path.is_file():
            raise RuntimeError('no persisted model response for REASON; do not attempt an automatic retry')
        response=read(raw_path)
        content=require_reason_content(_raw_obj(response['raw_text']))
        origin='LLM_REASONING_UNVERIFIED';raw_hash=file_sha(raw_path)
    return ResearchWorkProduct(work_id=wid,work_kind=req['work_kind'],execution_status='COMPLETED',
             inputs_sha256=work_input_digest(wid,req,c['manifest']['source_snapshot_sha256']),
             output_artifact_sha256=sha(content),origin=origin,content=content,
             raw_response_sha256=raw_hash,scientific_validation=False).model_dump(mode='json')


def _work_commit(out: Path, *, raw_path: Path | None = None, administrative_product: dict | None = None) -> dict:
    c=verify(out)
    p=c['state']['pending_work']
    if p is None or p['status']!='REQUESTED':
        raise ValueError('no unexecuted pending work')
    wd=out/'work'/p['work_id']
    rp=wd/'request.json'
    if not rp.is_file() or read(rp)!=p['request']:
        raise ValueError('work request missing/tampered')
    product=administrative_product if administrative_product is not None else _make_product(c,raw_path=raw_path)
    fp=wd/'product.json'
    if fp.exists():
        if read(fp)!=product:
            raise ValueError('persisted product mismatch; cannot overwrite')
    else:
        write_new(fp,product)
    links={str(rp.relative_to(out)):file_sha(rp),str(fp.relative_to(out)):file_sha(fp)}
    if raw_path is not None:
        links[str(raw_path.relative_to(out))]=file_sha(raw_path)
        request_file=wd/'llm_request.json'
        if not request_file.is_file() or read(request_file)!=_make_call_request(c,'work'):
            raise ValueError('work model request mismatch')
        links[str(request_file.relative_to(out))]=file_sha(request_file)
    event=_record_event(out,'WORK_PRODUCT',product,links=links)
    if not (wd/'receipt.json').exists():
        write_new(wd/'receipt.json',{'status':'COMPLETED_OR_RECOVERED_NO_CALL', 'event':event['index'],
                                      'work_id':p['work_id'],'work_kind':p['request']['work_kind']})
    return event


def execute(out: Path, *, allow_paid: bool=False, max_new_calls: int=1, backend=None) -> dict:
    c=verify(out)
    p=c['state']['pending_work']
    if p is None or p['status']!='REQUESTED':
        raise ValueError('no unexecuted work')
    wd=out/'work'/p['work_id']
    if p['request']['work_kind']=='READ_FROZEN_SOURCE':
        event=_work_commit(out)
        return {'status':'WORK_EXECUTED','event':event['index'],'kind':'READ_FROZEN_SOURCE','new_paid_calls_this_command':0}
    req=_make_call_request(c,'work')
    _require_paid(backend,allow_paid,max_new_calls)
    if any((wd/f).exists() for f in ('attempt.json','raw_response.json','product.json','llm_request.json')):
        raise RuntimeError(f'existing work attempt at {wd}; recover saved response, NEVER rerun it')
    write_new(wd/'llm_request.json',req)
    write_new(wd/'attempt.json',{'request_sha256':sha(req),'status':'STARTED_UNKNOWN_UNTIL_RECEIPT',
                                 'created_utc':datetime.now(timezone.utc).isoformat()})
    raw,usage,rid=_invoke(req,backend)
    write_new(wd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:
        event=_work_commit(out,raw_path=wd/'raw_response.json')
    except Exception as e:
        raise RuntimeError(f'saved REASON response needs offline recovery: {wd}') from e
    return {'status':'WORK_EXECUTED','event':event['index'],'kind':'REASON',
            'new_paid_calls_this_command':1 if backend is None else 0}



def record_work_failure(out: Path, *, execution_status: str, reason: str) -> dict:
    """Explicitly record inability to perform work; not a science conclusion or paid retry.

    Only for a not-yet-attempted REASON. Existing response/attempts are never
    reclassified as failures, to prevent masking an uncertain paid call.
    """
    if execution_status not in {'FAILED','BLOCKED'} or not isinstance(reason,str) or len(reason.strip()) < 12:
        raise ValueError('must supply a meaningful FAILED/BLOCKED execution explanation')
    c=verify(out); p=c['state']['pending_work']
    if not p or p['status']!='REQUESTED' or p['request']['work_kind']!='REASON':
        raise ValueError('only a pending REASON work can be explicitly blocked or failed')
    wd=out/'work'/p['work_id']
    if any((wd/f).exists() for f in ('attempt.json','raw_response.json','product.json','llm_request.json')):
        raise RuntimeError('an existing attempt/product cannot be turned into a clean failure receipt')
    content={'attempted_question':p['request']['research_question'],
             'premises_used':[], 'reasoning_or_operations':[],
             'candidate_consequence':None,'counterexample_or_obstruction':None,
             'limits_and_uncertainties':[reason], 'work_conclusion':'INCONCLUSIVE'}
    require_reason_content(content)
    product=ResearchWorkProduct(work_id=p['work_id'],work_kind='REASON',execution_status=execution_status,
              inputs_sha256=work_input_digest(p['work_id'],p['request'],c['manifest']['source_snapshot_sha256']),
              output_artifact_sha256=sha(content),origin='EXECUTION_STATUS_RECEIPT',content=content,
              failure_reason=reason,scientific_validation=False).model_dump(mode='json')
    row=_work_commit(out,administrative_product=product)
    return {'status':f'WORK_{execution_status}_NO_CALL','event':row['index'], 'work_id':p['work_id']}

def recover(out: Path) -> dict:
    c=verify(out,strict_state=False)
    if sha(read(out/'state.json'))!=sha(c['state']):
        refresh_state(out,c['state'])
        c=verify(out)
        return {'status':'STATE_RESTORED_FROM_VERIFIED_EVENTS_NO_CALL','event_count':c['state']['event_count']}
    s=c['state'];p=s['pending_work']
    if p and p['status']=='REQUESTED':
        wd=out/'work'/p['work_id']
        if p['request']['work_kind']=='READ_FROZEN_SOURCE':
            event=_work_commit(out)
            return {'status':'RECOVERED_FROZEN_SOURCE_NO_CALL','event':event['index']}
        if not (wd/'raw_response.json').is_file():
            raise RuntimeError('no persisted REASON response; uncertain billing, do not retry automatically')
        req=read(wd/'llm_request.json')
        if req!=_make_call_request(c,'work') or read(wd/'attempt.json')['request_sha256']!=sha(req):
            raise ValueError('saved REASON request does not match current replayed work state')
        event=_work_commit(out,raw_path=wd/'raw_response.json')
        return {'status':'RECOVERED_WORK_NO_CALL','event':event['index']}
    if s['ended']:
        raise RuntimeError('episode already ended; nothing to recover')
    call_dir=_decision_call_dir(out,s)
    if not (call_dir/'raw_response.json').is_file():
        raise RuntimeError('no persisted decision response; uncertain billing, do not retry automatically')
    event=_decision_commit(out,path=call_dir)
    return {'status':'RECOVERED_DECISION_NO_CALL','event':event['index']}


def inspect(out: Path) -> dict:
    c=verify(out)
    s=c['state']
    return {'schema_version':PROTOCOL,'event_count':s['event_count'],'decision_count':s['decision_count'],
            'max_decisions':c['manifest']['max_decisions'],'read_count':len(s['read_ids']),
            'work_count':len(s['work_history']),'interpretation_count':len(s['interpretations']),
            'pending_work':s['pending_work'] and {k:s['pending_work'][k] for k in ('work_id','status','execution_status','product_sha256')},
            'ended':s['ended'],'final_disposition':s['final_disposition'],
            'last_decision':s['decisions'][-1] if s['decisions'] else None,
            'state_sha256':sha(s),'new_paid_calls':0}
