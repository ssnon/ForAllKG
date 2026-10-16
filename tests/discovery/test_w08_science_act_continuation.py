"""The suite exercises new research behavior without paid model calls or original-episode mutation."""
from __future__ import annotations
import copy, json
from pathlib import Path
import pytest
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as a


def content(kind='DERIVATION_PROPOSED'):
    return {'attempted_question':'Can this autonomous native research question be developed?',
       'premises_used':['Hypothetical finite model, not a Ye measurement'],
       'reasoning_or_operations':['Construct states D and B and calculate stationary switching flux'],
       'candidate_consequence':'Conditional finite-state flux may depend on rate changes.',
       'counterexample_or_obstruction':'Other physical mechanisms may have the same Markov generator.',
       'limits_and_uncertainties':['The transition rates are illustrative, not fitted'],
       'work_conclusion':kind}


def response(*,probe=None,nxt='CONTINUE'):
    return json.dumps({'operation':'DERIVE','scientific_work_content':content(),
       'research_delta':'Completed a conditional model calculation, not a physical validation.',
       'next_action':nxt,'next_research_question':'Do the proposed candidate and null predict different fluxes?' if nxt=='CONTINUE' else None,
       'why_next_action':'The calculation can test this narrow theoretical condition.' if nxt=='CONTINUE' else 'No further justified analytical result is available.',
       'probe_request':probe})


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'root',lambda:tmp_path)
    ref=tmp_path/'source_w08';ref.mkdir();(ref/'manifest.json').write_text('{"test":1}')
    wid='work_0005_sample'
    req={'work_kind':'REASON','research_question':'An independently selected physical question?'}
    (ref/'work'/wid).mkdir(parents=True)
    (ref/'work'/wid/'request.json').write_text(json.dumps(req))
    (ref/'state.json').write_text('{}')
    state={'decision_count':5,'ended':False,'work_history':[],
       'pending_work':{'status':'REQUESTED','work_id':wid,'request':req}}
    store={'source_state':copy.deepcopy(state),'fork_state':copy.deepcopy(state)}
    def fake_verify(p):
        s=store['source_state'] if Path(p)==ref else store['fork_state']
        return {'state':copy.deepcopy(s),'manifest':{'max_decisions':12,'model':'mock-model','temperature':0.5,'max_tokens':4500}}
    monkeypatch.setattr(a.native,'verify',fake_verify)
    def make_req(c,role):
        assert role=='work'
        return {'model':'mock-model','temperature':0.5,'max_tokens':4500,'messages':[{'role':'user','content':'native W08 scientific context'}]}
    monkeypatch.setattr(a.native,'_make_call_request',make_req)
    run=tmp_path/'experiments'/'science_act_continuation'/'runs'/'case'
    return ref,run,wid,store


def probe():
    return {'models':[{'name':'molecule state','states':['D','B'],
        'transitions':[{'source':'D','target':'B','rate':1.,'group':'forward'},
                       {'source':'B','target':'D','rate':2.,'group':'back'}],
        'observables':[{'name':'bright','bright_states':['B']}]}],
        'sweep':{'rate_group':'forward','multipliers':[0.5,1.,2.]}}


def test_native_fork_research_and_commit(sandbox,monkeypatch):
    ref,run,wid,store=sandbox
    assert a.prepare(ref,run)['pending_work_kind']=='REASON'
    assert a.choose(run)['status']=='EXISTING_NATIVE_DECISION_RETAINED_NO_CALL'
    assert a.prepare_work(run)['work_id']==wid
    assert a.preview(run)['next_step']==1
    def first(req):return response(probe=probe()),None,'call1'
    r=a.next_step(run,backend=first)
    assert r['instrument_status']=='COMPUTED'
    sd=run/'companion'/wid/'step_0001'/'instrument_result.json'
    assert a.read(sd)['instrument_result']['result']['models'][0]['results'][0]['observables'][0]['bright_occupancy']>0
    a.next_step(run,backend=lambda req:(response(nxt='RETURN'),None,'call2'))
    assert a.finish(run)['steps']==2
    def fake_execute(p,backend,max_new_calls):
        req=a.native._make_call_request(a.native.verify(p),'work')
        raw,usage,rid=backend(req)
        product={'content':json.loads(raw),'execution_status':'COMPLETED'}
        (Path(p)/'work'/wid/'product.json').write_text(json.dumps(product))
        store['fork_state']['pending_work']['status']='EXECUTED'
        store['fork_state']['work_history'].append({'work_id':wid,'execution_status':'COMPLETED'})
        return {'event':11,'new_paid_calls_this_command':0}
    monkeypatch.setattr(a.native,'execute',fake_execute)
    assert a.commit_native(run)['research_steps']==2
    assert a.inspect(run)['research_work'][wid]['native_status']=='EXECUTED'
    assert (ref/'work'/wid/'product.json').exists() is False
    # A future native Decision can advance without erasing the earlier Work link.
    store['fork_state']['decision_count']=6
    store['fork_state']['pending_work']={'work_id':'another_work','status':'REQUESTED','request':{'work_kind':'REASON'}}
    assert a.inspect(run)['research_work'][wid]['native_status']=='FOLLOWED_BY_NEXT_DECISION'


def test_no_unnecessary_tool_and_return(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run);a.prepare_work(run)
    r=a.next_step(run,backend=lambda req:(response(nxt='RETURN'),None,'m'))
    assert r['instrument_status'] is None
    assert a.finish(run)['steps']==1
    with pytest.raises(ValueError,match='RESEARCHER_RETURNED|WORK_ALREADY'):
        a.preview(run)


def test_reject_source_mutation_and_attempted_native_work(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run)
    (run/'native'/'manifest.json').write_text('bad')
    with pytest.raises(ValueError,match='INHERITED_NATIVE_HISTORY_CHANGED'):a.verify(run)
    (ref/'work'/wid/'attempt.json').write_text('{}')
    with pytest.raises(ValueError,match='UNSETTLED_WORK'):a.prepare(ref,run.parent/'newrun')


def test_replay_rejects_tampered_science_and_no_retry(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run);a.prepare_work(run)
    a.next_step(run,backend=lambda req:(response(probe=probe()),None,'m'))
    p=run/'companion'/wid/'step_0001'/'normalized.json'
    x=a.read(p);x['research_delta']='Altered scientific claim';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='NORMALIZED_CHANGED'):a.verify(run)


def test_incomplete_attempt_never_repaid(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run);a.prepare_work(run)
    def crash(req):raise RuntimeError('simulated timeout after attempted billing')
    with pytest.raises(RuntimeError,match='timeout'):
        a.next_step(run,backend=crash)
    with pytest.raises(RuntimeError,match='UNSETTLED'):
        a.preview(run)
    with pytest.raises(RuntimeError,match='BILLING_UNCERTAIN'):
        a.recover(run)


def test_native_is_distinct_from_independent_pilot(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run)
    lock=a.read(run/'fork_lock.json')
    assert lock['reference_state_sha256']==a.digest(store['source_state'])
    assert a.read(run/'native'/'work'/wid/'request.json')['research_question']==store['source_state']['pending_work']['request']['research_question']


def test_reject_uninterpreted_last_probe(sandbox):
    ref,run,wid,store=sandbox
    a.prepare(ref,run);a.prepare_work(run)
    a.next_step(run,backend=lambda req:(response(nxt='CONTINUE'),None,'m1'))
    a.next_step(run,backend=lambda req:(response(nxt='CONTINUE'),None,'m2'))
    with pytest.raises(RuntimeError,match='RAW_SAVED_NO_RETRY'):
        a.next_step(run,backend=lambda req:(response(probe=probe()),None,'m3'))
    with pytest.raises(ValueError,match='LAST_STEP_HAS_UNINTERPRETED_PROBE'):
        a.recover(run)
