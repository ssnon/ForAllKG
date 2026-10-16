"""No network and no paid model invocations."""
import json
import pytest
from pipeline_core.discovery.scientific_research_control import scientific_claim_development as a
from scripts.discovery import run_claim_step2_json_recovery as fix


def step(probe=None):
    return {
      'operation':'TEST',
      'scientific_work_content': {
       'attempted_question':'Can conditional CTMC hypotheses reproduce threshold-event scaling?',
       'premises_used':['synthetic model not experimental data'],
       'reasoning_or_operations':['propose a bounded finite-state continuous-time generator'],
       'candidate_consequence':'A multi-site signal has a threshold-specific event flux.',
       'counterexample_or_obstruction':'The comparison is conditional upon transition rates.',
       'limits_and_uncertainties':['No experimental trace was provided'],
       'work_conclusion':'DERIVATION_PROPOSED'},
      'focal_claim':'Restricted independent and shared state models have distinct event fluxes.',
      'claim_maturity':'CONDITIONAL_DERIVATION',
      'research_delta':'Derived one model-dependent consequence that remains computationally unchecked.',
      'next_action':'CONTINUE',
      'next_research_question':'Does a direct CTMC probe match the proposed analytical event flux?',
      'why_next_action':'The concrete proposed generator must be tested before claiming a counterexample.',
      'probe_request':probe
    }


def probe():
    return {'kind':'ctmc_threshold','payload':{'models':[{'name':'m','conditions':{'c1':{
     'states':[{'name':'D','observable':[0]},{'name':'B','observable':[2]}],
     'transitions':[{'source':'D','target':'B','rate':1.0},{'source':'B','target':'D','rate':2.0}]
    }}}], 'conditions':['c1'],'thresholds':[0,1]}}


def test_merges_exactly_two_without_deleting_content():
    s=step(); raw=json.dumps(probe())+'\n'+json.dumps(s)
    combined=fix.canonical_step(raw)
    assert combined['probe_request']==probe()
    assert combined['scientific_work_content']==s['scientific_work_content']
    validated=a.ResearchStep.model_validate(combined)
    assert validated.operation=='TEST'
    assert a._check_probe(combined['probe_request'])['status']=='COMPUTED'


def test_process_local_parser_handles_multi_and_single():
    fix._activate()
    raw=json.dumps(probe())+'\n'+json.dumps(step())
    assert a.ResearchStep.model_validate_json(raw).probe_request==probe()
    assert a.ResearchStep.model_validate_json(json.dumps(step())).probe_request is None


@pytest.mark.parametrize('raw', [
    lambda x,y: x+'\n'+y+'\nhello',
    lambda x,y: x+' '+x+' '+y,
    lambda x,y: x+y,
])
def test_rejects_bad_extra_or_boundary(raw):
    x=json.dumps(probe());y=json.dumps(step())
    with pytest.raises((ValueError, json.JSONDecodeError)):
        fix.canonical_step(raw(x,y))


def test_rejects_conflicting_probe():
    bad=step(probe())
    with pytest.raises(ValueError,match='RESPONSE_ALREADY_HAS_PROBE'):
        fix.canonical_step(json.dumps(probe())+'\n'+json.dumps(bad))


def test_rejects_non_test_or_return():
    s=step();s['next_action']='RETURN';s['next_research_question']=None
    with pytest.raises(ValueError,match='NOT_A_TEST_CONTINUATION'):
        fix.canonical_step(json.dumps(probe())+'\n'+json.dumps(s))


def test_actual_saved_raw_recovery_and_replay_in_native_fork(tmp_path, monkeypatch):
    """End-to-end no-call replay: malformed paid raw persists unchanged."""
    import copy
    from pathlib import Path
    from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as prior
    monkeypatch.setattr(prior, 'root', lambda: tmp_path)
    src = tmp_path/'prior_native'; src.mkdir()
    (src/'manifest.json').write_text('{}','utf-8')
    (src/'state.json').write_text('{}','utf-8')
    wid = 'work_0005_mock'
    (src/'work'/wid).mkdir(parents=True)
    req = {'work_kind':'REASON','research_question':'Test a concrete independent-site model.'}
    state = {'decision_count':5, 'ended':False,
        'work_history':[{'work_id':'work_0004_1c8f243a3943','execution_status':'COMPLETED'}],
        'pending_work':{'status':'REQUESTED','work_id':wid,'request':req}}
    def nv(p):
        return {'state':copy.deepcopy(state), 'manifest':{'max_decisions':12,'model':'mock-model'}}
    monkeypatch.setattr(prior.native,'verify',nv)
    monkeypatch.setattr(prior.native,'_make_call_request',lambda c,role:{
        'model':'mock-model','temperature':0.2,'max_tokens':1500,
        'messages':[{'role':'user','content':'prior science context'}]})
    run=tmp_path/'experiments'/'science_act_continuation'/'runs'/'mock_case'
    a.prepare(src,run);a.prepare_work(run)
    # Step 1 is normal and fully committed.
    normal=json.dumps(step())
    a.next_step(run, backend=lambda request:(normal,None,'mock1'))
    # Step 2 gets real-world problematic two top-level JSON objects.
    abnormal=json.dumps(probe())+'\n'+json.dumps(step())
    # Simulate the original unpatched paid execution that saved the dual raw.
    monkeypatch.setattr(a.ResearchStep, 'model_validate_json', classmethod(fix._ORIGINAL_PARSE.__func__))
    with pytest.raises(RuntimeError,match='RAW_SAVED_USE_RECOVER_NO_RETRY'):
        a.next_step(run, backend=lambda request:(abnormal,None,'mock2'))
    rawfile=run/'claim_work'/'step_0002'/'raw_response.json'
    raw_before=rawfile.read_bytes()
    monkeypatch.setattr(fix, 'EXPECTED_RAW_SHA256',__import__('hashlib').sha256(raw_before).hexdigest())
    monkeypatch.setattr(fix, 'EXPECTED_WORK',wid)
    assert fix._guard(run)[-1]['probe_request']==probe()
    fix._activate()
    assert a.recover(run)['instrument_status']=='COMPUTED'
    verified=a.verify(run)
    assert verified['claim_work']['steps']==2
    assert verified['claim_work']['incomplete'] is None
    assert fix._audit(run,verified)['raw_response_unchanged']
    assert rawfile.read_bytes()==raw_before
    assert a.preview(run)['step']==3
