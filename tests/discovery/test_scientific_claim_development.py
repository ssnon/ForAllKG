"""Offline science-action and lineage integration tests; no paid LLM calls."""
from __future__ import annotations
import copy,json
from pathlib import Path
import pytest
from pipeline_core.discovery.scientific_research_control import scientific_claim_development as a
from pipeline_core.discovery.scientific_research_control import finite_observable_probe as probe
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as prior


def models():
    return {'models':[
        {'name':'independent','conditions':{'c=1':{'kind':'independent_binary_sites',
           'background':[0],'sites':[{'p_on':0.2,'emission':[1]},{'p_on':0.2,'emission':[1]}]}}},
        {'name':'shared','conditions':{'c=1':{'kind':'finite_shared_states',
           'states':[{'probability':0.64,'observable':[0]},{'probability':0.36,'observable':[2]}]}}}],
        'conditions':['c=1'],'thresholds':[0,1]}


def content():
    return {'attempted_question':'Can candidate physical generators lead to discriminating observables?',
        'premises_used':['All states are hypothetical, not Ye data'],
        'reasoning_or_operations':['Enumerate the exact distribution and compare two finite models'],
        'candidate_consequence':'Restricted models predict different threshold exceedance functions.',
        'counterexample_or_obstruction':'Flexible latent-state models can reproduce a given finite marginal.',
        'limits_and_uncertainties':['No timing or empirical physical identification is computed'],
        'work_conclusion':'DERIVATION_PROPOSED'}


def response(probe_request=None,action='CONTINUE',maturity='EXPLORATORY'):
    return json.dumps({'operation':'TEST' if probe_request else 'DERIVE',
        'scientific_work_content':content(),
        'focal_claim':'Threshold observations may discriminate restricted finite models.',
        'claim_maturity':maturity,'research_delta':'Derived or checked a limited mathematical condition, not a general physical fact.',
        'next_action':action,'next_research_question':'Does the actual computation disprove the claimed equivalence?' if action=='CONTINUE' else None,
        'why_next_action':'Concrete follow-up on quantitative difference is useful.' if action=='CONTINUE' else 'This particular restricted claim has been developed as far as possible.',
        'probe_request':probe_request})


def test_finite_observable_exact_counterexample():
    v=probe.evaluate(models())
    ia=v['models'][0]['conditions']['c=1']['threshold_exceedance']
    ib=v['models'][1]['conditions']['c=1']['threshold_exceedance']
    assert ia[0]['probabilities'][0]==pytest.approx(0.36)
    assert ia[1]['probabilities'][0]==pytest.approx(0.04)
    assert ib[0]['probabilities'][0]==pytest.approx(0.36)
    assert ib[1]['probabilities'][0]==pytest.approx(0.36)
    assert v['comparisons'][0]['total_variation_full_observable_distribution']==pytest.approx(0.32)
    assert not v['comparisons'][0]['full_distribution_equivalent_at_this_condition']


def test_finite_observable_identical_emission_distribution():
    m=models()
    m['models'][1]['conditions']['c=1']={'kind':'finite_shared_states','states':[
        {'probability':0.64,'observable':[0]},{'probability':0.32,'observable':[1]},
        {'probability':0.04,'observable':[2]}]}
    v=probe.evaluate(m)
    assert v['comparisons'][0]['full_distribution_equivalent_at_this_condition']
    assert v['comparisons'][0]['total_variation_full_observable_distribution']<1e-10


def test_finite_observable_rejects_bad_inputs():
    for bad in [{'probability':1.1,'observable':[0]}, {'probability':0.2,'observable':[True]}]:
        m=models();m['models'][1]['conditions']['c=1']['states']=[bad]
        with pytest.raises(ValueError):probe.evaluate(m)
    m=models();m['models'][0]['conditions']['c=1']['sites'][0]['emission']=[1.25]
    with pytest.raises(ValueError):probe.evaluate(m)


@pytest.fixture
def native_mock(tmp_path,monkeypatch):
    monkeypatch.setattr(prior,'root',lambda:tmp_path)
    src=tmp_path/'native_W08_after_decision5';src.mkdir()
    (src/'manifest.json').write_text('{"schema":"mock"}')
    wid='work_0005_native_selected'
    request={'work_kind':'REASON','research_question':'What constructive joint-observable constraint can be rigorously derived?'}
    (src/'work'/wid).mkdir(parents=True)
    (src/'work'/wid/'request.json').write_text(json.dumps(request))
    (src/'state.json').write_text('{}')
    state={'decision_count':5,'ended':False,
          'work_history':[{'work_id':'work_0004_1c8f243a3943','execution_status':'COMPLETED'}],
          'pending_work':{'status':'REQUESTED','work_id':wid,'request':request}}
    store={'src':copy.deepcopy(state),'fork':copy.deepcopy(state)}
    def nv(p):
        return {'state':copy.deepcopy(store['src' if Path(p)==src else 'fork']),
                'manifest':{'max_decisions':12,'model':'mock-LLM'}}
    monkeypatch.setattr(prior.native,'verify',nv)
    monkeypatch.setattr(prior.native,'_make_call_request',lambda c,role:{
      'model':'mock-LLM','temperature':0.2,'max_tokens':3500,
      'messages':[{'role':'user','content':'native past science, W04 failures, Decision5 request'}]})
    run=tmp_path/'experiments'/'science_act_continuation'/'runs'/'followup'
    return src,run,wid,store


def test_native_lineage_claim_work_and_commit(native_mock,monkeypatch):
    src,run,wid,store=native_mock
    r=a.prepare(src,run)
    assert r['decisions_inherited']==5
    assert a.prepare_work(run)['work_id']==wid
    assert a.preview(run)['step']==1
    first=a.next_step(run,backend=lambda req:(response({'kind':'finite_observable','payload':models()}),None,'test1'))
    assert first['instrument_status']=='COMPUTED'
    second=a.next_step(run,backend=lambda req:(response(action='RETURN',maturity='CONDITIONAL_DERIVATION'),None,'test2'))
    assert second['next_action']=='RETURN'
    assert a.finish(run)['steps']==2
    def native_commit(p,backend,max_new_calls):
        req=a.native._make_call_request(a.native.verify(p),'work')
        raw,usage,rid=backend(req)
        item={'content':json.loads(raw),'execution_status':'COMPLETED'}
        (Path(p)/'work'/wid/'product.json').write_text(json.dumps(item))
        store['fork']['pending_work']['status']='EXECUTED'
        store['fork']['work_history'].append({'work_id':wid,'execution_status':'COMPLETED'})
        return {'event':10,'new_paid_calls_this_command':0}
    monkeypatch.setattr(a.native,'execute',native_commit)
    assert a.commit_native(run)['status']=='NATIVE_CLAIM_WORK_COMMITTED_NO_PAID_CALL'
    assert a.verify(run)['claim_work']['last_claim_maturity']=='CONDITIONAL_DERIVATION'
    assert not (src/'work'/wid/'product.json').exists()
    store['fork']['decision_count']=6
    store['fork']['pending_work']={'work_id':'work_0006','status':'REQUESTED','request':{'work_kind':'REASON'}}
    assert a.verify(run)['claim_work']['finalized']


def test_early_return_is_respected(native_mock):
    src,run,_,_=native_mock;a.prepare(src,run);a.prepare_work(run)
    a.next_step(run,backend=lambda req:(response(action='RETURN',maturity='INCONCLUSIVE'),None,'m'))
    assert a.finish(run)['steps']==1
    with pytest.raises(ValueError,match='RESEARCH_ALREADY_FROZEN'):a.preview(run)


def test_tamper_detected(native_mock):
    src,run,_,_=native_mock;a.prepare(src,run);a.prepare_work(run)
    a.next_step(run,backend=lambda req:(response(action='RETURN'),None,'m'))
    p=run/'claim_work'/'step_0001'/'normalized.json'
    data=json.loads(p.read_text());data['research_delta']='Faked scientific result!';p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='SCIENTIFIC_RESULT_CHANGED'):a.verify(run)


def test_uncertain_paid_attempt_never_retried(native_mock):
    src,run,_,_=native_mock;a.prepare(src,run);a.prepare_work(run)
    with pytest.raises(RuntimeError):a.next_step(run,backend=lambda req:(_ for _ in ()).throw(RuntimeError('timeout after billing')))
    with pytest.raises(RuntimeError,match='UNSETTLED'):a.preview(run)
    with pytest.raises(RuntimeError,match='BILLING_UNCERTAIN'):a.recover(run)


def test_bad_instrument_model_reports_rejection(native_mock):
    src,run,_,_=native_mock;a.prepare(src,run);a.prepare_work(run)
    bad=models();bad['models'][0]['conditions']['c=1']['sites'][0]['p_on']=100
    result=a.next_step(run,backend=lambda req:(response({'kind':'finite_observable','payload':bad}),None,'m'))
    assert result['instrument_status']=='REJECTED_INVALID_MODEL'
    assert a.verify(run)['claim_work']['steps']==1


def test_ctmc_threshold_flux_with_native_solver():
    d={'models':[{'name':'physical_A','conditions':{
        'c1':{'states':[{'name':'D','observable':[0]},{'name':'B','observable':[2]}],
              'transitions':[{'source':'D','target':'B','rate':1.},
                             {'source':'B','target':'D','rate':2.}]},
        'c2':{'states':[{'name':'D','observable':[0]},{'name':'B','observable':[2]}],
              'transitions':[{'source':'D','target':'B','rate':2.},
                             {'source':'B','target':'D','rate':2.}]}
    }}], 'conditions':['c1','c2'], 'thresholds':[0,1,2]}
    v=probe.evaluate_kinetic(d)
    m1=v['models'][0]['conditions']['c1']['threshold_metrics']
    m2=v['models'][0]['conditions']['c2']['threshold_metrics']
    assert m1[0]['mode_metrics'][0]['occupancy']==pytest.approx(1/3)
    assert m1[0]['mode_metrics'][0]['entry_flux']==pytest.approx(2/3)
    assert m1[0]['mode_metrics'][0]['mean_bright_duration']==pytest.approx(0.5)
    assert m2[0]['mode_metrics'][0]['occupancy']==pytest.approx(0.5)
    assert m2[0]['mode_metrics'][0]['entry_flux']==pytest.approx(1.)
    assert m1[2]['mode_metrics'][0]['occupancy']==0
    assert m1[2]['mode_metrics'][0]['entry_flux']==0


def test_rejects_restarting_w08_work4_before_decision5(native_mock):
    src,run,_,store=native_mock
    store['src']['decision_count']=4
    with pytest.raises(ValueError,match='COMPLETE_W08_WORK4'):
        a.prepare(src,run)
    assert not run.exists()
