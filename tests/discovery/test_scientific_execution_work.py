"""Offline no-paid validation of scientifically meaningful instrument and workflow."""
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline_core.discovery.scientific_research_control import scientific_execution_probe as probe
from pipeline_core.discovery.scientific_research_control import scientific_execution_work as s
from pipeline_core.discovery.scientific_research_control import open_literature_research as a
from test_open_literature_research import opening, scenario, backend_of, FakeProvider


def model(name='physical A',on=.2,off=.3):
    return {'name':name,'states':['I','H'],'initial_probabilities':[1.,0.],
      'transitions':[{'source':'I','target':'H','rate':on,'group':'excitation'},
                     {'source':'H','target':'I','rate':off,'group':'relaxation'}],
      'observables':[{'name':'raman','values':[0.,2.]}],
      'event_sets':[{'name':'onset','bright_states':['H']}]}

def protocols():return [{'name':'light','segments':[{'duration':4.,'rate_multipliers':{'excitation':1.}}]}]

def step(tool=None,action='CONTINUE',op='MODEL'):
    return {'operation':op,'scientific_work_content':{
        'attempted_question':'Does a time-dependent finite-state mechanism generate the reported Raman blinking?',
        'premises_used':['Ye observations are frozen source data, not a model fit.'],
        'reasoning_or_operations':['Make a concrete stochastic mechanism, compute and compare an observable.'],
        'candidate_consequence':'A conditional model prediction follows from forward state propagation.',
        'counterexample_or_obstruction':'A relabeled molecule model can share all observable model dynamics.',
        'limits_and_uncertainties':['Unmeasured physical rates are model assumptions.'],
        'work_conclusion':'INCONCLUSIVE'},
        'focal_claim':'Two mechanisms could have conditional observable event-count differences.',
        'scientific_delta':'An executable model has been formulated instead of a qualitative narrative.',
        'literature_effect':'No new literature was accessed in this step.',
        'next_action':action,
        'next_research_question':'Which alternate microscopic generator gives a distinct burst-count prediction?' if action=='CONTINUE' else None,
        'why_next_action':'The assumptions require a concrete calculation and critical comparison.',
        'tool_request':tool}


def test_two_state_closed_form():
    q=probe.evaluate({'models':[model()], 'protocols':protocols()})
    s=q['results'][0]['protocols'][0]['segments'][0]
    expected=.2/(.2+.3)*(1-np.exp(-(.2+.3)*4))
    assert s['state_probabilities']['H']==pytest.approx(expected,rel=1e-9)
    assert s['observables'][0]['mean_at_segment_end']==pytest.approx(2*expected)
    # Integral of J_on = alpha * P(I); exact known expression.
    expected_count=.2*(4 - .2/(.2+.3)*(4-(1-np.exp(-2))/.5))
    assert s['events'][0]['expected_entries_in_segment']==pytest.approx(expected_count,rel=1e-8)


def test_different_hidden_label_same_observables_detected():
    x=model('surface');y=model('molecule')
    r=probe.evaluate({'models':[x,y],'protocols':protocols()})
    assert r['model_comparisons'][0]['status']=='NUMERICALLY_INDISTINGUISHABLE_UNDER_TESTED_PROTOCOL'
    assert r['model_comparisons'][0]['not_a_general_identifiability_proof']


def test_distinct_models_and_control_sequences():
    r=probe.evaluate({'models':[model('a',.1),model('b',.5)],'protocols':protocols()})
    assert r['model_comparisons'][0]['status']=='PREDICTIONS_DIFFER_UNDER_TESTED_PROTOCOL'
    dark={'name':'dark','segments':[{'duration':3,'rate_multipliers':{'excitation':0}}]}
    v=probe.evaluate({'models':[model()],'protocols':[dark]})
    assert v['results'][0]['protocols'][0]['segments'][0]['events'][0]['expected_entries_in_segment']==pytest.approx(0)


@pytest.mark.parametrize('modifier',[
    lambda r: r['models'][0].update(initial_probabilities=[.6,.5]),
    lambda r: r['models'][0]['transitions'][0].update(source='missing'),
    lambda r: r['protocols'][0]['segments'][0]['rate_multipliers'].update(other=1),
    lambda r: r['models'][0]['observables'][0].update(values=[1.]),
    lambda r: r['models'][0]['transitions'][0].update(rate='__import__("os")'),
])
def test_refuse_invalid_specs(modifier):
    r={'models':[model()],'protocols':protocols()}
    modifier(r)
    with pytest.raises(ValueError):probe.evaluate(r)


def test_fresh_opening_reused_and_source_unchanged(scenario,tmp_path):
    a.choose(scenario,backend=backend_of(opening()))
    original_hash=a.hfile(scenario/'opening'/'normalized.json')
    out=tmp_path/'science'
    result=s.prepare(scenario,out)
    assert 'mode' in result['opening_question'] or 'causal' in result['opening_question']
    assert result['source_untouched']
    assert original_hash==a.hfile(scenario/'opening'/'normalized.json')
    assert s.preview(out)['step']==1
    assert 'FINITE_CTMC' in s.request(out,[],1)['messages'][0]['content'] or 'finite CTMC' in s.request(out,[],1)['messages'][0]['content']


def test_scientific_execution_calculation_then_revision(scenario,tmp_path):
    a.choose(scenario,backend=backend_of(opening()))
    out=tmp_path/'work';s.prepare(scenario,out)
    tool={'kind':'ctmc_transient','payload':{'models':[model('surface'),model('molecule')], 'protocols':protocols()}}
    first=s.next_step(out,backend=backend_of(step(tool,op='MODEL')))
    assert first['tool_status']=='COMPUTED_CONDITIONAL_MODEL_RESULT'
    assert s.verify(out)['steps']==1
    req=s.request(out,s.rows(out)[0],2)
    assert 'NUMERICALLY_INDISTINGUISHABLE_UNDER_TESTED_PROTOCOL' in req['messages'][1]['content']
    follow=step(action='RETURN',op='REVISE');follow['scientific_delta']='Same emissions and generator imply this comparison did not discriminate mechanisms.'
    result=s.next_step(out,backend=backend_of(follow))
    assert result['next_action']=='RETURN'
    assert s.finish(out)['status']=='SCIENTIFIC_EXECUTION_FROZEN_NO_CALL'
    assert s.verify(out)['finalized']
    with pytest.raises(ValueError,match='ALREADY_FINALIZED'):s.preview(out)


def test_saved_raw_no_retry_and_recovery(scenario,tmp_path):
    a.choose(scenario,backend=backend_of(opening()))
    out=tmp_path/'work';s.prepare(scenario,out)
    bad='not json'
    with pytest.raises(RuntimeError,match='RAW_SAVED_USE_RECOVER_NO_RETRY'):
        s.next_step(out,backend=lambda _: (bad,None,'paid-response'))
    assert s.verify(out)['steps']==0
    assert s.verify(out)['incomplete']
    with pytest.raises(RuntimeError,match='SAVED_STEP_INCOMPLETE_DO_NOT_REPAY'):
        s.preview(out)
    with pytest.raises(ValueError,match='INVALID_SAVED_SCIENCE_JSON'):
        s.recover(out)
    assert s.read(out/'step_0001'/'raw_response.json')['raw_text']==bad


def test_parse_three_objects_matching_and_reject_conflict():
    tool={'kind':'ctmc_transient','payload':{'models':[model()],'protocols':protocols()}}
    raw='\n'.join([json.dumps(tool),json.dumps(tool),json.dumps(step(tool,op='EXPLORE'))])
    obj=s.parse(raw)
    assert obj['operation']=='EXPLORE' and obj['tool_request']==tool
    other=tool.copy();other['payload']={'models':[],'protocols':[]}
    with pytest.raises(ValueError):s.parse(json.dumps(tool)+'\n'+json.dumps(other)+'\n'+json.dumps(step(tool)))
    with pytest.raises(ValueError):s.parse(json.dumps(step())+'EXTRA')


def test_literature_search_optional(scenario,tmp_path):
    a.choose(scenario,backend=backend_of(opening()))
    out=tmp_path/'work';s.prepare(scenario,out)
    tool={'kind':'literature_search','payload':{'queries':['dynamic SERS hot spots'],'limit':2}}
    r=s.next_step(out,backend=backend_of(step(tool,op='EXPLORE')),search_provider=FakeProvider())
    assert r['tool_status']=='SEARCH_OK'
    assert s.verify(out)['known_papers']==1


def test_multiple_steps_without_fixed_budget(scenario,tmp_path):
    a.choose(scenario,backend=backend_of(opening()))
    out=tmp_path/'work';s.prepare(scenario,out)
    for _ in range(7):
        assert s.next_step(out,backend=backend_of(step()))['next_action']=='CONTINUE'
    assert s.verify(out)['steps']==7
    assert len(s.trajectory(out)['trajectory'])==7

def test_equal_stationary_different_transient_is_detected():
    # Same pi_H=1/3, but transition speed differs by factor 10.
    r=probe.evaluate({'models':[model('slow',.1,.2),model('fast',1.,2.)],
                      'protocols':[{'name':'short_pulse','segments':[{'duration':.5,'rate_multipliers':{}}]}]})
    assert r['model_comparisons'][0]['status']=='PREDICTIONS_DIFFER_UNDER_TESTED_PROTOCOL'
    low=r['results'][0]['protocols'][0]['segments'][0]['state_probabilities']['H']
    high=r['results'][1]['protocols'][0]['segments'][0]['state_probabilities']['H']
    assert high>low
