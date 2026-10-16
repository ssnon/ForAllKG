import copy
import json
from pathlib import Path
import pytest
from pipeline_core.discovery.scientific_research_control import open_literature_research as a


def opening():
    directions=[
      {'question':'What microscopically constrained intermittent process could explain Ye mode-specific blinking?',
       'scientific_reason':'The time-varying lines and concentration dependence suggest unresolved molecular kinetics.',
       'potential_route':'Build a stochastic mechanism with a new discriminating observable.',
       'overlap_or_uncertainty':'Prior blinking models may already cover this phenomenon.'},
      {'question':'Which distinct causal process could modulate several SERS vibrational modes nonuniformly?',
       'scientific_reason':'A fixed spectral gain cannot represent changed spectral directions by itself.',
       'potential_route':'Develop a physical response model and compare its predictions.',
       'overlap_or_uncertainty':'This alternative could restate known Raman selection physics.'}]
    return {'candidate_directions':directions,'selected_index':1,
      'selected_research_question':directions[1]['question'],
      'selection_rationale':'The latter problem may open genuinely distinct explanations rather than only refine a prior scalar null.',
      'possible_first_research_action':'Investigate existing physical explanations and model alternatives.',
      'prior_knowledge_status':'All saved ResearchWork conclusions are unverified and may overlap with published science.'}


def science(*,tool=None,action='CONTINUE',op='INVESTIGATE',effect='No new source has been read yet.'):
    return {'operation':op,'scientific_work_content':{
       'attempted_question':'Could an unfamiliar mode-selective causal process explain these signals?',
       'premises_used':['Ye observations supplied as primary source text.'],
       'reasoning_or_operations':['Compare causal mechanism and discriminating observational constraints.'],
       'candidate_consequence':'A candidate process could induce a changed spectral response.',
       'counterexample_or_obstruction':None,
       'limits_and_uncertainties':['Scientific novelty has not been independently established.'],
       'work_conclusion':'INCONCLUSIVE'},
       'focal_claim':'An unresolved mechanism may alter the spectral mode ratios.',
       'scientific_delta':'A new mechanism is being compared to prior published mechanisms.',
       'literature_effect':effect,
       'next_action':action,
       'next_research_question':('How do published mechanisms constrain this novel model?' if action=='CONTINUE' else None),
       'why_next_action':'We need to determine whether this is a real novelty or an existing model before stronger claims.',
       'tool_request':tool}

class FakeRecord:
    def to_dict(self):
        return {'paper_id':'literature:test','title':'Published physical blinking mechanism',
         'doi':'10.1111/example','year':2023,'abstract':'A previously published competing physical explanation.',
         'provider_references':[{'provider':'openalex','provider_id':'W1'}],
         'metadata':{'openalex_id':'https://openalex.org/W1'}}
class FakeProvider:
    def search(self,request):return [FakeRecord()]

@pytest.fixture
def scenario(monkeypatch,tmp_path):
    monkeypatch.setattr(a.lineage,'safe_path',lambda p:Path(p).expanduser().resolve())
    pages=[{'id':f'YE:P{i}','text':'Source observation Raman intermittent lines and concentrations','page':i} for i in range(1,9)]
    works=[{'id':'WORK:01','title':'Prior abstract','abstract':'Bliking model','origin':'SELECTED'}]
    source={'pages':pages,'works':works}
    content={'attempted_question':'What physical mechanism can explain blinking?',
     'candidate_consequence':'Previous conditional hypothesis not externally validated',
     'limits_and_uncertainties':['Cannot determine novelty'],'work_conclusion':'INCONCLUSIVE'}
    c={'state':{'pending_work':{'status':'EXECUTED'},'work_history':[{'work_id':'work_0006','execution_status':'COMPLETED','work_kind':'REASON','content':content}]},
      'sources':source,'manifest':{'model':'test','temperature':0.4,'max_tokens':4500}}
    monkeypatch.setattr(a.native,'verify',lambda _:copy.deepcopy(c))
    out=tmp_path/'free_open';a.prepare(tmp_path/'source_native',out)
    return out


def backend_of(value):return lambda req:(json.dumps(value),None,'fake-id')


def test_prepare_opening_and_same_input_arms(scenario):
    out=scenario
    assert a.verify(out)['opening']=='NOT_STARTED'
    assert a.choose(out,backend=backend_of(opening()))['new_paid_calls_this_command']==0
    a.prepare_arms(out)
    p=a.preview(out,'literature_enabled')
    q=a.preview(out,'no_new_literature')
    assert p['step']==q['step']==1
    assert a._step_request(out,'literature_enabled',[],1)['messages'][1]['content'].find(opening()['selected_research_question'])>=0
    assert a.verify(out)['source_history_preserved'] is True
    with pytest.raises(ValueError,match='OPENING_ALREADY_ATTEMPTED'):
        a.choose(out,backend=backend_of(opening()))


def test_search_then_literature_effect_replayed_and_control(scenario):
    out=scenario;a.choose(out,backend=backend_of(opening()));a.prepare_arms(out)
    search={'kind':'literature_search','payload':{'queries':['raman blinking microscopic origin'],'limit':2}}
    first=a.next_step(out,'literature_enabled',backend=backend_of(science(tool=search)),search_provider=FakeProvider())
    assert first['tool_status']=='SEARCH_OK' and first['papers_retrieved']==1
    assert a.verify(out)['arms']['literature_enabled']['papers_retrieved']==1
    second=science(action='RETURN',op='REVISE',effect='The discovered journal paper already discusses physical blinking kinetics.')
    a.next_step(out,'literature_enabled',backend=backend_of(second))
    a.next_step(out,'no_new_literature',backend=backend_of(science(action='RETURN',op='MODEL')))
    assert a.finish(out,'literature_enabled')['steps']==2
    assert a.finish(out,'no_new_literature')['steps']==1
    assert a.verify(out)['status']=='VERIFIED_NO_CALL'
    assert a.verify(out)['arms']['literature_enabled']['finalized']
    with pytest.raises(FileExistsError):a.finish(out,'literature_enabled')


def test_saved_tool_recovery_without_repeat_paid_call(scenario):
    out=scenario;a.choose(out,backend=backend_of(opening()));a.prepare_arms(out)
    sd=a.arm_dir(out,'literature_enabled')/'step_0001';sd.mkdir()
    req=a._step_request(out,'literature_enabled',[],1)
    a.put(sd/'request.json',req)
    a.put(sd/'attempt.json',{'request_sha256':a.sha(req),'previous_receipt_sha256':None,'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED'})
    a.put(sd/'raw_response.json',{'raw_text':json.dumps(science(tool={'kind':'literature_search','payload':{'queries':['raman blinking physical transitions']}})),'usage':None,'response_id':'already-paid'})
    with pytest.raises(RuntimeError,match='UNSETTLED_STEP'):a.preview(out,'literature_enabled')
    result=a.recover(out,'literature_enabled',search_provider=FakeProvider())
    assert result['tool_status']=='SEARCH_OK'
    assert a.verify(out)['arms']['literature_enabled']['steps']==1


def test_no_tool_in_control_arm(scenario):
    out=scenario;a.choose(out,backend=backend_of(opening()));a.prepare_arms(out)
    assert 'tool_request MUST be null' in a._step_request(out,'no_new_literature',[],1)['messages'][0]['content']
    with pytest.raises(RuntimeError,match='RAW_SAVED_USE_RECOVER_NO_RETRY|STEP_RAW_SAVED_USE_RECOVER_NO_RETRY'):
        a.next_step(out,'no_new_literature',backend=backend_of(science(tool={'kind':'literature_search','payload':{'queries':['raman blinking physical transitions']}})))
    assert a.verify(out)['arms']['no_new_literature']['incomplete']


def test_source_and_raw_tamper_detected(scenario):
    out=scenario;a.choose(out,backend=backend_of(opening()));a.prepare_arms(out)
    path=out/'sources.json';path.write_text('{}')
    with pytest.raises(ValueError,match='FROZEN_SCIENTIFIC_CONTEXT_CHANGED'):a.verify(out)


def test_opening_recovery_refuses_new_paid_invocation(scenario):
    out=scenario;op=out/'opening';op.mkdir()
    req=a._opening_request(out);a.put(op/'request.json',req)
    a.put(op/'attempt.json',{'request_sha256':a.sha(req),'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED'})
    a.put(op/'raw_response.json',{'raw_text':json.dumps(opening()),'usage':None,'response_id':'already-paid'})
    assert a.recover_opening(out)['status']=='OPENING_COMMITTED_NO_NEW_CALL'
    assert a.verify(out)['opening']=='READY'


def test_invalid_opening_selection_cannot_be_committed():
    d=opening();d['selected_index']=3
    with pytest.raises(ValueError):a.Opening.model_validate(d)


def test_no_native_mutation_ever_in_existing_run(scenario):
    out=scenario;a.choose(out,backend=backend_of(opening()));a.prepare_arms(out)
    all_paths=[str(p) for p in out.rglob('*')]
    assert all(str(out) in p for p in all_paths)
    assert not any('native' in p.split('/')[-1] for p in all_paths)
