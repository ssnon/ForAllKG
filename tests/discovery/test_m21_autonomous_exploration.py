"""M21 must change science exploration and evidence; never claim novelty from counts."""
from pathlib import Path
import json
import pytest

from pipeline_core.discovery.scientific_research_control import m21_autonomous_exploration as m
from pipeline_core.discovery.scientific_research_control import open_literature_research as openr
from pipeline_core.discovery.scientific_research_control import scientific_execution_work as exe
from test_open_literature_research import opening, backend_of, FakeProvider, scenario
from test_scientific_execution_work import step


def divergent():
    return {'directions':[
      {'question':f'What distinct physical causal explanation {i} could produce the mode-selective Raman observations?',
       'causal_possibility':f'A distinct family of physical changes {i} modifies the Raman tensor or local optical response.',
       'ye_observation':'Mode-selective blinking reported in original Ye figures.',
       'expected_or_testable_consequence':'A model should reproduce a measurable Raman spectral change with a justified physical constraint.',
       'reason_not_to_assume_novel':'Closely related theory might already exist in prior spectroscopy studies.',
       'research_action':'Read competing primary physics and derive a quantitative testable consequence.',
       'literature_queries':[f'mode-selective nanoscale Raman switching mechanism {i}']}
      for i in range(3)],
      'intellectual_search_rationale':'The three initial directions must be examined with actual prior papers before research is selected.'}


def selection():
    return {'selected_index':2,
      'selection_rationale':'This path could establish an independently checkable physical constraint that is not merely a relabeled finite state.',
      'first_executable_scientific_action':'Use retrieved physical assumptions to derive a concrete relationship for the measured Raman mode statistics.',
      'overlap_or_prior_art_caveat':'The available paper abstracts do not establish a genuinely novel hypothesis or exclude previous mechanisms.',
      'strongest_competing_explanation':'A spatially heterogeneous collection of independent switching molecular subpopulations.',
      'literature_assessment':[{'direction_index':2,'paper_id':'literature:test',
        'evidence_or_uncertainty':'The retrieved abstract indicates a related mechanism without enough details to establish equivalence.',
        'influence_on_direction':'The proposed research must first test whether its physical constraint is genuinely different.'}],
      'rejected_alternatives_reason':'The other directions could be researched later but appear less diagnostic under the present evidence.',
      'epistemic_status':'EXPLORATORY_DIRECTION_NOT_A_DISCOVERY'}


@pytest.fixture
def prepared(scenario,tmp_path):
    openr.choose(scenario,backend=backend_of(opening()))
    work=tmp_path/'m20_run'
    exe.prepare(scenario,work)
    exe.next_step(work,backend=backend_of(step(action='RETURN',op='RETURN')))
    assert exe.verify(work)['last_action']=='RETURN'
    out=tmp_path/'m21'
    assert m.prepare(scenario,work,out)['status']=='M21_PREPARED_NO_CALL'
    return scenario,work,out


def test_real_literature_retrieval_and_existing_m20_execution_bridge(prepared,tmp_path):
    op,work,out=prepared
    old=m.hfile(op/'opening'/'normalized.json')
    old_work=m.hfile(work/'step_0001'/'normalized.json')
    assert m.preview(out,'diverge')['next_call_is_paid']
    assert m.paid(out,'diverge',backend=backend_of(divergent()))['direction_count']==3
    assert m.search(out,provider=FakeProvider())['papers']==3
    assert m.preview(out,'select')['next_call_is_paid']
    assert m.paid(out,'select',backend=backend_of(selection()))['selected_index']==2
    target=tmp_path/'m21_science'
    result=m.export_work(out,target)
    assert result['new_execution_engine'] is False
    assert exe.verify(target)['steps']==0
    assert exe.preview(target)['step']==1
    ctx=m.read(target/'scientific_context.json')
    assert ctx['autonomous_opening']['selected_index']==2
    assert 'M21_external_literature_metadata_abstract' in ctx['scientific_sources']
    assert ctx['prior_memory_unverified']['m20_scientific_work']['not_a_discovery_claim']
    assert m.verify(out)['exported_work']=='VERIFIED'
    assert old==m.hfile(op/'opening'/'normalized.json')
    assert old_work==m.hfile(work/'step_0001'/'normalized.json')


def test_no_fabricated_paper_id(prepared):
    _,_,out=prepared
    m.paid(out,'diverge',backend=backend_of(divergent()))
    m.search(out,provider=FakeProvider())
    invented=selection();invented['literature_assessment'][0]['paper_id']='not-returned'
    with pytest.raises(RuntimeError,match='RAW_SAVED_USE_RECOVER_NO_RETRY'):
        m.paid(out,'select',backend=backend_of(invented))
    assert m.verify(out)['selection']=='RAW_SAVED_RECOVER_NO_RETRY'
    with pytest.raises(ValueError,match='FABRICATED'):
        m.recover(out,'select')
    with pytest.raises(ValueError,match='SELECT_NOT_READY'):
        m.preview(out,'select')


def test_no_double_paid_call_and_lock(prepared):
    _,_,out=prepared
    m.paid(out,'diverge',backend=backend_of(divergent()))
    with pytest.raises(ValueError,match='DIVERGE_ALREADY_ATTEMPTED'):
        m.paid(out,'diverge',backend=backend_of(divergent()))
    evidence=out/'evidence.json'
    evidence.write_text('{}')
    with pytest.raises(ValueError,match='M21_FROZEN_EVIDENCE_CHANGED'):
        m.verify(out)


def test_search_empty_does_not_certify_novelty(prepared):
    _,_,out=prepared
    m.paid(out,'diverge',backend=backend_of(divergent()))
    class NoPaper:
        def search(self,request):return []
    r=m.search(out,provider=NoPaper())
    assert r['papers']==0
    assert r['novelty_certified'] is False
    assert m.verify(out)['literature']=='READY'


def test_source_run_must_have_returned(scenario,tmp_path):
    openr.choose(scenario,backend=backend_of(opening()))
    w=tmp_path/'not_finished';exe.prepare(scenario,w)
    with pytest.raises(ValueError,match='M20_MUST_HAVE_RETURNED'):
        m.prepare(scenario,w,tmp_path/'m21')
