import json
from pathlib import Path
import pytest

from pipeline_core.discovery.scientific_research_control import value_guided_continuation as v
from test_open_literature_research import opening, science, backend_of, FakeProvider, scenario
from pipeline_core.discovery.scientific_research_control import open_literature_research as a


@pytest.fixture
def frozen_lit(scenario):
    out=scenario
    a.choose(out,backend=backend_of(opening()))
    a.prepare_arms(out)
    search={'kind':'literature_search','payload':{'queries':['dynamic SERS metal surface atoms'],'limit':2}}
    a.next_step(out,'literature_enabled',backend=backend_of(science(tool=search)),search_provider=FakeProvider())
    a.next_step(out,'literature_enabled',backend=backend_of(science(action='RETURN',op='RETURN')))
    a.finish(out,'literature_enabled')
    return out


def test_prepare_from_frozen_literature_and_history_immutable(frozen_lit,tmp_path):
    old=a.arm_dir(frozen_lit,'literature_enabled')/'final_science.json'
    before=a.hfile(old)
    out=tmp_path/'value_01'
    result=v.prepare(frozen_lit,'literature_enabled',out)
    assert result['source_steps']==2
    assert result['no_fixed_step_budget']
    assert v.verify(out)['steps']==0
    request=v._request(out,[],1)
    assert 'number of steps is NOT a scientific merit score' in request['messages'][0]['content']
    assert request['messages'][1]['content'].find('Published physical blinking mechanism')>=0
    assert a.hfile(old)==before


def test_cannot_prepare_unfinished_control(frozen_lit,tmp_path):
    with pytest.raises(ValueError,match='SOURCE_ARM_MUST_BE_FROZEN'):
        v.prepare(frozen_lit,'no_new_literature',tmp_path/'invalid')


def test_unbounded_scientific_continuation_more_than_six_steps(frozen_lit,tmp_path):
    out=tmp_path/'followup';v.prepare(frozen_lit,'literature_enabled',out)
    for i in range(1,9):
        p=v.preview(out)
        assert p['step']==i and p['paid_calls_if_executed']==1
        r=v.next_step(out,backend=backend_of(science(op='MODEL')))
        assert r['step']==i and r['next_action']=='CONTINUE'
    state=v.verify(out)
    assert state['steps']==8 and state['no_fixed_step_budget']
    assert len(v.trajectory(out)['trajectory'])==8
    assert 'max_steps' not in v._request(out,[],1)['messages'][1]['content']


def test_scientist_return_and_frozen_then_new_child(frozen_lit,tmp_path):
    out=tmp_path/'followup';v.prepare(frozen_lit,'literature_enabled',out)
    r=v.next_step(out,backend=backend_of(science(op='RETURN',action='RETURN')))
    assert r['next_action']=='RETURN'
    with pytest.raises(ValueError,match='SCIENTIST_RETURNED'):
        v.preview(out)
    v.finish(out)
    assert v.verify(out)['finalized']
    after=tmp_path/'new_direction'
    v.prepare(out,'literature_enabled',after)
    assert v.verify(after)['steps']==0
    assert len(v._read(after/'inherited_context.json')['completed_research_steps'])==3


def test_control_never_uses_new_external_literature(scenario,tmp_path):
    out0=scenario
    a.choose(out0,backend=backend_of(opening()));a.prepare_arms(out0)
    a.next_step(out0,'no_new_literature',backend=backend_of(science(op='RETURN',action='RETURN')))
    a.finish(out0,'no_new_literature')
    out=tmp_path/'ctrl';v.prepare(out0,'no_new_literature',out)
    assert 'tool_request MUST be null' in v._request(out,[],1)['messages'][0]['content']
    invalid=science(op='INVESTIGATE',tool={'kind':'literature_search','payload':{'queries':['raman spectral blinking']}})
    with pytest.raises(RuntimeError,match='STEP_RAW_SAVED_USE_RECOVER_NO_RETRY'):
        v.next_step(out,backend=backend_of(invalid))
    state=v.verify(out)
    assert state['steps']==0 and state['incomplete'] is not None


def test_duplicate_tool_raw_read_once_abstract(frozen_lit,tmp_path):
    out=tmp_path/'read';v.prepare(frozen_lit,'literature_enabled',out)
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'abstract'}}
    obj=science(op='EXPLORE',tool=tool)
    raw=json.dumps(tool)+'\n'+json.dumps(tool)+'\n'+json.dumps(obj)
    def backend(_):return raw,None,'id'
    result=v.next_step(out,backend=backend)
    assert result['tool_status']=='ABSTRACT_READ'
    assert v.verify(out)['known_papers']==1
    assert v._read(out/'step_0001'/'raw_response.json')['raw_text']==raw
    assert v._read(out/'step_0001'/'normalized.json')['operation']=='EXPLORE'


def test_incomplete_raw_never_repaid(frozen_lit,tmp_path):
    out=tmp_path/'raw';v.prepare(frozen_lit,'literature_enabled',out)
    with pytest.raises(RuntimeError,match='STEP_RAW_SAVED_USE_RECOVER_NO_RETRY'):
        v.next_step(out,backend=lambda _:('Not JSON',None,'id'))
    prior=v.verify(out)
    assert prior['incomplete']
    with pytest.raises(RuntimeError,match='DO_NOT_RETRY_PAID'):
        v.preview(out)
    with pytest.raises(ValueError):v.recover(out)
    assert v._read(out/'step_0001'/'raw_response.json')['raw_text']=='Not JSON'


def test_changed_immutable_inherited_context_fails(frozen_lit,tmp_path):
    out=tmp_path/'tamper';v.prepare(frozen_lit,'literature_enabled',out)
    (out/'inherited_context.json').write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='INHERITED_SCIENCE_CHANGED'):v.verify(out)
