import json
from pathlib import Path
import pytest

from test_open_literature_research import (
    scenario, science, opening, FakeProvider, backend_of,
)
from pipeline_core.discovery.scientific_research_control import open_explore_literature_compat as compat
from pipeline_core.discovery.scientific_research_control import open_literature_research as work
from pipeline_core.discovery.scientific_research_control import scientific_literature_work as lit


def test_preserves_explore_with_valid_literature_tool_only():
    tool={'kind':'literature_search','payload':{'queries':['SERS atomic reconstruction gold nanorod'],'limit':3}}
    row=science(tool=tool,op='EXPLORE')
    with pytest.raises(ValueError):lit._parse(json.dumps(row))
    parsed=compat.parse_compatible(json.dumps(row))
    assert parsed['operation']=='EXPLORE'
    assert parsed['tool_request']==tool
    assert parsed['next_action']=='CONTINUE'


def test_malformed_or_noncontinuing_response_refused():
    tool={'kind':'literature_search','payload':{'queries':['SERS atomic reconstruction']}}
    candidates=[
      science(tool=tool,op='EXPLORE',action='RETURN'),
      science(tool={'kind':'wrong_kind','payload':{}},op='EXPLORE'),
      science(tool={'kind':'literature_search','payload':'bad'},op='EXPLORE'),
      science(tool=tool,op='EXPLORE') | {'extra_unexpected_key':'wrong'},
      science(tool=tool,op='EXPLORE') | {'scientific_delta':'short'},
    ]
    for row in candidates:
        with pytest.raises(ValueError):compat.parse_compatible(json.dumps(row))


def test_recover_real_saved_explore_search_and_continue_without_second_paid(scenario, monkeypatch):
    out=scenario
    work.choose(out,backend=backend_of(opening()))
    work.prepare_arms(out)
    p=work.arm_dir(out,'literature_enabled')/'step_0001'
    p.mkdir()
    request=work._step_request(out,'literature_enabled',[],1)
    work.put(p/'request.json',request)
    work.put(p/'attempt.json',{
      'request_sha256':work.sha(request),
      'previous_receipt_sha256':None,
      'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED',
    })
    tool={'kind':'literature_search','payload':{'queries':['SERS gold atom reconstruction mechanism'],'limit':2}}
    work.put(p/'raw_response.json',{
      'raw_text':json.dumps(science(tool=tool,op='EXPLORE'),ensure_ascii=False),
      'usage':{'cost':'prior-paid'},'response_id':'saved-paid'
    })
    hash_before=work.hfile(p/'raw_response.json')
    engine_before=work.hfile(Path(work.__file__))
    literature_before=work.hfile(Path(lit.__file__))
    assert compat.inspect(out,'literature_enabled')['parser_compatibility'] is True
    original_dispatch=lit._dispatch
    monkeypatch.setattr(lit,'_dispatch', lambda tool_request,rows,**kwargs: original_dispatch(tool_request,rows,**(kwargs | {'search_provider':FakeProvider()})))
    recovered=compat.recover(out,'literature_enabled')
    assert recovered['step']==1
    assert compat.verify(out)['arms']['literature_enabled']['steps']==1
    assert work.read(p/'normalized.json')['operation']=='EXPLORE'
    assert work.hfile(p/'raw_response.json')==hash_before
    assert work.hfile(Path(work.__file__))==engine_before
    assert work.hfile(Path(lit.__file__))==literature_before
    assert work.read(p/'explore_tool_compat_audit.json')['scientific_text_modified'] is False
    assert compat.preview(out,'literature_enabled')['step']==2
    with compat.compatibility_scope():
        work.next_step(out,'literature_enabled',backend=backend_of(science(op='REVISE',action='RETURN')))
    assert compat.finish(out,'literature_enabled')['steps']==2
    assert compat.verify(out)['arms']['literature_enabled']['finalized'] is True


def test_control_arm_tool_rejected_even_if_explore(scenario):
    out=scenario;work.choose(out,backend=backend_of(opening()));work.prepare_arms(out)
    p=work.arm_dir(out,'no_new_literature')/'step_0001';p.mkdir()
    req=work._step_request(out,'no_new_literature',[],1)
    work.put(p/'request.json',req)
    work.put(p/'attempt.json',{'request_sha256':work.sha(req),'previous_receipt_sha256':None})
    work.put(p/'raw_response.json',{'raw_text':json.dumps(science(op='EXPLORE',tool={'kind':'literature_search','payload':{'queries':['SERS blinking']}}))})
    with pytest.raises(ValueError,match='CONTROL_ARM_CANNOT_USE_LITERATURE_TOOL'):
        compat.recover(out,'no_new_literature')
    assert compat.verify(out)['arms']['no_new_literature']['incomplete']


def test_ordinary_investigate_tool_unchanged():
    tool={'kind':'literature_search','payload':{'queries':['Raman nanorod']}}
    row=science(op='INVESTIGATE',tool=tool)
    assert compat.parse_compatible(json.dumps(row))==lit._parse(json.dumps(row))
