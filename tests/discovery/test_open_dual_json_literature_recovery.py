import json
from pathlib import Path
import pytest

from test_open_literature_research import scenario, science, opening, FakeProvider, backend_of
from pipeline_core.discovery.scientific_research_control import open_dual_json_literature_recovery as compat
from pipeline_core.discovery.scientific_research_control import open_literature_research as work
from pipeline_core.discovery.scientific_research_control import open_explore_literature_compat as previous
from pipeline_core.discovery.scientific_research_control import scientific_literature_work as literature


def envelope(tool, row):
    return json.dumps(tool,ensure_ascii=False) + '\n' + json.dumps(row,ensure_ascii=False)


def test_dual_read_envelope_with_explore_recovered_without_science_edit():
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'fulltext'}}
    row=science(op='EXPLORE',tool=None)
    raw=envelope(tool,row)
    with pytest.raises(ValueError):previous.parse_compatible(raw)
    parsed=compat.parse_saved(raw)
    assert parsed['operation']=='EXPLORE'
    assert parsed['tool_request']==tool
    assert parsed['scientific_work_content']==row['scientific_work_content']
    assert parsed['literature_effect']==row['literature_effect']
    assert parsed['scientific_delta']==row['scientific_delta']


def test_dual_investigate_and_single_explore_unchanged():
    tool={'kind':'literature_search','payload':{'queries':['atomic scale SERS blinking']}}
    ordinary=science(op='EXPLORE',tool=tool)
    assert compat.parse_saved(json.dumps(ordinary))==previous.parse_compatible(json.dumps(ordinary))
    double=envelope(tool,science(op='INVESTIGATE',tool=None))
    assert compat.parse_saved(double)==literature._parse(double)


def test_conflicting_or_invalid_envelopes_rejected():
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'abstract'}}
    different={'kind':'literature_search','payload':{'queries':['SERS gold atomic rearrangement']}}
    bad=[
        envelope(tool,science(op='EXPLORE',tool=different)),
        envelope(tool,science(op='EXPLORE',action='RETURN')),
        envelope({'kind':'other','payload':{}},science(op='EXPLORE')),
        envelope({'kind':'literature_search','payload':'not-dict'},science(op='EXPLORE')),
        envelope(tool,science(op='EXPLORE')|{'fabricated_extra':'no'}),
        envelope(tool,science(op='EXPLORE')|{'scientific_delta':'short'}),
        envelope(tool,science(op='EXPLORE')) + '\n' + json.dumps(science(op='EXPLORE')),
        'leading narrative\n' + envelope(tool,science(op='EXPLORE')),
    ]
    for raw in bad:
        with pytest.raises(ValueError):compat.parse_saved(raw)


def test_real_saved_step3_recovered_with_read_of_previous_search(scenario):
    out=scenario
    work.choose(out,backend=backend_of(opening()))
    work.prepare_arms(out)
    search={'kind':'literature_search','payload':{'queries':['atomic structure optical SERS blinking'],'limit':2}}
    with compat.recovery_scope():
        first=work.next_step(out,'literature_enabled',backend=backend_of(science(op='EXPLORE',tool=search)),search_provider=FakeProvider())
        second=work.next_step(out,'literature_enabled',backend=backend_of(science(op='EXPLORE',tool=search)),search_provider=FakeProvider())
    assert first['tool_status']=='SEARCH_OK' and second['tool_status']=='SEARCH_OK'
    p=work.arm_dir(out,'literature_enabled')/'step_0003'
    with compat.recovery_scope():
        rows,_=work._read_rows(out,'literature_enabled')
        request=work._step_request(out,'literature_enabled',rows,3)
    p.mkdir()
    work.put(p/'request.json',request)
    work.put(p/'attempt.json',{
      'request_sha256':work.sha(request),
      'previous_receipt_sha256':work.sha(rows[-1]['receipt']),
      'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED'
    })
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'abstract'}}
    raw=envelope(tool,science(op='EXPLORE',tool=None))
    work.put(p/'raw_response.json',{'raw_text':raw,'usage':{'saved':True},'response_id':'paid-already'})
    old_sha=work.hfile(p/'raw_response.json')
    original_sha=work.hfile(Path(work.__file__))
    literature_sha=work.hfile(Path(literature.__file__))
    diagnostic=compat.inspect(out,'literature_enabled')
    assert diagnostic['dual_json_envelope'] is True
    assert diagnostic['operation']=='EXPLORE'
    assert diagnostic['read_target_in_previous_search'] is True
    assert diagnostic['read_target_title']=='Published physical blinking mechanism'
    recovered=compat.recover(out,'literature_enabled')
    assert recovered['tool_status']=='ABSTRACT_READ'
    assert compat.verify(out)['arms']['literature_enabled']['steps']==3
    assert work.read(p/'normalized.json')['operation']=='EXPLORE'
    assert work.read(p/'tool_result.json')['tool_result']['source_depth']=='abstract'
    assert work.hfile(p/'raw_response.json')==old_sha
    assert work.hfile(Path(work.__file__))==original_sha
    assert work.hfile(Path(literature.__file__))==literature_sha
    assert work.read(p/'dual_json_recovery_audit.json')['scientific_text_modified'] is False
    assert compat.preview(out,'literature_enabled')['step']==4
    with compat.recovery_scope():
        work.next_step(out,'literature_enabled',backend=backend_of(science(op='REVISE',action='RETURN')))
    assert compat.finish(out,'literature_enabled')['steps']==4
    assert compat.verify(out)['arms']['literature_enabled']['finalized'] is True
    # Integrity audit catches any subsequent audit tampering.
    audit=p/'dual_json_recovery_audit.json'
    content=work.read(audit);content['scientific_text_modified']=True
    audit.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='DUAL_JSON_RECOVERY_AUDIT_CHANGED'):
        compat.verify(out)


def test_control_arm_denies_dual_json_tool(scenario):
    out=scenario;work.choose(out,backend=backend_of(opening()));work.prepare_arms(out)
    p=work.arm_dir(out,'no_new_literature')/'step_0001';p.mkdir()
    req=work._step_request(out,'no_new_literature',[],1)
    work.put(p/'request.json',req)
    work.put(p/'attempt.json',{'request_sha256':work.sha(req),'previous_receipt_sha256':None})
    tool={'kind':'literature_search','payload':{'queries':['SERS dynamic hot spots']}}
    work.put(p/'raw_response.json',{'raw_text':envelope(tool,science(op='EXPLORE',tool=None))})
    with pytest.raises(ValueError,match='CONTROL_ARM_CANNOT_USE_LITERATURE_TOOL'):
        compat.recover(out,'no_new_literature')
    assert compat.verify(out)['arms']['no_new_literature']['incomplete']
