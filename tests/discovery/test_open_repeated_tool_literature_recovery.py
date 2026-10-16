import json
from pathlib import Path
import pytest

from test_open_literature_research import scenario, science, opening, FakeProvider, backend_of
from pipeline_core.discovery.scientific_research_control import open_repeated_tool_literature_recovery as newer
from pipeline_core.discovery.scientific_research_control import open_dual_json_literature_recovery as prior
from pipeline_core.discovery.scientific_research_control import open_literature_research as engine


def triple(tool, row):
    return '\n'.join(json.dumps(v, ensure_ascii=False) for v in (tool,tool,row))


def test_exact_three_valid_and_other_existing_formats_unmodified():
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'fulltext'}}
    final=science(op='EXPLORE',tool=tool)
    raw=triple(tool,final)
    with pytest.raises(ValueError):prior.parse_saved(raw)
    with newer.scope():
        parsed=prior.parse_saved(raw)
    assert parsed['operation']=='EXPLORE'
    assert parsed['tool_request']==tool
    assert parsed['scientific_work_content']==final['scientific_work_content']
    assert parsed['scientific_delta']==final['scientific_delta']
    assert parsed['literature_effect']==final['literature_effect']
    assert newer._repeated_only(raw)==(tool,final)
    assert newer._parse_objects(raw)==[tool,tool,final]
    assert prior.parse_saved(json.dumps(final))==prior.PREVIOUS_PARSE(json.dumps(final))
    assert prior.parse_saved(json.dumps(tool)+'\n'+json.dumps(final))['operation']=='EXPLORE'


def test_conflicts_and_extra_material_strictly_rejected():
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'fulltext'}}
    different={'kind':'literature_read','payload':{'paper_id':'literature:OTHER','depth':'fulltext'}}
    cases=[
        '\n'.join(json.dumps(x) for x in (tool,different,science(op='EXPLORE',tool=tool))),
        '\n'.join(json.dumps(x) for x in (tool,tool,science(op='EXPLORE',tool=different))),
        '\n'.join(json.dumps(x) for x in (tool,tool,science(op='EXPLORE',tool=None))),
        triple({'kind':'literature_search','payload':{'queries':['SERS']}},science(op='EXPLORE',tool=tool)),
        triple(tool,science(op='EXPLORE',action='RETURN',tool=tool)),
        triple(tool,science(op='EXPLORE',tool=tool))+'\n3',
        triple(tool,science(op='EXPLORE',tool=tool))+'\n'+json.dumps({}),
        triple(tool,science(op='EXPLORE',tool=tool))+' commentary',
    ]
    for raw in cases:
        with pytest.raises(ValueError):newer._repeated_only(raw)
        with newer.scope():
            with pytest.raises(ValueError):prior.parse_saved(raw)


def test_step3_read_realistic_saved_triple_and_verify_audit(scenario):
    out=scenario
    engine.choose(out,backend=backend_of(opening()))
    engine.prepare_arms(out)
    search={'kind':'literature_search','payload':{'queries':['atomic hot spots'],'limit':2}}
    with newer.scope(),prior.recovery_scope():
        for _ in range(2):
            res=engine.next_step(out,'literature_enabled',backend=backend_of(science(op='EXPLORE',tool=search)),search_provider=FakeProvider())
            assert res['tool_status']=='SEARCH_OK'
    p=engine.arm_dir(out,'literature_enabled')/'step_0003'
    with newer.scope(), prior.recovery_scope():
        rows,_=engine._read_rows(out,'literature_enabled')
        req=engine._step_request(out,'literature_enabled',rows,3)
    p.mkdir()
    engine.put(p/'request.json',req)
    engine.put(p/'attempt.json',{
        'request_sha256':engine.sha(req),'previous_receipt_sha256':engine.sha(rows[-1]['receipt']),
        'status':'BILLING_UNCERTAIN_UNTIL_RAW_SAVED'
    })
    tool={'kind':'literature_read','payload':{'paper_id':'literature:test','depth':'abstract'}}
    original_step=science(op='EXPLORE',tool=tool)
    engine.put(p/'raw_response.json',{'raw_text':triple(tool,original_step),'response_id':'paid-saved','usage':{'fixed':True}})
    old_hash=engine.hfile(p/'raw_response.json')
    inspect=newer.inspect(out,'literature_enabled')
    assert inspect['three_json_objects'] is True
    assert inspect['duplicate_tool_envelopes_identical'] is True
    assert inspect['matches_final_tool_request'] is True
    assert inspect['read_target_in_previous_search'] is True
    assert inspect['only_one_literature_read_will_be_dispatched'] is True
    assert inspect['requested_depth']=='abstract'
    res=newer.recover(out,'literature_enabled')
    assert res['tool_status']=='ABSTRACT_READ'
    assert res['literature_reads_dispatched']==1
    assert engine.hfile(p/'raw_response.json')==old_hash
    assert engine.read(p/'normalized.json')['operation']=='EXPLORE'
    assert engine.read(p/'normalized.json')['scientific_work_content']==original_step['scientific_work_content']
    assert engine.read(p/'tool_result.json')['tool_result']['source_depth']=='abstract'
    assert newer.verify(out)['arms']['literature_enabled']['steps']==3
    assert newer.preview(out,'literature_enabled')['step']==4
    # The prior audit is still checked. Our extra audit additionally checks exact dedup.
    a=engine.read(p/'repeated_tool_recovery_audit.json')
    assert a['identical_repetitions_verified'] is True
    assert a['original_paid_raw_sha256']==old_hash
    a['paper_id']='manipulated'
    (p/'repeated_tool_recovery_audit.json').write_text(json.dumps(a))
    with pytest.raises(ValueError,match='REPEATED_TOOL_RECOVERY_AUDIT_MISSING_OR_CHANGED'):
        newer.verify(out)


def test_unknown_paper_is_never_read(scenario):
    out=scenario
    engine.choose(out,backend=backend_of(opening()))
    engine.prepare_arms(out)
    p=engine.arm_dir(out,'literature_enabled')/'step_0001'
    p.mkdir()
    req=engine._step_request(out,'literature_enabled',[],1)
    engine.put(p/'request.json',req)
    engine.put(p/'attempt.json',{'request_sha256':engine.sha(req),'previous_receipt_sha256':None})
    tool={'kind':'literature_read','payload':{'paper_id':'literature:does_not_exist','depth':'fulltext'}}
    engine.put(p/'raw_response.json',{'raw_text':triple(tool,science(op='EXPLORE',tool=tool))})
    assert newer.inspect(out,'literature_enabled')['read_target_in_previous_search'] is False
    with pytest.raises(ValueError,match='READ_TARGET_NOT_IN_PREVIOUS_SEARCH'):
        newer.recover(out,'literature_enabled')
    assert not (p/'normalized.json').exists()
