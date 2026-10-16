from __future__ import annotations
import copy
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from pipeline_core.discovery.scientific_research_control import scientific_literature_work as lw
from pipeline_core.discovery.scientific_research_control import research_literature_tools as lt


@dataclass
class FakeRequest:
    query:str
    mechanism_bucket:str
    limit:int


class FakeRecord:
    def __init__(self,id,title,doi,abstract=None):
        self.d={'paper_id':id,'title':title,'doi':doi,'year':2024,'venue':'Journal',
            'abstract':abstract,'source_depth':'abstract' if abstract else 'metadata',
            'provider_references':[{'provider':'openalex','provider_id':'W123'}],
            'metadata':{'openalex_id':'https://openalex.org/W123','best_oa_location':{'is_oa':True,'pdf_url':'https://example.org/read.pdf'}}}
    def to_dict(self):return copy.deepcopy(self.d)

class FakeProvider:
    def search(self,request):
        return [FakeRecord('literature:abc','A Relevant Experiment','10.1234/test',
                           'An experiment reports a prior causal mechanism and a contrasting prediction.')]
    def get_work(self,id):return {'best_oa_location':{'is_oa':True,'pdf_url':'https://example.org/read.pdf'}}

@pytest.fixture(autouse=True)
def mock_request(monkeypatch):
    name='pipeline_core.literature.discovery.providers.base'
    m=types.ModuleType(name);m.LiteratureSearchRequest=FakeRequest
    monkeypatch.setitem(sys.modules,name,m)


def science(tool=None,action='CONTINUE',operation='INVESTIGATE'):
    return {'operation':operation,'scientific_work_content':{
         'attempted_question':'Can a causal model predict a discriminating physical experiment?',
         'premises_used':['Unverified research memory, not experimental fact'],
         'reasoning_or_operations':['Compare independently observable predictions.'],
         'candidate_consequence':'If two rates differ, the mechanism can be rejected.',
         'counterexample_or_obstruction':None,'limits_and_uncertainties':['Unverified'],
         'work_conclusion':'INCONCLUSIVE'},
      'focal_claim':'Potential mechanism constrained by published kinetics',
      'scientific_delta':'Investigate an unresolved physical mechanism using literature.',
      'literature_effect':'No external source has been read yet.',
      'next_action':action,'next_research_question':'Which primary paper tests these rate laws?' if action=='CONTINUE' else None,
      'why_next_action':'A proper comparison is necessary before claiming a new scientific relation.',
      'tool_request':tool}


def test_actual_fakg_search_adapter_and_dedup():
    got=lt.search({'queries':['SERS blinking microscopic kinetics','ensemble Raman dynamics'],'limit':3},provider=FakeProvider())
    assert got['status']=='SEARCH_OK' and len(got['papers'])==1
    assert got['papers'][0]['source_depth']=='abstract'
    assert got['novelty_certified'] is False


def test_search_fails_without_faking_novelty():
    class Failure:
        def search(self,request):raise RuntimeError('API key SHOULD NOT LEAK')
    got=lt.search({'queries':['microbial dynamics kinetics']},provider=Failure())
    assert got['status']=='SEARCH_FAILED'
    assert got['failures'][0]['error_type']=='RuntimeError'
    assert 'SHOULD NOT LEAK' not in str(got)


def test_abstract_read_and_no_unsearched_read():
    paper=lt.search({'queries':['gold nanoparticle blinking']},provider=FakeProvider())['papers']
    result=lt.read({'paper_id':'literature:abc','depth':'abstract'},known_papers=paper)
    assert result['status']=='ABSTRACT_READ' and result['source_depth']=='abstract'
    with pytest.raises(ValueError,match='PREVIOUSLY_DISCOVERED'):
        lt.read({'paper_id':'literature:fake','depth':'fulltext'},known_papers=paper)


def test_fulltext_fallback_is_not_fulltext(monkeypatch):
    paper=lt.search({'queries':['gold nanoparticle blinking']},provider=FakeProvider())['papers']
    name='pipeline_core.literature.acquisition.openalex_access'
    module=types.ModuleType(name)
    module._locations_from_openalex_work=lambda **kwargs:[types.SimpleNamespace(url_for_pdf='https://example.org/read.pdf',automatic_download_eligible=True)]
    monkeypatch.setitem(sys.modules,name,module)
    output=lt.read({'paper_id':'literature:abc','depth':'fulltext'},known_papers=paper,
        provider=FakeProvider(),pdf_fetcher=lambda url:b'not a PDF')
    assert output['status']=='FULLTEXT_UNAVAILABLE'
    assert output['source_depth']=='abstract' and output['novelty_certified'] is False


def test_two_object_valid_json_envelope():
    tool={'kind':'literature_search','payload':{'queries':['test blinking kinetics']}}
    obj=science(action='CONTINUE');obj.pop('tool_request')
    joined=json.dumps(tool)+'\n'+json.dumps(obj)
    a=lw._parse(joined)
    assert a['tool_request']==tool and a['operation']=='INVESTIGATE'
    with pytest.raises(ValueError):lw._parse(joined+' other text')


def test_invalid_tool_dispatch_never_calls_provider():
    res=lw._dispatch({'kind':'arbitrary_python','payload':{}},[])
    assert res['status']=='INVALID_TOOL_REQUEST'


def test_previous_search_ids_required():
    out=lw._dispatch({'kind':'literature_read','payload':{'paper_id':'literature:unknown'}},[])
    assert out['status']=='INVALID_TOOL_REQUEST'


def test_noninvestigate_tool_is_rejected():
    obj=science(tool={'kind':'literature_search','payload':{'queries':['test blinking kinetics']}},operation='DERIVE')
    with pytest.raises(ValueError):lw._parse(json.dumps(obj))


def test_persisted_tool_output_replayed_without_search(monkeypatch,tmp_path):
    """Full 2-step research, replay verify, and no future search while verifying."""
    folder=tmp_path/'episode';folder.mkdir()
    native_dir=folder/'native';native_dir.mkdir()
    workid='work_0006_test'
    task={'status':'REQUESTED','work_id':workid,'request':{'work_kind':'REASON','research_question':'Which published mechanism constrains observed blinking?'}}
    state={'pending_work':task,'ended':False,'work_history':[],'decision_count':6}
    request={'model':'fake-science','temperature':0.5,'max_tokens':2500,'messages':[]}
    ctxt={'state':state,'manifest':{'model':'fake-science','max_decisions':12}}
    original=lw.native.verify
    monkeypatch.setattr(lw.lineage,'verify',lambda out:{'status':'VERIFIED_NO_CALL','native_decisions':6})
    monkeypatch.setattr(lw.lineage,'native_dir',lambda out:native_dir)
    monkeypatch.setattr(lw.lineage,'safe_path',lambda out:Path(out))
    monkeypatch.setattr(lw.native,'verify',lambda out:ctxt)
    monkeypatch.setattr(lw.native,'_make_call_request',lambda c,kind:request)
    monkeypatch.setattr(lw.lineage,'_valid_pending',lambda c:task)
    lw.put(folder/'literature_policy.json',{'protocol':lw.PROTOCOL,'literature_enabled':True})
    lw.prepare_work(folder)
    responses=[science({'kind':'literature_search','payload':{'queries':['gold nanoparticle blinking physics'],'limit':2}}),
               science(None,'RETURN','REVISE')]
    def backend(req):
        return json.dumps(responses.pop(0)),None,'mock-id'
    def fail_search(_):raise AssertionError('Must not re-query verified snapshot')
    lw.next_step(folder,backend=backend,search_provider=FakeProvider())
    p=folder/'literature_work'/'step_0001'/'tool_result.json'
    assert lw.read(p)['tool_result']['status']=='SEARCH_OK'
    monkeypatch.setattr(lt,'search',fail_search)
    assert lw.verify(folder)['literature_work']['steps']==1
    lw.next_step(folder,backend=backend)
    assert lw.verify(folder)['literature_work']['steps']==2
    assert lw.verify(folder)['literature_work']['last_action']=='RETURN'
    assert lw.finish(folder)['steps']==2
    assert lw.verify(folder)['literature_work']['finalized']


def test_control_arm_prompt_has_no_live_search():
    lock={'model':'test','literature_enabled':False}
    original={'temperature':0.3,'max_tokens':2400}
    req=lw._request(lock,original,[],1)
    content=req['messages'][0]['content']
    assert 'CONTROL ARM: live literature tools are NOT available' in content
    assert req['messages'][1]['content'].find('"literature_access_enabled": false')>=0


def test_search_payload_limits_enforced():
    for payload in ({'queries':[]},{'queries':['small'],'limit':100},
                    {'queries':['some valid query']*3}):
        with pytest.raises(ValueError):lt.search(payload,provider=FakeProvider())
