import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import pytest

HERE=Path(__file__).resolve().parents[2]
SCRIPT=HERE/'scripts/discovery/run_m25_two_intervention_holdout.py'
spec=importlib.util.spec_from_file_location('m25_runner',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fake_backend(req):
    assert 'observed_ratio' not in json.dumps(req)
    assert 'reported_PEI_max_relative_shift_percent' not in json.dumps(req)
    assert '0.70/0.24' not in json.dumps(req)
    assert 's41467' not in json.dumps(req)
    return json.dumps(valid_output()),{'total_tokens':100},'fake-id'

def valid_output():
    return {'chosen_primary_id':'ligand',
            'research_question':'How should ligand permittivity and rod geometry affect the maximum relative LSPR response?',
            'predictions':[
            {'id':'ligand','predicted_ratio':7.7/3.3,
             'derivation':'At matched effective geometry and photovoltage, C proportional epsilon, so ratio 7.7/3.3.',
             'assumptions':'Effective ligand thickness and field screening are comparable in both coating classes.',
             'alternative_mechanism':'Ligand dependent charge exchange or film thickness can affect observed shifts.',
             'confidence':'CONDITIONAL'},
            {'id':'size','predicted_ratio':25/10,
             'derivation':'In geometrically similar rods surface divided by volume scales as inverse linear size.',
             'assumptions':'Aspect ratios and photovoltage remain sufficiently close across rods of two sizes.',
             'alternative_mechanism':'Size dependent absorption and surface chemistry could alter charge generation.',
             'confidence':'CONDITIONAL'}],
            'cross_system_transfer_limit':'These liquid solvent data cannot immediately justify solid PVA Raman blinking mechanism.',
            'what_result_would_revise_model':'A large systematic deviation from expected ratios suggests violations of constant capacitance driving or geometric scaling.'}

@pytest.fixture
def stub():
    # No real API connection or original repository needed for offline dry runs.
    roots={}
    for name in ['pipeline_core','pipeline_core.discovery','pipeline_core.discovery.scientific_research_control']:
        obj=types.ModuleType(name);obj.__path__=[];roots[name]=sys.modules.get(name);sys.modules[name]=obj
    name='pipeline_core.discovery.scientific_research_control.research_work_v12_stable_runner'
    prev=sys.modules.get(name)
    obj=types.ModuleType(name)
    def require(backend,allow_paid,n):
        assert n==1
        if backend is None:raise PermissionError('No live paid call in test')
    obj._require_paid=require
    obj._invoke=lambda req,backend:backend(req)
    sys.modules[name]=obj
    yield
    if prev is None:sys.modules.pop(name,None)
    else:sys.modules[name]=prev
    for name,old in roots.items():
        if old is None:sys.modules.pop(name,None)
        else:sys.modules[name]=old

def test_source_values_traceable():
    h=m.sealed_holdout()
    assert h['provenance']['doi']=='10.1038/s41467-025-67130-8'
    assert h['targets']['size']['observed_ratio']==pytest.approx(0.7/0.24)
    assert h['targets']['ligand']['observed_ratio']==2.5

def test_no_holdout_in_model_request(tmp_path):
    dest=tmp_path/'run';m.prepare(dest,model='offline-model')
    req=m.request(dest)
    s=json.dumps(req)
    assert 'observed_ratio' not in s and '0.28' not in s
    assert 'article' not in s.lower() and 'zenodo' not in s.lower()
    assert m.verify(dest)['prediction']=='NOT_STARTED'
    assert m.preview(dest)['next_call_is_paid']
    assert m.preview(dest)['public_inputs_only']

def test_fake_paid_result_and_holdout_evaluation(tmp_path,stub):
    dest=tmp_path/'run';m.prepare(dest,model='offline-model')
    r=m.propose(dest,backend=fake_backend)
    assert r['new_paid_calls']==0
    assert m.verify(dest)['evaluation']=='SEALED'
    assert not (dest/'evaluation.json').exists()
    x=m.reveal(dest)
    assert x['tested_interventions']==2
    assert x['both_beats_baseline']
    v=m.read(dest/'evaluation.json')
    assert v['holdout']['ligand']['predicted_ratio']==pytest.approx(7.7/3.3)
    assert v['holdout']['size']['reported_observed_ratio']==pytest.approx(0.7/0.24)
    assert m.verify(dest)['evaluation']=='REVEALED'
    with pytest.raises(ValueError):m.reveal(dest)
    with pytest.raises(ValueError):m.preview(dest)

def test_recover_saved_raw_without_new_model_call(tmp_path,stub):
    dest=tmp_path/'run';m.prepare(dest,model='offline-model')
    request=m.request(dest);pd=dest/'propose';pd.mkdir()
    m.put(pd/'request.json',request)
    m.put(pd/'attempt.json',{'request_sha256':m.sha_bytes(m.canon(request)), 'status':'BILLING_UNKNOWN_UNTIL_RAW_SAVED'})
    m.put(pd/'raw_response.json',{'raw_text':json.dumps(valid_output()),'usage':None,'response_id':'already_paid'})
    assert m.verify(dest)['prediction']=='RAW_SAVED_RECOVER_NO_RETRY'
    done=m.recover(dest)
    assert done['status']=='M25_SAVED_PREDICTION_RECOVERED_NO_REPAY'
    assert m.verify(dest)['prediction']=='READY'

def test_invalid_raw_does_not_make_retry_possible(tmp_path,stub):
    dest=tmp_path/'run';m.prepare(dest,model='offline-model')
    bad=valid_output();bad['predictions'][0]['predicted_ratio']=-1
    count=[0]
    def backend(req):
        count[0]+=1
        return json.dumps(bad),None,'mock'
    with pytest.raises(RuntimeError,match='DO_NOT_REPAY'):
        m.propose(dest,backend=backend)
    assert count==[1]
    assert m.verify(dest)['prediction']=='RAW_SAVED_RECOVER_NO_RETRY'
    with pytest.raises(ValueError):m.preview(dest)
    with pytest.raises(ValueError):m.recover(dest)
    assert (dest/'propose/raw_response.json').is_file()

def test_tampering_fail_closed(tmp_path):
    dest=tmp_path/'run';m.prepare(dest,model='offline-model')
    hold=dest/'private_holdout.json';x=m.read(hold)
    x['targets']['ligand']['observed_ratio']=999
    hold.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='CHANGED'):m.verify(dest)

def test_abstention_scored_without_false_success():
    x=valid_output()
    for pred in x['predictions']:
        pred['predicted_ratio']=None;pred['confidence']='ABSTAIN'
    m.validate_prediction(x)
    s=m.score(x,m.sealed_holdout(),{'record':'fake'})
    assert s['tested_interventions']==0
    assert s['both_beats_no_change_baseline'] is False
    assert s['mean_absolute_percentage_error_over_nonabstained'] is None
