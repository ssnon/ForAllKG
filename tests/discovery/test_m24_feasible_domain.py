import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.discovery import run_m24_feasible_domain as m


def put(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')


def source(tmp_path):
    x=tmp_path/'m23'
    put(x/'input.json',{'verified_SOURCE_EXCERPTS':[
        {'id':'HOT:0','pdf_source_span':'It generates carriers at a dephasing timescale of 1–100 fs with the mentioned geometry.'},
        {'id':'HOT:1','pdf_source_span':'The indirect process shows very low efficiency (typically<2%) under special conditions.'},
        {'id':'HOT:2','pdf_source_span':'Back transfer competes with long-lived carrier injection.'}],
        'prior':'hypotheses are not verified'})
    put(x/'lock.json',{'model':'test-model','temperature':0,'max_tokens':6000})
    put(x/'develop/normalized.json',{'selected_question':'Can charge transfer produce a measurably different Raman response?'})
    put(x/'challenge/normalized.json',{'final_disposition':'REVISE','other':'Allow sign-odd structural nulls'})
    out=tmp_path/'m24'
    with patch.object(m.m23,'verify',return_value={'develop':'READY','challenge':'READY'}):
        assert m.prepare(x,out)['registered_facts']==2
    return out


def proposal():
    fact='hot electron injection efficiency via indirect pathway (NOT total charge trapping yield)'
    return {'selected_scientific_question':'Under a conditional indirect-transfer pathway, what is the largest generated flux?',
        'independent_evidence_needed':'A GNR-PVA-specific transfer yield and measured absorption flux would be required.',
        'scientific_model_or_bridge':'A hypothetical constant source gives accumulated charge proportional to injected flux times effective survival time.',
        'scientific_scope':'YE_GNR_PVA',
        'variables':[
            {'symbol':'eta_indirect','meaning':fact,'unit':'fraction',
             'assumed_min':None,'assumed_max':1,'source_fact_id':'HOT:INDIRECT_ETA_TYPICAL'},
            {'symbol':'phi','meaning':'absorbed photon flux in a hypothetical specimen','unit':'1/s',
             'assumed_min':100.,'assumed_max':1000.,'source_fact_id':None},
            {'symbol':'tau','meaning':'effective residence time of trapped electrons','unit':'s',
             'assumed_min':0.1,'assumed_max':10.,'source_fact_id':None}],
        'assumed_constraints':[],
        'test_quantity':{'name':'conditional occupied charge','powers':{'eta_indirect':1,'phi':1,'tau':1},
                         'positive_factor':1.,'units_as_assumed':'electrons',
                         'why_physically_relevant':'Would bound trap occupancy if the measured parameters and model assumptions applied.'},
        'candidate_conclusion':'Only a conditional scenario cap can be obtained, not any actual Ye-specific quantitative bound.',
        'what_would_change_after_evidence':'A direct calibrated injection yield and residence time could convert the case to an actual bound.',
        'unmeasured_quantities':['Specific gold nanorod absorption and survival time'],
        'alternative_explanation':'Thermal and polymer-interface reconstruction can change Raman response without stored electrons.',
        'source_transfer_risks':['Reported typical efficiency is not a hard gold nanorod in PVA bound.']}


def test_sourced_scenario_reduces_only_typical_domain(tmp_path):
    p=source(tmp_path)
    packet=m.read(p/'input.json')
    obj=m.parse(json.dumps(proposal()),packet)
    res=m.evaluate(obj,packet)
    assert res['model_assumption_only']['status']=='FEASIBLE_CONDITIONAL_LOG_LINEAR_DOMAIN'
    assert res['source_context_changes_feasible_domain'] is False
    assert res['typical_scenario_changes_feasible_domain'] is True
    assert abs(res['model_assumption_only']['upper_bound']['value']-10000.)<1e-8
    assert abs(res['with_typical_indirect_transfer_scenario']['upper_bound']['value']-200.)<1e-8
    assert res['YE_case_feasible_domain_reduced_by_independent_measurement'] is False
    assert res['hard_ye_specific_source_constraints_added']==0


def test_source_dephasing_tightens_context_but_never_trap_lifetime(tmp_path):
    p=source(tmp_path);pkt=m.read(p/'input.json')
    obj=proposal();obj['variables']=[{'symbol':'plasmon_dephasing_fs',
          'meaning':'plasmon dephasing time (NOT trapped charge lifetime)','unit':'fs',
          'assumed_min':None,'assumed_max':None,'source_fact_id':'HOT:DEPHASE_FS'}]
    obj['test_quantity']={'name':'dephasing interval','powers':{'plasmon_dephasing_fs':1},
         'positive_factor':1.,'units_as_assumed':'fs',
         'why_physically_relevant':'Plasmon dephasing time, not charge trapping lifetime, sets one ultrafast process.'}
    norm=m.parse(json.dumps(obj),pkt)
    r=m.evaluate(norm,pkt)
    assert r['source_context_changes_feasible_domain'] is True
    assert r['with_review_reported_context']['lower_bound']['value']==1.
    assert abs(r['with_review_reported_context']['upper_bound']['value']-100.)<1e-8
    assert r['YE_case_feasible_domain_reduced_by_independent_measurement'] is False


def test_wrong_physical_quantity_binding_rejected(tmp_path):
    p=source(tmp_path);pkt=m.read(p/'input.json');data=proposal()
    data['variables'][0]['meaning']='Trapped charge lifetime rather than transfer efficiency'
    with pytest.raises(ValueError,match='SEMANTIC_REBINDING'):
        m.parse(json.dumps(data),pkt)
    data=proposal();data['variables'][0]['source_fact_id']='HOT:NONEXISTENT'
    with pytest.raises(ValueError,match='NONEXISTENT_SOURCE'):
        m.parse(json.dumps(data),pkt)


def test_incompatible_model_assumptions_report_infeasible(tmp_path):
    p=source(tmp_path);pkt=m.read(p/'input.json');data=proposal()
    data['assumed_constraints']=[
       {'name':'force low transfer','powers':{'eta_indirect':1},'relation':'LE','rhs':0.01,
        'physical_justification':'Synthetic toy model assumption for contradiction test only.',
        'status':'MODEL_ASSUMPTION'},
       {'name':'force high transfer','powers':{'eta_indirect':1},'relation':'GE','rhs':0.5,
        'physical_justification':'Synthetic contradictory toy model for contradiction test.',
        'status':'MODEL_ASSUMPTION'}]
    r=m.evaluate(m.parse(json.dumps(data),pkt),pkt)
    assert r['model_assumption_only']['status']=='INFEASIBLE_ASSUMPTION_SET'


def test_paid_saved_and_no_retry(tmp_path):
    p=source(tmp_path);seen=[]
    def backend(req):
        seen.append(req)
        return json.dumps(proposal()),None,'test-response'
    assert m.preview(p)['next_call_is_paid']
    result=m.propose(p,backend=backend)
    assert result['proposal']=='READY' and len(seen)==1
    assert m.verify(p)['proposal']=='READY'
    with pytest.raises(ValueError,match='ALREADY_STARTED'):
        m.propose(p,backend=backend)
    assert len(seen)==1
    assert m.report(p)['science_and_novelty_not_certified']
    file=p/'propose/feasibility.json';data=m.read(file);data['scientific_validation']=True
    put(file,data)
    with pytest.raises(ValueError,match='FEASIBILITY_CHANGED'):
        m.verify(p)


def test_error_on_saved_raw_blocks_repay(tmp_path):
    p=source(tmp_path)
    raw=proposal();raw['variables'][0]['source_fact_id']='HOT:NONSENSE'
    def backend(req):return json.dumps(raw),None,'test-invalid'
    with pytest.raises(RuntimeError,match='PAID_RAW_SAVED_DO_NOT_REPAY'):
        m.propose(p,backend=backend)
    assert m.verify(p)['proposal']=='RAW_SAVED_RECOVER_NO_REPAY'
    with pytest.raises(ValueError,match='ALREADY_STARTED'):
        m.preview(p)
    assert (p/'propose/raw_response.json').is_file()


def test_model_cannot_mark_source_numeric_constraint(tmp_path):
    p=source(tmp_path);pkt=m.read(p/'input.json');bad=proposal()
    bad['assumed_constraints']=[{'name':'invented source evidence',
        'powers':{'eta_indirect':1},'relation':'LE','rhs':0.00001,
        'physical_justification':'A fabricated external physics number for testing rejection.',
        'status':'SOURCE_REPORTED'}]
    with pytest.raises(ValueError):m.parse(json.dumps(bad),pkt)
