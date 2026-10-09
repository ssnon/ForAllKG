"""M4-C4 synthetic benchmark; no empirical, novelty or production authorization."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pytest

from pipeline_core.discovery.conditional_covariance_confrontation_m4c4 import (
    CASE, PARENTS, _metrics, _origins, fixture, run,
)
from scripts.discovery.evaluate_conditional_covariance_m4c4 import execute


@pytest.fixture
def sources():
    a1={"status":"DRAFT_CONFRONTATIONS_NEED_REVIEW","cases":[{
        "case_id":CASE,"status":"REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED",
        "proposed_experiment_UNREVIEWED":{"pair_relationship":"ADJACENT_NOT_EXCLUSIVE"},
        "input_hypotheses":[
            {"ref":"B03","terminal_idea_id":"research_idea:46af2e8dcdc0e3a1802a"},
            {"ref":"B06","terminal_idea_id":"research_idea:d26c4653802153c0d499"}]}]}
    scenarios = ["A_ONLY_IN_SCOPE", "B_ONLY_IN_SCOPE", "BOTH_COMPATIBLE", "NEITHER_COMPATIBLE", "NOT_IDENTIFIABLE"]
    c2={"status":"SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING",
        "input_sha256":{"m4a1":"a"},"rows":[{
            "case_id": CASE,"synthetic_scenario":scenarios[i], "scenario_kind":"SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED", "research_idea_node_created":False, "draft_scientific_kernel_NOT_CREATED":({"x":1} if i < 4 else None),
            "original_terminal_idea_ids_PRESERVED":sorted(PARENTS)} for i in range(5)]}
    c3={"status":"INTEGRITY_PASS_SCIENTIFIC_DELTA_UNPROVEN",
        "input_sha256":{"m4a1":"a","m4c2":"b"},
        "substantive_revisions_scientifically_confirmed":0,
        "rows":[{"case_id": CASE, "scenario":scenarios[i], "structural_integrity":"PASS", "independent_science_confirmed":False, "scientific_delta_state":"NOT_PROVEN"} for i in range(5)]}
    return a1,c2,c3


def test_output_and_non_authority(sources):
    a,b,c=sources
    original=copy.deepcopy(sources)
    x=run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')
    assert sources==original
    assert x['status'].startswith('SYNTHETIC_')
    assert not x['scientific_improvement_certified']
    assert not x['research_idea_nodes_created']
    assert not x['empirical_data_consumed']
    assert x['llm_or_network_calls']==0
    assert set(x['design']['parent_idea_ids_preserved_separately'])==PARENTS
    assert not x['design']['causal_claim']
    assert not x['design']['mechanisms_mutually_exclusive']
    assert x['synthetic_fixtures']['COVARIANCE_EFFECT_INJECTED']['delta_mse'] > .05
    assert x['synthetic_fixtures']['NO_COVARIANCE_EFFECT']['delta_mse'] < .01
    assert x['synthetic_fixtures']['EXACT_COLLINEARITY']['status']=='UNIDENTIFIABLE_DESIGN'


@pytest.mark.parametrize('side,field,value',[
    ('a1','status','BAD'),('c2','status','BAD'),('c3','status','BAD'),
    ('c2','input_sha256',{'m4a1':'bad'}),('c3','input_sha256',{'m4a1':'a','m4c2':'bad'}),
    ('c3','substantive_revisions_scientifically_confirmed',1),
])
def test_pin_mismatch(sources,side,field,value):
    a,b,c=sources
    d={'a1':a,'c2':b,'c3':c}[side]
    d[field]=value
    with pytest.raises(ValueError,match='M4C4_INTEGRITY_FAILURE'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_nonexclusive_is_required(sources):
    a,b,c=sources
    a['cases'][0]['proposed_experiment_UNREVIEWED']['pair_relationship']='MUTUALLY_EXCLUSIVE'
    with pytest.raises(ValueError,match='nonexclusive'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_parent_lineage(sources):
    a,b,c=sources
    b['rows'][1]['original_terminal_idea_ids_PRESERVED']=['research_idea:lost']
    with pytest.raises(ValueError,match='parent preservation'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_missing_one_parent(sources):
    a,b,c=sources
    a['cases'][0]['input_hypotheses'].pop()
    with pytest.raises(ValueError,match='source parent'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_missing_synthetic_draft(sources):
    a,b,c=sources
    b['rows'][0]['draft_scientific_kernel_NOT_CREATED']=None
    with pytest.raises(ValueError,match='four draft'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_injected_signal_group_holdout():
    p = _metrics(fixture(effect=1.15))
    assert p['fold_count']==4 and p['group_count']==16 and p['n_observations']==128
    assert p['positive_delta_all_folds']
    assert all(x['train_group_count']+x['test_group_count']==16 for x in p['folds'])
    assert sum(x['test_n'] for x in p['folds'])==128


def test_null_effect_is_not_supported():
    p = _metrics(fixture(effect=0.0))
    assert p['delta_mse'] < .01


def test_nonidentified_is_detected():
    p=_metrics(fixture(effect=2.,collinear=True))
    assert p['status']=='UNIDENTIFIABLE_DESIGN'
    assert not p['effect_claim_allowed']


def test_reject_target_derived_covariance():
    with pytest.raises(ValueError,match='leaky'):
        _metrics(fixture(effect=1.15,leakage=True))


@pytest.mark.parametrize('violation', ['missing_feature','nan','non_numeric','bad_source','missing_group','too_few_rows'])
def test_incomplete_or_unapproved_fixture(violation):
    rows=fixture(effect=1.15)
    if violation=='missing_feature': rows[0]['features'].pop('active_orientation')
    if violation=='nan': rows[0]['features']['mean_orientation']=float('nan')
    if violation=='non_numeric':rows[0]['target_band_ratio']='bad'
    if violation=='bad_source':rows[0]['measurement_source']='REAL_INSTRUMENT'
    if violation=='missing_group':rows[0]['group_id']=None
    if violation=='too_few_rows': rows=rows[:30]
    with pytest.raises(ValueError,match='M4C4_INTEGRITY_FAILURE'):
        _metrics(rows)


def test_deterministic(sources):
    a,b,c=sources
    x=run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')
    y=run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')
    assert x==y


def test_cli_input_guards(tmp_path,sources):
    a,b,c=sources
    p=[]
    for i,obj in enumerate(sources):
        path=tmp_path/f'{i}.json'
        path.write_text(json.dumps(obj),encoding='utf-8')
        p.append(path)
    with pytest.raises(ValueError,match='SHA mismatch'):
        execute(m4a1=p[0],m4c2=p[1],m4c3=p[2],expected_m4a1_sha256='wrong',output_dir=tmp_path/'out')
    h=hashlib.sha256(p[0].read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='lineage'):
        execute(m4a1=p[0],m4c2=p[1],m4c3=p[2],expected_m4a1_sha256=h,output_dir=tmp_path/'out')


def test_cli_overwrite_and_repo_safety(tmp_path,sources):
    a,b,c=sources
    p=[]
    for i,obj in enumerate(sources):
        path=tmp_path/f'{i}.json'
        path.write_text(json.dumps(obj),encoding='utf-8')
        p.append(path)
    with pytest.raises(ValueError,match='outside repo'):
        execute(m4a1=p[0],m4c2=p[1],m4c3=p[2],expected_m4a1_sha256='x',output_dir=Path(__file__).resolve().parents[2]/'fakeout')
    out=tmp_path/'occupied'
    out.mkdir()
    with pytest.raises(FileExistsError):
        execute(m4a1=p[0],m4c2=p[1],m4c3=p[2],expected_m4a1_sha256='x',output_dir=out)


def test_scenario_pairing_guard(sources):
    a,b,c = sources
    c['rows'][0]['scenario'] = 'B_ONLY_IN_SCOPE'
    with pytest.raises(ValueError, match='scenario identity'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')


def test_no_science_promotion_guard(sources):
    a,b,c = sources
    c['rows'][0]['independent_science_confirmed'] = True
    with pytest.raises(ValueError, match='scientific authorization'):
        run(a1=a,c2=b,c3=c,sha_a1='a',sha_c2='b',sha_c3='c')
