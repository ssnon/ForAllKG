"""Contract and adversarial tests for metadata-only M4-B2 triage."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.discovery import triage_empirical_evidence_candidates_m4b2 as m


def source():
    # Contract-compatible synthetic M4-B1 result; no dependency on private Q-A data.
    root = '/tmp/m4b2_test_data'
    def f(name, fields, suffix='.csv', status='HEADER_SAMPLED'):
        return {
            'path_PRIVATE':root+'/'+name,
            'relative_path':name,
            'root_PRIVATE':root,
            'suffix':suffix,
            'size_bytes':100,
            'columns_or_keys':fields,
            'inspection_status':status,
            'path_tags_HEURISTIC':[],
            'field_tags_HEURISTIC':[],
            'small_file_sha256':'a'*64,
            'actual_measurements_read':False,
            'measurement_independence_verified':False,
        }
    obs = [
      ('time-resolved relative SERS band ratios','TARGET_OUTCOME',3),
      ('molecular orientation independent of relative SERS band ratios','INDEPENDENT_DISCRIMINATOR',3),
      ('surface and local-field evolution','INDEPENDENT_DISCRIMINATOR',0),
    ]
    rows=[
      f('a_structured_data.csv',['sample_id','exposure_time_s','raman_band_ratio','orientation_angle']),
      f('b_header_only.csv',['sample_id','raman_band_ratio']),
      f('raman_sers_summary.json',['sample_id','exposure_time_s','raman_band_ratio'],'.json','JSON_KEYS_SAMPLED'),
      f('orientation_sfg_experiment.csv',['sample_id','orientation_angle','SFG_signal']),
    ]
    cands=[{'name':x,'role':y,'candidate_file_count':z,'candidate_files_HEURISTIC':[],
            'independent_measurement_witness_verified':False,'observations_validated':False} for x,y,z in obs]
    return {
      'schema_version':m.SOURCE_SCHEMA,
      'status':'INVENTORY_COMPLETE_WITHIN_DECLARED_SCOPE_REVIEW_REQUIRED',
      'm4a2_input_sha256':'a'*64,
      'data_roots_PRIVATE':[root],
      'file_count':len(rows),
      'files':rows,
      'cases':[{'case_id':'QA_B12_AIR_EXPOSURE','readiness':'UNKNOWN_PENDING_DATA_LINKAGE',
                'empirical_discrimination_established':False,'observables':cands}],
      'scan_truncated':False,'scan_warnings':[],
      'science_truth_or_novelty_certified':False,
      'physical_feasibility_certified':False,
      'measurement_independence_certified':False,
      'source_files_modified':False,
      'automatic_research_idea_or_hypothesis_promotion':False,
      'claims_about_unscanned_data':False,
      'llm_or_network_calls':0,
    }


def test_structured_fields_prioritize_measured_header_cues():
    result=m.triage(source(),input_sha='b'*64,top_k=5)
    req=result['cases'][0]['requirements'][0]
    assert req['top_unreviewed_files_PRIVATE'][0]['relative_path_PRIVATE']=='a_structured_data.csv'
    assert req['m4b1_weak_overlap_count']==3
    assert req['strict_metadata_cue_count_NOT_VERIFIED']==2
    assert not req['measurements_available_verified']


def test_no_parent_directory_keyword_leak():
    data=source()
    data['files'][1]['path_PRIVATE']='/tmp/SERS_RAMAN_CORPUS/b_header_only.csv'
    # This file still has the Raman tag from its field but lacks TIME.
    result=m.triage(data,input_sha='a'*64,top_k=5)
    assert all(x['relative_path_PRIVATE']!='b_header_only.csv' for x in result['cases'][0]['requirements'][0]['top_unreviewed_files_PRIVATE'])


def test_reports_deprioritized():
    req=m.triage(source(),input_sha='b'*64,top_k=5)['cases'][0]['requirements'][0]
    assert req['top_unreviewed_files_PRIVATE'][-1]['signal_tier']=='DERIVED_OR_REPORT_ARTIFACT_VERIFY_ORIGIN'


def test_orientation_header_not_independent_witness():
    req=m.triage(source(),input_sha='b'*64,top_k=5)['cases'][0]['requirements'][1]
    assert req['top_unreviewed_files_PRIVATE'][0]['measurement_independence_verified'] is False
    assert req['independence_verified'] is False


def test_method_string_is_not_measurement_qualification():
    req=m.triage(source(),input_sha='b'*64,top_k=5)['cases'][0]['requirements'][1]
    picked=next(x for x in req['top_unreviewed_files_PRIVATE'] if x['relative_path_PRIVATE']=='orientation_sfg_experiment.csv')
    assert picked['method_cue']=='METHOD_STRING_CUE_UNVERIFIED'
    assert picked['scientific_evidence_confirmed'] is False


def test_unmatched_requirement_not_falsely_absent():
    req=m.triage(source(),input_sha='b'*64,top_k=5)['cases'][0]['requirements'][2]
    assert req['status']=='NO_STRICT_METADATA_CUE_IN_SCANNED_FILES'
    assert req['top_unreviewed_files_PRIVATE']==[]


def test_empty_specificity_can_abstain():
    a=source();a['cases'][0]['observables'][0]['name']='foo'
    req=m.triage(a,input_sha='c'*64,top_k=5)['cases'][0]['requirements'][0]
    # A renamed/unrecognized observable must not inherit another one's
    # case-specific hint set merely because it has the same role.
    assert req['status']=='NO_SPECIFIC_METADATA_QUERY_SAFE_TO_USE'
    a['cases'][0]['case_id']='OTHER'
    assert m.triage(a,input_sha='c'*64,top_k=5)['cases'][0]['requirements'][0]['status']=='NO_SPECIFIC_METADATA_QUERY_SAFE_TO_USE'


@pytest.mark.parametrize('field,bad',[
    ('schema_version','fake'),
    ('status','SCIENTIFICALLY_VERIFIED'),
    ('science_truth_or_novelty_certified',True),
    ('measurement_independence_certified',True),
    ('source_files_modified',True),
    ('llm_or_network_calls',1),
    ('automatic_research_idea_or_hypothesis_promotion',True),
    ('m4a2_input_sha256','invalid'),
])
def test_fail_closed_unsafe_source_flags(field,bad):
    a=source();a[field]=bad
    with pytest.raises(ValueError,match='M4B2_INTEGRITY_FAILURE'):
        m.triage(a,input_sha='c'*64,top_k=5)


def test_fail_closed_duplicate_file():
    a=source();a['files'].append(copy.deepcopy(a['files'][0]));a['file_count']+=1
    with pytest.raises(ValueError,match='file identity'):
        m.validate_inventory(a)


def test_fail_closed_bad_candidate_pointer():
    a=source();a['cases'][0]['observables'][0]['candidate_files_HEURISTIC']=[{'file_path_PRIVATE':'/abs/not-in-index'}]
    with pytest.raises(ValueError,match='unknown file'):
        m.validate_inventory(a)


def test_fail_closed_promoted_case():
    a=source();a['cases'][0]['empirical_discrimination_established']=True
    with pytest.raises(ValueError): m.validate_inventory(a)


def test_fail_closed_promoted_observable():
    a=source();a['cases'][0]['observables'][0]['observations_validated']=True
    with pytest.raises(ValueError): m.validate_inventory(a)


def test_fail_closed_file_count():
    a=source();a['file_count']+=1
    with pytest.raises(ValueError): m.validate_inventory(a)


@pytest.mark.parametrize('top',[0,21])
def test_bounded_shortlist(top):
    with pytest.raises(ValueError): m.triage(source(),input_sha='a'*64,top_k=top)


def test_write_outputs_no_mutation(tmp_path):
    source_path=tmp_path/'input.json'
    data=source();source_path.write_text(json.dumps(data),encoding='utf-8')
    old=source_path.read_bytes()
    out=tmp_path/'out'
    result=m.execute(argparse.Namespace(m4b1_inventory=source_path,output_dir=out,top_k=3,expected_m4b1_sha256=m.sha256(source_path)))
    assert source_path.read_bytes()==old
    assert result['source_files_modified'] is False
    csv_path=out/'M4B2_REVIEW_TEMPLATE_PRIVATE.csv'
    rows=list(csv.DictReader(csv_path.open(encoding='utf-8')))
    assert len(rows)>0 and all(x['review_outcome']=='UNREVIEWED' for x in rows)
    assert (out/'M4B2_REPORT_PRIVATE.md').is_file()
    assert (out/'M4B2_METADATA_TRIAGE_PRIVATE.json').is_file()


def test_wrong_expected_pin_refuses_new_output(tmp_path):
    p=tmp_path/'src.json';p.write_text(json.dumps(source()))
    out=tmp_path/'badout'
    with pytest.raises(ValueError,match='SHA mismatch'):
        m.execute(argparse.Namespace(m4b1_inventory=p,output_dir=out,top_k=3,expected_m4b1_sha256='f'*64))
    assert not out.exists()


def test_no_overwrite(tmp_path):
    p=tmp_path/'src.json';p.write_text(json.dumps(source()))
    d=tmp_path/'exists';d.mkdir()
    with pytest.raises(ValueError,match='output exists'):
        m.execute(argparse.Namespace(m4b1_inventory=p,output_dir=d,top_k=3,expected_m4b1_sha256=None))


def test_partial_scanning_not_mistaken_for_total_coverage():
    a=source();a['scan_truncated']=True;a['status']='PARTIAL_INVENTORY_REVIEW_REQUIRED'
    r=m.triage(a,input_sha='c'*64,top_k=3)
    assert r['source_scan_truncated'] is True and r['no_claims_about_unscanned_data'] is True


def test_four_case_synthetic_contract_no_science_claims():
    a = source()
    cases = [
        ("QA_B12_AIR_EXPOSURE", [
            ("time-resolved relative SERS band ratios", "TARGET_OUTCOME"),
            ("molecular orientation independent of relative SERS band ratios", "INDEPENDENT_DISCRIMINATOR"),
            ("surface and local-field evolution", "INDEPENDENT_DISCRIMINATOR"),
        ]),
        ("QA_B01_ELECTRONIC_VS_EM", [
            ("anchoring-sensitive relative SERS band ratios", "TARGET_OUTCOME"),
            ("interfacial electronic state", "INDEPENDENT_DISCRIMINATOR"),
            ("electromagnetic field response and accessibility", "INDEPENDENT_DISCRIMINATOR"),
        ]),
        ("QA_B03_B06_PROXY_COVARIANCE", [
            ("spatial or polarization-resolved relative SERS ratios", "TARGET_OUTCOME"),
            ("active-subpopulation-sensitive orientation", "INDEPENDENT_DISCRIMINATOR"),
            ("registered orientation-by-field covariance", "INDEPENDENT_DISCRIMINATOR"),
        ]),
        ("QA_B17_DYNAMIC_VS_STATIC", [
            ("time-resolved SERS band ratios", "TARGET_OUTCOME"),
            ("molecular residence or orientation states", "INDEPENDENT_DISCRIMINATOR"),
            ("hotspot intermittency or near-field activity", "INDEPENDENT_DISCRIMINATOR"),
        ]),
    ]
    a["cases"] = [{
        "case_id": cid,
        "readiness": "UNKNOWN_PENDING_DATA_LINKAGE",
        "empirical_discrimination_established": False,
        "observables": [{
            "name": n, "role": role, "candidate_file_count": 4,
            "candidate_files_HEURISTIC": [],
            "independent_measurement_witness_verified": False,
            "observations_validated": False,
        } for n, role in obs],
    } for cid, obs in cases]
    r = m.triage(a, input_sha="d"*64, top_k=2)
    assert len(r["cases"]) == 4
    assert sum(len(x["requirements"]) for x in r["cases"]) == 12
    assert r["measured_values_read"] is False
    assert r["empirical_truth_or_falsification_certified"] is False
    assert all(not c["case_empirically_testable_certified"] for c in r["cases"])
