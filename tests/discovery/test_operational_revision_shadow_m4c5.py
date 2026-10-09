"""M4-C5 deterministic safety/operational-delta tests. No paid calls/data writes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.conditional_covariance_confrontation_m4c4 import fixture, _metrics
from pipeline_core.discovery.operational_revision_shadow_m4c5 import (
    B03, B06, execute_shadow, render_report,
)
from scripts.discovery.replay_operational_revision_shadow_m4c5 import run


def _kernel(suffix: str) -> dict:
    return {
        "schema_version":"research-idea-kernel-v1",
        "canonical_intent":"Study SERS intensity orientation relationship "+suffix,
        "core_scientific_commitments":["Orientation and local fields may affect SERS intensity "+suffix],
        "scope_commitments":["Matched Au Ag surfaces "+suffix],
        "contrastive_commitments":["Compare orientation and field mechanisms "+suffix],
        "question_commitment":"Which mechanism explains observed intensity "+suffix+"?",
    }


def _traces() -> dict:
    return {
        "status":"SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW",
        "authoritative_science_judgment":False,
        "original_data_mutated":False,
        "rows":[{
            "blind_id":blind,"terminal_idea_id":idea,"lineage_validation":"PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY",
            "steps_earliest_to_latest":[{"idea_id":idea,"kernel":_kernel(blind)}],
        } for blind,idea in (("B03",B03),("B06",B06))],
    }


def _m4c4() -> dict:
    return {
        "schema_version":"m4c4-conditional-predictive-increment-shadow-v1",
        "status":"SYNTHETIC_DISCRIMINATION_DEMO_ONLY_NO_SCIENTIFIC_REVISION",
        "case_id":"QA_B03_B06_PROXY_COVARIANCE",
        "source_files_mutated":False,"research_idea_nodes_created":False,
        "production_selection_changed":False,"empirical_data_consumed":False,
        "scientific_improvement_certified":False,"scientific_truth_or_falsification_authority":False,
        "llm_or_network_calls":0,
        "lineage":{"b03_idea_id":B03,"b06_idea_id":B06},
        "source_sha256":{"m4a1":"cabf847646a23eb292aa9cc1deb5097a07cf4c927605d9f6f4631da5e29732f7"},
        "design":{
            "parent_idea_ids_preserved_separately":sorted((B03,B06)),
            "comparison_type":"NESTED_NONEXCLUSIVE_PREDICTIVE_INCREMENT",
            "causal_claim":False,"mechanisms_mutually_exclusive":False,
            "scientific_question":"Does independently registered orientation–local-field covariance improve held-out SERS prediction?",
            "operational_estimand":"baseline minus augmented held-out mean squared error (delta_MSE)",
            "baseline_model":"f(active orientation, mean orientation, field, substrate)",
            "augmented_model":"baseline + orientation–field covariance",
            "required_measurement_witnesses":["independent orientation assay","independent field map",
                "registered covariance measured without target", "SERS target and error","sample group linkage"],
            "required_controls":["substrate", "reporter", "coverage", "optics", "registration"],
        },
        "synthetic_fixtures":{
            "COVARIANCE_EFFECT_INJECTED":_metrics(fixture(effect=1.15)),
            "NO_COVARIANCE_EFFECT":_metrics(fixture(effect=0.0)),
            "EXACT_COLLINEARITY":_metrics(fixture(effect=1.15,collinear=True)),
        },
    }


def _run(a=None,t=None):
    return execute_shadow(m4c4=a or _m4c4(),trajectories=t or _traces(),
                          sha_m4c4="a"*64,sha_traces="b"*64)


def test_two_distinct_operational_drafts_and_identifiability_abstention():
    r = _run()
    assert r['kernel_drafts']==2 and r['abstentions']==1
    assert r['rows'][-1]['status']=='ABSTAIN_NON_IDENTIFIABLE'
    assert r['rows'][-1]['kernel_draft_NOT_CREATED'] is None
    assert len({x['draft_id_NOT_RESEARCH_IDEA_ID'] for x in r['rows'][:2]}) == 2


def test_parent_core_is_preserved_and_new_estimand_is_explicit():
    r=_run()
    old=_traces()['rows'][1]['steps_earliest_to_latest'][-1]['kernel']
    for row in r['rows'][:2]:
        kernel=row['kernel_draft_NOT_CREATED']
        assert kernel['core_scientific_commitments']==old['core_scientific_commitments']
        assert len(kernel['contrastive_commitments'])==len(old['contrastive_commitments'])+4
        assert 'delta_MSE' in row['protocol']['estimand']
        assert 'covariance' in row['protocol']['augmented']
        assert row['comparison_parent_idea_id_PRESERVED']==B03
        assert row['seed_parent_idea_id_PRESERVED']==B06
        assert row['sis_identity_diagnostic_ONLY'] in {'DIFFERENT_IDEA','INDETERMINATE','SAME_IDEA'}
        assert len(row['sis_facet_diagnostics_ONLY'])==4
        assert row['semantic_or_scientific_improvement_certified'] is False


def test_positive_and_null_reframes_are_not_the_same_question():
    r=_run()
    assert r['rows'][0]['protocol']['question'] != r['rows'][1]['protocol']['question']
    assert r['rows'][0]['protocol']['revision_dimension']=='OPERATIONAL_INCREMENT_RESEARCH_QUESTION'
    assert r['rows'][1]['protocol']['revision_dimension']=='REGIME_BOUNDARY_RESEARCH_QUESTION'
    assert 'globally' in r['rows'][0]['protocol']['scoped_non_confirmation']
    assert 'regime' in r['rows'][1]['protocol']['scoped_non_confirmation']


def test_no_science_or_sis_node_authority():
    r=_run()
    for f in ['new_research_idea_nodes_created','scientific_semantic_improvement_certified',
              'empirical_measurements_consumed','production_selection_changed','original_idea_nodes_mutated']:
        assert r[f] is False
    assert r['llm_or_network_calls']==0
    assert 'HUMAN_SPECIFIED' in r['candidate_generation_method']
    assert 'No ResearchIdeaNode' in render_report(r)


def test_deterministic_replay():
    r=_run()
    assert json.dumps(r,sort_keys=True)==json.dumps(_run(),sort_keys=True)


@pytest.mark.parametrize('field,bad',[
    ('status','SIS_EVOLUTION_COMPLETED'),('empirical_data_consumed',True),
    ('research_idea_nodes_created',True),('scientific_improvement_certified',True),
    ('production_selection_changed',True),('scientific_truth_or_falsification_authority',True),
    ('llm_or_network_calls',1),('source_files_mutated',True),
])
def test_reject_input_science_promotion(field,bad):
    a=_m4c4();a[field]=bad
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)


def test_reject_parent_merge_or_causal_promotion():
    for mutate in (lambda a:a['design'].update(mechanisms_mutually_exclusive=True),
                   lambda a:a['design'].update(causal_claim=True),
                   lambda a:a['design'].update(parent_idea_ids_preserved_separately=[B03]),
                   lambda a:a['lineage'].update(b06_idea_id=B03)):
        a=_m4c4(); mutate(a)
        with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
            _run(a=a)


def test_reject_invalid_fixture_positive_or_null():
    a=_m4c4();a['synthetic_fixtures']['COVARIANCE_EFFECT_INJECTED']['delta_mse']=-.2
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)
    a=_m4c4();a['synthetic_fixtures']['NO_COVARIANCE_EFFECT']['delta_mse']=.4
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)


def test_reject_bad_parent_kernel_or_trajectory():
    t=_traces();t['rows'][1]['terminal_idea_id']=B03
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(t=t)
    t=_traces();t['rows'][0]['steps_earliest_to_latest'][-1]['idea_id']=B06
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(t=t)


def test_reject_unknown_fixture_and_wrong_case():
    a=_m4c4();a['synthetic_fixtures']['ARTIFICIAL']=a['synthetic_fixtures']['NO_COVARIANCE_EFFECT']
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)
    a=_m4c4();a['case_id']='QA_B12_AIR_EXPOSURE'
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)


def test_prospective_measurement_plan_missing_rejected():
    a=_m4c4();a['design']['required_measurement_witnesses']=[]
    with pytest.raises(ValueError,match='M4C5_INTEGRITY_FAILURE'):
        _run(a=a)


def test_real_file_runner_does_not_overwrite_or_accept_unpinned(tmp_path):
    a=tmp_path/'a.json'; b=tmp_path/'b.json'
    a.write_text(json.dumps(_m4c4()),encoding='utf-8')
    b.write_text(json.dumps(_traces()),encoding='utf-8')
    getsha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='SHA mismatch'):
        run(m4c4=a,m3c1=b,expected_m4c4_sha256='0'*64,
            expected_m3c1_sha256=getsha(b),output_dir=out)
    assert not out.exists()
    report=run(m4c4=a,m3c1=b,expected_m4c4_sha256=getsha(a),
               expected_m3c1_sha256=getsha(b),output_dir=out)
    assert report['kernel_drafts']==2
    assert (out/'M4C5_REPORT_PRIVATE.md').exists()
    with pytest.raises(FileExistsError,match='overwrite'):
        run(m4c4=a,m3c1=b,expected_m4c4_sha256=getsha(a),
            expected_m3c1_sha256=getsha(b),output_dir=out)


def test_source_files_are_not_mutated(tmp_path):
    a=tmp_path/'a.json'; b=tmp_path/'b.json'
    a.write_text(json.dumps(_m4c4()),encoding='utf-8')
    b.write_text(json.dumps(_traces()),encoding='utf-8')
    dig=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before=[dig(a),dig(b)]
    run(m4c4=a,m3c1=b,expected_m4c4_sha256=before[0],
        expected_m3c1_sha256=before[1],output_dir=tmp_path/'out')
    assert [dig(a),dig(b)]==before
