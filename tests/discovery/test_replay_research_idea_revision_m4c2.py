from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline_core.discovery.empirical_revision_policy_m4c0 import build_policy_replay
from pipeline_core.discovery.research_idea_revision_m4c2 import (
    build_revision_replay, render_report,
)
from scripts.discovery.replay_research_idea_revision_m4c2 import run


def _bytes(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def _sha_bytes(obj):
    return hashlib.sha256(_bytes(obj)).hexdigest()


def fixture(*, dual=False):
    ids=['research_idea:example_A'] + (['research_idea:example_B'] if dual else [])
    refs=['B12'] + (['B06'] if dual else [])
    hyps=[{'ref':ref,'hypothesis_id':'hypothesis:'+ref,'terminal_idea_id':idea,
           'scientific_intent':'Test difference between related mechanisms'}
         for ref,idea in zip(refs,ids)]
    rel='ADJACENT_NOT_EXCLUSIVE' if dual else 'COMPETING_PARTIAL'
    case={
        'case_id':'QA_EXAMPLE', 'hypothesis_refs':refs,'input_hypotheses':hyps,
        'status':'REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED',
        'confrontation_validated':False,
        'experiment_physically_feasible_certified':False,
        'falsification_logic_certified':False,
        'novelty_or_truth_authority':False,
        'proposed_experiment_UNREVIEWED':{'pair_relationship':rel},
        'proposed_mechanisms_UNREVIEWED':[
            {'role':'A','explanation':'Orientation mediator can affect band ratios',
             'predicted_pattern':'Independent orientation readout changes'},
            {'role':'B','explanation':'Surface environment can affect band ratios',
             'predicted_pattern':'Independent surface measure changes'}],
    }
    a1={
        'schema_version':'m4a1-scientific-confrontation-v1',
        'status':'DRAFT_CONFRONTATIONS_NEED_REVIEW','case_count':1,
        'external_evidence_validated':False,'human_or_expert_science_certification':False,
        'hypothesis_cards_modified':False,'source_files_mutated':False,
        'production_selection_changed':False,'llm_or_network_calls':0,
        'cases':[case],
    }
    sha_a1=_sha_bytes(a1)
    c0=build_policy_replay(a1,source_sha256=sha_a1)
    c1={
        'schema_version':'m4c1-evidence-event-audit-v1',
        'status':'EVIDENCE_EVENT_INTAKE_INCOMPLETE',
        'input_m4a1_sha256':sha_a1,'input_m4c0_sha256':_sha_bytes(c0),
        'case_count':1,'file_sha256_match_count':0,
        'real_measurement_values_read':False,'new_llm_or_network_calls':0,
        'hypothesis_cards_modified':False,'independence_certified':False,
        'new_idea_generated':False,'original_research_ideas_modified':False,
        'production_selection_changed':False,'scientific_truth_or_falsification_authority':False,
        'source_files_mutated':False,'cases':[
            {'case_id':'QA_EXAMPLE','original_terminal_idea_ids_PRESERVED':ids,
             'independent_measurements_certified':False,'claim_adjudication_authorized':False}
        ],
    }
    traces={
        'schema_version':'m3c1-kernel-trajectories-v1',
        'status':'SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW',
        'root_set_conserved_across_arms':True,
        'original_data_mutated':False,'new_llm_calls':0,
        'authoritative_science_judgment':False,
        'rows':[
            {'blind_id':ref,'lineage_validation':'PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY',
             'hypothesis_id':'hypothesis:'+ref,'terminal_idea_id':idea,
             'steps_earliest_to_latest':[
                 {'idea_id':idea,'kernel':{
                     'schema_version':'research-idea-kernel-v1',
                     'canonical_intent':'Investigate orientation and spectral change '+ref,
                     'core_scientific_commitments':['SERS band ratios can be influenced by orientation '+ref],
                     'scope_commitments':['Matched metal substrates '+ref],
                     'contrastive_commitments':['Orientation and environment are different mediators '+ref],
                     'question_commitment':'What mediates spectral change '+ref+'?'
                 }}]}
            for ref,idea in zip(refs,ids)
        ],
    }
    args={'a1':a1,'c0':c0,'c1':c1,'trajectories':traces,
          'sha_a1':sha_a1,'sha_c0':_sha_bytes(c0),
          'sha_c1':_sha_bytes(c1),'sha_trajectories':_sha_bytes(traces)}
    return args


def test_fixture_all_scenarios_and_draft_counts():
    result=build_revision_replay(**fixture())
    assert result['scenario_count']==5
    assert result['draft_kernel_count']==4
    assert result['no_revision_count']==1
    assert result['status']=='SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING'


def test_no_real_scientific_idea_created():
    result=build_revision_replay(**fixture())
    assert result['research_idea_nodes_created'] is False
    assert result['empirical_scientific_learning_demonstrated'] is False
    assert result['new_llm_or_network_calls']==0
    assert result['scientific_truth_or_falsification_authority'] is False
    assert result['production_selection_changed'] is False
    assert all(row['research_idea_node_created'] is False for row in result['rows'])
    assert all(row['draft_id_NOT_RESEARCH_IDEA_ID'] is None or row['draft_id_NOT_RESEARCH_IDEA_ID'].startswith('m4c2_kernel_draft:') for row in result['rows'])


def test_all_parent_cores_are_preserved_and_draft_content_changes():
    fixture_kwargs=fixture(dual=True)
    result=build_revision_replay(**fixture_kwargs)
    for row in result['rows']:
        assert row['original_terminal_idea_ids_PRESERVED']==['research_idea:example_A','research_idea:example_B']
        assert row['seed_parent_id']=='research_idea:example_A'
        assert row['comparison_parent_ids_PRESERVED_SEPARATELY']==['research_idea:example_B']
        assert row['multi_parent_composition_NOT_CLAIMED'] is True
        assert len(row['original_parent_scientific_core_by_id'])==2
        if row['draft_scientific_kernel_NOT_CREATED'] is not None:
            draft=row['draft_scientific_kernel_NOT_CREATED']
            assert draft['core_scientific_commitments']==row['original_parent_scientific_core_by_id']['research_idea:example_A']
            assert 'Conditional revision objective' in draft['canonical_intent']
            assert draft['question_commitment']
            assert len(draft['scope_commitments'])>=3
            assert len(draft['contrastive_commitments'])>=3


def test_scenarios_produce_distinct_scientific_research_questions():
    result=build_revision_replay(**fixture())
    questions=[r['draft_scientific_kernel_NOT_CREATED']['question_commitment']
        for r in result['rows'] if r['draft_scientific_kernel_NOT_CREATED']]
    assert len(set(questions))==4


def test_unidentifiable_is_not_revised():
    result=build_revision_replay(**fixture())
    r=[x for x in result['rows'] if x['synthetic_scenario']=='NOT_IDENTIFIABLE'][0]
    assert r['draft_scientific_kernel_NOT_CREATED'] is None
    assert r['draft_kernel_sha256'] is None
    assert r['draft_id_NOT_RESEARCH_IDEA_ID'] is None
    assert 'RESOLVE_MEASUREMENT_INDEPENDENCE' in r['preferred_policy_actions_NOT_EXECUTED']


def test_nonexclusive_never_excludes_competing_mechanism():
    result=build_revision_replay(**fixture(dual=True))
    for row in result['rows']:
        assert row['pair_relationship']=='ADJACENT_NOT_EXCLUSIVE'
        assert 'NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION' in row['scientific_cautions']
        assert row['original_research_ideas_modified'] is False


def test_idempotent():
    f=fixture(dual=True)
    assert build_revision_replay(**f)==build_revision_replay(**f)


def test_no_input_mutation():
    f=fixture(dual=True);saved=copy.deepcopy(f)
    build_revision_replay(**f)
    assert f==saved


def test_render_boundary_notice():
    report=render_report(build_revision_replay(**fixture()))
    assert 'NOT OBSERVED' in report and 'no scientific falsification' in report


@pytest.mark.parametrize('case,change',[
    ('a1',lambda d:d.update(status='SCIENTIFIC_CERTIFIED')),
    ('a1',lambda d:d.update(external_evidence_validated=True)),
    ('c0',lambda d:d.update(status='EMPIRICAL_LEARNING_COMPLETE')),
    ('c0',lambda d:d.update(real_measurements_consumed=True)),
    ('c0',lambda d:d['cases'][0].update(hypothesis_falsified=True)),
    ('c0',lambda d:d['cases'][0].update(scenario_kind='ACTUAL_OBSERVATION')),
    ('c0',lambda d:d['cases'][0].update(original_terminal_idea_ids_PRESERVED=['research_idea:spoof'])),
    ('c0',lambda d:d['cases'].pop()),
    ('c0',lambda d:d['cases'].append(copy.deepcopy(d['cases'][0]))),
    ('c1',lambda d:d.update(status='EVIDENCE_EVENT_INTAKE_COMPLETE')),
    ('c1',lambda d:d.update(file_sha256_match_count=1)),
    ('c1',lambda d:d.update(real_measurement_values_read=True)),
    ('c1',lambda d:d.update(independence_certified=True)),
    ('c1',lambda d:d['cases'][0].update(claim_adjudication_authorized=True)),
    ('trajectories',lambda d:d.update(status='UNKNOWN')),
    ('trajectories',lambda d:d.update(authoritative_science_judgment=True)),
    ('trajectories',lambda d:d['rows'][0].update(terminal_idea_id='research_idea:tampered')),
    ('trajectories',lambda d:d['rows'][0]['steps_earliest_to_latest'][0]['kernel'].update(core_scientific_commitments=[])),
])
def test_fail_closed_for_tampering(case,change):
    f=fixture()
    change(f[case])
    with pytest.raises((ValueError,TypeError),match='M4C2_INTEGRITY_FAILURE|M4C0_INTEGRITY_FAILURE|ValidationError'):
        build_revision_replay(**f)


def test_wrong_a1_sha():
    f=fixture();f['sha_a1']='f'*64
    with pytest.raises(ValueError,match='SHA linkage'):
        build_revision_replay(**f)


def test_wrong_c0_sha():
    f=fixture();f['sha_c0']='0'*64
    with pytest.raises(ValueError,match='C1/C0'):
        build_revision_replay(**f)


def _write_sources(tmp_path, fixture_):
    sources={}
    for source in ('a1','c0','c1','trajectories'):
        path=tmp_path/(source+'.json')
        path.write_bytes(_bytes(fixture_[source]))
        sources[source]=path
    return sources


def test_cli_safe_output_creation(tmp_path):
    f=fixture();ps=_write_sources(tmp_path,f)
    out=tmp_path/'out'
    report=run(a1=ps['a1'],c0=ps['c0'],c1=ps['c1'],trajectories=ps['trajectories'],
               expected_trajectories_sha256=f['sha_trajectories'],output_dir=out)
    assert report['draft_kernel_count']==4
    assert (out/'M4C2_REPORT_PRIVATE.md').is_file()
    assert (out/'M4C2_KERNEL_DRAFTS_PRIVATE.json').is_file()


def test_cli_rejects_bad_m3c1_sha_before_output(tmp_path):
    f=fixture();ps=_write_sources(tmp_path,f)
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='pinned SHA-256 mismatch'):
        run(a1=ps['a1'],c0=ps['c0'],c1=ps['c1'],trajectories=ps['trajectories'],
            expected_trajectories_sha256='0'*64,output_dir=out)
    assert not out.exists()


def test_cli_existing_output_preserved(tmp_path):
    f=fixture();ps=_write_sources(tmp_path,f)
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('original')
    with pytest.raises(FileExistsError,match='overwrite'):
        run(a1=ps['a1'],c0=ps['c0'],c1=ps['c1'],trajectories=ps['trajectories'],
            expected_trajectories_sha256=f['sha_trajectories'],output_dir=out)
    assert (out/'keep').read_text()=='original'


def test_cli_invalid_a1_never_creates_output(tmp_path):
    f=fixture();f['a1']['status']='CERTIFIED';ps=_write_sources(tmp_path,f)
    out=tmp_path/'out'
    with pytest.raises(ValueError):
        run(a1=ps['a1'],c0=ps['c0'],c1=ps['c1'],trajectories=ps['trajectories'],
            expected_trajectories_sha256=f['sha_trajectories'],output_dir=out)
    assert not out.exists()
