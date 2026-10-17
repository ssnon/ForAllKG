"""No network, no paid calls, source hashes and scientific inference safeguards."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.discovery import run_m23_evidence_constrained_reasoning as m


class FakePage:
    def __init__(self, text):self.text=text
    def extract_text(self):return self.text
class FakeReader:
    def __init__(self,pages):self.pages=[FakePage(t) for t in pages]


def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False),encoding='utf-8')


@pytest.fixture
def prepared(tmp_path):
    p=tmp_path/'source_file.pdf';p.write_bytes(b'%PDF-test-static-bytes')
    sentence='Subse-\nquently, the plasmon can generate carriers at 1-100 fs under these example conditions.'
    sentence2='After injection, hot elec-\ntrons may return across the interface in this illustrative example.'
    claimed1='Subsequently, the plasmon can generate carriers at 1-100 fs under these example conditions.'
    claimed2='After injection, hot electrons may return across the interface in this illustrative example.'
    fake_reader=FakeReader([sentence,sentence2])
    m21=tmp_path/'m21';put(m21/'step_0006/normalized.json',{'next_action':'CONTINUE','physics':'Earlier hypothesis is conditional, not established'})
    charge=tmp_path/'charge'
    put(charge/'lock.json',{'model':'fake-model','temperature':0,'max_tokens':4500})
    put(charge/'plan/normalized.json',{'research_question':'Can a physical model bound the charge-trapping mechanism?',
                                      'missing_physical_constraint':'Unknown recombination lifetime and optical coupling.'})
    put(charge/'pdf_evidence.json',{'status':'PDF_PAGES_RETRIEVED','paper_id':m.PAPER_ID,
         'pdf_sha256':m.digest(p),'selected_pages':[{'page':1,'text':sentence},{'page':2,'text':sentence2}]})
    analysis={'anchors':[
        {'page':1,'exact_source_span':claimed1,'source_conditions':'Only this synthetic regime',
         'physical_relation_or_parameter':'Short-timescale generation'},
        {'page':2,'exact_source_span':claimed2,'source_conditions':'Transfer competition',
         'physical_relation_or_parameter':'Back transfer'}],
        'scientific_relation':'Evidence of competing paths','conditions_and_domain':'Metal interface'}
    put(charge/'analyze/raw_response.json',{'raw_text':json.dumps(analysis)})
    out=tmp_path/'output'
    with patch.object(m,'PdfReader',return_value=fake_reader):
        status=m.prepare(m21,charge,p,out)
    assert status['verified_pdf_anchors']==2
    return out,m21,charge,p


def development():
    return dict(
      selected_question='What physical condition would distinguish possible charge accumulation from loss?',
      scientific_action='RESTRICT',reason_for_selection='This tests whether source evidence can constrain a model rather than decorate it.',
      premises=[dict(kind='SOURCE',statement='The synthetic source permits hot-carrier generation in a metal.',source_ids=['HOT:0'],applicability_limit='Only synthetic illustrative conditions'),
                dict(kind='MODEL_ASSUMPTION',statement='Assume constant generation with linear first-order loss.',source_ids=[],applicability_limit='Only unsaturated dilute trap dynamics')],
      derivation_steps=['Suppose a nonnegative current J enters an initially empty trap population and is lost at a first order rate.','Under stationary linear balance production equals first-order loss.'],
      result_kind='NECESSARY_CONDITION',
      explicit_result='If linear first-order trapping holds and a target charge is required, injected flux must exceed its loss rate; no material-specific estimate follows.',
      physical_variable_definitions='J is the trap injection flux (s^-1); N is trap occupancy (dimensionless); tau is the residence time (s).',
      strongest_competing_explanation='A thermally modulated structural state might produce similar optical dynamics independently of trap charge.',
      testable_discriminator_or_obstruction='Independent bounds on optical generation and trap lifetime are needed to exclude a thermal competing explanation.',
      unsupported_leaps=['Neither a trap lifetime nor a Raman Stark coefficient is independently determined.'],
      numeric_or_symbolic_probe=dict(kind='ARITHMETIC_IDENTITY',left_expression='J*tau',right_expression='tau*J',
         variables={'J':2.0,'tau':3.0},relation='EQ',physical_meaning='Arithmetic consistency of a simple linear balance model.'),
      what_changed_from_M21_and_M22='The arbitrary optical contrast is replaced by an explicit conditional flux-lifetime constraint, not yet grounded numerically.',
      epistemic_status='CONDITIONAL_MODEL_RESULT')


def challenge():
    return dict(
      strongest_null_mechanism='A thermally driven reversible hotspot state with mode-selective response could mimic the same observations.',
      attempted_counterexample_or_derivation='Choose a thermal state with the same output correlation and matched optical observation functions; neither trap lifetime nor interfacial field is observed.',
      constraints_shared_by_both_models=['Both models must reproduce the reported time-dependent Raman response.'],
      previous_result_survives='YES_CONDITIONALLY',
      survival_conditions=['The mass-balance equation holds only under the explicit linear loss assumption.'],
      objection_or_repair='A flux-lifetime product does not establish causation or give a discriminating spectroscopic prediction.',
      new_research_information='The derived identity leaves the thermal alternative viable; independently measured coupling and kinetics are required.',
      revised_falsification_or_measurement='Measure the trap lifetime and field-sensitive vibrational response independently under controlled heating.',
      final_disposition='RETAIN_CONDITIONAL',evidence_ids_used=['HOT:0','HOT:1'])


def test_prepare_retains_verbatim_source_and_science_is_unverified(prepared):
    out,_,charge,p=prepared
    pkt=m.load(out/'input.json')
    assert len(pkt['verified_SOURCE_EXCERPTS'])==2
    assert 'Subse-\nquently' in pkt['verified_SOURCE_EXCERPTS'][0]['pdf_source_span']
    assert 'elec-\ntrons' in pkt['verified_SOURCE_EXCERPTS'][1]['pdf_source_span']
    assert m.verify(out)['develop']=='NOT_STARTED'
    assert not (charge/'analyze/normalized.json').exists()


def test_develop_and_challenge_paid_raw_saved_and_no_repay(prepared):
    out,*_=prepared
    callbacks=[]
    def backend_develop(req):
        callbacks.append(req)
        return json.dumps(development()),None,'test-response-1'
    assert m.preview(out,'develop')['next_call_is_paid']
    res=m.execute(out,'develop',backend=backend_develop)
    assert res['new_paid_calls_this_command']==0 and len(callbacks)==1
    with pytest.raises(ValueError,match='PHASE_ALREADY_STARTED'):
        m.execute(out,'develop',backend=backend_develop)
    assert len(callbacks)==1
    assert m.load(out/'develop/numeric_probe.json')['status']=='DIAGNOSTIC_NUMERIC_PROBE_PASSED'
    def backend_challenge(req):
        callbacks.append(req)
        return json.dumps(challenge()),None,'test-response-2'
    m.execute(out,'challenge',backend=backend_challenge)
    assert len(callbacks)==2
    report=m.report(out)
    assert report['status']=='M23_SCIENTIFIC_AUDIT_READY_NO_CALL'
    assert report['level_ii_certified'] is False
    assert report['independent_scientific_review_required'] is True
    assert m.verify(out)['challenge']=='READY'


def test_recover_without_second_call_after_saving_raw(prepared):
    out,*_=prepared
    req=m.request(out,'develop')
    put(out/'develop/request.json',req)
    put(out/'develop/attempt.json',{'request_sha256':m.hashobj(req)})
    raw=json.dumps(development())
    put(out/'develop/raw_response.json',{'raw_text':raw,'usage':None,'response_id':'old'})
    assert m.verify(out)['develop']=='RAW_SAVED_RECOVER_NO_REPAY'
    m.recover(out,'develop')
    assert m.verify(out)['develop']=='READY'
    assert m.load(out/'develop/raw_response.json')['raw_text']==raw


def test_reject_invented_source_id_preserving_raw(prepared):
    out,*_=prepared
    bad=development();bad['premises'][0]['source_ids']=['HOT:99']
    with pytest.raises(RuntimeError,match='M23_PAID_RAW_SAVED_DO_NOT_REPAY'):
        m.execute(out,'develop',backend=lambda req:(json.dumps(bad),None,'id'))
    assert m.verify(out)['develop']=='RAW_SAVED_RECOVER_NO_REPAY'
    with pytest.raises(ValueError,match='UNREGISTERED_SOURCE_ANCHOR'):
        m.recover(out,'develop')


def test_original_input_mutation_rejected(prepared):
    out,_,charge,_=prepared
    put(charge/'plan/normalized.json',{'changed':'true'})
    with pytest.raises(ValueError,match='M23_ORIGINAL_SOURCE_CHANGED'):
        m.verify(out)


def test_numeric_probe_is_only_a_pointwise_check():
    p=development()['numeric_or_symbolic_probe']
    assert m.evaluate(p)['status']=='DIAGNOSTIC_NUMERIC_PROBE_PASSED'
    p['right_expression']='J*tau+1'
    assert m.evaluate(p)['status']=='DIAGNOSTIC_NUMERIC_PROBE_FAILED'
    p['right_expression']='open("/etc/passwd")'
    assert m.evaluate(p)['status']=='PROBE_REJECTED'


def test_conflicting_verbatim_or_numbers_not_silently_repaired():
    text='The physical efficiency is 0.2 and the hot elec-\ntrons may be transferred.'
    assert m.locate_verbatim(text,'The physical efficiency is 0.2 and the hot electrons may be transferred.')['start_offset']==0
    with pytest.raises(ValueError,match='CLAIMED_ANCHOR_NOT_FOUND'):
        m.locate_verbatim(text,'The physical efficiency is 0.9 and the hot electrons may be transferred.')
    with pytest.raises(ValueError,match='AMBIGUOUS_ANCHOR'):
        m.locate_verbatim(text+' '+text,'The physical efficiency is 0.2 and the hot electrons may be transferred.')
