"""Offline, fail-closed M22 saved-analysis tests; no LLM or network."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.discovery import run_m22_source_span_recovery as r
from scripts.discovery import run_m22_plan_terms_recovery as plan
from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m
from test_m22_evidence_to_calc import sample_pdf
from test_m22_plan_terms_recovery import source


def put(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False))


def sample_analysis(span,page=1,k=0.2):
    return {'attempted_question':'How quickly does the sample rate contribute to a decay?',
        'anchors':[{'page':page,'exact_source_span':span,
        'physical_relation_or_parameter':'A synthetic rate coefficient that is only a test.',
        'source_conditions':'Applicable only to synthetic conditions in the paper.',
        'uncertainties':'No experimental uncertainty estimate is in this toy example.'}],
        'scientific_relation':'This synthetic rate is not a measured SERS hot-carrier parameter.',
        'conditions_and_domain':'The synthetic source applies only under its specified conditions.',
        'what_is_not_supported':'Nothing can be concluded about Ye SERS blinking from this synthetic example.',
        'calculation':{'expression':'k*dt','variables':{'k':{
            'value':k,'unit':'s^-1','origin':'SOURCE_REPORTED','supporting_anchor_index':0,
            'justification':'Rate constant in the quoted synthetic source.'},
            'dt':{'value':5,'unit':'s','origin':'EXPLICIT_ASSUMPTION','supporting_anchor_index':None,
            'justification':'An assumed interval of five seconds.'}},
            'result_unit':'dimensionless','modeled_mapping_to_source':'Assume a uniform first-order rate over this invented interval.',
            'interpretation_and_limits':'This is an arithmetic example and is not scientific validation.'},
        'research_next_action':'RETURN','reason_for_action':'The toy example cannot validate the physics of the real question.',
        'epistemic_status':'SOURCE_ANCHORED_INTERPRETATION_NOT_VALIDATED'}


def create_failed_run(src,tmp_path,span,*,pagenum=1,source_lines=None,k=0.2):
    out=tmp_path/'pdf_read';plan.fork(src,out)
    local=tmp_path/'source.pdf'
    local.write_bytes(sample_pdf(source_lines or ['Rate constant k = 0.2 per second under 300 K, synthetic test only.']))
    plan.compatible('fetch',out,pdf_file=local,paper_id='p1')
    obj=sample_analysis(span,pagenum,k)
    p=out/'analyze';p.mkdir()
    with r.plan_scope(out):req=m._phase_req(out,'analyze')
    put(p/'request.json',req);put(p/'attempt.json',{'request_sha256':m.sha(req)})
    put(p/'raw_response.json',{'raw_text':json.dumps(obj,ensure_ascii=False)})
    with r.plan_scope(out):
        assert m.verify(out)['analysis']=='RAW_SAVED_RECOVER_NO_RETRY'
        with pytest.raises(ValueError,match='SOURCE_SPAN_NOT_VERBATIM'):
            m.recover(out,'analyze')
    return out


def test_normalized_unique_whitespace_fork(source,tmp_path):
    src,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        failed=create_failed_run(src,tmp_path,'Rate   constant k = 0.2 per second under 300 K')
        result=r.inspect(failed)
        assert result['status']=='M22_VERBATIM_RECOVERY_ELIGIBLE_NO_CALL'
        target=tmp_path/'verbatim';ans=r.fork(failed,target)
        assert ans['paid_calls']==0 and ans['calculation_status']=='SOURCE_ANCHORED_CONDITIONAL_ARITHMETIC_EXECUTED'
        data=m.read(target/'analyze/normalized.json')
        assert data['anchors'][0]['exact_source_span']=='Rate constant k = 0.2 per second under 300 K'
        assert m.read(target/'analyze/calculation_result.json')['result']==1.0
        assert r.verify(target)['analysis']=='READY'
        assert r.report(target)['scientific_validation'] is False
        assert not (failed/'analyze/normalized.json').exists()
        # Native base runner remains strict (no silent changes to source code).
        with r.plan_scope(target),pytest.raises(ValueError,match='SOURCE_SPAN_NOT_VERBATIM'):
            m.verify(target)


def test_unique_page_relocation(source,tmp_path):
    src,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        failed=create_failed_run(src,tmp_path,'Rate constant k = 0.2 per second under 300 K',pagenum=1,
                  source_lines=['Introductory generic paper page.',
                                'Rate constant k = 0.2 per second under 300 K, synthetic test only.'])
        x=r.inspect(failed)
        assert x['safe_matches'][0]['verbatim_page']==2
        target=tmp_path/'relocated';r.fork(failed,target)
        assert m.read(target/'analyze/normalized.json')['anchors'][0]['page']==2


def test_no_repair_for_changed_number(source,tmp_path):
    src,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        failed=create_failed_run(src,tmp_path,'Rate constant k = 0.3 per second under 300 K')
        result=r.inspect(failed)
        assert result['status']=='M22_UNVERIFIED_QUOTES_NEED_MANUAL_REVIEW'
        assert len(result['unresolved_anchors'])==1
        with pytest.raises(ValueError,match='UNMATCHED_SOURCE_QUOTES'):
            r.fork(failed,tmp_path/'dont_make')
        assert not (tmp_path/'dont_make').exists()


def test_no_repair_for_invented_quote(source,tmp_path):
    src,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        failed=create_failed_run(src,tmp_path,'Experimental evidence proves trapped charge is the only SERS cause.')
        result=r.inspect(failed)
        assert result['status']=='M22_UNVERIFIED_QUOTES_NEED_MANUAL_REVIEW'


def test_all_science_fields_preserved(source,tmp_path):
    src,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        failed=create_failed_run(src,tmp_path,'Rate\nconstant k = 0.2 per second under 300 K')
        target=tmp_path/'verified';r.fork(failed,target)
        raw=json.loads(m.read(failed/'analyze/raw_response.json')['raw_text'])
        repaired=m.read(target/'analyze/normalized.json')
        for key in raw:
            if key!='anchors': assert repaired[key]==raw[key]
        assert raw['anchors'][0]['physical_relation_or_parameter']==repaired['anchors'][0]['physical_relation_or_parameter']
        assert r.fhash(target/'analyze/raw_response.json')==r.fhash(failed/'analyze/raw_response.json')
        # Unrecorded tampering invalidates verified analysis.
        val=m.read(target/'analyze/normalized.json');val['scientific_relation']='Tampered!'
        put(target/'analyze/normalized.json',val)
        with pytest.raises(ValueError,match='FORK_NORMALIZED_CHANGED'):
            r.verify(target)


def test_typographic_ligatures_only():
    needle='A fi correction is plausible under 300 K'
    pages={1:'A \ufb01 correction is plausible under 300 K for an optical device.'}
    found=r.exact_selected_span(needle,pages)
    assert found[1]=='A \ufb01 correction is plausible under 300 K'
    assert r.exact_selected_span('a fi correction is plausible under 300 K',pages) is None # case-significant
    assert r.exact_selected_span('A fi correction is plausible under 400 K',pages) is None # number-significant


def test_rejects_ambiguous_duplicate_span():
    pages={1:'Rate constant k = 0.2 per second under 300 K',2:'Rate constant k = 0.2 per second under 300 K'}
    assert r.exact_selected_span('Rate constant k = 0.2 per second under 300 K',pages) is None
