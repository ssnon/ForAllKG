import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from scripts.discovery import run_m22_plan_terms_recovery as recovery
from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m
from test_m22_evidence_to_calc import sample_pdf


def put(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False))


@pytest.fixture
def source(tmp_path):
    with patch.object(m, 'root', side_effect=lambda p: Path(p).resolve()):
        prior=tmp_path/'m21';work=tmp_path/'research';out=tmp_path/'m22'
        (prior/'select').mkdir(parents=True)
        (work/'step_0001').mkdir(parents=True)
        put(prior/'lock.json',{'model':'fake','temperature':0.3,'max_tokens':6000})
        put(prior/'select'/'normalized.json',{'selection_rationale':'A scientific problem with competing explanations.'})
        put(prior/'literature.json',{'directions':[{'papers':[{'paper_id':'p1','title':'A synthetic source','abstract':'A rate law','source_depth':'abstract','doi':'10.0000/example'}]}]})
        put(work/'scientific_context.json',{'autonomous_opening':{'selected_research_question':'What is the relevant rate for this scientific mechanism?'}})
        put(work/'step_0001'/'normalized.json',{'operation':'MODEL'})
        dummy={'normalized':{'operation':'MODEL','scientific_delta':'A proposed model needs grounded rate coefficients.',
            'scientific_work_content':{'candidate_consequence':'An untested relation'},
            'next_research_question':'Does reported rate matter?'},'tool_result':None}
        with patch.object(m.m21,'root',side_effect=lambda p:Path(p).resolve()),\
             patch.object(m,'_m21_work_state',return_value=({'steps':1},[dummy])):
            m.prepare(prior,work,out)
        model_plan={'research_question':'What rate relates to this synthetic physical phenomenon?',
              'missing_physical_constraint':'A quantitative rate under stated temperature and source conditions.',
              'why_it_changes_science':'Its magnitude determines whether this mechanism could act fast enough.',
              'ranked_paper_ids':['p1'],
              'pdf_search_terms':['rate constant temperature dependence in confined nanostructures with unusual adsorption environments and pressure', 'activation energy'],
              'source_limitations':'This synthetic reference is insufficient for actual SERS mechanism claims.'}
        phase=out/'plan';phase.mkdir();req=m._phase_req(out,'plan')
        put(phase/'request.json',req);put(phase/'attempt.json',{'request_sha256':m.sha(req)})
        put(phase/'raw_response.json',{'raw_text':json.dumps(model_plan)})
        assert m.verify(out)['plan']=='RAW_SAVED_RECOVER_NO_RETRY'
        yield out,model_plan


def test_eligible_and_fork_immutable(source,tmp_path):
    out,raw_plan=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        pre=recovery.inspect(out)
        assert pre['status']=='M22_OVERLONG_PLAN_TERMS_ELIGIBLE_NO_CALL'
        assert pre['search_terms'][0]['modified']
        before=recovery.file_sha(out/'plan/raw_response.json')
        with pytest.raises(ValueError,match='INVALID_PDF_SEARCH_TERM'):
            m.recover(out,'plan')
        new=tmp_path/'m22_fixed'
        r=recovery.fork(out,new)
        assert r['paid_calls']==0 and r['plan']=='READY'
        assert recovery.file_sha(out/'plan/raw_response.json')==before
        assert not (out/'plan/normalized.json').exists()
        norm=m.read(new/'plan/normalized.json')
        assert norm['ranked_paper_ids']==raw_plan['ranked_paper_ids']
        assert all(norm[k]==raw_plan[k] for k in raw_plan if k!='pdf_search_terms')
        assert len(norm['pdf_search_terms'][0])<=80
        assert norm['pdf_search_terms'][1]=='activation energy'
        assert recovery.compatible('verify',new)['plan']=='READY'
        # Native engine deliberately refuses the fork without a documented scope.
        with pytest.raises(ValueError, match='INVALID_PDF_SEARCH_TERM'):
            m.verify(new)
        assert m.Plan.model_validate_json(json.dumps(raw_plan)).pdf_search_terms==raw_plan['pdf_search_terms']


def test_fork_fetch_local_pdf_without_paid_calls(source,tmp_path):
    out,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        target=tmp_path/'m22_fixed';recovery.fork(out,target)
        local=tmp_path/'paper.pdf'
        local.write_bytes(sample_pdf(['Rate constant k = 0.2 per second under 300 K, synthetic test only.']))
        result=recovery.compatible('fetch',target,pdf_file=local,paper_id='p1')
        assert result['status']=='PDF_PAGES_RETRIEVED' and result['paid_calls']==0
        assert recovery.compatible('verify',target)['pdf']=='PDF_PAGES_RETRIEVED'
        assert recovery.compatible('preview',target,phase='analyze')['phase']=='analyze'
        assert recovery.file_sha(out/'plan/raw_response.json')==recovery.file_sha(target/'plan/raw_response.json')


def test_refuse_missing_or_malformed_search_terms(source,tmp_path):
    out,raw_plan=source
    for new_terms,expected in [(['k',raw_plan['pdf_search_terms'][0]],'OTHER_INVALID_SEARCH_HINT'),
              (['regular keyword'],'NO_OVERLONG_SEARCH_HINT')]:
        mutated=dict(raw_plan,pdf_search_terms=new_terms)
        put(out/'plan/raw_response.json',{'raw_text':json.dumps(mutated)})
        with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
            with pytest.raises(ValueError,match=expected):recovery.inspect(out)


def test_refuse_unrecognized_paper_id(source):
    out,raw_plan=source
    put(out/'plan/raw_response.json',{'raw_text':json.dumps(dict(raw_plan,ranked_paper_ids=['never-returned']))})
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        with pytest.raises(ValueError,match='INVALID_OR_UNRETRIEVED_PAPER_ID'):
            recovery.inspect(out)


def test_reject_any_tampering_with_paid_raw(source,tmp_path):
    out,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        target=tmp_path/'m22_fixed';recovery.fork(out,target)
        raw=m.read(target/'plan/raw_response.json')
        raw['raw_text'] += '\n'
        put(target/'plan/raw_response.json',raw)
        with pytest.raises(ValueError,match='FORK_PAID_RAW_CHANGED'):
            recovery.compatible('verify',target)


def test_reject_unsplittable_long_query(source):
    out,raw_plan=source
    put(out/'plan/raw_response.json',{'raw_text':json.dumps(dict(raw_plan,pdf_search_terms=['x'*120]))})
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        with pytest.raises(ValueError,match='LONG_UNSPLITTABLE_SEARCH_TERM'):
            recovery.inspect(out)


def test_full_offline_source_anchored_calculation(source,tmp_path):
    out,_=source
    with patch.object(m,'root',side_effect=lambda p:Path(p).resolve()):
        target=tmp_path/'m22_fixed';recovery.fork(out,target)
        local=tmp_path/'paper.pdf'
        local.write_bytes(sample_pdf(['Rate constant k = 0.2 per second under 300 K, synthetic test only.']))
        assert recovery.compatible('fetch',target,pdf_file=local,paper_id='p1')['status']=='PDF_PAGES_RETRIEVED'
        phase=target/'analyze';phase.mkdir()
        result={'attempted_question':'How rapidly does this synthetic measured rate predict changes?',
              'anchors':[{'page':1,'exact_source_span':'Rate constant k = 0.2 per second under 300 K',
                          'physical_relation_or_parameter':'The reported synthetic rate coefficient.',
                          'source_conditions':'Synthetic test at 300 K only.',
                          'uncertainties':'Not physically validated for the studied nanorod.'}],
              'scientific_relation':'In a synthetic test the rate coefficient is 0.2 per second.',
              'conditions_and_domain':'Only applicable to this artificial demonstration at 300 K.',
              'what_is_not_supported':'No conclusion about the Ye experiment is supported by this test.',
              'calculation':{'expression':'k*dt','variables':{
                   'k':{'value':0.2,'unit':'1/s','origin':'SOURCE_REPORTED','supporting_anchor_index':0,
                        'justification':'The cited source gives 0.2 per second.'},
                   'dt':{'value':5,'unit':'s','origin':'EXPLICIT_ASSUMPTION','supporting_anchor_index':None,
                        'justification':'An arbitrary five-second time interval.'}},
                   'result_unit':'dimensionless',
                   'modeled_mapping_to_source':'Assume that the source-reported rate remains constant.',
                   'interpretation_and_limits':'Conditional arithmetic, not a real-world mechanistic inference.'},
              'research_next_action':'RETURN',
              'reason_for_action':'This synthetic test does not provide a scientifically useful new constraint.',
              'epistemic_status':'SOURCE_ANCHORED_INTERPRETATION_NOT_VALIDATED'}
        req=m._phase_req(target,'analyze')
        put(phase/'request.json',req)
        put(phase/'attempt.json',{'request_sha256':m.sha(req)})
        put(phase/'raw_response.json',{'raw_text':json.dumps(result)})
        done=recovery.compatible('recover',target)
        assert done['state']['analysis']=='READY'
        report=recovery.compatible('report',target)
        assert report['calculation_result']['result']==1.0
        assert report['scientific_validation'] is False
