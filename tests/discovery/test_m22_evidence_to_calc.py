"""Focused offline tests; never invoke paid LLM or external network."""
import unittest

def sample_pdf(lines):
    # Self-contained elementary text PDF, no reportlab dependency on the user's host.
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>', None, b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    kids=[]
    for label in lines:
        page_n=len(objects)+1
        stream_n=page_n+1
        kids.append(f'{page_n} 0 R')
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_n} 0 R >>'.encode())
        text=label.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
        stream=f'BT /F1 12 Tf 40 700 Td ({text}) Tj ET'.encode()
        objects.append(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream')
    objects[1]=('<< /Type /Pages /Kids ['+' '.join(kids)+'] /Count '+str(len(kids))+' >>').encode()
    result=bytearray(b'%PDF-1.4\n');offsets=[0]
    for n,obj in enumerate(objects,1):
        offsets.append(len(result))
        result.extend(f'{n} 0 obj\n'.encode()+obj+b'\nendobj\n')
    start=len(result)
    result.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:result.extend(f'{offset:010d} 00000 n \n'.encode())
    result.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF'.encode())
    return bytes(result)

from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m

class M22Tests(unittest.TestCase):
    def test_exact_page_anchor_accepted(self):
        pdf={'status':'PDF_PAGES_RETRIEVED','selected_pages':[{'page':4,'text':'The measured rate is k = 0.2 s^-1 at 300 K. Valid only in regime A.'}]}
        a={'anchors':[{'page':4,'exact_source_span':'k = 0.2 s^-1 at 300 K',
                       'physical_relation_or_parameter':'Measured rate at temperature',
                       'source_conditions':'Only regime A','uncertainties':'Limited evidence'}],
           'calculation':{'expression':'k * t','variables':{'k':{'value':0.2,'unit':'s^-1','origin':'SOURCE_REPORTED','supporting_anchor_index':0,'justification':'Quoted from page four.'},'t':{'value':5,'unit':'s','origin':'EXPLICIT_ASSUMPTION','supporting_anchor_index':None,'justification':'A hypothetical five-second period.'}},'result_unit':'dimensionless','modeled_mapping_to_source':'Assume a constant rate in this regime.', 'interpretation_and_limits':'A rate-times-time quantity not experimental validation.'}}
        m._validate_anchors(a,pdf)
        self.assertAlmostEqual(m._execute(a)['result'],1)
    def test_fabricated_anchor_rejected(self):
        a={'anchors':[{'page':4,'exact_source_span':'A fabricated law k = 999 s^-1'}],'calculation':None}
        pdf={'status':'PDF_PAGES_RETRIEVED','selected_pages':[{'page':4,'text':'The measured rate is k = 0.2 s^-1 at 300 K.'}]}
        with self.assertRaisesRegex(ValueError,'NOT_VERBATIM'):
            m._validate_anchors(a,pdf)
    def test_source_numbers_need_numeric_anchor(self):
        a={'anchors':[{'page':4,'exact_source_span':'The result increased with temperature.'}],
           'calculation':{'expression':'k*2','variables':{'k':{'value':0.2,'unit':'s^-1','origin':'SOURCE_REPORTED','supporting_anchor_index':0}}}}
        pdf={'status':'PDF_PAGES_RETRIEVED','selected_pages':[{'page':4,'text':'The result increased with temperature.'}]}
        with self.assertRaisesRegex(ValueError,'REPORTED_NUMERIC_VALUE_NOT_PRESENT'):
            m._validate_anchors(a,pdf)
    def test_unsafe_expr_blocked(self):
        for s in ['__import__("os").system("id")','(1).__class__','x[0]','x if x else 1',
                  '[x for x in [1]]','lambda: 1','2**99']:
            with self.subTest(s=s),self.assertRaises(ValueError):m._numeric_expr(s,{'x':1.})
    def test_numeric_math(self):
        self.assertAlmostEqual(m._numeric_expr('exp(-Ea/(kB*T))',{'Ea':0.1,'kB':8.617333e-5,'T':300}),
                               __import__('math').exp(-0.1/(8.617333e-5*300)))
    def test_model_rejects_unattributed_numeric_claim(self):
        v={'value':0.2,'unit':'s^-1','origin':'SOURCE_REPORTED','justification':'a reported measurement'}
        with self.assertRaises(ValueError):m.Variable.model_validate(v)
    def test_pdf_page_selection_not_head_only(self):
        blob=sample_pdf(['Equation rate constant appears here' if i==4 else 'Introduction unrelated words' for i in range(5)])
        out=m._page_extract(blob,['rate constant'])
        self.assertEqual(out['selected_pages'][0]['page'],5)
        self.assertEqual(out['page_count'],5)

if __name__=='__main__':unittest.main()

class OfflineFullRoute(unittest.TestCase):
    def test_existing_paper_id_to_grounded_calculation(self):
        import tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        bypass=patch.object(m,'root',side_effect=lambda p:Path(p).resolve())
        bypass.start();self.addCleanup(bypass.stop)

        with tempfile.TemporaryDirectory() as td:
            home=Path(td);prior=home/'m21';work=home/'research';out=home/'m22'
            (prior/'select').mkdir(parents=True);(work/'step_0001').mkdir(parents=True)
            def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False))
            save(prior/'lock.json',{'model':'fake','temperature':0.3,'max_tokens':6000})
            save(prior/'select'/'normalized.json',{'selection_rationale':'A scientific problem with competing explanations.'})
            save(prior/'literature.json',{'directions':[{'papers':[{'paper_id':'p1','title':'A synthetic source','abstract':'A rate law','source_depth':'abstract','doi':'10.0000/example'}]}]})
            save(work/'scientific_context.json',{'autonomous_opening':{'selected_research_question':'What is the relevant rate for this scientific mechanism?'}})
            save(work/'step_0001'/'normalized.json',{'operation':'MODEL'})
            dummy={'normalized':{'operation':'MODEL','scientific_delta':'A proposed model needs grounded rate coefficients.',
                'scientific_work_content':{'candidate_consequence':'An untested relation'},'next_research_question':'Does reported rate matter?'},
                'tool_result':None}
            with patch.object(m.m21,'root',side_effect=lambda p:Path(p).resolve()),\
                 patch.object(m,'_m21_work_state',return_value=({'steps':1},[dummy])):
                status=m.prepare(prior,work,out)
            self.assertEqual(status['papers_available'],1)
            self.assertEqual(m.verify(out)['plan'],'NOT_STARTED')
            plan={'research_question':'What rate relates to this synthetic physical phenomenon?',
                  'missing_physical_constraint':'A quantitative rate under stated temperature and source conditions.',
                  'why_it_changes_science':'Its magnitude determines whether this mechanism could act fast enough.',
                  'ranked_paper_ids':['p1'],'pdf_search_terms':['rate','temperature'],
                  'source_limitations':'This synthetic reference is insufficient for actual SERS mechanism claims.'}
            p=out/'plan';p.mkdir();req=m._phase_req(out,'plan')
            save(p/'request.json',req);save(p/'attempt.json',{'request_sha256':m.sha(req)})
            save(p/'raw_response.json',{'raw_text':json.dumps(plan)})
            self.assertEqual(m.recover(out,'plan')['phase'],'plan')
            local=home/'example.pdf';local.write_bytes(sample_pdf(['k = 0.2 per second at 300 K, measured in synthetic conditions.']))
            self.assertEqual(m.fetch(out,local_pdf=local,paper_id='p1')['status'],'PDF_PAGES_RETRIEVED')
            scan=m.read(out/'pdf_evidence.json');self.assertEqual(scan['acquisition'],'USER_SUPPLIED_LOCAL_PDF_IDENTITY_UNVERIFIED')
            text=scan['selected_pages'][0]['text'];self.assertIn('k = 0.2 per second',text)
            result={'attempted_question':'How rapidly does the synthetic physical event occur?',
                'anchors':[{'page':1,'exact_source_span':'k = 0.2 per second at 300 K',
                            'physical_relation_or_parameter':'A synthetic rate coefficient.',
                            'source_conditions':'Reported under synthetic conditions only.',
                            'uncertainties':'No experimental uncertainty supplied.'}],
                'scientific_relation':'A synthetic rate of 0.2 per second, not a SERS constraint.',
                'conditions_and_domain':'Valid only in this synthetic demonstration at 300 K.',
                'what_is_not_supported':'No assertion about gold nanorods or physical mechanism is justified.',
                'calculation':{'expression':'k * dt','variables':{
                  'k':{'value':0.2,'unit':'1/s','origin':'SOURCE_REPORTED','supporting_anchor_index':0,'justification':'Literal numeric claim in source text.'},
                  'dt':{'value':5,'unit':'s','origin':'EXPLICIT_ASSUMPTION','supporting_anchor_index':None,'justification':'A hypothetical five-second window.'}},
                  'result_unit':'dimensionless','modeled_mapping_to_source':'Assume a homogeneous time-independent rate for five seconds.',
                  'interpretation_and_limits':'Arithmetic only; physical applicability and units need independent review.'},
                'research_next_action':'CONTINUE','reason_for_action':'The computed conditional number alone does not identify the causal mechanism.',
                'epistemic_status':'SOURCE_ANCHORED_INTERPRETATION_NOT_VALIDATED'}
            p=out/'analyze';p.mkdir();req=m._phase_req(out,'analyze')
            save(p/'request.json',req);save(p/'attempt.json',{'request_sha256':m.sha(req)})
            save(p/'raw_response.json',{'raw_text':json.dumps(result)})
            self.assertEqual(m.recover(out,'analyze')['state']['analysis'],'READY')
            self.assertEqual(m.report(out)['calculation_result']['result'],1.)
            self.assertFalse(m.report(out)['scientific_validation'])
