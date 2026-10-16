"""No-cost raw-preserving M19-E1 citation recovery tests."""
from __future__ import annotations
import copy,json,unittest
from pathlib import Path
from test_m19_e1 import M19E1Tests, mock_output
from pipeline_core.discovery.scientific_research_control import m19_e1_reference_recovery as r
from pipeline_core.discovery.scientific_research_control import m19_deep_work_e1 as m
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b

class M19ReferenceRecoveryTests(unittest.TestCase):
    home=(Path(m.__file__).resolve().parents[3]/'experiments'/'m19_e1')
    @classmethod
    def setUpClass(cls):cls.home.mkdir(parents=True,exist_ok=True)
    def setUp(self):M19E1Tests.setUp(self)

    def _invalid_b1(self,mentions=False):
        raw=mock_output('B1')
        raw['source_or_prior_work_refs']=['WORK:01','work_0003_cd32f5d823b4','work_0004_5ce1a98db391']
        if mentions:raw['research_delta']+=' based on work_0004_5ce1a98db391 prior results.'
        with self.assertRaisesRegex(RuntimeError,'RAW_SAVED_OFFLINE_RECOVER_REQUIRED'):
            m.next_step(self.out,'B',backend=lambda req:(json.dumps(raw),{'prompt_tokens':1},'mock-b1'))
        self.assertEqual(r.verify(self.out)['incomplete']['B']['status'],'SAVED_RAW_NEEDS_RECOVER_NO_CALL')
        return raw

    def test_invalid_citations_quarantined_and_recovered_without_raw_changes(self):
        self._invalid_b1()
        sd=m._step_dir(self.out,'B',1)
        orig_digest=b.file_sha(sd/'raw_response.json')
        audit=r.audit(self.out,'B',1)
        self.assertEqual(audit['invalid_refs'],['work_0003_cd32f5d823b4','work_0004_5ce1a98db391'])
        with self.assertRaises(PermissionError):r.quarantine(self.out,'B',1)
        status=r.quarantine(self.out,'B',1,accept_invalid_refs=True)
        self.assertTrue(status['saved_raw_unchanged'])
        self.assertEqual(r.recover(self.out)['status'],'RECOVERED_SAVED_RAW_NO_CALL')
        self.assertEqual(b.file_sha(sd/'raw_response.json'),orig_digest)
        norm=b.read(sd/'normalized.json')
        self.assertEqual(norm['source_or_prior_work_refs'],['WORK:01'])
        self.assertEqual(norm['research_delta'],mock_output('B1')['research_delta'])
        self.assertEqual(r.verify(self.out)['b_count'],1)
        self.assertEqual(r.preview(self.out,'C')['request'],r.preview(self.out,'B')['request'])
        self.assertEqual(len(r.inspect(self.out)['reference_quarantine']),1)

    def test_invalid_prose_not_silently_fixed_and_visible(self):
        self._invalid_b1(mentions=True)
        audit=r.audit(self.out,'B',1)
        self.assertEqual(audit['invalid_refs_mentioned_in_science_prose']['work_0004_5ce1a98db391'],1)
        r.quarantine(self.out,'B',1,accept_invalid_refs=True)
        r.recover(self.out)
        text=b.read(m._step_dir(self.out,'B',1)/'normalized.json')['research_delta']
        self.assertIn('work_0004_5ce1a98db391',text)
        self.assertEqual(r.inspect(self.out)['reference_quarantine'][0]['science_prose_invalid_ref_mentions']['work_0004_5ce1a98db391'],1)

    def test_sidecar_tampering_detected(self):
        self._invalid_b1()
        r.quarantine(self.out,'B',1,accept_invalid_refs=True)
        r.recover(self.out)
        qp=m._step_dir(self.out,'B',1)/r.SIDECAR
        content=b.read(qp);content['invalid_refs']=[]
        qp.write_text(json.dumps(content),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'QUARANTINE_CONTENT_OR_SOURCE_CHANGED'):
            r.verify(self.out)

    def test_missing_sidecar_cannot_be_bypassed(self):
        self._invalid_b1()
        with self.assertRaisesRegex(ValueError,'UNKNOWN_SOURCE_REF_UNQUARANTINED'):
            r.recover(self.out)
        with self.assertRaisesRegex(RuntimeError,'INCOMPLETE_OR_FINALIZED_STEPS_REQUIRE_REVIEW'):
            r.preview(self.out,'C')

if __name__=='__main__': unittest.main()
