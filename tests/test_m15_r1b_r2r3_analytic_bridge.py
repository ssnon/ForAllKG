import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.discovery import run_m15_r1b_r2r3_analytic_bridge as m


class TestAnalyticBridge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root/'out'
        self.decision = self.root/'M15R_BACKEND_DECISION.json'
        self.decision.write_text(json.dumps({
            'schema_version':'m15r-backend-decision-v1',
            'science_backend_choice':'NO_QUALIFIED_BACKEND',
            'science_x0_x1_status':'BLOCKED',
            'scientific_interpretation_permitted':False,
            'physics_claim_authority':False,
        }))

    def _prepare(self):
        m.prepare(self.out,self.decision,1502)

    def _manual_g1(self):
        a={
            'prediction_um2':0.0186,
            'prediction_interval_um2':[0.012,0.027],
            'preferred_model':'rayleigh',
            'conditional_claim':'Rayleigh is an approximate description when the sphere is much smaller than the wavelength.',
            'falsification':'A full-wave Mie value outside the stated error interval challenges this approximation.',
            'reasoning':'Without numerical reference data, I use the dipole scaling as a provisional bound.',
        }
        b={**a,'prediction_um2':0.023,'preferred_model':'full_mie',
           'conditional_claim':'A higher-order full wave term can modify the scattering cross section at this sphere size.',
           'reasoning':'The x0 exact result differs from a dipole model, so I broaden the prediction at x1.'}
        p1,p2=self.root/'a.json',self.root/'b.json'
        p1.write_text(json.dumps(a));p2.write_text(json.dumps(b))
        return p1,p2

    def _manual_g2(self):
        a={
            'prediction_x2_um2':0.0113,
            'prediction_interval_um2':[0.009,0.015],
            'preferred_model':'rayleigh',
            'next_test_wavelength_nm':760.0,
            'decision':'REFINE',
            'conditional_claim':'A longer wavelength should lower the scattering cross section for this dielectric sphere.',
            'falsification':'A high-quality exact result outside the predeclared interval would falsify this extrapolation.',
            'reason_for_change':'I update the previous prediction after seeing the held-out numerical result.',
        }
        b={**a,'prediction_x2_um2':0.0140,'preferred_model':'full_mie',
           'conditional_claim':'Higher-order Mie corrections can persist in the new held-out wavelength.',
           'reason_for_change':'The observed difference between Mie and Rayleigh supports the full-wave alternative.'}
        p1,p2=self.root/'ga.json',self.root/'gb.json'
        p1.write_text(json.dumps(a));p2.write_text(json.dumps(b))
        return p1,p2

    def test_exact_mie_nonnegative_and_rayleigh_differs(self):
        result=m.mie_sphere_cross_section_um2(475)
        self.assertTrue(0.09 < result < 0.12)
        approx=m.rayleigh_cross_section_um2(475)
        self.assertGreater(abs(result-approx)/result,0.3)
        self.assertGreater(result,0)

    def test_equal_indices_zero_scattering(self):
        self.assertLess(m.mie_sphere_cross_section_um2(600,n_particle=1,n_medium=1),1e-25)

    def test_rayleigh_agrees_small_sphere(self):
        exact=m.mie_sphere_cross_section_um2(1000,radius_nm=8,n_particle=1.5)
        approx=m.rayleigh_cross_section_um2(1000,radius_nm=8,n_particle=1.5)
        self.assertLess(abs(exact-approx)/exact,0.03)

    def test_reject_unqualified_decision_changed(self):
        self._prepare()
        d=json.loads(self.decision.read_text())
        d['scientific_interpretation_permitted']=True
        self.decision.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'GATE_NOT_BLOCKED|DECISION_TAMPERED|AUTHORITY'):
            m.preflight(self.out)

    def test_prepare_and_preflight_do_not_reveal_x1(self):
        self._prepare()
        self.assertFalse((self.out/'M15R_X1_FEEDBACK.json').exists())
        self.assertFalse((self.out/'M15R_X2_FEEDBACK.json').exists())
        plan,spec=m.preflight(self.out)
        self.assertEqual(plan['quantitative_nanorod_evidence'],'BLOCKED')
        control=spec['prompts']['A']['user']
        treatment=spec['prompts']['B']['user']
        self.assertNotIn('0.105836',control)
        self.assertIn('0.105836',treatment)
        self.assertNotIn('0.024527592',str((self.out/'M15R_G1_PROMPTS.json').read_text()))

    def test_two_generation_complete_and_scores(self):
        self._prepare()
        ga,gb=self._manual_g1()
        self.assertEqual(m.freeze_g1(self.out,ga,gb)['status'],'G1_FROZEN')
        self.assertFalse((self.out/'M15R_X1_FEEDBACK.json').exists())
        result=m.reveal_x1(self.out)
        self.assertEqual(result['status'],'X1_REVEALED_G2_PROMPTS_READY')
        self.assertGreater(result['g1_relative_error_by_arm']['A'],result['g1_relative_error_by_arm']['B'])
        self.assertFalse((self.out/'M15R_X2_FEEDBACK.json').exists())
        g2a,g2b=self._manual_g2()
        self.assertEqual(m.freeze_g2(self.out,g2a,g2b)['status'],'G2_FROZEN')
        end=m.reveal_x2(self.out)
        self.assertEqual(end['status'],'BENCHMARK_COMPUTATIONAL_FEEDBACK_WORKFLOW_FUNCTIONAL')
        self.assertFalse(end['scientific_novelty_certified'])
        self.assertEqual(end['nanorod_science_gate'],'BLOCKED')
        self.assertTrue((self.out/'M15R_BENCHMARK_REPORT.json').exists())
        report=m.load(self.out/'M15R_BENCHMARK_REPORT.json')
        self.assertEqual(report['g2_x2_errors']['B']['decision'],'REFINE')
        self.assertFalse(report['canonical_graph_mutated'])

    def test_cannot_reveal_without_frozen_predictions(self):
        self._prepare()
        with self.assertRaises(FileNotFoundError):
            m.reveal_x1(self.out)
        a,b=self._manual_g1()
        m.freeze_g1(self.out,a,b)
        with self.assertRaises(FileNotFoundError):
            m.reveal_x2(self.out)

    def test_reject_g1_invalid_number_and_duplicate_json(self):
        self._prepare()
        a,b=self._manual_g1()
        payload=json.loads(a.read_text())
        payload['prediction_um2']=-1
        a.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'INVALID_POSITIVE_NUMBER'):
            m.freeze_g1(self.out,a,b)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):
            m._strict_dict('{"a":1,"a":2}')

    def test_no_freeze_g1_after_x1_reveal(self):
        self._prepare()
        a,b=self._manual_g1()
        m.freeze_g1(self.out,a,b)
        m.reveal_x1(self.out)
        with self.assertRaisesRegex(ValueError,'X1_ALREADY_REVEALED'):
            m.freeze_g1(self.out,a,b)

    def test_hashed_source_cannot_be_changed_after_g1(self):
        self._prepare()
        a,b=self._manual_g1()
        m.freeze_g1(self.out,a,b)
        obj=json.loads(a.read_text());obj['prediction_um2']=0.03
        a.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'G1_SOURCE_TAMPERED'):
            m.reveal_x1(self.out)

    def test_paid_explicit_authorization(self):
        self._prepare()
        with self.assertRaisesRegex(ValueError,'EXPLICIT_PAID_AUTHORIZATION_REQUIRED'):
            m.paid_generate(self.out,'g1','dummy',1000,2,False,False,'OPENROUTER_API_KEY','https://openrouter.ai/api/v1')

    def test_invalid_g2_wavelength(self):
        a,b=self._manual_g2()
        obj=json.loads(a.read_text())
        obj['next_test_wavelength_nm']=m.X1
        with self.assertRaisesRegex(ValueError,'G2_NEXT_WAVELENGTH_INVALID'):
            m.validate_g2(obj)

    def test_local_probe_never_qualifies(self):
        result=m.probe_local()
        self.assertFalse(result['matched_Au_Ag_nanorod_backend_qualified'])
        self.assertIn('scipy',result['python_modules'])

    def test_preflight_detects_tampered_prompt(self):
        self._prepare()
        p=self.out/'M15R_G1_PROMPTS.json'
        d=m.load(p)
        d['prompts']['B']['user']='forged'
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'G1_PROMPTS_TAMPERED'):
            m.preflight(self.out)


if __name__=='__main__':unittest.main()
