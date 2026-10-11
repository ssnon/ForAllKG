import copy
import math
import unittest
from scripts.discovery.run_m15_r6b_pft_calibration import (parse_pft,compare,Z0_OHM)

HEADER='''# scuff-scatter test run
# data file columns:
# 1 omega
# 2 surface label
# 3 absorbed power (watts)
# 4 scattered power (watts)
# 5 x-force (nanoNewtons)
# 6 y-force (nanoNewtons)
# 7 z-force (nanoNewtons)
# 8 x-torque (nanoNewtons*microns)
# 9 y-torque (nanoNewtons*microns)
# 10 z-torque (nanoNewtons*microns)
'''
OMEGA=2*math.pi/0.633
ROW=f'{OMEGA:.14g} TestSphere 0 1.47393310828701e-7 0 0 0 0 0 0\n'

class M15R6BTests(unittest.TestCase):
    def test_good_pft(self):
        p=parse_pft(HEADER+ROW,expected_omega=OMEGA)
        self.assertEqual(p['surface_label'],'TestSphere')
        self.assertAlmostEqual(p['scattered_power_w'],1.47393310828701e-7)
    def test_expected_cross_section(self):
        r=compare(parse_pft(HEADER+ROW,expected_omega=OMEGA),0.0001110550564421231)
        self.assertAlmostEqual(r['relative_error'],0,places=10)
        self.assertAlmostEqual(r['incident_intensity_w_per_um2'],1/(2*Z0_OHM))
    def test_no_header_rejected(self):
        with self.assertRaisesRegex(ValueError,'HEADER'):
            parse_pft(ROW)
    def test_wrong_header_rejected(self):
        with self.assertRaisesRegex(ValueError,'HEADER'):
            parse_pft(HEADER.replace('4 scattered power','4 scattered energy')+ROW)
    def test_wrong_label_rejected(self):
        with self.assertRaisesRegex(ValueError,'LABEL'):
            parse_pft(HEADER+ROW.replace('TestSphere','OtherSphere'))
    def test_wrong_frequency_rejected(self):
        with self.assertRaisesRegex(ValueError,'FREQUENCY'):
            parse_pft(HEADER+ROW,expected_omega=4)
    def test_two_rows_rejected(self):
        with self.assertRaisesRegex(ValueError,'ONE_DATA_ROW'):
            parse_pft(HEADER+ROW+ROW)
    def test_negative_scattered_rejected(self):
        with self.assertRaisesRegex(ValueError,'NONPOSITIVE'):
            parse_pft(HEADER+ROW.replace('1.47393310828701e-7','-1.0'))
    def test_truncated_rejected(self):
        with self.assertRaisesRegex(ValueError,'TEN_COLUMNS'):
            parse_pft(HEADER+ROW.rsplit(' ',1)[0]+'\n')
    def test_bad_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'MIE_REFERENCE'):
            compare(parse_pft(HEADER+ROW),0)
    def test_lossless_abs_flag(self):
        r=compare(parse_pft(HEADER+ROW.replace(' 0 1.4739',' 0.001 1.4739')),0.00011105)
        self.assertFalse(r['lossless_sphere_absorption_small_10pct_of_scatter'])
    def test_mismatch_flag(self):
        r=compare(parse_pft(HEADER+ROW),0.02)
        self.assertFalse(r['single_mesh_reference_within_10pct'])

if __name__=='__main__':unittest.main()

class M15R6BIntegrationTests(unittest.TestCase):
    def test_audit_and_verify_synthetic_frozen_output(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.discovery import run_m15_r6b_pft_calibration as tool
        with tempfile.TemporaryDirectory() as temp:
            r6=Path(temp)/'r6';r6.mkdir()
            pft=(HEADER+ROW)
            (r6/'sphere.PFT').write_text(pft)
            pft_sha=hashlib.sha256(pft.encode()).hexdigest()
            plan={'sphere_parameters':{'radius_um':.05,'lambda_vacuum_um':.633,
                  'epsilon_relative':2.25,'mesh_edge_um':.012},
                  'sphere_mesh_audit':{'faces':300},
                  'sphere_analytic_mie_scattering_um2':0.0001110550564421231}
            (r6/'M15R6_CONTROL_PLAN.json').write_text(json.dumps(plan))
            receipt={'exit_code':0,'timed_out':False,'nanorod_science_gate':'BLOCKED',
                     'science_backend_qualified':False,'pft_sha256':pft_sha,
                     'command':['scuff-scatter','--pwPolarization','1','0','0',
                                '--pwDirection','0','0','1']}
            (r6/'M15R6_SCUFF_RUN.json').write_text(json.dumps(receipt))
            out=Path(temp)/'audit'
            with patch('scripts.discovery.run_m15_r6_scuff_control.check',return_value={
                    'scuff_run':'SCUFF_PFT_PRODUCED_UNCALIBRATED'}):
                result=tool.audit(r6,out)
                self.assertEqual(result['status'],'SINGLE_MESH_COMPARISON_RECORDED')
                self.assertTrue((out/'M15R6B_SINGLE_MESH_REPORT.md').exists())
                self.assertEqual(tool.check(out)['status'],'M15R6B_CHECK_OK')
                with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
                    tool.audit(r6,out)
                (r6/'sphere.PFT').write_text(pft+'# modified\n')
                with self.assertRaisesRegex(ValueError,'SOURCE_CHANGED'):
                    tool.check(out)
