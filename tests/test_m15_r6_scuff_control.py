import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from types import SimpleNamespace

from scripts.discovery import run_m15_r6_scuff_control as r6


class R6UnitTests(TestCase):
    def test_mie_is_positive(self):
        value=r6.mie_sphere_scattering()
        self.assertGreater(value,0)
        self.assertLess(value, 0.02)

    def test_mie_reproducible(self):
        self.assertEqual(r6.mie_sphere_scattering(),r6.mie_sphere_scattering())

    def test_mie_vanishes_low_contrast(self):
        self.assertLess(r6.mie_sphere_scattering(epsilon=1.0),1.e-25)

    def test_more_contrast_larger_scattering(self):
        self.assertGreater(r6.mie_sphere_scattering(epsilon=3.0),r6.mie_sphere_scattering(epsilon=2.25))

    def test_missing_scuff_closed(self):
        with self.assertRaises(ValueError):
            r6.executable('/path/that/does/not/exist/scuff-scatter')

    def test_preexisting_output_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            with mock.patch.object(r6,'validate_r5',return_value={'nanorod_science_gate':'BLOCKED'}),mock.patch.object(r6,'executable',return_value='/bin/true'):
                with self.assertRaises(FileExistsError):
                    r6.prepare(p,p,'gmsh',60)

    def test_prepare_requires_runtime(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            with mock.patch.object(r6,'validate_r5',return_value={'nanorod_science_gate':'BLOCKED'}):
                with mock.patch.object(r6,'executable',side_effect=ValueError('EXECUTABLE_NOT_FOUND')):
                    with self.assertRaisesRegex(ValueError,'EXECUTABLE_NOT_FOUND'):
                        r6.prepare(p,p/'new','bad',60)

    def test_run_requires_real_scuff(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            with mock.patch.object(r6,'verify',return_value={}),mock.patch.object(r6,'executable',side_effect=ValueError('EXECUTABLE_NOT_FOUND')):
                with self.assertRaisesRegex(ValueError,'EXECUTABLE_NOT_FOUND'):
                    r6.run_scuff(p,'bad',60)

    def test_run_no_duplicate(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'sphere.PFT').write_text('existing data')
            with mock.patch.object(r6,'verify',return_value={}),mock.patch.object(r6,'executable',return_value='/bin/true'):
                with self.assertRaises(FileExistsError):
                    r6.run_scuff(p,'true',60)

    def test_run_writes_receipt_without_science_gate(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'M15R6_CONTROL_PLAN.json').write_text('{}')
            def fake_run(*args,**kwargs):
                (p/'sphere.PFT').write_text('# omega PAbs PScat\n1.0 0 0.001\n')
                return SimpleNamespace(returncode=0,stdout='ok',stderr='')
            with mock.patch.object(r6,'verify',return_value={}),mock.patch.object(r6,'executable',return_value='/bin/true'):
                with mock.patch.object(r6.subprocess,'run',side_effect=fake_run):
                    receipt=r6.run_scuff(p,'true',60)
            self.assertTrue(receipt['pft_present'])
            data=json.loads((p/'M15R6_SCUFF_RUN.json').read_text())
            self.assertFalse(data['pft_to_cross_section_calibrated'])
            self.assertFalse(data['science_backend_qualified'])

    def test_timeout_records_failure(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'M15R6_CONTROL_PLAN.json').write_text('{}')
            import subprocess
            with mock.patch.object(r6,'verify',return_value={}),mock.patch.object(r6,'executable',return_value='/bin/true'):
                with mock.patch.object(r6.subprocess,'run',side_effect=subprocess.TimeoutExpired(cmd='scuff',timeout=60)):
                    receipt=r6.run_scuff(p,'true',60)
            self.assertEqual(receipt['status'],'SCUFF_EXECUTION_FAILED_REVIEW_LOGS')
            self.assertFalse(receipt['pft_present'])

    def test_timeout_limits(self):
        with self.assertRaises(ValueError):r6.run_scuff(Path('.'),'bad',0)
        with self.assertRaises(ValueError):r6.prepare(Path('.'),Path('.'),'bad',0)


if __name__=='__main__':
    import unittest
    unittest.main()
