import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery import sers_successor_challenge_m14c3 as sc
from scripts.discovery import run_sers_successor_challenge_m14c3 as cli
from pipeline_core.discovery.sers_evidence_successor_m14c2 import canon
from scripts.discovery.run_sers_evidence_successor_m14c2 import _validate_submission

class NumericalScience(unittest.TestCase):
    def test_exact_observational_counterexample(self):
        rows,sweep,summary=sc.calculate()
        self.assertEqual(len(rows),20)
        self.assertEqual(summary['flexible_one_population_max_abs_residual'],0)
        self.assertFalse(summary['flexible_one_population_null_excluded'])
    def test_negative_control_not_falsely_excluded_at_five_percent(self):
        _,sw,summary=sc.calculate()
        self.assertTrue(summary['synthetic_negative_control_passes_0_05_bound'])
        hit=[v for v in sw if v['dataset']=='synthetic_one_population_negative_control' and v['independently_calibrated_max_response_deviation_required']==.05]
        self.assertEqual(hit[0]['violations_if_that_bound_were_externally_verified'],0)
    def test_two_population_conditional_falsification_under_strict_bound(self):
        _,sweep,_=sc.calculate()
        x=[v for v in sweep if v['dataset']=='synthetic_two_population' and v['independently_calibrated_max_response_deviation_required']==.05][0]
        self.assertGreater(x['violations_if_that_bound_were_externally_verified'],0)
    def test_data_are_synthetic_not_certified(self):
        _,_,summary=sc.calculate()
        self.assertEqual(summary['measured_data_rows'],0)
        self.assertFalse(summary['literature_novelty_certified'])
        self.assertFalse(summary['physical_mechanism_confirmed'])
    def test_all_normalized_coverage_positive(self):
        for v,b in sc.grid():self.assertGreater(sc.coverage(v,b),0)
    def test_calibration_bounds_sweep(self):
        _,s,_=sc.calculate()
        self.assertEqual(len(s),2*len(sc.BOUNDS))
        for dataset in set(x['dataset'] for x in s):
            vals=[x for x in s if x['dataset']==dataset]
            self.assertEqual([x['violations_if_that_bound_were_externally_verified'] for x in vals],
                sorted((x['violations_if_that_bound_were_externally_verified'] for x in vals),reverse=True))
    def test_csv_repeatability(self):
        r,_,_=sc.calculate()
        self.assertEqual(sc.rows_csv(r),sc.rows_csv(sc.calculate()[0]))
        self.assertEqual(len(list(csv.DictReader(io.StringIO(sc.rows_csv(r).decode())))),20)
    def test_canon_cannot_hash_nan(self):
        with self.assertRaises(ValueError):sc.canonical({'a':float('nan')})

class Integration(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory();self.addCleanup(self.d.cleanup)
        self.c2=Path(self.d.name)/'C2';self.c3=Path(self.d.name)/'C3'
        self.c2.mkdir()
        self.scientific_source={'problem_id':'m14gap:feceb611df04bd85','candidate_scientific_null':'Single population null',
            'candidate_prediction':'Measured across Raman/IR','source_files_sha256':{'test':'v1'},
            'original_source_receipt_verified':True}
        self.mock=patch.object(cli,'_validate_c2',return_value=self.scientific_source)
        self.mock.start();self.addCleanup(self.mock.stop)
        self.p=type('Args',(),{'out_dir':self.c3,'c2_dir':self.c2})()
    def _start(self):cli.cmd_prepare(self.p);cli.cmd_execute(self.p)
    def test_end_to_end(self):
        cli.cmd_prepare(self.p)
        a=cli.cmd_execute(self.p)
        self.assertEqual(a['signal_rows'],20)
        b=cli.cmd_verify(self.p)
        self.assertEqual(b['successor_action'],'SUSPEND')
        self.assertEqual(b['new_model_calls'],0)
        self.assertTrue((self.c3/cli.POLICY).exists())
    def test_idempotent_verification(self):
        self._start();x=cli.cmd_verify(self.p);sha1=sc.sha((self.c3/cli.POLICY).read_bytes())
        y=cli.cmd_verify(self.p);sha2=sc.sha((self.c3/cli.POLICY).read_bytes())
        self.assertEqual(x,y);self.assertEqual(sha1,sha2)
    def test_no_overwrite_on_prepare(self):
        cli.cmd_prepare(self.p)
        with self.assertRaises(FileExistsError):cli.cmd_prepare(self.p)
    def test_no_overwrite_execution(self):
        self._start()
        with self.assertRaises(FileExistsError):cli.cmd_execute(self.p)
    def test_observable_tamper_fails(self):
        self._start();(self.c3/cli.OBS).write_bytes(b'junk')
        with self.assertRaises(ValueError):cli.cmd_verify(self.p)
        self.assertFalse((self.c3/cli.POLICY).exists())
    def test_receipt_tamper_fails(self):
        self._start();r=cli._read(self.c3/cli.RECEIPT);r['outputs_sha256'][cli.OBS]='x'
        (self.c3/cli.RECEIPT).write_bytes(sc.canonical(r))
        with self.assertRaises(ValueError):cli.cmd_verify(self.p)
    def test_pre_registration_modified_fails(self):
        cli.cmd_prepare(self.p);d=cli._read(self.c3/cli.PLAN);d['challenge_code_sha256']='bad'
        (self.c3/cli.PLAN).write_bytes(sc.canonical(d))
        with self.assertRaises(ValueError):cli.cmd_execute(self.p)
    def test_source_changed_fails(self):
        cli.cmd_prepare(self.p)
        source=dict(self.scientific_source);source['source_files_sha256']={'test':'changed'}
        with patch.object(cli,'_validate_c2',return_value=source):
            with self.assertRaises(ValueError):cli.cmd_execute(self.p)
    def test_no_missing_receipt_verification(self):
        cli.cmd_prepare(self.p)
        with self.assertRaises(FileNotFoundError):cli.cmd_verify(self.p)
    def test_no_identification_claim_in_policy(self):
        self._start();cli.cmd_verify(self.p);p=cli._read(self.c3/cli.POLICY)
        self.assertTrue(p['does_not_falsify_two_population_mechanism'])
        self.assertTrue(p['not_a_general_SERS_identifiability_theorem'])
        self.assertEqual(p['external_empirical_observations'],0)

class ProvenanceGate(unittest.TestCase):
    def test_invalid_original_receipt_digest_rejected(self):
        # Without an original C2 paid receipt, the high-trust source gate fails.
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):cli._validate_c2(Path(d))
    def test_c2_validation_logic_is_not_bypassed_by_own_math(self):
        # All challenge math can be executed alone, but no sourced policy may
        # be created without a frozen C2 response and exact audit chain.
        rows,_,_=sc.calculate();self.assertEqual(len(rows),20)
        with tempfile.TemporaryDirectory() as d:
            args=type('Args',(),{'out_dir':Path(d)/'C3','c2_dir':Path(d)/'C2'})()
            with self.assertRaises((ValueError,FileNotFoundError)):
                cli.cmd_prepare(args)
            self.assertFalse(args.out_dir.exists())

if __name__=='__main__':unittest.main()
