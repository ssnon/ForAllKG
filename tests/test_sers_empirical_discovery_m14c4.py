import csv, json, hashlib, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as e
from scripts.discovery import run_sers_empirical_discovery_m14c4 as runner


def fixture(path, labs=4, concentration_levels=4, replicates=3, quadratic=False):
 headers=['labcode','substrate','laser','method','sample','type','conc','batch','replica','X715','730','750','1100']
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(headers)
  for lab in range(labs):
   for k in range(concentration_levels):
    for r in range(replicates):
     x=-7+k;y=10+3*x+(2*x*x if quadratic else 0)+lab*.01+r*.01
     w.writerow([f'L{lab}','Au','785','methodA','adenine','calibration',str(10.**x),f'b{lab}',str(r),
                 y,y+1,y+2,'NA'])

class EmpiricalScienceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.base=Path(self.tmp.name);self.raw=self.base/'source.csv';fixture(self.raw)
  self.entry=e.DATASETS['ILS_ADENINE_2020'].copy()
  self.entry['md5']=e.digest_file(self.raw,'md5')
  self.patcher=patch.dict(e.DATASETS,{'ILS_ADENINE_2020':self.entry})
  self.patcher.start();self.addCleanup(self.patcher.stop)
  self.out=self.base/'study'
 def test_parse_real_band_metadata_and_strict_concentration(self):
  x=e.parse_csv(self.raw)
  self.assertEqual(len(x['rows']),48)
  self.assertEqual(x['spectral_columns'],3)
  self.assertAlmostEqual(x['rows'][0]['y'],-347.5) # trapezoidal band-integrated raw units
  self.assertEqual(e.concentration('1e-6 M'),None)
  self.assertEqual(e.concentration('0'),None)
  self.assertEqual(e.concentration('1e-6'),1e-6)
 def test_screen_selection_uses_metadata_and_labs(self):
  z=e.screening(e.parse_csv(self.raw)['rows'])
  self.assertEqual(len(z['selected_cohort_keys']),1)
  self.assertEqual(z['candidate_cohorts'][0]['n_labs'],4)
 def test_group_folds_never_contain_train_lab(self):
  pts=e.parse_csv(self.raw)
  z=e.screening(pts['rows'])
  folds,skips=e.compare(pts,z)
  self.assertEqual(len(folds),12)
  self.assertFalse(skips)
  self.assertTrue(all(r['train_n']==36 and r['test_n']==12 for r in folds))
 def test_linear_over_no_effect(self):
  pts=e.parse_csv(self.raw);folds,_=e.compare(pts,e.screening(pts['rows']))
  from collections import defaultdict
  d=defaultdict(dict)
  for v in folds:d[v['heldout_lab']][v['model']]=v['test_rmse']
  self.assertTrue(all(v['LINEAR_LOG_CONC']<v['NO_CONCENTRATION'] for v in d.values()))
  y=e.infer(folds,e.screening(pts['rows']))
  self.assertEqual(y[0]['action'],'REFINE')
  self.assertFalse(y[0]['novelty_certified'])
 def test_quadratic_model_can_win_in_heldout_groups(self):
  fixture(self.raw,quadratic=True)
  pts=e.parse_csv(self.raw);z=e.screening(pts['rows'])
  folds,_=e.compare(pts,z)
  res=e.infer(folds,z)
  self.assertEqual(res[0]['action'],'BRANCH')
  self.assertEqual(res[0]['quadratic_beats_linear_by_10_percent_folds'],4)
  self.assertFalse(res[0]['new_mechanism_identified'])
 def test_insufficient_labs_abstain(self):
  fixture(self.raw,labs=2)
  pts=e.parse_csv(self.raw);z=e.screening(pts['rows'])
  self.assertFalse(z['selected_cohort_keys'])
  folds,_=e.compare(pts,z)
  self.assertEqual(e.infer(folds,z),[])
 def test_catalog_separates_unmapped_source(self):
  self.assertFalse(e.DATASETS['WATER_TITRATION_638_2022']['analysis_ready'])
  with self.assertRaisesRegex(ValueError,'DATASET_ADAPTER_UNAVAILABLE'):
   runner.init(self.out,'WATER_TITRATION_638_2022')
 def test_prepare_acquire_requires_explicit_consent(self):
  runner.init(self.out,'ILS_ADENINE_2020')
  with self.assertRaises(PermissionError):runner.acquire(self.out,False)
  self.assertFalse((self.out/runner.RAW).exists())
 def test_full_offline_attach_evaluate_decide(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  a=runner.screen(self.out);self.assertEqual(a['usable_rows'],48)
  b=runner.evaluate(self.out);self.assertEqual(b['heldout_fold_model_rows'],12)
  c=runner.decide(self.out);self.assertEqual(c['empirical_problem_cards'],1)
  d=json.loads((self.out/runner.SUMMARY).read_text())
  self.assertTrue(d['observations_from_real_published_experiments']) # fixture exercises path only
  self.assertFalse(d['scientific_novelty_certified'])
  self.assertEqual(c,runner.decide(self.out)) # deterministic idempotency
 def test_source_tamper_blocks(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  with (self.out/runner.RAW).open('a') as f:f.write('\n')
  with self.assertRaisesRegex(ValueError,'MD5_MISMATCH'):runner.screen(self.out)
 def test_mutated_pre_registration_fails(self):
  runner.init(self.out,'ILS_ADENINE_2020')
  p=self.out/runner.PLAN;p.write_text(p.read_text()+' ')
  with self.assertRaisesRegex(ValueError,'PREREGISTRATION_TAMPERED'):
   runner.attach(self.out,self.raw)
 def test_decision_refuses_fold_tampering(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  runner.screen(self.out);runner.evaluate(self.out)
  (self.out/runner.FOLDS).write_text('fake')
  with self.assertRaisesRegex(ValueError,'Frozen upstream artifact changed'):runner.decide(self.out)
 def test_nonnumeric_band_safely_dropped(self):
  with self.raw.open('a') as f:f.write('L9,Au,785,methodA,adenine,calibration,1e-6,b9,1,NA,0,1,0\n')
  parsed=e.parse_csv(self.raw)
  self.assertEqual(parsed['rejections']['missing_or_invalid_band'],1)
 def test_single_frozen_output_refuses_alteration(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  runner.screen(self.out);(self.out/runner.SCREEN).write_text('modified')
  with self.assertRaises(Exception):runner.screen(self.out)
 def test_no_reanalysis_for_underspecified_titration(self):
  self.assertIn('titration',e.DATASETS['WATER_TITRATION_638_2022']['description'].lower())
  self.assertNotIn('metadata',e.DATASETS['WATER_TITRATION_638_2022'])
 def test_rank_deficiency_returns_none(self):
  data=[{'x':1,'y':3},{'x':1,'y':4},{'x':1,'y':5}]
  self.assertIsNone(e.fit(data,2))
 def test_actual_source_identity_constants(self):
  self.assertEqual(e.DATASETS['ILS_ADENINE_2020']['doi'],'10.5281/zenodo.3572359')
  self.assertEqual(e.BAND,(715.0,750.0))
 def test_no_empirical_claim_created_from_empty_folds(self):
  z={'selected_cohort_keys':['a']};r=e.infer([],z)
  self.assertEqual(r[0]['status'],'EVIDENCE_INSUFFICIENT')
 def test_heldout_fold_metadata_no_raw_spectrum_export(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  runner.screen(self.out);runner.evaluate(self.out)
  self.assertNotIn('spectrum', (self.out/runner.FOLDS).read_text().lower())
 def test_repeated_evaluation_is_idempotent(self):
  runner.init(self.out,'ILS_ADENINE_2020');runner.attach(self.out,self.raw)
  runner.screen(self.out)
  self.assertEqual(runner.evaluate(self.out),runner.evaluate(self.out))

if __name__=='__main__':unittest.main()
