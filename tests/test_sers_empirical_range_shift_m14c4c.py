import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as old
from pipeline_core.discovery import sers_empirical_transfer_m14c4b as b
from pipeline_core.discovery import sers_empirical_range_shift_m14c4c as c
from scripts.discovery import run_sers_empirical_discovery_m14c4 as acli
from scripts.discovery import run_sers_empirical_transfer_m14c4b as bcli
from scripts.discovery import run_sers_empirical_range_shift_m14c4c as cli

def fixture(path,labs=4,outside=True):
 with path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['labcode','substrate','laser','method','sample','type','conc','batch','replica','715','730','750'])
  for lab in range(labs):
   for typ,levels in [('train',[-9,-8,-7,-6,-5]),('test',[-10,-9,-4] if outside else [-8,-7,-6])]:
    for step,x in enumerate(levels):
     for replicate in range(3):
      y=100+2*x+0.1*x*x +lab*0.1+replicate*.01
      w.writerow([f'L{lab}','sAg','532','sAg@532',f'{typ}{step}',typ,10**x,f'B{lab}',replicate,y,y+1,y+2])

class C4CTests(unittest.TestCase):
 def setUp(self):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
  self.base=Path(temp.name);self.csv=self.base/'source.csv';fixture(self.csv)
  orig=old.DATASETS['ILS_ADENINE_2020'].copy();orig['md5']=old.digest_file(self.csv,'md5')
  p=patch.dict(old.DATASETS,{'ILS_ADENINE_2020':orig});p.start();self.addCleanup(p.stop)
  self.a=self.base/'a';self.b=self.base/'b';self.c=self.base/'c';self.fore=self.base/'forensics.json'
 def setup_studies(self):
  acli.init(self.a,'ILS_ADENINE_2020');acli.attach(self.a,self.csv)
  acli.screen(self.a);acli.evaluate(self.a);acli.decide(self.a)
  self.fore.write_text(json.dumps({'status':'M14C4_READ_ONLY_COHORT_FORENSICS_COMPLETE',
     'source':{'raw_csv_sha256':old.digest_file(self.a/acli.RAW),
               'frozen_screen_sha256':old.digest_file(self.a/acli.SCREEN)},
     'original_pre_registered_screen':{'screen_eligible_cohorts':0}}))
  bcli.prepare(self.a,self.fore,self.b);bcli.screen(self.b);bcli.evaluate(self.b);bcli.decide(self.b)
  self.assertEqual(bcli.load(self.b/bcli.ASSESS)['heldout_folds'],0)
 def prepared(self):
  self.setup_studies();return cli.prepare(self.b,self.c)
 def test_split_has_out_of_range_and_holdout(self):
  parsed,rows=b.data(self.csv);folds,excluded=c.eligible_splits(rows)
  self.assertEqual(len(folds),4);self.assertEqual(excluded,[])
  for row in folds:
   self.assertEqual(row['n_test_below'],3)
   self.assertEqual(row['n_test_above'],3)
   self.assertEqual(row['n_test_in_range'],3)
   self.assertNotIn(row['heldout_lab'],row['train_labs'])
   self.assertFalse(row['author_test_used_to_fit'])
   self.assertAlmostEqual(row['max_log10_extrapolation_below'],1)
   self.assertAlmostEqual(row['max_log10_extrapolation_above'],1)
   self.assertEqual(len(row['predictive_errors']),3)
 def test_no_extrapolation_abstains(self):
  fixture(self.csv,outside=False)
  _,rows=b.data(self.csv);folds,excluded=c.eligible_splits(rows)
  self.assertEqual(folds,[])
  self.assertEqual(len(excluded),4)
  self.assertTrue(all('NO_OUT_OF_RANGE_TEST_ROWS' in f['reasons'] for f in excluded))
 def test_too_few_labs_abstains(self):
  fixture(self.csv,labs=2)
  _,rows=b.data(self.csv);folds,excluded=c.eligible_splits(rows)
  self.assertEqual(folds,[])
  self.assertTrue(excluded)
 def test_summary_not_novelty(self):
  _,rows=b.data(self.csv);folds,excl=c.eligible_splits(rows)
  prior=b.metadata_screen(*b.data(self.csv))['protocols']
  res=c.summarize(folds,excl,prior)
  self.assertEqual(res['fold_count'],4)
  self.assertFalse(res['scientific_novelty_certified'])
  self.assertTrue(res['model_comparison_is_predictive_not_physical_mechanism_test'])
  self.assertEqual(res['protocols'][0]['n_evaluated_folds'],4)
  self.assertTrue(res['protocols'][0]['no_automatic_hypothesis_generation'])
 def test_requires_completed_previous_abstention(self):
  self.setup_studies();summary=self.b/bcli.ASSESS
  data=bcli.load(summary);data['heldout_folds']=1
  summary.write_bytes(old.canonical(data))
  with self.assertRaisesRegex(ValueError,'zero-fold abstention'):cli.prepare(self.b,self.c)
 def test_full_pipeline_and_idempotent_replay(self):
  self.prepared();self.assertEqual(cli.screen(self.c)['potential_range_shift_folds'],4)
  self.assertEqual(cli.evaluate(self.c)['evaluated_folds'],4)
  self.assertEqual(cli.decide(self.c)['evaluated_folds'],4)
  self.assertEqual(cli.main(['verify','--out-dir',str(self.c)])['status'],'M14C4C_REPRODUCIBILITY_VERIFIED')
  self.assertEqual(cli.decide(self.c)['evaluated_folds'],4)
 def test_source_immutability(self):
  self.prepared()
  oldfiles={str(f):old.digest_file(f) for root in (self.a,self.b) for f in root.glob('*') if f.is_file()}
  cli.screen(self.c);cli.evaluate(self.c);cli.decide(self.c)
  self.assertEqual(oldfiles,{str(f):old.digest_file(f) for root in (self.a,self.b) for f in root.glob('*') if f.is_file()})
 def test_plan_tamper(self):
  self.prepared();(self.c/cli.PLAN).write_text('{}')
  with self.assertRaisesRegex(ValueError,'PLAN_HASH_CHANGED'):cli.screen(self.c)
 def test_source_tamper(self):
  self.prepared();(self.a/acli.RAW).write_text('altered')
  with self.assertRaises(Exception):cli.screen(self.c)
 def test_record_tamper(self):
  self.prepared();cli.screen(self.c);cli.evaluate(self.c)
  (self.c/cli.FOLDS).write_text('[]')
  with self.assertRaisesRegex(ValueError,'REPLAY_DIFFERENCE'):cli.decide(self.c)
 def test_plan_discloses_design_adaptation(self):
  self.prepared();plan=cli.load(self.c/cli.PLAN)
  self.assertTrue(plan['why_new_study'])
  self.assertIn('after viewing C4B metadata',plan['limitations'][0])
  self.assertIn('held-out',plan['test'])
 def test_install_no_model_calls(self):
  self.assertEqual(cli.PLAN,'M14C4C_RANGE_SHIFT_PLAN.json')
  self.assertFalse(c.summarize([],[],[])['scientific_novelty_certified'])

if __name__=='__main__':unittest.main()
