import csv, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as original
from pipeline_core.discovery import sers_empirical_transfer_m14c4b as m
from scripts.discovery import run_sers_empirical_discovery_m14c4 as original_runner
from scripts.discovery import run_sers_empirical_transfer_m14c4b as r


def fixture(path, labs=4, train_levels=5, test_levels=3, test_outside=False, square=False):
 with path.open('w', encoding='utf-8', newline='') as f:
  w=csv.writer(f)
  w.writerow(['labcode','substrate','laser','method','sample','type','conc','batch','replica','X715','730','750'])
  for lab in range(labs):
   for part,nlevels,tag in [('train',train_levels,'C'),('test',test_levels,'X')]:
    for k in range(nlevels):
     x= -9+k if part=='train' else (-8+k if not test_outside else 1+k)
     for rep in range(3):
      y=50+10*x+(1*x*x if square else 0)+.02*lab+.01*rep
      w.writerow([f'L{lab}','sAg','532','sAg@532',f'{tag}{k}',part,10.**x,f'B{lab}',rep,y,y+1,y+2])

class EvidenceTransferTests(unittest.TestCase):
 def setUp(self):
  td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
  self.root=Path(td.name);self.raw=self.root/'data.csv';fixture(self.raw)
  info=original.DATASETS['ILS_ADENINE_2020'].copy();info['md5']=original.digest_file(self.raw,'md5')
  p=patch.dict(original.DATASETS,{'ILS_ADENINE_2020':info});p.start();self.addCleanup(p.stop)
  self.c4a=self.root/'c4a';self.out=self.root/'c4b';self.forensics=self.root/'forensic.json'
 def source(self):
  original_runner.init(self.c4a,'ILS_ADENINE_2020')
  original_runner.attach(self.c4a,self.raw)
  s=original_runner.screen(self.c4a)
  self.assertEqual(s['selected_cohorts'],0)
  original_runner.evaluate(self.c4a);original_runner.decide(self.c4a)
  self.forensics.write_text(json.dumps({
    'status':'M14C4_READ_ONLY_COHORT_FORENSICS_COMPLETE',
    'source':{'raw_csv_sha256':original.digest_file(self.c4a/'ILSdata.csv'),
              'frozen_screen_sha256':original.digest_file(self.c4a/original_runner.SCREEN)},
    'original_pre_registered_screen':{'screen_eligible_cohorts':0}
  }),encoding='utf-8')
 def prepared(self):
  self.source();r.prepare(self.c4a,self.forensics,self.out)
 def test_sample_is_not_a_stratum(self):
  p,rows=m.data(self.raw)
  s=m.metadata_screen(p,rows)
  self.assertEqual(len(s['protocols']),1)
  self.assertEqual(s['potential_eligible_protocols'],1)
  self.assertEqual(s['protocols'][0]['potential_full_laboratory_holdout_folds'],4)
  self.assertEqual(s['rows_with_published_train_or_test_partition'],96)
 def test_fit_other_labs_train_and_evaluate_heldout_test_only(self):
  p,rows=m.data(self.raw);s=m.metadata_screen(p,rows)
  folds,excluded=m.fold_comparison(rows,s)
  self.assertEqual(len(folds),4)
  self.assertEqual(excluded,[])
  for f in folds:
   self.assertEqual(f['train_n'],45)
   self.assertEqual(f['test_n'],9)
   self.assertNotIn(f['heldout_lab'],f['train_labs'])
   self.assertFalse(f['test_partition_seen_in_fitting'])
   self.assertFalse(f['heldout_lab_seen_in_fitting'])
   self.assertEqual(set(f['test_rmse']),{name for name,_ in m.MODELS})
 def test_original_zero_cohorts_new_design_valid(self):
  original_screen=original.screening(original.parse_csv(self.raw)['rows'])
  self.assertEqual(len(original_screen['selected_cohort_keys']),0)
  s=m.metadata_screen(*m.data(self.raw))
  self.assertEqual(s['potential_eligible_protocols'],1)
 def test_out_of_support_folds_not_used(self):
  fixture(self.raw,test_outside=True)
  p,rows=m.data(self.raw);s=m.metadata_screen(p,rows)
  self.assertEqual(s['potential_eligible_protocols'],0)
  folds,ex=m.fold_comparison(rows,s)
  self.assertFalse(folds)
  self.assertIn('HELDOUT_TEST_OUTSIDE_TRAIN_RANGE',s['protocols'][0]['fold_screen'][0]['reasons'])
 def test_insufficient_labs_abstain(self):
  fixture(self.raw,labs=2)
  p,rows=m.data(self.raw);s=m.metadata_screen(p,rows)
  self.assertEqual(s['potential_eligible_protocols'],0)
  self.assertFalse(m.fold_comparison(rows,s)[0])
 def test_single_test_concentration_abstain(self):
  fixture(self.raw,test_levels=1)
  p,rows=m.data(self.raw);s=m.metadata_screen(p,rows)
  self.assertFalse(s['protocols'][0]['analysis_eligible'])
 def test_published_partition_is_case_insensitive(self):
  self.assertEqual(m.part({'cohort':['m','s','l','x',' TRAIN ']}),'train')
 def test_protocol_stratifies_physics_relevant_factors(self):
  self.assertEqual(m.prot({'cohort':['method','substrate','laser','sample','type']}),('method','substrate','laser'))
 def test_no_claim_of_confirmatory_discovery(self):
  p,rows=m.data(self.raw);s=m.metadata_screen(p,rows)
  folds,_=m.fold_comparison(rows,s);result=m.summarize(s,folds)
  self.assertTrue(result['no_independent_new_discovery_certified'])
  self.assertFalse(result['protocol_results'][0]['hypothesis_generation_permitted'])
  self.assertFalse(result['protocol_results'][0]['scientific_novelty_certified'])
 def test_strict_negative_source_id(self):
  self.prepared();self.assertIn('source_artifacts_sha256',r.pre(self.out)[0])
 def test_secondary_plan_discloses_metadata_review(self):
  self.prepared();plan=json.loads((self.out/r.PLAN).read_text())
  self.assertTrue(plan['not_independent_replication'])
  self.assertIn('NOT preregistered before observing metadata',plan['why_secondary'])
 def test_repeated_prepare_refused(self):
  self.prepared()
  with self.assertRaises(FileExistsError):r.prepare(self.c4a,self.forensics,self.out)
 def test_full_secondary_study(self):
  self.prepared();a=r.screen(self.out);self.assertEqual(a['potentially_eligible_protocols'],1)
  b=r.evaluate(self.out);self.assertEqual(b['folds'],4)
  c=r.decide(self.out);self.assertEqual(c['heldout_folds'],4)
  self.assertEqual(c,r.decide(self.out))
  self.assertEqual(r.main(['verify','--out-dir',str(self.out)])['status'],
                   'M14C4B_SAVED_RESULTS_REPLAY_VERIFIED_NO_EXTERNAL_CALLS')
 def test_source_modification_rejected(self):
  self.prepared();(self.c4a/'ILSdata.csv').write_text('BAD')
  with self.assertRaises(ValueError):r.screen(self.out)
 def test_old_plan_mutation_rejected(self):
  self.prepared();with_plan=self.c4a/original_runner.PLAN
  with_plan.write_text('not json',encoding='utf-8')
  with self.assertRaises(Exception):r.screen(self.out)
 def test_forensics_modification_rejected(self):
  self.prepared();self.forensics.write_text('tamper')
  with self.assertRaisesRegex(ValueError,'FORENSICS_REPORT_TAMPERED'):r.screen(self.out)
 def test_secondary_plan_mutation_rejected(self):
  self.prepared();(self.out/r.PLAN).write_text('{}')
  with self.assertRaisesRegex(ValueError,'PLAN_TAMPERED'):r.screen(self.out)
 def test_original_study_unmodified(self):
  self.prepared();before={x.name:original.digest_file(x) for x in self.c4a.iterdir() if x.is_file()}
  r.screen(self.out);r.evaluate(self.out);r.decide(self.out)
  after={x.name:original.digest_file(x) for x in self.c4a.iterdir() if x.is_file()}
  self.assertEqual(before,after)
 def test_result_tamper_rejected(self):
  self.prepared();r.screen(self.out);r.evaluate(self.out)
  (self.out/r.FOLDS).write_text('[]')
  with self.assertRaisesRegex(ValueError,'REPLAY_MISMATCH'):r.decide(self.out)
 def test_original_screen_must_be_abstention(self):
  self.source();f=json.loads(self.forensics.read_text());f['original_pre_registered_screen']['screen_eligible_cohorts']=2
  self.forensics.write_text(json.dumps(f))
  with self.assertRaisesRegex(ValueError,'Requires observed C4A null screen'):
   r.prepare(self.c4a,self.forensics,self.out)
 def test_baseline_regression_preexisting_models_unchanged(self):
  self.assertEqual(original.COHORT_FIELDS,('method','substrate','laser','sample','type'))
  self.assertEqual(set(name for name,_ in m.MODELS),{'H0_CONSTANT','H1_LINEAR_LOG_CONC','H2_QUADRATIC_LOG_CONC'})

if __name__=='__main__':unittest.main()
