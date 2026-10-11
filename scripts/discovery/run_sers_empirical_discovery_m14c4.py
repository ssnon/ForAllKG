"""M14-C4 real-data evidence acquisition and conservative problem discovery.

No LLM or paid provider. `acquire` is the only command using network;
requires --allow-download. Source is verified against Zenodo-published MD5.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile, urllib.request
from pathlib import Path
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as e

PLAN='M14C4_PREREGISTRATION.json'
PIN='M14C4_PREREGISTRATION_SHA256.txt'
RAW='ILSdata.csv'
SCREEN='M14C4_METADATA_SCREEN.json'
FOLDS='M14C4_HELDOUT_LAB_MODEL_COMPARISON.csv'
SKIPPED='M14C4_FOLD_EXCLUSIONS.json'
RECEIPT='M14C4_REAL_DATA_COMPUTATION_RECEIPT.json'
ASSESS='M14C4_EMPIRICAL_RESEARCH_PROBLEMS.json'
SUMMARY='M14C4_SCIENTIFIC_ADMISSIBILITY.json'

def load(path):return json.loads(path.read_bytes())
def freeze(path,raw):
 if path.exists():
  if path.read_bytes()!=raw:raise ValueError('Existing frozen artifact differs: '+str(path))
 else:
  with path.open('xb') as f:f.write(raw)

def init(out,dataset):
 if out.exists():raise FileExistsError('Output directory exists; choose a fresh preregistration')
 src=e.DATASETS[dataset]
 if not src['analysis_ready']:raise ValueError('DATASET_ADAPTER_UNAVAILABLE: '+src.get('reason',''))
 plan={'schema':e.SCHEMA,'status':'M14C4_PREREGISTERED_BEFORE_DATA_ANALYSIS',
  'dataset_id':dataset,'dataset':src,'source_expected_md5':src['md5'],
  'analysis_code_sha256':e.digest_file(Path(e.__file__)),
  'preregistered_hypotheses':[
   'H0: 715-750 cm^-1 band area has no transferable concentration dependence (constant response)',
   'H1: a global affine response in log10 concentration transfers across laboratories',
   'H2: a global quadratic response in log10 concentration transfers across laboratories'],
  'wavenumber_window_cm_minus_1':list(e.BAND),
  'observed_feature':'Trapezoidal integral of raw ILS spectral intensity over band; no per-lab test-set calibration',
  'concentration_requirement':'strictly positive numeric conc, not inferred from labels',
  'cohort_fields':list(e.COHORT_FIELDS),'heldout_group':e.GROUP_FIELD,
  'min_labs':e.MIN_LABS,'min_samples':e.MIN_SAMPLES,'max_cohorts':e.MAX_COHORTS,
  'cohort_selection':'Eligible by metadata AND spectral-band completeness, then rank by lab count, row count, and stable hash; never by intensity magnitude or model fit',
  'analysis_protocol':'Leave-one-lab-out OLS, constant vs linear vs quadratic; no hyperparameter selection on test lab',
  'success_rule':'At least three labs and 75% of heldout labs show >10% reduction in RMSE; descriptive only',
  'critical_limits':['No independent surface orientation, field, coverage or molecular population measurements',
    'A published adenine band/calibration analysis already exists; no novelty may be inferred from this result',
    'No independent replication beyond the existing published dataset',
    'No independent mechanism adjudication, causal effect, or new scientific discovery'],
  'paid_model_calls_authorized':False,'novelty_certified':False}
 out.mkdir(parents=True)
 freeze(out/PLAN,e.canonical(plan))
 freeze(out/PIN,(e.digest_file(out/PLAN)+'\n').encode())
 return {'status':plan['status'],'out_dir':str(out),'dataset_id':dataset,'new_paid_calls':0}

def ctx(out,need_raw=True):
 plan=load(out/PLAN)
 if (out/PIN).read_text(encoding='ascii').strip()!=e.digest_file(out/PLAN):
  raise ValueError('M14C4_PREREGISTRATION_TAMPERED: original plan sha256 mismatch')
 if plan.get('schema')!=e.SCHEMA or plan.get('dataset_id')!='ILS_ADENINE_2020':
  raise ValueError('Incorrect preregistration schema or adapter')
 if plan['analysis_code_sha256']!=e.digest_file(Path(e.__file__)):
  raise ValueError('Scientific analysis code changed after preregistration; fork a NEW study')
 if need_raw:
  raw=out/RAW
  if not raw.exists():raise FileNotFoundError('Real CSV absent; run acquire or attach')
  if e.digest_file(raw,'md5')!=plan['source_expected_md5']:
   raise ValueError('M14C4_RAW_SOURCE_MD5_MISMATCH: original bytes altered or wrong data')
 return plan

def acquire(out,allow_download):
 plan=ctx(out,False);dest=out/RAW
 if dest.exists():ctx(out,True);return {'status':'M14C4_SOURCE_ALREADY_VERIFIED','new_downloads':0}
 if not allow_download:raise PermissionError('Explicit --allow-download required; do not silently access network')
 url=plan['dataset']['source_url']
 if not url.startswith('https://zenodo.org/records/3572359/files/ILSdata.csv?'):
  raise ValueError('Untrusted data origin')
 # Do not leave a valid-looking partial dataset on download failure.
 with tempfile.NamedTemporaryFile(prefix='.m14c4-download-',dir=out,delete=False) as f:
  tmp=Path(f.name)
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ForAllKG-M14C4-Research-Verification/1.0'})
  size=0
  with urllib.request.urlopen(req,timeout=80) as resp, tmp.open('wb') as f:
   for block in iter(lambda:resp.read(1048576),b''):
    size+=len(block)
    if size>25000000:raise ValueError('Source exceeds expected 25 MB safety bound')
    f.write(block)
  if e.digest_file(tmp,'md5')!=plan['source_expected_md5']:
   raise ValueError('Source MD5 mismatch; upstream data may have changed')
  # O_EXCL prevents accidental replacement of previous original.
  with dest.open('xb') as f, tmp.open('rb') as inp:shutil.copyfileobj(inp,f)
 finally:tmp.unlink(missing_ok=True)
 return {'status':'M14C4_AUTHENTIC_EMPIRICAL_SOURCE_BOUND','source_bytes':dest.stat().st_size,
         'sha256':e.digest_file(dest),'md5_verified':True,'new_paid_calls':0}

def attach(out,path):
 ctx(out,False);dest=out/RAW
 if not path.is_file():raise FileNotFoundError(path)
 if e.digest_file(path,'md5')!=e.DATASETS['ILS_ADENINE_2020']['md5']:
  raise ValueError('Local file failed original Zenodo MD5; attachment rejected')
 if dest.exists():ctx(out,True);return {'status':'M14C4_SOURCE_ALREADY_VERIFIED','new_paid_calls':0}
 with dest.open('xb') as f,path.open('rb') as source:shutil.copyfileobj(source,f)
 return {'status':'M14C4_LOCAL_ORIGINAL_SOURCE_BOUND','source_sha256':e.digest_file(dest),'new_paid_calls':0}

def screen(out):
 ctx(out)
 parsed=e.parse_csv(out/RAW);info=e.screening(parsed['rows'])
 screening={'schema':e.SCHEMA,'status':'M14C4_METADATA_ONLY_SCREEN_FROZEN',
  'data_sha256':e.digest_file(out/RAW),'source_record':e.DATASETS['ILS_ADENINE_2020']['record'],
  'raw_rows':parsed['raw_rows'],'usable_rows':len(parsed['rows']),
  'rejected_rows':parsed['rejections'],'spectral_band_column_count':parsed['spectral_columns'],
  'actual_band_range':parsed['band_observed_range'],
  'selection':info,'intensity_residuals_examined_for_selection':False,'spectral_missingness_used_for_eligibility':True,
  'empirical_science_novelty_certified':False}
 freeze(out/SCREEN,e.canonical(screening))
 return {'status':screening['status'],'raw_rows':screening['raw_rows'],
   'usable_rows':screening['usable_rows'],'selected_cohorts':len(info['selected_cohort_keys']),
   'rejected_rows':screening['rejected_rows'],'new_paid_calls':0}

def recompute(out):
 ctx(out);parsed=e.parse_csv(out/RAW);expected=e.screening(parsed['rows']);scr=load(out/SCREEN)
 if scr['data_sha256']!=e.digest_file(out/RAW) or scr['selection']!=expected:
  raise ValueError('Screening result changed or corrupted')
 folds,skips=e.compare(parsed,expected)
 return folds,skips,expected

def evaluate(out):
 folds,skips,selected=recompute(out)
 if not selected['selected_cohort_keys']:
  # No analysis is legitimate; still leave auditable abstention receipt.
  outcome='M14C4_ABSTAIN_NO_ELIGIBLE_COHORTS'
 else:outcome='M14C4_REAL_HELDOUT_LAB_EVALUATION_COMPLETE'
 freeze(out/FOLDS,e.csv_bytes(folds))
 freeze(out/SKIPPED,e.canonical(skips))
 receipt={'schema':e.SCHEMA,'status':outcome,'source_sha256':e.digest_file(out/RAW),
  'preregistration_sha256':e.digest_file(out/PLAN),
  'screen_sha256':e.digest_file(out/SCREEN),
  'folds_sha256':e.digest_file(out/FOLDS),
  'excluded_folds_sha256':e.digest_file(out/SKIPPED),
  'holdout':'Full laboratory held out; no fold-based model tuning',
  'true_experimental_measurements':True,'independent_new_scientific_experiment':False,
  'external_fulltext_prior_art_verified_per_problem':False,'novelty_certified':False,
  'new_model_calls':0}
 freeze(out/RECEIPT,e.canonical(receipt))
 return {'status':outcome,'heldout_fold_model_rows':len(folds),'excluded_fold_rows':len(skips),
  'source_sha256':receipt['source_sha256'],'new_paid_calls':0}

def decide(out):
 folds,skips,selected=recompute(out)
 receipt=load(out/RECEIPT)
 checks={PLAN:receipt['preregistration_sha256'],SCREEN:receipt['screen_sha256'],
  FOLDS:receipt['folds_sha256'],SKIPPED:receipt['excluded_folds_sha256']}
 for k,v in checks.items():
  if e.digest_file(out/k)!=v:raise ValueError('Frozen upstream artifact changed: '+k)
 if (out/FOLDS).read_bytes()!=e.csv_bytes(folds) or (out/SKIPPED).read_bytes()!=e.canonical(skips):
  raise ValueError('Real-data analysis no longer bytewise reproducible')
 problems=e.infer(folds,selected)
 status='M14C4_EMPIRICAL_PROBLEMS_PROVISIONAL_NOT_NEW_DISCOVERY'
 if not problems:status='M14C4_NO_SUPPORTED_EMPIRICAL_PROBLEM_ABSTAIN'
 results={'schema':e.SCHEMA,'status':status,'source_dataset_id':'ILS_ADENINE_2020',
  'evidence_receipt_sha256':e.digest_file(out/RECEIPT),
  'analysis_cohort_count':len(problems),'problems':problems,
  'research_problem_discovery_scope':'Observed calibration/transfer problems in established adenine SERS, not novel physical gaps',
  'hypothesis_proposals_admissible_only_after_new_independent_source_anchor':True,
  'candidate_problem_cards':[{'problem_id':'m14c4:'+e.sha(e.canonical([e.DATASETS['ILS_ADENINE_2020']['doi'],p['cohort_key'],p['status']]))[:20],
       'scientific_question':p.get('problem_question','Insufficient independent laboratories for model comparison'),
       'action_proposal':p['action'],
       'empirical_source_doi':e.DATASETS['ILS_ADENINE_2020']['doi'],
       'evidence_sha256':e.digest_file(out/RECEIPT),
       'gap_confirmed':False,'novelty_certified':False,
       'scientific_review_required_before_generation':True}
      for p in problems],
  'requires_independent_external_observation':True,
  'independent_literature_novelty_review_per_problem':False,
  'novelty_certified':False,'model_calls':0}
 freeze(out/ASSESS,e.canonical(results))
 counts={}
 for p in problems:counts[p['action']]=counts.get(p['action'],0)+1
 summary={'schema':e.SCHEMA,'status':status,'raw_source_sha256':e.digest_file(out/RAW),
   'result_sha256':e.digest_file(out/ASSESS),
   'observations_from_real_published_experiments':True,
   'new_empirical_observations_generated_by_system':False,
   'actual_scientific_mechanism_competition_performed':False,
   'competing_models_are_statistical_calibration_hypotheses':True,
   'separate_train_test_laboratories':True,
   'actions':counts,'real_gap_certified':False,'scientific_novelty_certified':False,
   'next_science_action':('Investigate uncertainty, omitted controls, and an independently held-out SERS problem before generating mechanism claims' if problems else 'Acquire different evidence source; do not hallucinate a gap'),
   'suspended_existing_C3_case_unmodified':True,
   'new_paid_calls':0}
 freeze(out/SUMMARY,e.canonical(summary))
 return {'status':status,'empirical_problem_cards':len(problems),'actions':counts,
    'novelty_certified':False,'new_paid_calls':0}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('catalog')
 for name in ('prepare','acquire','attach','screen','evaluate','decide'):
  a=sub.add_parser(name);a.add_argument('--out-dir',type=Path,required=True)
  if name=='prepare':a.add_argument('--dataset',choices=sorted(e.DATASETS),default='ILS_ADENINE_2020')
  if name=='acquire':a.add_argument('--allow-download',action='store_true')
  if name=='attach':a.add_argument('--csv',type=Path,required=True)
 args=p.parse_args(argv)
 if args.cmd=='catalog':result={'status':'M14C4_CATALOG','datasets':e.DATASETS,'new_paid_calls':0}
 elif args.cmd=='prepare':result=init(args.out_dir,args.dataset)
 elif args.cmd=='acquire':result=acquire(args.out_dir,args.allow_download)
 elif args.cmd=='attach':result=attach(args.out_dir,args.csv)
 elif args.cmd=='screen':result=screen(args.out_dir)
 elif args.cmd=='evaluate':result=evaluate(args.out_dir)
 else:result=decide(args.out_dir)
 print(json.dumps(result,indent=2,ensure_ascii=False));return result

if __name__=='__main__':main()
