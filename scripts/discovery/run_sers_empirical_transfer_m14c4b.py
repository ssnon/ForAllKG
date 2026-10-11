"""M14-C4B: metadata-informed *secondary* transfer analysis, preserving C4A."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as old
from pipeline_core.discovery import sers_empirical_transfer_m14c4b as e
from scripts.discovery import run_sers_empirical_discovery_m14c4 as prior

PLAN='M14C4B_SECONDARY_ANALYSIS_PLAN.json';PIN='M14C4B_PLAN_SHA256.txt'
SCREEN='M14C4B_METADATA_SUPPORT.json';FOLDS='M14C4B_STRICT_CROSSLAB_TEST_FOLDS.json'
EXCLUDED='M14C4B_FOLD_EXCLUSIONS.json';RECEIPT='M14C4B_EVIDENCE_RECEIPT.json';ASSESS='M14C4B_PROVISIONAL_RESEARCH_QUESTIONS.json'

def load(p):return json.loads(p.read_bytes())
def freeze(path,data):
 raw=old.canonical(data)
 if path.exists():
  if path.read_bytes()!=raw:raise ValueError('M14C4B_EXISTING_ARTIFACT_DIFFERENT: '+str(path))
 else:
  with path.open('xb') as f:f.write(raw)

def pre(out):
 plan=load(out/PLAN)
 if old.digest_file(out/PLAN)!=(out/PIN).read_text().strip():raise ValueError('M14C4B_PLAN_TAMPERED')
 if old.digest_file(Path(e.__file__))!=plan['m14c4b_core_code_sha256']:raise ValueError('M14C4B_CORE_CODE_CHANGED')
 if old.digest_file(Path(__file__))!=plan['m14c4b_runner_code_sha256']:raise ValueError('M14C4B_RUNNER_CODE_CHANGED')
 src=Path(plan['original_c4a_directory']);prior.ctx(src)
 for n,hashvalue in plan['source_artifacts_sha256'].items():
  if old.digest_file(src/n)!=hashvalue:raise ValueError('M14C4B_ORIGINAL_SOURCE_TAMPERED: '+n)
 forensic=Path(plan['read_only_forensic_report'])
 if old.digest_file(forensic)!=plan['read_only_forensic_sha256']:
  raise ValueError('M14C4B_FORENSICS_REPORT_TAMPERED')
 if not load(forensic)['status']=='M14C4_READ_ONLY_COHORT_FORENSICS_COMPLETE':
  raise ValueError('M14C4B_INCORRECT_FORENSICS')
 return plan,src

def prepare(c4a,forensics,out):
 if out.exists():raise FileExistsError('New study output already exists: '+str(out))
 prior.ctx(c4a)
 f=load(forensics)
 if f.get('status')!='M14C4_READ_ONLY_COHORT_FORENSICS_COMPLETE':raise ValueError('Incorrect diagnostic report')
 if f['source']['raw_csv_sha256']!=old.digest_file(c4a/prior.RAW):raise ValueError('Forensic source differs from raw CSV')
 srcfiles=[prior.PLAN,prior.PIN,prior.RAW,prior.SCREEN,prior.RECEIPT,prior.ASSESS]
 for n in srcfiles:
  if not (c4a/n).exists():raise FileNotFoundError('Original completed C4A artifact missing: '+n)
 if f['source']['frozen_screen_sha256']!=old.digest_file(c4a/prior.SCREEN):raise ValueError('Forensic screen hash differs')
 if f['original_pre_registered_screen']['screen_eligible_cohorts']!=0:raise ValueError('Requires observed C4A null screen')
 plan={'schema':e.SCHEMA,'status':'SECONDARY_PROTOCOL_FIXED_BEFORE_C4B_MODEL_OUTCOMES',
  'original_c4a_directory':str(c4a.resolve()),'read_only_forensic_report':str(forensics.resolve()),
  'read_only_forensic_sha256':old.digest_file(forensics),
  'source_artifacts_sha256':{n:old.digest_file(c4a/n) for n in srcfiles},
  'source_md5':old.digest_file(c4a/prior.RAW,'md5'),
  'm14c4b_core_code_sha256':old.digest_file(Path(e.__file__)),
  'm14c4b_runner_code_sha256':old.digest_file(Path(__file__)),
  'why_secondary':'C4A failed by grouping on sample labels which encode concentration. Forensics informed regrouping; C4B is NOT preregistered before observing metadata.',
  'protocol_fields':list(e.PROTOCOL_FIELDS),'sample_role':'concentration/sample identifier; never grouping variable',
  'published_partition_role':{'train':'model fitting in other laboratories only','test':'out-of-lab evaluation only'},
  'heldout':'Leave one entire laboratory out; all held-out test measurements used solely for evaluation',
  'protocol_policy':'Evaluate all protocols with three or more eligible folds; no intensity-based protocol selection',
  'required_train_samples':e.MIN_TRAIN_SAMPLES,'required_test_samples':e.MIN_TEST_SAMPLES,
  'required_train_concentration_levels':e.MIN_TRAIN_LEVELS,'required_test_concentration_levels':e.MIN_TEST_LEVELS,
  'extrapolation_policy':'Exclude held-out test-fold concentrations outside fitting train concentration range',
  'models':[name for name,_ in e.MODELS],
  'feature':'Original raw 715-750 cm^-1 band area only; no per-lab baseline or test-data scaling',
  'model_evaluation':'Held-out lab RMSE per protocol; descriptive effect sizes, no p-values or mechanism identification',
  'secondary_decision_rule':'At least 3 folds and 75% labs showing >10% RMSE advantage; unconfirmed research question only',
  'observed_before_plan':'Only the previously completed C4A abstention and eligibility/metadata grouping diagnostics',
  'not_independent_replication':True,'novelty_certified':False,'paid_model_calls_authorized':False}
 out.mkdir(parents=True)
 freeze(out/PLAN,plan)
 (out/PIN).write_text(old.digest_file(out/PLAN)+'\n',encoding='ascii')
 return {'status':'M14C4B_NEW_SECONDARY_PLAN_FROZEN','out_dir':str(out),
         'c4a_preserved':True,'read_only_forensic_report_hash':plan['read_only_forensic_sha256'],'paid_calls':0}

def calc(out):
 _,src=pre(out)
 parsed,entries=e.data(src/prior.RAW)
 screen=e.metadata_screen(parsed,entries)
 folds,excluded=e.fold_comparison(entries,screen)
 result=e.summarize(screen,folds)
 return screen,folds,excluded,result

def screen(out):
 s,_,_,_=calc(out)
 freeze(out/SCREEN,s)
 return {'status':'M14C4B_METADATA_SCREEN_COMPLETE','protocols':len(s['protocols']),
  'potentially_eligible_protocols':s['potential_eligible_protocols'],
  'source_usable_rows':s['usable_positive_conc_and_complete_band'],'paid_calls':0}

def evaluate(out):
 s,folds,excluded,_=calc(out)
 if (out/SCREEN).read_bytes()!=old.canonical(s):raise ValueError('Frozen metadata screen changed')
 freeze(out/FOLDS,folds);freeze(out/EXCLUDED,excluded)
 plan,_=pre(out)
 receipt={'schema':e.SCHEMA,'status':'M14C4B_SECONDARY_REAL_DATA_EVALUATION_COMPLETED' if folds else 'M14C4B_ABSTAIN_NO_VALID_FOLDS',
  'source_sha256':plan['source_artifacts_sha256'][prior.RAW],
  'new_secondary_plan_sha256':old.digest_file(out/PLAN),'forensic_sha256':plan['read_only_forensic_sha256'],
  'metadata_sha256':old.digest_file(out/SCREEN),'folds_sha256':old.digest_file(out/FOLDS),
  'fold_exclusions_sha256':old.digest_file(out/EXCLUDED),
  'heldout_lab_and_published_test_partition_both_enforced':True,
  'model_fit_never_sees_published_test_partition':True,
  'dataset_already_published':True,'adaptation_after_metadata_review':True,
  'author_train_test_partition_not_a_new_independent_replication':True,
  'novelty_certified':False,'paid_calls':0}
 freeze(out/RECEIPT,receipt)
 return {'status':receipt['status'],'folds':len(folds),'excluded_folds':len(excluded),
         'novelty_certified':False,'paid_calls':0}

def decide(out):
 s,f,ex,result=calc(out)
 for name,obj in [(SCREEN,s),(FOLDS,f),(EXCLUDED,ex)]:
  if (out/name).read_bytes()!=old.canonical(obj):raise ValueError('M14C4B_REPLAY_MISMATCH: '+name)
 rec=load(out/RECEIPT)
 if rec['new_secondary_plan_sha256']!=old.digest_file(out/PLAN):raise ValueError('Plan hash altered')
 for k,v in [(SCREEN,'metadata_sha256'),(FOLDS,'folds_sha256'),(EXCLUDED,'fold_exclusions_sha256')]:
  if old.digest_file(out/k)!=rec[v]:raise ValueError('Receipt lineage mismatch: '+k)
 result['source_evidence_receipt_sha256']=old.digest_file(out/RECEIPT)
 result['evidence_is_secondary_to_metadata_forensics']=True
 result['independent_physical_mechanism_test_performed']=False
 result['existing_c4a_abstention_remains_valid']=True
 freeze(out/ASSESS,result)
 return {'status':result['status'],'protocols_with_folds':result['protocols_with_evaluated_folds'],
  'heldout_folds':result['heldout_folds'],'potential_scientific_novelty_certified':False,'paid_calls':0}

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
 p=sub.add_parser('prepare');p.add_argument('--c4a-dir',type=Path,required=True);p.add_argument('--forensics',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
 for name in ('screen','evaluate','decide','verify'):
  p=sub.add_parser(name);p.add_argument('--out-dir',type=Path,required=True)
 args=ap.parse_args(argv)
 if args.cmd=='prepare':r=prepare(args.c4a_dir,args.forensics,args.out_dir)
 elif args.cmd=='screen':r=screen(args.out_dir)
 elif args.cmd=='evaluate':r=evaluate(args.out_dir)
 elif args.cmd=='decide':r=decide(args.out_dir)
 else:
  r=decide(args.out_dir);r['status']='M14C4B_SAVED_RESULTS_REPLAY_VERIFIED_NO_EXTERNAL_CALLS'
 print(json.dumps(r,indent=2,ensure_ascii=False));return r

if __name__=='__main__':main()
