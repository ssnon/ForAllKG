"""Run additive M14-C4C *secondary* extrapolative transfer study; historical studies immutable."""
import argparse, json
from pathlib import Path
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as old
from pipeline_core.discovery import sers_empirical_transfer_m14c4b as b
from pipeline_core.discovery import sers_empirical_range_shift_m14c4c as c
from scripts.discovery import run_sers_empirical_discovery_m14c4 as ar
from scripts.discovery import run_sers_empirical_transfer_m14c4b as br

PLAN='M14C4C_RANGE_SHIFT_PLAN.json';PIN='M14C4C_PLAN_SHA256.txt'
SCREEN='M14C4C_RANGE_SHIFT_METADATA.json';FOLDS='M14C4C_OUT_OF_RANGE_HELDOUT_FOLDS.json'
EXCLUSIONS='M14C4C_FOLD_EXCLUSIONS.json';RECEIPT='M14C4C_EVIDENCE_RECEIPT.json';RESULT='M14C4C_RESEARCH_ACTIONS.json'

def load(p):return json.loads(p.read_bytes())
def freeze(path, obj):
    content=old.canonical(obj)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('Existing C4C artifact differs: '+str(path))
    else:
        with path.open('xb') as fd:fd.write(content)

def prepare(c4b,out):
    if out.exists():raise FileExistsError(out)
    oldplan,src=br.pre(c4b)
    sourcefiles=[br.PLAN,br.PIN,br.SCREEN,br.FOLDS,br.EXCLUDED,br.RECEIPT,br.ASSESS]
    for n in sourcefiles:
        if not (c4b/n).is_file():raise FileNotFoundError('Missing C4B result '+n)
    previous=load(c4b/br.ASSESS)
    if previous['status']!='M14C4B_ABSTAIN_NO_VALID_EMPIRICAL_FOLDS' or previous['heldout_folds']!=0:
        raise ValueError('C4C analysis designed after observing C4B zero-fold abstention')
    met=load(c4b/br.SCREEN)
    rejections=[r for p in met['protocols'] for f in p['fold_screen'] for r in f['reasons']]
    if not rejections or set(rejections)!={'HELDOUT_TEST_OUTSIDE_TRAIN_RANGE'}:
        raise ValueError('C4C protocol only supports the observed outside-train-range-only exclusion')
    plan={'schema':c.SCHEMA,'status':'SECONDARY_PLAN_CREATED_AFTER_C4B_ABSTENTION',
          'c4b_path':str(c4b.resolve()),'c4a_path':str(src.resolve()),
          'c4b_artifact_hashes':{n:old.digest_file(c4b/n) for n in sourcefiles},
          'source_csv_sha256':old.digest_file(src/ar.RAW),
          'c4c_core_sha256':old.digest_file(Path(c.__file__)),
          'c4c_runner_sha256':old.digest_file(Path(__file__)),
          'why_new_study':'The C4B metadata screening revealed train/test ranges do not nest for every one of 39 laboratory candidates.',
          'selection_rule':'All six prior C4B protocols, all labs, no intensity outcome-based selection',
          'train':'Only train partition in laboratories other than test lab',
          'test':'Only author test partition in fully held-out laboratory',
          'range_policy':'Include and label range-shift/out-of-range test rows, do not pretend interpolation',
          'reported':'RMSE full-test, extrapolation-only, low, high, and in-range; held-out lab counts; direction/distance',
          'predefined_models':[name for name,_ in c.MODELS],
          'min_folds':c.MIN_FOLDS,
          'limitations':['Analysis selected after viewing C4B metadata; exploratory only',
                         'Original study already analyzes calibration/test/transfer; not new observation',
                         'Raw 715-750 band area is unnormalized across laboratories',
                         'Intensity RMSE is not a physical mechanism discrimination',
                         'Polynomial extrapolation may be numerically poor or unstable',
                         'No external dataset or newly acquired measurements'],
          'novelty_certified':False,'network_calls':0,'model_calls':0}
    out.mkdir(parents=True)
    freeze(out/PLAN,plan)
    (out/PIN).write_text(old.digest_file(out/PLAN)+'\n',encoding='ascii')
    return {'status':'M14C4C_NEW_SECONDARY_ANALYSIS_PREREGISTERED','out_dir':str(out),'source_c4b_status':previous['status'],'network_calls':0,'new_paid_calls':0}

def pre(out):
    plan=load(out/PLAN)
    if (out/PIN).read_text().strip()!=old.digest_file(out/PLAN):raise ValueError('M14C4C_PLAN_HASH_CHANGED')
    if old.digest_file(Path(c.__file__))!=plan['c4c_core_sha256'] or old.digest_file(Path(__file__))!=plan['c4c_runner_sha256']:
        raise ValueError('M14C4C_CODE_CHANGED')
    prior=Path(plan['c4b_path']);br.pre(prior)
    for name,digest in plan['c4b_artifact_hashes'].items():
        if old.digest_file(prior/name)!=digest:raise ValueError('M14C4C_UPSTREAM_CHANGED: '+name)
    src=Path(plan['c4a_path'])
    if old.digest_file(src/ar.RAW)!=plan['source_csv_sha256']:raise ValueError('M14C4C_SOURCE_CHANGED')
    return plan,src,prior

def calc(out):
    plan,src,prior=pre(out)
    parsed,entries=b.data(src/ar.RAW)
    protocols=load(prior/br.SCREEN)['protocols']
    folds,excluded=c.eligible_splits(entries)
    extra={'schema':c.SCHEMA,'source_csv_sha256':plan['source_csv_sha256'],
           'usable_rows':len(entries),
           'protocols_examined':len(protocols),
           'prior_C4B_no_valid_folds':True,
           'predeclared_no_intensity_based_filtering':True,
           'author_train_and_test_not_mixed_in_model_fitting':True,
           'out_of_range_explicit':True,
           'observed_metadata_before_new_study':True}
    result=c.summarize(folds,excluded,protocols)
    return extra,folds,excluded,result

def screen(out):
    screen,folds,_,_=calc(out)
    freeze(out/SCREEN,screen)
    return {'status':'M14C4C_NEW_STUDY_METADATA_SCREEN','potential_range_shift_folds':len(folds),'network_calls':0,'new_paid_calls':0}

def evaluate(out):
    meta,folds,excluded,_=calc(out)
    if (out/SCREEN).read_bytes()!=old.canonical(meta):raise ValueError('M14C4C_METADATA_CHANGED')
    freeze(out/FOLDS,folds);freeze(out/EXCLUSIONS,excluded)
    receipt={'schema':c.SCHEMA,'status':'M14C4C_SECONDARY_OBSERVED_DATA_EVALUATED' if folds else 'M14C4C_ABSTAIN',
             'plan_sha256':old.digest_file(out/PLAN),
             'metadata_sha256':old.digest_file(out/SCREEN),
             'folds_sha256':old.digest_file(out/FOLDS),
             'exclusions_sha256':old.digest_file(out/EXCLUSIONS),
             'original_source_sha256':load(out/PLAN)['source_csv_sha256'],
             'both_lab_and_test_partition_held_out':True,'confirmatory':False,'novelty_certified':False,
             'model_calls':0,'network_calls':0}
    freeze(out/RECEIPT,receipt)
    return {'status':receipt['status'],'evaluated_folds':len(folds),'excluded_folds':len(excluded),'network_calls':0,'new_paid_calls':0}

def decide(out):
    meta,folds,excluded,result=calc(out)
    for n,v in ((SCREEN,meta),(FOLDS,folds),(EXCLUSIONS,excluded)):
        if (out/n).read_bytes()!=old.canonical(v):raise ValueError('M14C4C_REPLAY_DIFFERENCE: '+n)
    receipt=load(out/RECEIPT)
    for n,key in ((PLAN,'plan_sha256'),(SCREEN,'metadata_sha256'),(FOLDS,'folds_sha256'),(EXCLUSIONS,'exclusions_sha256')):
        if old.digest_file(out/n)!=receipt[key]:raise ValueError('M14C4C_RECEIPT_LINEAGE_CHANGED: '+n)
    result['source_evidence_receipt_sha256']=old.digest_file(out/RECEIPT)
    freeze(out/RESULT,result)
    return {'status':result['status'],'evaluated_folds':result['fold_count'],
            'protocols_with_minimum_folds':sum(p['n_evaluated_folds']>=c.MIN_FOLDS for p in result['protocols']),
            'novelty_certified':False,'network_calls':0,'new_paid_calls':0}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--c4b-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    for k in ('screen','evaluate','decide','verify'):
        p=sub.add_parser(k);p.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args(argv)
    actions={'prepare':lambda:prepare(a.c4b_dir,a.out_dir),'screen':lambda:screen(a.out_dir),
             'evaluate':lambda:evaluate(a.out_dir),'decide':lambda:decide(a.out_dir),
             'verify':lambda:{**decide(a.out_dir),'status':'M14C4C_REPRODUCIBILITY_VERIFIED'}}
    result=actions[a.cmd]()
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result

if __name__=='__main__':main()
