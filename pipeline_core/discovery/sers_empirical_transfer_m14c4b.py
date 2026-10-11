"""M14-C4B: secondary empirical analysis with correct concentration/sample roles.

Observational published dataset. Metadata-informed SECONDARY study, not confirmatory
scientific novelty. Hold out *entire* laboratory + author's `test` partition.
Standard library only. Old M14-C4A plan and scripts are not modified.
"""
from __future__ import annotations
from collections import defaultdict
import csv, io, math, statistics
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as old

SCHEMA = 'm14c4b-metadata-informed-transfer-evaluation-v1'
PROTOCOL_FIELDS = ('method', 'substrate', 'laser')
TYPE_INDEX = 4
SAMPLE_INDEX = 3
MIN_PROTOCOL_LABS = 3
MIN_TRAIN_SAMPLES = 12
MIN_TEST_SAMPLES = 3
MIN_TRAIN_LEVELS = 3
MIN_TEST_LEVELS = 2
MIN_VALID_FOLDS = 3
MODELS = (('H0_CONSTANT', 0), ('H1_LINEAR_LOG_CONC', 1), ('H2_QUADRATIC_LOG_CONC', 2))

def prot(row): return tuple(row['cohort'][:3])
def part(row): return str(row['cohort'][TYPE_INDEX]).strip().lower()
def stable_key(protocol): return old.sha(old.canonical(list(protocol)))[:20]

def data(source):
    parsed = old.parse_csv(source)
    out=[]
    for row in parsed['rows']:
        p=part(row)
        if p in ('train','test'):
            out.append(row)
    return parsed, out

def metadata_screen(parsed, entries):
    groups=defaultdict(list)
    for row in entries: groups[prot(row)].append(row)
    result=[]
    for protocol, group in sorted(groups.items()):
        train=[r for r in group if part(r)=='train']
        test=[r for r in group if part(r)=='test']
        trainlabs={r['lab'] for r in train};testlabs={r['lab'] for r in test}
        both=trainlabs & testlabs
        candidates=[]
        for lab in sorted(both):
            atrain=[r for r in train if r['lab']!=lab]
            atest=[r for r in test if r['lab']==lab]
            reasons=[]
            if len(atrain)<MIN_TRAIN_SAMPLES:reasons.append('TRAIN_ROWS_LT_12')
            if len(atest)<MIN_TEST_SAMPLES:reasons.append('HELDOUT_TEST_ROWS_LT_3')
            if len({r['x'] for r in atrain})<MIN_TRAIN_LEVELS:reasons.append('TRAIN_CONC_LEVELS_LT_3')
            if len({r['x'] for r in atest})<MIN_TEST_LEVELS:reasons.append('HELDOUT_TEST_CONC_LEVELS_LT_2')
            if atrain and atest and (min(r['x'] for r in atest)<min(r['x'] for r in atrain)-1e-8 or max(r['x'] for r in atest)>max(r['x'] for r in atrain)+1e-8):
                reasons.append('HELDOUT_TEST_OUTSIDE_TRAIN_RANGE')
            candidates.append({'heldout_lab':lab,'train_rows':len(atrain),'test_rows':len(atest),
                'train_levels':len({r['x'] for r in atrain}),'test_levels':len({r['x'] for r in atest}),
                'train_labs_excluding_heldout':len({r['lab'] for r in atrain}),'eligible':not reasons,'reasons':reasons})
        eligible=sum(v['eligible'] for v in candidates)
        result.append({'protocol':list(protocol),'protocol_key':stable_key(protocol),'n_rows':len(group),
           'train_rows':len(train),'test_rows':len(test),'all_labs':len({r['lab'] for r in group}),
           'train_labs':len(trainlabs),'test_labs':len(testlabs),'labs_with_both_partitions':len(both),
           'levels_train':len({r['x'] for r in train}),'levels_test':len({r['x'] for r in test}),
           'potential_full_laboratory_holdout_folds':eligible,
           'analysis_eligible':eligible>=MIN_VALID_FOLDS,'fold_screen':candidates})
    return {'schema':SCHEMA,'raw_rows':parsed['raw_rows'],'usable_positive_conc_and_complete_band':len(parsed['rows']),
         'rows_with_published_train_or_test_partition':len(entries),
         'other_partition_rows_excluded':len(parsed['rows'])-len(entries),
         'rows_rejected_during_source_parse':parsed['rejections'],
         'protocol_fields':list(PROTOCOL_FIELDS),'partition_field':'type','non_stratification_field':'sample',
         'protocols':result,'potential_eligible_protocols':sum(v['analysis_eligible'] for v in result),
         'policy':'All eligible protocols predeclared by metadata; not selected by intensity, residuals or accuracy',
         'research_status':'METADATA_INFORMED_SECONDARY_ANALYSIS_NOT_CONFIRMATORY_DISCOVERY'}

def fold_comparison(entries,screen):
    group=defaultdict(list)
    for row in entries:group[stable_key(prot(row))].append(row)
    folds=[]; exclusions=[]
    for profile in screen['protocols']:
        ck=profile['protocol_key']
        if not profile['analysis_eligible']:
            exclusions.append({'protocol_key':ck,'reason':'PROTOCOL_FEWER_THAN_THREE_POSSIBLE_HOLDOUT_FOLDS'});continue
        rows=group[ck]
        for fold in profile['fold_screen']:
            lab=fold['heldout_lab']
            if not fold['eligible']:
                exclusions.append({'protocol_key':ck,'lab':lab,'reasons':fold['reasons']});continue
            train=[r for r in rows if r['lab']!=lab and part(r)=='train']
            test=[r for r in rows if r['lab']==lab and part(r)=='test']
            assert all(r['lab']!=lab and part(r)=='train' for r in train)
            assert all(r['lab']==lab and part(r)=='test' for r in test)
            metrics={}
            for name,degree in MODELS:
                predictor=old.fit(train,degree)
                if predictor is None:
                    metrics={};exclusions.append({'protocol_key':ck,'lab':lab,'reason':'SINGULAR_MODEL_FIT','model':name});break
                rmse=old.rmse([predictor(r['x'])-r['y'] for r in test])
                metrics[name]=rmse
            if len(metrics)!=len(MODELS):continue
            folds.append({'protocol_key':ck,'protocol':list(profile['protocol']),
             'heldout_lab':lab,'fitting_partition':'train','evaluation_partition':'test',
             'heldout_lab_seen_in_fitting':False,'test_partition_seen_in_fitting':False,
             'train_labs':sorted({r['lab'] for r in train}),'train_n':len(train),'test_n':len(test),
             'train_concentration_levels':len({r['x'] for r in train}),'test_concentration_levels':len({r['x'] for r in test}),
             'train_log10_conc_range':[min(r['x'] for r in train),max(r['x'] for r in train)],
             'test_log10_conc_range':[min(r['x'] for r in test),max(r['x'] for r in test)],
             'test_rmse':metrics,'observed_test_intensity_mean':statistics.mean(r['y'] for r in test)})
    return folds,exclusions

def summarize(screen,folds):
    by=defaultdict(list)
    for r in folds:by[r['protocol_key']].append(r)
    outcomes=[]
    for row in screen['protocols']:
        key=row['protocol_key'];fs=by[key];n=len(fs)
        decision='ABSTAIN';status='EVIDENCE_INSUFFICIENT'
        if n>=MIN_VALID_FOLDS:
            linear_better=sum(v['test_rmse']['H1_LINEAR_LOG_CONC'] < .9 * v['test_rmse']['H0_CONSTANT'] for v in fs)
            quadratic_better=sum(v['test_rmse']['H2_QUADRATIC_LOG_CONC'] < .9 * v['test_rmse']['H1_LINEAR_LOG_CONC'] for v in fs)
            need=max(MIN_VALID_FOLDS,math.ceil(.75*n))
            if quadratic_better>=need:
                decision='PROPOSE_MODEL_COMPLEXITY_CHECK';status='QUADRATIC_TRANSFER_SIGNAL_DESCRIPTIVE'
            elif linear_better>=need:
                decision='PROPOSE_CALIBRATION_TRANSFER_AUDIT';status='LINEAR_CONCENTRATION_TRANSFER_SIGNAL_DESCRIPTIVE'
            else:
                decision='REQUEST_INDEPENDENT_CALIBRATION';status='TRANSFER_SIGNAL_WEAK_OR_HETEROGENEOUS'
        else:
            linear_better=quadratic_better=0
        outcomes.append({'protocol_key':key,'protocol':row['protocol'],'heldout_folds_evaluated':n,
         'model_comparison_status':status,'next_action_proposal':decision,
         'linear_better_than_constant_by_ten_percent_labs':linear_better,
         'quadratic_better_than_linear_by_ten_percent_labs':quadratic_better,
         'scientific_question':('What independently measured laboratory, baseline, substrate and batch effects explain concentration-response transferability in this published SERS protocol?'
            if n>=MIN_VALID_FOLDS else 'Which missing concentration/partition supports prevent out-of-laboratory SERS calibration evaluation?'),
         'hypothesis_generation_permitted':False,'scientific_gap_confirmed':False,
         'mechanism_identified':False,'scientific_novelty_certified':False,
         'independent_dataset_confirmation_required':True,
         'is_post_metadata_secondary_design':True})
    return {'schema':SCHEMA,'status':'M14C4B_SECONDARY_EMPIRICAL_EVIDENCE_EVALUATED' if folds else 'M14C4B_ABSTAIN_NO_VALID_EMPIRICAL_FOLDS',
            'protocol_results':outcomes,'protocols_with_evaluated_folds':sum(v['heldout_folds_evaluated']>=MIN_VALID_FOLDS for v in outcomes),
            'heldout_folds':len(folds),'no_new_empirical_observations_created':True,
            'no_independent_new_discovery_certified':True,'new_model_calls':0}
