"""M14-C4C: SECONDARY cross-lab, author-test, out-of-range empirical calibration evaluation.

No new scientific discovery, experimental intervention or physical-mechanism inference.
The dataset and C4B exclusion results were examined when designing this analysis.
No network/provider calls, and no mutation of C4A/C4B artifacts.
"""
from collections import defaultdict
import math, statistics
from pipeline_core.discovery import sers_empirical_discovery_m14c4 as old
from pipeline_core.discovery import sers_empirical_transfer_m14c4b as prev

SCHEMA = 'm14c4c-post-abstention-extrapolation-transfer-v1'
MIN_FOLDS = 3
MODELS = prev.MODELS
EPS = 1e-8

def eligible_splits(rows):
    groups=defaultdict(list)
    for row in rows:groups[prev.stable_key(prev.prot(row))].append(row)
    records=[];excluded=[]
    for key in sorted(groups):
        group=groups[key]
        labs=sorted({r['lab'] for r in group if prev.part(r)=='train'} &
                    {r['lab'] for r in group if prev.part(r)=='test'})
        for lab in labs:
            tr=[r for r in group if r['lab']!=lab and prev.part(r)=='train']
            te=[r for r in group if r['lab']==lab and prev.part(r)=='test']
            why=[]
            if len(tr)<prev.MIN_TRAIN_SAMPLES:why.append('TRAIN_ROWS_LT_12')
            if len(te)<prev.MIN_TEST_SAMPLES:why.append('TEST_ROWS_LT_3')
            if len({r['x'] for r in tr})<prev.MIN_TRAIN_LEVELS:why.append('TRAIN_LEVELS_LT_3')
            if len({r['x'] for r in te})<prev.MIN_TEST_LEVELS:why.append('TEST_LEVELS_LT_2')
            if len({r['lab'] for r in tr})<2:why.append('TRAIN_LABS_LT_2')
            if not tr or not te:
                excluded.append({'protocol_key':key,'lab':lab,'reasons':why});continue
            lower=min(r['x'] for r in tr);upper=max(r['x'] for r in tr)
            low=[r for r in te if r['x']<lower-EPS]
            high=[r for r in te if r['x']>upper+EPS]
            inside=[r for r in te if r not in low and r not in high]
            if not low and not high:why.append('NO_OUT_OF_RANGE_TEST_ROWS')
            if why:
                excluded.append({'protocol_key':key,'lab':lab,'reasons':why});continue
            metrics={}
            for name,degree in MODELS:
                f=old.fit(tr,degree)
                if f is None or not all(math.isfinite(f(r['x'])) for r in te):
                    why.append('NON_IDENTIFIABLE_OR_NONFINITE_FIT_'+name);break
                def summary(group):
                    return {'n':len(group),'rmse':old.rmse([f(r['x'])-r['y'] for r in group]) if group else None,
                            'mean_observed_band_area':statistics.mean(r['y'] for r in group) if group else None}
                metrics[name]={'all_test':summary(te),'extrapolative':summary(low+high),
                               'below_train_range':summary(low),'above_train_range':summary(high),
                               'inside_train_range':summary(inside)}
            if why:
                excluded.append({'protocol_key':key,'lab':lab,'reasons':why});continue
            records.append({'protocol_key':key,'protocol':list(prev.prot(tr[0])),
                'heldout_lab':lab,'train_labs':sorted({r['lab'] for r in tr}),
                'model_fit_partition':'other_labs_author_train_only',
                'evaluation_partition':'heldout_lab_author_test_only',
                'heldout_lab_used_to_fit':False,'author_test_used_to_fit':False,
                'n_train':len(tr),'n_test':len(te),'n_test_below':len(low),
                'n_test_above':len(high),'n_test_in_range':len(inside),
                'train_log10_range':[lower,upper],
                'test_log10_range':[min(r['x'] for r in te),max(r['x'] for r in te)],
                'max_log10_extrapolation_below':max([lower-r['x'] for r in low],default=0.),
                'max_log10_extrapolation_above':max([r['x']-upper for r in high],default=0.),
                'predictive_errors':metrics})
    return records,excluded

def summarize(records, excluded, known_protocols):
    groups=defaultdict(list)
    for r in records:groups[r['protocol_key']].append(r)
    results=[]
    for p in known_protocols:
        key=p['protocol_key'];fs=groups[key]
        status='SECONDARY_OUT_OF_RANGE_TRANSFER_EVALUATED' if len(fs)>=MIN_FOLDS else 'ABSTAIN_TOO_FEW_VALID_HOLDOUT_LABS'
        errors={name:statistics.median([r['predictive_errors'][name]['extrapolative']['rmse'] for r in fs]) if fs else None for name,_ in MODELS}
        decisions=[]
        if len(fs)>=MIN_FOLDS:
            decisions=['REQUEST_INDEPENDENT_RANGE_SHIFT_VALIDATION', 'EVALUATE_UNMEASURED_INSTRUMENT_AND_BATCH_EFFECTS']
        else: decisions=['SUSPEND_AND_DISCOVER_OTHER_DATA_SOURCE']
        results.append({'protocol':p['protocol'],'protocol_key':key,'n_evaluated_folds':len(fs),'status':status,
           'out_of_range_rows':sum(r['n_test_below']+r['n_test_above'] for r in fs),
           'below_rows':sum(r['n_test_below'] for r in fs),
           'above_rows':sum(r['n_test_above'] for r in fs),
           'median_out_of_range_band_area_rmse_by_model':errors,'next_actions':decisions,
           'candidate_problem':('Do established adenine SERS calibration models generalize across labs to different test concentration ranges, and which independent calibration factors explain failures?' if fs else None),
           'research_status':'PREVIOUSLY_STUDIED_SERS_TRANSFERABILITY_NOT_A_NEW_PHYSICAL_GAP',
           'physical_mechanism_identified':False,'novelty_certified':False,
           'no_automatic_hypothesis_generation':True})
    return {'schema':SCHEMA,'status':'M14C4C_SECONDARY_EXTRAPOLATION_EVIDENCE' if records else 'M14C4C_ABSTAIN_NO_VALID_FOLDS',
        'fold_count':len(records),'excluded_lab_fold_count':len(excluded),'protocols':results,
        'raw_measurements_from_published_work':True,
        'independent_new_observation':False,'independent_novelty_assessment':False,
        'model_comparison_is_predictive_not_physical_mechanism_test':True,
        'design_informed_by_C4B_all_folds_abstained':True,
        'no_claim_of_generalization_to_other_datasets':True,
        'scientific_novelty_certified':False,'new_paid_calls':0,'network_calls':0}
