"""M25: two-condition literature holdout. Add-only experimental runner.

Scientific scope: Gold-nanorod charging in liquid PVP/PEI conditions, NOT Ye/PVA.
Observed outcomes are isolated from the model request. The published article is
public, so this is a protocol-level (NOT pretraining-proof) holdout.
No network during prepare/reveal. Never re-call after saved raw exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

PROTOCOL = 'm25-two-intervention-holdout-v1'
SOURCE_DOI = '10.1038/s41467-025-67130-8'
SOURCE_URL = 'https://www.nature.com/articles/s41467-025-67130-8'


def canon(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'),allow_nan=False).encode()


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(p.read_bytes())


def read(p: Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def put(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def path(p):
    return Path(p).expanduser().resolve()


def public_packet():
    # Deliberately avoids observed numerical outcomes, citation, paper title, or paper fitting coefficients.
    return {
        'domain':'photoinduced charging and small-shift optical readout of solvated gold nanorods',
        'training_information_status':'SOURCE_REPORTED_RELATIONS_AND_MATERIAL_CONDITIONS__NOT_MEASURED_TARGETS',
        'shared_laws':[
            'For a nanocapacitor under matched photo-driving voltage, maximum stored excess charge is proportional to capacitance.',
            'For a matched nanorod optical geometry in a small-shift regime, fractional LSPR frequency shift is proportional to excess charge divided by baseline conduction-electron count.',
            'Capacitance dependence on dielectric environment and on exposed area must be justified using additional assumptions; they are not universal across interfaces.'
        ],
        'conditions':[
            {'id':'ligand','question':'Predict ratio of maximum fractional LSPR shifts: PVP-stabilized divided by PEI-stabilized rods.',
             'facts':{'relative_permittivity_PVP_approx':7.7,'relative_permittivity_PEI_approx':3.3,
                      'same_nominal_rod_size':True,'nominal_transverse_diameter_nm':10,
                      'same_nominal_illumination_mW':820},
             'possible_bridge':'Assume equal effective coating geometry, comparable photo-driving voltage, and effective capacitance proportional to ligand-layer permittivity. Verify whether these assumptions suffice for a numerical prediction.'},
            {'id':'size','question':'Predict ratio of maximum fractional LSPR shifts: small rods divided by large rods.',
             'facts':{'small_transverse_diameter_nm':10,'large_transverse_diameter_nm':25,
                      'same_aspect_ratio':True,'same_ligand_PVP':True,'same_nominal_illumination_mW':820},
             'possible_bridge':'Assume geometrically similar rods, baseline conduction-electron number proportional to volume and accumulated charge proportional to surface area, with similar photo-driving voltage.'}
        ],
        'instructions':'For each condition predict a dimensionless ratio or explicitly abstain. Do not conflate optical LSPR shifts with Raman blinking, and do not treat same nominal settings as proof of equal absorbed power. State the conditional prediction and at least one credible confounder.'
    }


def sealed_holdout():
    # Direct numerical statements from Stete et al. Fig. 7, NOT synthesized measurements.
    return {
        'provenance':{'doi':SOURCE_DOI,'url':SOURCE_URL,'title':'Capacitive photocharging of gold nanorods',
                      'source_location':'Results / Parameters influencing the charge accumulation / Figure 7',
                      'source_type':'ARTICLE_TEXT_NUMERIC_VALUES_NOT_RAW_ZENODO_TRACES',
                      'notes':'Both observed ratios were reported for the same paper, NOT GNR-PVA-dye.'},
        'targets':{
            'ligand':{'ratio_definition':'PVP relative maximum shift / PEI relative maximum shift',
                      'observed_ratio':2.5,'reported_PEI_max_relative_shift_percent':0.28,
                      'statement':'Source reports a factor 2.5 higher PVP response than PEI under matched nominal conditions.'},
            'size':{'ratio_definition':'10 nm rod relative maximum shift / 25 nm rod relative maximum shift',
                    'observed_ratio':0.70/0.24,'reported_small_max_relative_shift_percent':0.70,
                    'reported_large_max_relative_shift_percent':0.24,
                    'statement':'Source reports 0.70% for smaller and 0.24% for larger transverse diameter rods.'}
        },
        'warning':'Source ratios are experimental reports with approximations; not held-out numerical traces or an unseen paper.'
    }

SYSTEM = '''You are a research scientist assessing real gold nanorod photocharging physics.
Derive quantitative, conditional predictions from the supplied physical relationships
and intervention-specific inputs. You have NOT been provided the measured outcomes.
No browser, literature lookup, or auxiliary tools are available in this call.
Return exactly one JSON object, no Markdown. Schema:
{"chosen_primary_id":"ligand"|"size", "research_question":string,
 "predictions":[{"id":"ligand"|"size","predicted_ratio":positive_number_or_null,
 "derivation":string,"assumptions":string,"alternative_mechanism":string,
 "confidence":"CONDITIONAL"|"ABSTAIN"}],
 "cross_system_transfer_limit":string,"what_result_would_revise_model":string}
Exactly one prediction per condition, no duplicates. A null prediction must use ABSTAIN.
All non-null predictions must have CONFIDENCE CONDITIONAL and clear assumptions.
Do not represent model assumptions as measurements or imply applicability to polymer-embedded
Raman blinking. Mathematical identities alone and arbitrary parameter assignments are not successes.
'''


def validate_prediction(obj):
    if not isinstance(obj,dict) or set(obj)!={'chosen_primary_id','research_question','predictions','cross_system_transfer_limit','what_result_would_revise_model'}:
        raise ValueError('INVALID_PREDICTION_FIELDS')
    if obj['chosen_primary_id'] not in ('ligand','size'):
        raise ValueError('BAD_PRIMARY_ID')
    for k in ('research_question','cross_system_transfer_limit','what_result_would_revise_model'):
        if not isinstance(obj[k],str) or len(obj[k].strip())<15:
            raise ValueError('MISSING_SCIENTIFIC_EXPLANATION_'+k)
    if not isinstance(obj['predictions'],list) or len(obj['predictions'])!=2:
        raise ValueError('TWO_CONDITIONS_REQUIRED')
    seen=set()
    for row in obj['predictions']:
        if not isinstance(row,dict) or set(row)!={'id','predicted_ratio','derivation','assumptions','alternative_mechanism','confidence'}:
            raise ValueError('INVALID_CONDITION_RESULT')
        k=row['id']
        if k not in ('ligand','size') or k in seen:
            raise ValueError('DUPLICATE_OR_UNKNOWN_CONDITION')
        seen.add(k)
        for key in ('derivation','assumptions','alternative_mechanism'):
            if not isinstance(row[key],str) or len(row[key].strip())<15:
                raise ValueError('MISSING_EXPLANATION_'+key)
        y=row['predicted_ratio']
        if y is None:
            if row['confidence']!='ABSTAIN':raise ValueError('NULL_MUST_ABSTAIN')
        else:
            if type(y) not in (float,int) or not math.isfinite(y) or y<=0 or y>=100:
                raise ValueError('BAD_FINITE_RATIO')
            if row['confidence']!='CONDITIONAL':raise ValueError('NUMBER_MUST_BE_CONDITIONAL')
    return obj


def prior_settings(m24):
    from scripts.discovery import run_m24_feasible_domain as m24mod
    status=m24mod.verify(m24)
    if status['proposal']!='READY': raise ValueError('PRIOR_M24_NOT_READY')
    lock=read(m24/'lock.json')
    return {'model':lock['model'],'temperature':lock['temperature'],
            'max_tokens':max(3500,lock['max_tokens'])}, {str(m24/'propose/normalized.json'):sha_file(m24/'propose/normalized.json'),
                             str(m24/'propose/feasibility.json'):sha_file(m24/'propose/feasibility.json')}


def prepare(out,m24=None,model=None):
    out=path(out)
    if out.exists():raise FileExistsError('FRESH_RUN_PATH_REQUIRED')
    origins={}
    if m24:
        cfg,origins=prior_settings(path(m24))
    else:
        if not model:raise ValueError('PROVIDE_M24_OR_MODEL')
        cfg={'model':model,'temperature':0.2,'max_tokens':5000}
    out.mkdir(parents=True)
    try:
        put(out/'public_input.json',public_packet())
        put(out/'private_holdout.json',sealed_holdout())
        put(out/'lock.json',{'protocol':PROTOCOL,'script_sha256':sha_file(path(__file__)),
            'public_sha256':sha_file(out/'public_input.json'),
            'private_sha256':sha_file(out/'private_holdout.json'),
            'research_origins':origins,'model_settings':cfg,
            'scientific_truth_not_certified':True,'holdout_measured_targets_sent_to_model':False})
        verify(out)
    except BaseException:
        shutil.rmtree(out);raise
    return {'status':'M25_PREPARED_NO_CALL','two_measured_holdouts_sealed':True,
            'prior_runs_modified':False,'paper_source_data_traces_used':False,
            'out':str(out)}


def check(out):
    out=path(out)
    lock=read(out/'lock.json')
    if lock['protocol']!=PROTOCOL or lock['script_sha256']!=sha_file(path(__file__)):
        raise ValueError('M25_EXECUTOR_CHANGED')
    if lock['public_sha256']!=sha_file(out/'public_input.json') or lock['private_sha256']!=sha_file(out/'private_holdout.json'):
        raise ValueError('M25_SOURCE_OR_HOLDOUT_CHANGED')
    for src,h in lock['research_origins'].items():
        if sha_file(Path(src))!=h:raise ValueError('PRIOR_ORIGIN_CHANGED')
    if read(out/'public_input.json')!=public_packet() or read(out/'private_holdout.json')!=sealed_holdout():
        raise ValueError('PACKET_IS_NOT_PREREGISTERED')
    return out,lock


def request(out):
    out,lock=check(out)
    return {'model':lock['model_settings']['model'],
       'temperature':lock['model_settings']['temperature'],
       'max_tokens':lock['model_settings']['max_tokens'],
       'messages':[{'role':'system','content':SYSTEM},
                   {'role':'user','content':json.dumps(read(out/'public_input.json'),ensure_ascii=False,sort_keys=True)}]}


def verify(out):
    out,lock=check(out)
    pd=out/'propose'
    if not pd.exists():
        if (out/'evaluation.json').exists():raise ValueError('EVALUATED_WITHOUT_PREDICTION')
        return {'status':'M25_VERIFIED_NO_CALL','prediction':'NOT_STARTED','evaluation':'NOT_STARTED'}
    req=request(out)
    if read(pd/'request.json')!=req or read(pd/'attempt.json')['request_sha256']!=sha_bytes(canon(req)):
        raise ValueError('M25_REQUEST_CHANGED')
    if not (pd/'raw_response.json').exists():
        return {'status':'M25_VERIFIED_NO_CALL','prediction':'ATTEMPT_NO_RAW_DO_NOT_RETRY','evaluation':'NOT_STARTED'}
    if not (pd/'prediction.json').exists() or not (pd/'receipt.json').exists():
        return {'status':'M25_VERIFIED_NO_CALL','prediction':'RAW_SAVED_RECOVER_NO_RETRY','evaluation':'NOT_STARTED'}
    p=validate_prediction(json.loads(read(pd/'raw_response.json')['raw_text']))
    if read(pd/'prediction.json')!=p:raise ValueError('MODEL_PREDICTION_MUTATED')
    receipt={'request_sha256':sha_bytes(canon(req)),
             'raw_sha256':sha_file(pd/'raw_response.json'),
             'prediction_sha256':sha_file(pd/'prediction.json')}
    if read(pd/'receipt.json')!=receipt:raise ValueError('M25_RECEIPT_CHANGED')
    if (out/'evaluation.json').exists():
        expected=score(p,read(out/'private_holdout.json'),receipt)
        if read(out/'evaluation.json')!=expected:raise ValueError('EVALUATION_CHANGED')
        return {'status':'M25_VERIFIED_NO_CALL','prediction':'READY','evaluation':'REVEALED'}
    return {'status':'M25_VERIFIED_NO_CALL','prediction':'READY','evaluation':'SEALED'}


def preview(out):
    state=verify(out)
    if state['prediction']!='NOT_STARTED':raise ValueError('NOT_A_FRESH_PREVIEW')
    return {'status':'M25_PREVIEW_NO_CALL','request_sha256':sha_bytes(canon(request(out))),
            'next_call_is_paid':True,'public_inputs_only':True,
            'holdout_ratio_values_not_in_request':True}


def recover(out):
    out=path(out)
    state=verify(out)
    if state['prediction'] not in ('RAW_SAVED_RECOVER_NO_RETRY','READY'):
        raise ValueError('NO_SAVED_RESPONSE_TO_RECOVER')
    pd=out/'propose'
    original_hash=sha_file(pd/'raw_response.json')
    p=validate_prediction(json.loads(read(pd/'raw_response.json')['raw_text']))
    if not (pd/'prediction.json').exists():put(pd/'prediction.json',p)
    elif read(pd/'prediction.json')!=p:raise ValueError('PREDICTION_CONFLICT')
    receipt={'request_sha256':sha_bytes(canon(request(out))),
             'raw_sha256':sha_file(pd/'raw_response.json'),
             'prediction_sha256':sha_file(pd/'prediction.json')}
    if not (pd/'receipt.json').exists():put(pd/'receipt.json',receipt)
    elif read(pd/'receipt.json')!=receipt:raise ValueError('RECEIPT_CONFLICT')
    if sha_file(pd/'raw_response.json')!=original_hash:raise ValueError('PAID_RAW_CHANGED')
    return {'status':'M25_SAVED_PREDICTION_RECOVERED_NO_REPAY','verification':verify(out)}


def propose(out,allow_paid=False,backend=None):
    prev=preview(out)
    from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base
    base._require_paid(backend,allow_paid,1)
    out=path(out);pd=out/'propose';pd.mkdir()
    req=request(out)
    put(pd/'request.json',req)
    put(pd/'attempt.json',{'request_sha256':prev['request_sha256'],
                          'status':'BILLING_UNKNOWN_UNTIL_RAW_SAVED'})
    raw,usage,rid=base._invoke(req,backend)
    put(pd/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:done=recover(out)
    except Exception as exc:raise RuntimeError('M25_SAVED_RAW_DO_NOT_REPAY_USE_RECOVER') from exc
    return {'new_paid_calls':int(backend is None),**done}


def score(prediction,holdout,receipt):
    found={p['id']:p for p in prediction['predictions']}
    results={}
    for k,obs in holdout['targets'].items():
        predicted=found[k]['predicted_ratio']
        actual=obs['observed_ratio']
        if predicted is None:
            results[k]={'prediction_status':'ABSTAIN','reported_observed_ratio':actual,
                        'predicted_ratio':None,'absolute_percentage_error':None,
                        'null_ratio_1_absolute_percentage_error':abs(1-actual)/actual,
                        'beats_null_ratio_1':False}
        else:
            err=abs(predicted-actual)/actual
            baseline=abs(1-actual)/actual
            results[k]={'prediction_status':'PREDICTED','reported_observed_ratio':actual,
                        'predicted_ratio':predicted,'absolute_percentage_error':err,
                        'null_ratio_1_absolute_percentage_error':baseline,
                        'beats_null_ratio_1':err<baseline}
    predicted_rows=[v for v in results.values() if v['prediction_status']=='PREDICTED']
    return {'status':'M25_EVALUATED_AGAINST_PUBLISHED_FIG7_NUMBERS',
      'paper':holdout['provenance'],
      'method':'Single held-out observation ratio per intervention, conditional laws exposed. No training on measured targets.',
      'holdout':results,
      'mean_absolute_percentage_error_over_nonabstained':sum(r['absolute_percentage_error'] for r in predicted_rows)/len(predicted_rows) if predicted_rows else None,
      'tested_interventions':len(predicted_rows),'both_beats_no_change_baseline':len(predicted_rows)==2 and all(r['beats_null_ratio_1'] for r in predicted_rows),
      'scientific_evaluation':'TEST_OF_SOURCE_TO_PREDICTION_TRANSFER_NOT_DISCOVERY_OR_CROSS_SYSTEM_VALIDATION',
      'limitations':['Only two published summary measurements; not withheld raw time series.',
         'Public paper can appear in model pretraining; isolation is from run request only.',
         'Approximate dielectric constants and geometry; matched driving voltage is not measured.',
         'Agreement may reflect supplied scaling premises, not independent mechanism discovery.',
         'Cannot establish applicability to Ye solid PVA/dye SERS blinking.'],
      'prediction_receipt':receipt}


def reveal(out):
    state=verify(out)
    if state['prediction']!='READY' or state['evaluation']!='SEALED':
        raise ValueError('MUST_HAVE_COMMITTED_PREDICTION_AND_NO_PRIOR_REVEAL')
    out=path(out)
    ev=score(read(out/'propose/prediction.json'),read(out/'private_holdout.json'),read(out/'propose/receipt.json'))
    put(out/'evaluation.json',ev)
    verify(out)
    return {'status':'M25_HOLDOUT_REVEALED_NO_PAID_CALL',
            'tested_interventions':ev['tested_interventions'],
            'mean_absolute_percentage_error':ev['mean_absolute_percentage_error_over_nonabstained'],
            'both_beats_baseline':ev['both_beats_no_change_baseline']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for c in ('prepare','preview','propose','recover','verify','reveal'):
        p=sub.add_parser(c);p.add_argument('--out',required=True)
        if c=='prepare':
            p.add_argument('--m24');p.add_argument('--model')
        if c=='propose':p.add_argument('--allow-paid',action='store_true')
    a=ap.parse_args()
    if a.cmd=='prepare':result=prepare(a.out,a.m24,a.model)
    elif a.cmd=='preview':result=preview(a.out)
    elif a.cmd=='propose':result=propose(a.out,a.allow_paid)
    elif a.cmd=='recover':result=recover(a.out)
    elif a.cmd=='verify':result=verify(a.out)
    else:result=reveal(a.out)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
