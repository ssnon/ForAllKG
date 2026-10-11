"""M14-C3 conditional identifiability challenge for the C2 SERS/SEIRA successor.

All outputs are exactly computed SYNTHETIC counterexamples, never data or
confirmation of the successor's physical premise. No LLM, network or randomness.
"""
from __future__ import annotations
import csv, io, json, math, hashlib

SCHEMA='m14c3-conditional-identifiability-challenge-v1'
POTENTIAL=(-1.,-.5,0.,.5,1.)
BLOCKER=(0,1)
BOUNDS=(.02,.05,.10,.20,.40,.80)
MODALITIES=('SERS','SEIRA')
# SYNTHETIC, pre-declared example model parameters. NOT inferred from observations.
RESPONSES={'SERS':(1.70,.72),'SEIRA':(.60,1.35)}

def canonical(data):
    return (json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')

def sha(data):return hashlib.sha256(data).hexdigest()

def _sigmoid(t):return 1/(1+math.exp(-t))

def grid():
    return [(v,b) for v in POTENTIAL for b in BLOCKER]

def coverage(v,b):return .85+.05*v-.10*b

def fraction(v,b):return _sigmoid(-.35+1.2*v-1.15*b)

def _effective_two_population(modality,v,b):
    a,c=RESPONSES[modality];f=fraction(v,b)
    return a*f+c*(1-f)

def _baseline(modality):
    return _effective_two_population(modality,0.,0)

def calculate():
    """Compute two-pop signal, flexible one-pop witness and negative control.

    Flexible one-pop q_m(v,b) is *exactly* the effective response of the
    synthetic two-pop model, so model matching is algebraic by construction.
    This establishes non-uniqueness when tensor/optical response is unconstrained;
    it does NOT demonstrate a physically calibrated one-pop alternative.
    """
    rows=[]
    for v,b in grid():
        theta=coverage(v,b)
        if theta<=0:raise ValueError('synthetic coverage invalid')
        for modality in MODALITIES:
            two_gain=_effective_two_population(modality,v,b)
            y_two=theta*two_gain
            # Single chemical population with free modality/context response.
            single_gain=two_gain
            y_null=theta*single_gain
            # Negative control: a genuine single-pop with small, calibrated drift.
            control_gain=_baseline(modality)+(.018*v+.008*b if modality=='SERS' else -.016*v+.009*b)
            rows.append({'potential_normalized':v,'site_blocker':b,'modality':modality,
                'synthetic_coverage':theta,'synthetic_two_pop_fraction_A':fraction(v,b),
                'two_population_intensity':y_two,
                'single_population_flexible_response':single_gain,
                'single_population_flexible_intensity':y_null,
                'single_population_negative_control_intensity':theta*control_gain,
                'two_population_effective_response_deviation':abs(two_gain-_baseline(modality)),
                'negative_control_response_deviation':abs(control_gain-_baseline(modality))})
    assert len(rows)==len(POTENTIAL)*len(BLOCKER)*len(MODALITIES)
    # Exact constructive equivalence, not an empirical inference.
    max_equivalence=max(abs(x['two_population_intensity']-x['single_population_flexible_intensity']) for x in rows)
    if max_equivalence>1e-12:raise RuntimeError('analytic counterexample does not reproduce')
    sweep=[]
    for bound in BOUNDS:
        for dataset,field in (('synthetic_two_population','two_population_effective_response_deviation'),
                              ('synthetic_one_population_negative_control','negative_control_response_deviation')):
            deviations=[r[field] for r in rows]
            failures=sum(v>bound+1e-12 for v in deviations)
            sweep.append({'dataset':dataset,'independently_calibrated_max_response_deviation_required':bound,
                'violations_if_that_bound_were_externally_verified':failures,
                'maximum_required_response_drift':max(deviations),
                'identification_conditional_on_real_calibration':True})
    expected={'schema':SCHEMA,'status':'M14C3_SYNTHETIC_COMPETING_MODELS_COMPUTED',
      'observational_contexts':len(grid()),'signal_rows':len(rows),
      'flexible_one_population_max_abs_residual':max_equivalence,
      'flexible_one_population_null_excluded':False,
      'strong_null_identification':'NOT_IDENTIFIED_WITH_UNMEASURED_CONTEXT_DEPENDENT_RESPONSE',
      'synthetic_negative_control_passes_0_05_bound':
          not any(r['negative_control_response_deviation']>.05+1e-12 for r in rows),
      'conditional_distinguishability_only_if_response_tensor_drift_calibrated':True,
      'assumed_bounds_not_measurements':True,'measured_data_rows':0,
      'literature_novelty_certified':False,'physical_mechanism_confirmed':False,
      'future_evidence_needed':[
         'Independently constrain SERS and SEIRA orientation/selection-rule tensor drift under potential and blocker changes',
         'Measure site-selective surface occupancy beyond total coverage; characterize whether blocker changes chemical and optical responses',
         'Calibrate wavelength-dependent local enhancement and water-dependent optical response in matched environments',
         'Preregister held-out isotope and site-blocking predictions under calibrated nuisance uncertainty']}
    return rows,sweep,expected

def rows_csv(rows):
    if not rows:raise ValueError('empty data')
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]))
    w.writeheader();w.writerows(rows);return s.getvalue().encode('utf-8')
