"""M14-C: controlled, numerical, falsification-oriented SERS model challenges.

A synthetic benchmark is NOT natural evidence or a demonstration of novel physics.
Results delimit identifiable observables under preregistered model assumptions.
No network, external models, randomness or paid providers are used.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from typing import Any

SCHEMA = 'm14c-executed-computational-challenge-v1'
CASE_PHASE = 'LINEAR_PHASE_CLASSICAL_COUNTERMODEL'
CASE_INJECTION = 'INJECTION_THERMAL_CONFOUNDING'
CASE_BY_LANE = {'MECHANISM_SEARCH': CASE_PHASE, 'OPEN_FRONTIER': CASE_INJECTION}
STATUS_PHASE = 'CLASSICAL_NULL_OBSERVATIONALLY_EQUIVALENT_IN_TESTED_LINEAR_OBSERVABLES'
STATUS_INJECTION = 'RESTRICTED_NULL_DIFFERENTIATED_EXPANDED_THERMAL_NULL_NOT_EXCLUDED'
FIELDNAMES = ('case_id','split','condition','intervention','input_value','null_prediction',
              'hypothesis_prediction','abs_difference','countermodel_prediction','countermodel_abs_error')


def canonical(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(case: str, split: str, condition: str, intervention: str, value: float,
            null: float, candidate: float, counter: float) -> dict:
    vals=(value,null,candidate,counter)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('Nonfinite numerical value')
    return {'case_id':case,'split':split,'condition':condition,'intervention':intervention,
            'input_value':f'{value:.9g}','null_prediction':f'{null:.12g}',
            'hypothesis_prediction':f'{candidate:.12g}',
            'abs_difference':f'{abs(candidate-null):.12g}',
            'countermodel_prediction':f'{counter:.12g}',
            'countermodel_abs_error':f'{abs(candidate-counter):.12g}'}


def phase_probe() -> tuple[list[dict],dict]:
    """A classical 2-state linear oscillator is observationally equivalent to a
    complex coherence amplitude in chosen *linear* measured channels.

    z(t) = exp((-gamma+i*omega)t). Re/Im(z) are real classical states
    x'=-gamma x - omega*y; y'=omega*x-gamma*y. Identical outputs for all
    linear readouts c*x+d*y. This is constructive, not a quantum-vs-classical
    general theorem about higher-order observables.
    """
    rows=[]
    for detuning in (-0.8,-0.3,0.2,0.65):
        omega=1.6+detuning
        for t_index in range(1,61):
            t=t_index/20.0
            decay=math.exp(-0.35*t)
            real=decay*math.cos(omega*t)
            imag=decay*math.sin(omega*t)
            for name,(a,b) in {'in_phase':(1.0,0.0),'quadrature':(0.0,1.0),
                                'mixed_readout':(0.6,-0.8)}.items():
                coherent=a*real+b*imag
                classical=a*real+b*imag  # same dynamical solution, different interpretation
                rows.append(_record(CASE_PHASE,'held_out_analytical',f'detuning={detuning};t={t}',
                                    name,detuning,classical,coherent,classical))
    delta=max(float(r['abs_difference']) for r in rows)
    summary={'case_id':CASE_PHASE,'outcome':STATUS_PHASE,'data_origin':'DETERMINISTIC_ANALYTICAL_SYNTHETIC',
             'n_observations':len(rows),'detunings':[-0.8,-0.3,0.2,0.65],
             'linear_readouts':['in_phase','quadrature','mixed_readout'],
             'max_abs_residual_vs_classical_hidden_oscillator':delta,
             'preregistered_equivalence_tolerance':1e-12,
             'equivalent_under_selected_observables':delta<=1e-12,
             'can_certify_quantum_coherence':False,
             'external_empirical_validation':False,
             'limitation':'The mathematical countermodel matches linear phase readouts by construction; nonlinear and calibrated quantum correlations were not tested.'}
    return rows,summary


def thermal_rate(t1: float,t2: float) -> float:
    """Dimensionless two-hotspot Arrhenius index, fixed coverage and weights."""
    if min(t1,t2)<250 or max(t1,t2)>400:
        raise ValueError('Temperature outside pre-registered demonstrator range')
    activation_kelvin=0.32/8.617333262145e-5
    return 0.65*math.exp(activation_kelvin*(1/310-1/t1)) + \
           0.35*math.exp(activation_kelvin*(1/310-1/t2))


def find_thermal_shift(rate_target:float,t1:float,t2:float) -> float:
    """An unmeasured common hotspot-temperature shift can explain injection signal."""
    lo,hi=0.,10.
    if thermal_rate(t1+hi,t2+hi)<rate_target:
        raise ValueError('No confounder within preregistered 10-K range')
    for _ in range(75):
        mid=(lo+hi)/2
        if thermal_rate(t1+mid,t2+mid)<rate_target:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


def injection_probe() -> tuple[list[dict],dict]:
    """A deliberately synthetic causal-identification challenge.

    Two-temperature thermal null: R_th(T1,T2).
    Restricted alternative: R=R_th + beta*injection, beta=0.12.
    Independent injection intervention is a *simulation assumption*, not an
    empirical demonstration that it can be isolated in an actual nanogap.
    Allowing unobserved barrier-dependent hotspot temperature permits thermal
    null to fit the synthetic injection response exactly.
    """
    beta=0.12
    train=(0.25,0.65,1.05)
    holdout=(0.45,0.85,1.25)
    rows=[]; shifts=[]; differences=[]
    for split,powers in [('training',train),('held_out',holdout)]:
        for power in powers:
            t1=299+6*power
            t2=302+8*power
            null=thermal_rate(t1,t2)
            for injected in (0,1):
                alternative=null+beta*injected
                shift=find_thermal_shift(alternative,t1,t2) if injected else 0.0
                counter=thermal_rate(t1+shift,t2+shift)
                shifts.append(shift)
                if injected:
                    differences.append(alternative-null)
                rows.append(_record(CASE_INJECTION,split,f'power={power:.2f};T1={t1:.2f};T2={t2:.2f}',
                                    f'injection={injected}',power,null,alternative,counter))
    holdout_deltas=[float(r['abs_difference']) for r in rows if r['split']=='held_out' and r['intervention']=='injection=1']
    max_cerror=max(float(r['countermodel_abs_error']) for r in rows)
    summary={'case_id':CASE_INJECTION,'outcome':STATUS_INJECTION,'data_origin':'DETERMINISTIC_SYNTHETIC_MODEL_OUTPUT',
             'n_observations':len(rows),'train_powers':list(train),'heldout_powers':list(holdout),
             'injection_coefficient_assumed':beta,'temperature_measurement_assumed_exact_in_restricted_null':True,
             'preregistered_restricted_null_difference_threshold':0.04,
             'minimum_heldout_restricted_null_abs_difference':min(holdout_deltas),
             'restricted_null_differentiated':min(holdout_deltas)>0.04,
             'expanded_thermal_countermodel_max_abs_residual':max_cerror,
             'expanded_thermal_countermodel_max_temperature_shift_K':max(shifts),
             'expanded_thermal_null_remains_viable':max_cerror<1e-9 and max(shifts)<=10.0,
             'causal_identification_in_real_SERS':False,
             'external_empirical_validation':False,
             'required_next_evidence':'Spatial thermometry and covariate-controlled intervention proving injection can vary without hidden temperature, coverage, field, potential, and transport changes.',
             'limitation':'Synthetic oracle has injection by construction. Restricted-null separation is not a real-world discovery; expanded thermal null exactly mimics these observations with unmeasured hotspot shifts.'}
    return rows,summary


def execute_cases() -> tuple[bytes,dict]:
    a,sa=phase_probe();b,sb=injection_probe()
    buf=io.StringIO(newline='')
    writer=csv.DictWriter(buf,fieldnames=FIELDNAMES,lineterminator='\n')
    writer.writeheader();writer.writerows(a+b)
    return buf.getvalue().encode('utf-8'), {'schema':SCHEMA,'case_results':{CASE_PHASE:sa,CASE_INJECTION:sb},
        'total_rows':len(a)+len(b),'model_calls':0,'network_calls':0,
        'observational_scientific_novelty_certified':False,'contains_real_experimental_observations':False,
        'tested_models_come_from_preregistered_illustrative_equations':True}


def effective_action(case: str, outcome: str, evidence_verified: bool) -> dict:
    """Outcome-aware M14-C overlay; never treats any verified SHA as positive novelty."""
    if not evidence_verified:
        raise ValueError('Unverified numerical evidence must not affect policy')
    if case==CASE_PHASE and outcome==STATUS_PHASE:
        return {'action':'ESCAPE','reason':'CONSTRUCTIVE_CLASSICAL_EQUIVALENCE_NO_LINEAR_IDENTIFICATION',
                'next_test':'Seek nonlinear calibrated observables or independently constrained hidden-state transfer before claiming coherence.'}
    if case==CASE_INJECTION and outcome==STATUS_INJECTION:
        return {'action':'SUSPEND','reason':'RESTRICTED_NULL_DIFFERS_BUT_EXPANDED_THERMAL_NULL_CAN_MATCH',
                'next_test':'Acquire independent hotspot-resolved thermometry and decoupled carrier intervention; otherwise return to alternative research questions.'}
    raise ValueError('Unregistered challenge/outcome; fail closed')
