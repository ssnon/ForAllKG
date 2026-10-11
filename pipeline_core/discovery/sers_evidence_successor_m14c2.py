"""M14-C2, forward-only evidence-conditioned hypothesis proposals.

Test code or model-generated predictions do not establish new scientific facts.
Scientific signatures and polynomial test coefficients are hypotheses, not data.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any

SCHEMA='m14c2-evidence-conditioned-successor-v1'
LANE_ESCAPE='MECHANISM_SEARCH'
LANE_SUSPEND='OPEN_FRONTIER'
GRID=(0.15,0.45,0.85,1.15,1.45)


def canon(x:Any)->bytes:
    return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def prompt_for(contract:dict)->dict:
    escaped=contract['lanes'][LANE_ESCAPE]
    return {'system':('You are a speculative SERS scientific hypothesis generator. '
                      'Search aggressively, claim conservatively. Output a single JSON object, no fences. '
                      'Do not infer novelty from missing literature. No fabricated data, DOIs or citations. '
                      'M14-C1 numerical evidence was synthetic, not a real-world finding.'),
            'user':('Create ONE forward successor for the ESCAPE lane, not two. The other lane is SUSPEND: do not generate for it. '
                    'Choose exactly one of the anchored alternative problem cards listed below. '
                    'Change the scientific problem, mechanism family, causal driver and generating assumption. '
                    'Include a strong physical null, a measurable conditional prediction, independent intervention, falsifier, '
                    'and a computable proposed mathematical comparison. '
                    'The mathematical comparison is a synthetic PROPOSAL, not evidence. '
                    'Output exactly {"successor": {"problem_id":str,"family":str,"novelty_certified":false, '
                    '"mechanism_signature":{"mechanism_family_id":str,"causal_driver":str,"core_assumption":str, '
                    '"predicted_observable":str,"independent_intervention":str,"strongest_null":str,"falsification":str}, '
                    '"conditional_predictions":[str],"independent_observables":[str], '
                    '"test_proposal":{"independent_variable":str,"null_coefficients":[a,b,c], '
                    '"hypothesis_coefficients":[a,b,c],"nuisance_envelope":positive_number, '
                    '"units":str,"physical_scope_and_limitations":str}},"suspended_lane_successor":null}. '
                    'Each polynomial is a+b*x+c*x*x on a preregistered abstract x grid. '
                    'Coefficient separation has ZERO novelty or empirical-certification value. '
                    +json.dumps({'previous':escaped['previous'],'computational_feedback':contract['verified_c1_findings'],
                                  'available_anchored_problem_cards':escaped['alternative_cards'],
                                  'previous_interpretation_restrictions':contract['limitations']},ensure_ascii=False))}


def test_model_vectors(spec:dict)->dict:
    if not isinstance(spec,dict) or set(spec)!= {'independent_variable','null_coefficients','hypothesis_coefficients','nuisance_envelope','units','physical_scope_and_limitations'}:
        raise ValueError('Complete constrained prospective test spec required')
    if not all(isinstance(spec[k],str) and spec[k].strip() for k in ('independent_variable','units','physical_scope_and_limitations')):
        raise ValueError('Test needs independent variable, units, physical limitations')
    for k in ('null_coefficients','hypothesis_coefficients'):
        arr=spec[k]
        if not isinstance(arr,list) or len(arr)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in arr):
            raise ValueError('Numerically bounded three-coefficient polynomial required')
    nuisance=spec['nuisance_envelope']
    if type(nuisance) not in (int,float) or not math.isfinite(nuisance) or nuisance<=0 or nuisance>1e6:
        raise ValueError('Positive bounded nuisance interval required')
    def eval_poly(v,x):return v[0]+v[1]*x+v[2]*x*x
    outputs=[]
    for x in GRID:
        n=eval_poly(spec['null_coefficients'],x)
        h=eval_poly(spec['hypothesis_coefficients'],x)
        if not all(math.isfinite(v) and abs(v)<1e10 for v in (n,h)):
            raise ValueError('Unphysical numerical scale')
        outputs.append({'x':x,'proposed_null':n,'proposed_hypothesis':h,'formal_abs_difference':abs(h-n),
                        'out_of_assumed_null_envelope':abs(h-n)>nuisance})
    return {'status':'M14C2_SYNTHETIC_PROSPECTIVE_TEST_EXECUTED', 'grid':list(GRID),'rows':outputs,
            'formal_grid_points_outside_assumed_envelope':sum(int(v['out_of_assumed_null_envelope']) for v in outputs),
            'nuisance_bounds_are_model_proposed_not_measured':True,
            'physically_identifiable':False,'literature_novelty_certified':False,
            'real_world_observations':False,'independent_empirical_support':False}
