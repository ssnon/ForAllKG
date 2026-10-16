"""Data-only exact finite-state CTMC forward instrument.

No LLM-authored executable code or formulas are evaluated. The user/model supplies
inert transition tables, observation weights, and piecewise-constant controls.
A computational result is a *conditional model prediction*, not empirical evidence.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import expm

MAX_STATES = 12
MAX_MODELS = 2
MAX_PROTOCOLS = 3
MAX_SEGMENTS = 8
MAX_OBSERVABLES = 5
MAX_EVENTS = 3


def number(x: Any, name: str, low: float = 0.0, high: float = 1e5) -> float:
    if type(x) not in (int, float) or not math.isfinite(x) or not low <= x <= high:
        raise ValueError('INVALID_FINITE_NUMBER:' + name)
    return float(x)


def names(v: Any, name: str, maximum: int, minimum: int = 1) -> list[str]:
    if not isinstance(v, list) or not minimum <= len(v) <= maximum or any(
        not isinstance(s, str) or not 1 <= len(s) <= 64 for s in v
    ) or len(v) != len(set(v)):
        raise ValueError('INVALID_IDENTIFIERS:' + name)
    return v


def parse_model(obj: dict) -> dict:
    if not isinstance(obj, dict) or set(obj) != {'name', 'states', 'initial_probabilities', 'transitions', 'observables', 'event_sets'}:
        raise ValueError('INVALID_MODEL_KEYS')
    if not isinstance(obj['name'], str) or not 1 <= len(obj['name']) <= 80:
        raise ValueError('INVALID_MODEL_NAME')
    states = names(obj['states'], 'states', MAX_STATES, 2)
    n = len(states)
    init = obj['initial_probabilities']
    if not isinstance(init, list) or len(init) != n:
        raise ValueError('INVALID_INITIAL_DISTRIBUTION')
    p = np.array([number(x, 'initial_probability', 0, 1) for x in init])
    if abs(float(sum(p)) - 1) > 1e-7:
        raise ValueError('INITIAL_PROBABILITIES_NOT_NORMALIZED')
    if not isinstance(obj['transitions'], list) or not 1 <= len(obj['transitions']) <= n * (n - 1):
        raise ValueError('INVALID_TRANSITION_COUNT')
    ids = {v: i for i, v in enumerate(states)}
    edges = []
    seen = set()
    for e in obj['transitions']:
        if not isinstance(e, dict) or set(e) != {'source','target','rate','group'}:
            raise ValueError('INVALID_TRANSITION_FORMAT')
        if e['source'] not in ids or e['target'] not in ids or e['source'] == e['target']:
            raise ValueError('INVALID_TRANSITION_ENDPOINT')
        if (e['source'],e['target']) in seen:
            raise ValueError('DUPLICATE_EDGE_USE_SINGLE_TOTAL_RATE')
        seen.add((e['source'],e['target']))
        if not isinstance(e['group'], str) or not 1 <= len(e['group']) <= 64:
            raise ValueError('INVALID_GROUP')
        edges.append((ids[e['source']],ids[e['target']],number(e['rate'],'rate'),e['group']))
    obs = obj['observables']
    if not isinstance(obs, list) or not 1 <= len(obs) <= MAX_OBSERVABLES:
        raise ValueError('INVALID_OBSERVABLES')
    labels = []
    for o in obs:
        if not isinstance(o, dict) or set(o) != {'name','values'} or not isinstance(o['name'], str) or not o['name']:
            raise ValueError('INVALID_OBSERVABLE')
        if not isinstance(o['values'], list) or len(o['values']) != n:
            raise ValueError('INVALID_OBSERVABLE_VECTOR')
        vals=[number(v,'observable_value',-1e5,1e5) for v in o['values']]
        labels.append((o['name'],np.array(vals)))
    events = obj['event_sets']
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError('INVALID_EVENT_SETS')
    event_labels = []
    for ev in events:
        if not isinstance(ev, dict) or set(ev) != {'name','bright_states'} or not isinstance(ev['name'],str) or not ev['name']:
            raise ValueError('INVALID_EVENT_SET')
        bright=names(ev['bright_states'],'bright_states',n)
        if len(bright) == n or any(s not in ids for s in bright):
            raise ValueError('INVALID_EVENT_PARTITION')
        event_labels.append((ev['name'],{ids[s] for s in bright}))
    all_labels=[v[0] for v in labels+event_labels]
    if len(all_labels)!=len(set(all_labels)):
        raise ValueError('DUPLICATE_OUTPUT_NAME')
    return {'name':obj['name'],'states':states,'initial':p,'edges':edges,'observables':labels,'events':event_labels}


def parse_protocol(obj: dict) -> dict:
    if not isinstance(obj, dict) or set(obj) != {'name','segments'}:
        raise ValueError('INVALID_PROTOCOL_KEYS')
    if not isinstance(obj['name'],str) or not 1 <= len(obj['name']) <= 80:
        raise ValueError('INVALID_PROTOCOL_NAME')
    seg=obj['segments']
    if not isinstance(seg,list) or not 1 <= len(seg) <= MAX_SEGMENTS:
        raise ValueError('INVALID_SEGMENT_COUNT')
    parsed=[]
    for s in seg:
        if not isinstance(s,dict) or set(s) != {'duration','rate_multipliers'} or not isinstance(s['rate_multipliers'],dict):
            raise ValueError('INVALID_SEGMENT')
        mul={}
        for key,val in s['rate_multipliers'].items():
            if not isinstance(key,str) or not 1 <= len(key) <= 64:
                raise ValueError('INVALID_RATE_GROUP')
            mul[key]=number(val,'rate_multiplier',0,1e3)
        parsed.append({'duration':number(s['duration'],'segment_duration',1e-8,1e4),'multipliers':mul})
    return {'name':obj['name'],'segments':parsed}


def generator(m: dict, multipliers: dict) -> np.ndarray:
    n=len(m['states']);Q=np.zeros((n,n))
    groups={g for _,_,_,g in m['edges']}
    if not set(multipliers)<=groups:
        raise ValueError('PROTOCOL_REFERENCES_UNKNOWN_GROUP:'+str(sorted(set(multipliers)-groups)))
    for i,j,r,g in m['edges']:
        Q[i,j] += r*multipliers.get(g,1.)
    np.fill_diagonal(Q,-Q.sum(axis=1))
    return Q


def evaluate(request: dict) -> dict:
    if not isinstance(request,dict) or set(request) != {'models','protocols'}:
        raise ValueError('INVALID_PROBE_REQUEST')
    raw_models=request['models'];raw_protocols=request['protocols']
    if not isinstance(raw_models,list) or not 1 <= len(raw_models) <= MAX_MODELS:
        raise ValueError('INVALID_MODEL_COUNT')
    if not isinstance(raw_protocols,list) or not 1 <= len(raw_protocols) <= MAX_PROTOCOLS:
        raise ValueError('INVALID_PROTOCOL_COUNT')
    models=[parse_model(x) for x in raw_models]
    protocols=[parse_protocol(x) for x in raw_protocols]
    if len({x['name'] for x in models})!=len(models) or len({x['name'] for x in protocols})!=len(protocols):
        raise ValueError('DUPLICATE_MODEL_OR_PROTOCOL_NAMES')
    results=[]
    for model in models:
        per_protocol=[]
        for protocol in protocols:
            p=model['initial'].copy();t=0.;segments=[];totals={ev[0]:0. for ev in model['events']}
            for segment in protocol['segments']:
                dt=segment['duration'];Q=generator(model,segment['multipliers'])
                # Prevent extreme model-imposed numerical workloads.
                if np.linalg.norm(Q,ord=np.inf)*dt > 2000:
                    raise ValueError('STIFF_OR_EXTREME_SEGMENT_REFUSED')
                A=Q.T
                n=len(p)
                block=np.zeros((2*n,2*n))
                block[:n,:n]=A
                block[:n,n:]=np.eye(n)
                E=expm(block*dt)
                occupation=E[:n,n:]@p
                end=expm(A*dt)@p
                if not np.all(np.isfinite(end)) or min(end)<-1e-7 or abs(float(sum(end))-1)>1e-7:
                    raise ValueError('INVALID_PROPAGATION')
                p=np.maximum(end,0);p/=sum(p)
                occ=np.maximum(occupation,0)
                event_out=[]
                for name,bright in model['events']:
                    dark=[i for i in range(n) if i not in bright]
                    q_event=np.array([sum(Q[i,j] for j in bright) if i in dark else 0. for i in range(n)])
                    events=float(q_event@occ)
                    totals[name]+=events
                    event_out.append({'name':name,'expected_entries_in_segment':events,
                                      'entry_rate_at_segment_end':float(q_event@p)})
                t+=dt
                segments.append({'end_time':t,'state_probabilities':dict(zip(model['states'],map(float,p))),
                    'observables':[{'name':label,'mean_at_segment_end':float(values@p),
                                    'time_integrated_mean_in_segment':float(values@occ)}
                                   for label,values in model['observables']],
                    'events':event_out})
            per_protocol.append({'protocol':protocol['name'],'segments':segments,
                                 'total_expected_event_entries':totals})
        results.append({'model':model['name'],'protocols':per_protocol})
    comparisons=[]
    if len(models)==2:
        for pi,protocol in enumerate(protocols):
            # Compare *observable* numeric signatures, not hidden-state labels.
            # Only like-named output series can be compared meaningfully.
            a,b=models
            if [o[0] for o in a['observables']]!=[o[0] for o in b['observables']] or [e[0] for e in a['events']]!=[e[0] for e in b['events']]:
                comparisons.append({'protocol':protocol['name'],'status':'INCOMPARABLE_OUTPUT_DEFINITIONS'})
                continue
            sig=[]
            for mi in range(2):
                r=results[mi]['protocols'][pi]
                vector=[]
                for segment in r['segments']:
                    vector += [v['mean_at_segment_end'] for v in segment['observables']]
                    vector += [v['expected_entries_in_segment'] for v in segment['events']]
                sig.append(np.array(vector))
            d=float(np.max(np.abs(sig[0]-sig[1]))) if sig[0].size else 0.
            scale=float(max(1.,np.max(np.abs(sig[0])) if sig[0].size else 0.,np.max(np.abs(sig[1])) if sig[1].size else 0.))
            comparisons.append({'protocol':protocol['name'],
                'status':'NUMERICALLY_INDISTINGUISHABLE_UNDER_TESTED_PROTOCOL' if d<=1e-8*scale else 'PREDICTIONS_DIFFER_UNDER_TESTED_PROTOCOL',
                'max_absolute_signature_difference':d,
                'not_a_general_identifiability_proof':True})
    return {'status':'COMPUTED_CONDITIONAL_MODEL_RESULT','tool':'FINITE_CTMC_TRANSIENT_FORWARD',
        'method':'piecewise-constant CTMC matrix exponential, exact integrated state occupancy',
        'epistemic_status':'THEORETICAL_PREDICTIONS_NOT_EXPERIMENTAL_VALIDATION',
        'limitations':['States/rates/emissions are assumptions, not identified physical mechanisms.',
                       'Event entries are latent state transitions, not thresholded measured Raman bursts.',
                       'Numeric equality under supplied protocols does not establish global non-identifiability.',
                       'Piecewise-constant rates do not represent an unmodeled continuous temperature profile.'],
        'results':results,'model_comparisons':comparisons}
