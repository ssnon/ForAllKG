"""One data-only scientific instrument: finite CTMC observables and parameter sweeps.

No model-authored Python, imports, filesystem paths, shell or dynamic expressions.
A computed result is conditional mathematics, not experimental observation.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

MAX_STATES=12
MAX_EDGES=120
MAX_OBSERVABLES=8
MAX_SWEEP=12


def _finite_pos(v: Any, label: str, *, allow_zero: bool=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v):
        raise ValueError('INVALID_NUMBER: '+label)
    f=float(v)
    if not (0 <= f <= 1e5 if allow_zero else 0 < f <= 1e5):
        raise ValueError('OUT_OF_RANGE: '+label)
    return f


def _names(xs: Any, label: str, limit: int, min_size: int=1) -> list[str]:
    if not isinstance(xs,list) or not min_size <= len(xs) <= limit:
        raise ValueError('INVALID_LIST: '+label)
    if any(not isinstance(x,str) or not x or len(x)>64 for x in xs) or len(set(xs))!=len(xs):
        raise ValueError('INVALID_IDENTIFIERS: '+label)
    return xs


def _one_model(m: dict[str,Any], factors: list[float], group: str) -> dict[str,Any]:
    if not isinstance(m,dict) or set(m)!={'name','states','transitions','observables'}:
        raise ValueError('INVALID_MODEL_KEYS')
    states=_names(m['states'],'states',MAX_STATES,2)
    obs=m['observables'];edges=m['transitions']
    if not isinstance(m['name'],str) or not m['name'] or len(m['name'])>80:
        raise ValueError('INVALID_MODEL_NAME')
    if not isinstance(edges,list) or not 2<=len(edges)<=MAX_EDGES:
        raise ValueError('INVALID_EDGES')
    if not isinstance(obs,list) or not 1<=len(obs)<=MAX_OBSERVABLES:
        raise ValueError('INVALID_OBSERVABLES')
    idx={n:i for i,n in enumerate(states)}
    parsed=[]
    for e in edges:
        if not isinstance(e,dict) or set(e)!={'source','target','rate','group'}:
            raise ValueError('INVALID_EDGE_KEYS')
        if e['source'] not in idx or e['target'] not in idx or e['source']==e['target']:
            raise ValueError('INVALID_EDGE')
        if not isinstance(e['group'],str) or not e['group'] or len(e['group'])>64:
            raise ValueError('INVALID_RATE_GROUP')
        parsed.append((idx[e['source']],idx[e['target']],_finite_pos(e['rate'],'rate'),e['group']))
    parsed_obs=[]
    obs_names=[]
    for o in obs:
        if not isinstance(o,dict) or set(o)!={'name','bright_states'}:
            raise ValueError('INVALID_OBSERVABLE_KEYS')
        if not isinstance(o['name'],str) or not o['name'] or len(o['name'])>80:
            raise ValueError('INVALID_OBSERVABLE_NAME')
        bright=_names(o['bright_states'],'bright_states',len(states))
        if any(x not in idx for x in bright) or len(bright)==len(states):
            raise ValueError('INVALID_BRIGHT_PARTITION')
        obs_names.append(o['name'])
        parsed_obs.append((o['name'],[idx[x] for x in bright]))
    if len(set(obs_names))!=len(obs_names):
        raise ValueError('DUPLICATE_OBSERVABLE')
    if group and group not in [g for _,_,_,g in parsed]:
        raise ValueError('SWEEP_GROUP_NOT_IN_MODEL: '+m['name'])
    outputs=[]
    for factor in factors:
        Q=np.zeros((len(states),len(states)),dtype=float)
        for s,t,rate,g in parsed:
            Q[s,t]+=rate*(factor if g==group else 1.)
        for s in range(len(states)):
            Q[s,s]=-sum(Q[s,t] for t in range(len(states)) if t!=s)
        A=Q.T.copy();A[-1,:]=1.;b=np.zeros(len(states));b[-1]=1.
        if np.linalg.matrix_rank(A, tol=1e-10)<len(states):
            raise ValueError('NONUNIQUE_STATIONARY_DISTRIBUTION')
        pi=np.linalg.solve(A,b)
        if min(pi)<-1e-8 or np.max(np.abs(pi@Q))>1e-7 or abs(sum(pi)-1)>1e-7:
            raise ValueError('INVALID_STATIONARY_SOLUTION')
        pi=np.maximum(pi,0.);pi=pi/pi.sum()
        metrics=[]
        for name,bright in parsed_obs:
            br=set(bright);dark=[i for i in range(len(states)) if i not in br]
            occupancy=float(pi[bright].sum())
            flux=float(sum(pi[i]*Q[i,j] for i in dark for j in bright))
            exitflux=float(sum(pi[j]*Q[j,i] for j in bright for i in dark))
            if abs(flux-exitflux)>1e-7*max(1.,flux):
                raise ValueError('FLUX_IMBALANCE')
            metrics.append({'name':name,'bright_occupancy':occupancy,
                'episode_entry_rate':flux,
                'mean_bright_episode_duration':float(occupancy/flux) if flux>1e-12 else None,
                'mean_dark_episode_duration':float((1.-occupancy)/flux) if flux>1e-12 else None})
        outputs.append({'sweep_multiplier':factor,'stationary_probabilities':dict(zip(states,map(float,pi))),
                        'observables':metrics})
    return {'name':m['name'],'results':outputs}


def evaluate(request: dict[str,Any]) -> dict[str,Any]:
    """Evaluate 1-2 user-authored mechanistic CTMCs; all inputs are inert JSON."""
    if not isinstance(request,dict) or set(request)!={'models','sweep'}:
        raise ValueError('INVALID_PROBE_REQUEST')
    models=request['models'];sw=request['sweep']
    if not isinstance(models,list) or not 1<=len(models)<=2:
        raise ValueError('INVALID_MODEL_COUNT')
    if len({m.get('name') for m in models if isinstance(m,dict)})!=len(models):
        raise ValueError('DUPLICATE_MODEL_NAMES')
    if not isinstance(sw,dict) or set(sw)!={'rate_group','multipliers'}:
        raise ValueError('INVALID_SWEEP')
    group=sw['rate_group']
    if not isinstance(group,str) or len(group)>64:
        raise ValueError('INVALID_SWEEP_GROUP')
    factors=sw['multipliers']
    if not isinstance(factors,list) or not 1<=len(factors)<=MAX_SWEEP:
        raise ValueError('INVALID_SWEEP_VALUES')
    f=[_finite_pos(v,'sweep_multiplier') for v in factors]
    if group=='':
        if f != [1.]: raise ValueError('EMPTY_GROUP_REQUIRES_UNIT_FACTOR')
    r=[_one_model(m,f,group) for m in models]
    return {'tool':'NUMERICAL_STATIONARY_FINITE_CTMC',
        'epistemic_status':'NUMERICALLY_COMPUTED_CONDITIONAL_MODEL_RESULT_NOT_REAL_WORLD_EVIDENCE',
        'numerical_method':'float64 solve of stationary linear system with residual checks; near-degenerate regimes can be rejected',
        'meaning':{'bright_occupancy':'stationary P(observable bright)',
                   'episode_entry_rate':'stationary transitions from complement into bright set per unit model time',
                   'mean_bright_episode_duration':'stationary bright occupancy / episode entry flux'},
        'models':r}
