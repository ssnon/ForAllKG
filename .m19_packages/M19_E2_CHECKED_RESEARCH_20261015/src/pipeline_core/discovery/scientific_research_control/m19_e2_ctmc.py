"""Bounded exact CTMC calculator for independent on/off sites.

Mathematical check of a declared toy model, NOT a Ye/SERS physical verification.
No eval/exec, arbitrary code, internet, imported scientific hypothesis or paid calls.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict) or set(spec) != {'on_rates','off_rates','weights','threshold'}:
        raise ValueError('SPEC_KEYS_REQUIRED: on_rates,off_rates,weights,threshold')
    for k in ('on_rates','off_rates','weights'):
        if not isinstance(spec[k],list) or not 1<=len(spec[k])<=3:
            raise ValueError(f'{k}: list length 1..3 required')
    n=len(spec['on_rates'])
    if any(len(spec[k])!=n for k in ('off_rates','weights')):raise ValueError('UNEQUAL_SITE_COUNTS')
    out={}
    for k in ('on_rates','off_rates','weights'):
        val=[]
        for x in spec[k]:
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not .00001 <=x <=100:
                raise ValueError('INVALID_POSITIVE_SITE_PARAMETER: '+k)
            val.append(float(x))
        out[k]=val
    h=spec['threshold']
    if isinstance(h,bool) or not isinstance(h,(int,float)) or not math.isfinite(h) or not 0<h<sum(out['weights']):
        raise ValueError('INVALID_THRESHOLD')
    out['threshold']=float(h)
    return out


def evaluate(spec:dict[str,Any])->dict[str,Any]:
    s=normalize_spec(spec);a=s['on_rates'];b=s['off_rates'];w=s['weights'];h=s['threshold'];n=len(a);size=1<<n
    q=np.zeros((size,size),dtype=float)
    p=[x/(x+y) for x,y in zip(a,b)]
    pi=np.array([math.prod(p[i] if mask&(1<<i) else 1-p[i] for i in range(n)) for mask in range(size)])
    bright=np.array([sum(w[i] for i in range(n) if mask & (1<<i))>h for mask in range(size)],dtype=bool)
    for mask in range(size):
        for i in range(n):
            dest=mask ^(1<<i)
            rate=b[i] if mask&(1<<i) else a[i]
            q[mask,dest]=rate
        q[mask,mask]=-sum(q[mask])
    bi=np.flatnonzero(bright);di=np.flatnonzero(~bright)
    if not len(bi) or not len(di):raise ValueError('NO_ALTERNATING_BRIGHT_DARK_STATES')
    entry_flux=pi[di]@q[np.ix_(di,bi)]
    entry_rate=float(entry_flux.sum())
    if entry_rate<=0:raise ValueError('NO_BRIGHT_ENTRIES')
    initial=entry_flux/entry_rate
    qb=q[np.ix_(bi,bi)]
    bright_fraction=float(pi[bi].sum())
    mean_bright=float(initial@np.linalg.solve(-qb,np.ones(len(bi))))
    mean_by_balance=bright_fraction/entry_rate
    if not math.isclose(mean_bright,mean_by_balance,rel_tol=1e-10):
        raise ArithmeticError('PH_STATIONARY_BALANCE_FAILED')
    return {'schema':'m19-e2-independent-ctmc-exact-v1', 'model_spec':s,
      'bright_rule':'sum_i(weights[i]*X_i) > threshold',
      'bright_stationary_fraction':bright_fraction,
      'stationary_bright_entry_rate':entry_rate,
      'mean_bright_interval':mean_bright,
      'mean_dark_interval':(1-bright_fraction)/entry_rate,
      'reachable_bright_states':int(len(bi)), 'num_states':size,
      'math_method':'Product Bernoulli stationary distribution and threshold-subgenerator phase-type first passage; E[Tbright] = fraction_bright / entry_rate.',
      'interpretation_limits':['Conditional on independent stationary continuous-time 2-state Markov site processes with fixed positive rates and nonnegative intensity weights.',
       'No measured Ye time series or SERS mechanism fit; does not validate a physical cause.',
       'The mean and stationary fraction are exact for the specified mathematical model, not for all ensemble blinking models.']}
