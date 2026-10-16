"""Finite-state observable comparison. Data-only probability models; no dynamic code.

Purpose: explore *conditional* mathematical claims by enumerating independent binary
sites or arbitrary finite shared-state models. Not a SERS validator; no timing/kinetics.
"""
from __future__ import annotations
from itertools import product
from collections import defaultdict
import math

MAX_STATES = 512
MAX_SITES = 9
MAX_MODELS = 2
MAX_CONDITIONS = 5
MAX_MODES = 4
MAX_THRESHOLDS = 12


def _num(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError('INVALID_'+name.upper())
    return float(x)


def _vec(v, n=None):
    if not isinstance(v,list) or not 1 <= len(v) <= MAX_MODES or (n is not None and len(v)!=n):
        raise ValueError('INVALID_OBSERVABLE_VECTOR')
    if any(isinstance(x,bool) or not isinstance(x,int) or not -1000 <= x <= 1000 for x in v):
        raise ValueError('OBSERVABLES_MUST_BE_BOUNDED_INTEGERS')
    return tuple(v)


def _id(x):
    if not isinstance(x,str) or not 1 <= len(x) <= 80:
        raise ValueError('INVALID_IDENTIFIER')
    return x


def _pmf(spec):
    if not isinstance(spec,dict) or 'kind' not in spec:
        raise ValueError('INVALID_MODEL_SPEC')
    kind=spec['kind']
    out=defaultdict(float)
    if kind=='independent_binary_sites':
        if set(spec)!={'kind','background','sites'}:raise ValueError('INVALID_INDEPENDENT_KEYS')
        bg=_vec(spec['background']);sites=spec['sites']
        if not isinstance(sites,list) or not 1 <= len(sites) <= MAX_SITES:
            raise ValueError('INVALID_SITE_COUNT')
        items=[]
        for s in sites:
            if not isinstance(s,dict) or set(s)!={'p_on','emission'}:raise ValueError('INVALID_SITE')
            p=_num(s['p_on'],'site_probability',0,1)
            e=_vec(s['emission'],len(bg))
            if any(v<0 for v in e):raise ValueError('NEGATIVE_SITE_EMISSION')
            items.append((p,e))
        if 2**len(items)>MAX_STATES:raise ValueError('STATE_BUDGET')
        for bits in product((0,1),repeat=len(items)):
            prob=1.;vec=list(bg)
            for bit,(p,e) in zip(bits,items):
                prob*= p if bit else 1-p
                if bit:
                    vec=[a+b for a,b in zip(vec,e)]
            if prob:out[tuple(vec)]+=prob
    elif kind=='finite_shared_states':
        if set(spec)!={'kind','states'}:raise ValueError('INVALID_SHARED_KEYS')
        states=spec['states']
        if not isinstance(states,list) or not 1 <= len(states) <= MAX_STATES:raise ValueError('STATE_BUDGET')
        n=None;total=0.
        for s in states:
            if not isinstance(s,dict) or set(s)!={'probability','observable'}:
                raise ValueError('INVALID_STATE')
            p=_num(s['probability'],'probability',0,1);v=_vec(s['observable'],n)
            n=len(v);total+=p
            out[v]+=p
        if abs(total-1.)>1e-9:raise ValueError('PROBABILITIES_DO_NOT_SUM_TO_ONE')
    else:raise ValueError('UNKNOWN_MODEL_KIND')
    mass=sum(out.values())
    if abs(mass-1.)>1e-9:raise ValueError('INVALID_PROBABILITY_MASS')
    return dict(sorted((k,v/mass) for k,v in out.items() if v>1e-14))


def _stats(dist,thresholds):
    n=len(next(iter(dist)))
    mean=[sum(prob*x[i] for x,prob in dist.items()) for i in range(n)]
    cov=[[sum(prob*(x[i]-mean[i])*(x[j]-mean[j]) for x,prob in dist.items())
           for j in range(n)] for i in range(n)]
    exceed=[{'threshold':t,'probabilities':[sum(prob for x,prob in dist.items() if x[i]>t) for i in range(n)],
             'any_mode_probability':sum(prob for x,prob in dist.items() if any(v>t for v in x)),
             'all_modes_probability':sum(prob for x,prob in dist.items() if all(v>t for v in x))}
            for t in thresholds]
    return {'support_size':len(dist),'mean':mean,'covariance':cov,'threshold_exceedance':exceed,
            'distribution_preview':[{'observable':list(x),'probability':prob} for x,prob in sorted(dist.items(), key=lambda x:-x[1])[:24]],
            'distribution_preview_truncated':len(dist)>24}


def evaluate(request):
    """Compare two user-defined conditional generators, across matched conditions.

    Returned probabilities describe observation at one time, NOT excursion/entry rates,
    persistence durations, empirical SERS intensity, or real-world validity.
    """
    if not isinstance(request,dict) or set(request)!={'models','conditions','thresholds'}:
        raise ValueError('INVALID_PROBE_KEYS')
    models=request['models'];conds=request['conditions'];ths=request['thresholds']
    if not isinstance(models,list) or not 1 <= len(models) <= MAX_MODELS:
        raise ValueError('INVALID_MODEL_COUNT')
    if not isinstance(conds,list) or not 1 <= len(conds) <= MAX_CONDITIONS:
        raise ValueError('INVALID_CONDITIONS')
    conds=[_id(x) for x in conds]
    if len(set(conds))!=len(conds):raise ValueError('DUPLICATE_CONDITIONS')
    if not isinstance(ths,list) or not 1 <= len(ths) <= MAX_THRESHOLDS:
        raise ValueError('INVALID_THRESHOLDS')
    ths=[_num(t,'threshold',-10000,10000) for t in ths]
    names=[];mapped=[];dims=set()
    for m in models:
        if not isinstance(m,dict) or set(m)!={'name','conditions'}:
            raise ValueError('INVALID_MODEL_ENTRY')
        name=_id(m['name']);names.append(name)
        cases=m['conditions']
        if not isinstance(cases,dict) or set(cases)!=set(conds):
            raise ValueError('MODEL_CONDITIONS_MUST_MATCH')
        result={}
        for c in conds:
            dist=_pmf(cases[c]);dims.add(len(next(iter(dist))))
            result[c]=dist
        mapped.append(result)
    if len(set(names))!=len(names):raise ValueError('DUPLICATE_MODEL_NAMES')
    if len(dims)!=1:raise ValueError('MODEL_OBSERVABLE_DIMENSION_MISMATCH')
    output=[]
    for name,cases in zip(names,mapped):
        output.append({'name':name,'conditions':{c:_stats(cases[c],ths) for c in conds}})
    comparisons=[]
    if len(mapped)==2:
        for c in conds:
            a=mapped[0][c];b=mapped[1][c]
            tv=0.5*sum(abs(a.get(x,0.)-b.get(x,0.)) for x in set(a)|set(b))
            ax=output[0]['conditions'][c]['threshold_exceedance']
            bx=output[1]['conditions'][c]['threshold_exceedance']
            diffs=[{'threshold':t,'max_abs_probability_difference':max(
                max(abs(u-v) for u,v in zip(pa['probabilities'],pb['probabilities'])),
                abs(pa['any_mode_probability']-pb['any_mode_probability']),
                abs(pa['all_modes_probability']-pb['all_modes_probability']))}
                for t,pa,pb in zip(ths,ax,bx)]
            comparisons.append({'condition':c,'total_variation_full_observable_distribution':tv,
              'full_distribution_equivalent_at_this_condition':tv<1e-10,
              'threshold_differences':diffs})
    return {'tool':'FINITE_INSTANTANEOUS_OBSERVABLE_ENUMERATION',
       'epistemic_status':'CONDITIONAL_EXHAUSTIVE_ENUMERATION_FLOAT_ARITHMETIC_NOT_REAL_WORLD_EVIDENCE',
       'limitations':'Instantaneous finite discrete models only; no event rate, dwell time, kinetics, fitted parameters, or physical causal identification.',
       'models':output,'comparisons':comparisons}


def evaluate_kinetic(request):
    """Stationary threshold kinetics for finite Markov states with discrete emissions.

    Reuses the already-tested CTMC generator solver; no new stochastic solver.
    Threshold crossing means direct CTMC transitions between the two state sets,
    not experiment-specific burst detection with temporal binning or noisy data.
    """
    from . import science_act_probe
    if not isinstance(request,dict) or set(request)!={'models','conditions','thresholds'}:
        raise ValueError('INVALID_KINETIC_KEYS')
    mods=request['models'];conds=request['conditions'];ths=request['thresholds']
    if not isinstance(mods,list) or not 1 <= len(mods) <= MAX_MODELS:
        raise ValueError('INVALID_MODEL_COUNT')
    if not isinstance(conds,list) or not 1 <= len(conds) <= MAX_CONDITIONS:
        raise ValueError('INVALID_CONDITIONS')
    conds=[_id(x) for x in conds]
    if len(set(conds))!=len(conds):raise ValueError('DUPLICATE_CONDITIONS')
    if not isinstance(ths,list) or not 1 <= len(ths) <= MAX_THRESHOLDS:
        raise ValueError('INVALID_THRESHOLDS')
    ths=[_num(x,'threshold',-10000,10000) for x in ths]
    names=[];output=[];n_modes=set()
    for model in mods:
        if not isinstance(model,dict) or set(model)!={'name','conditions'}:
            raise ValueError('INVALID_MODEL_KEYS')
        names.append(_id(model['name']));cases=model['conditions']
        if not isinstance(cases,dict) or set(cases)!=set(conds):
            raise ValueError('MODEL_CONDITIONS_MUST_MATCH')
        results={}
        for c in conds:
            item=cases[c]
            if not isinstance(item,dict) or set(item)!={'states','transitions'}:
                raise ValueError('INVALID_KINETIC_CONDITION')
            states=item['states'];trans=item['transitions']
            if not isinstance(states,list) or not 2 <= len(states) <= science_act_probe.MAX_STATES:
                raise ValueError('INVALID_STATES')
            names_i=[];em={};nm=None
            for s in states:
                if not isinstance(s,dict) or set(s)!={'name','observable'}:
                    raise ValueError('INVALID_KINETIC_STATE')
                name=_id(s['name']);names_i.append(name)
                val=_vec(s['observable'],nm);nm=len(val);em[name]=val
            if len(set(names_i))!=len(names_i):raise ValueError('DUPLICATE_STATE_NAMES')
            n_modes.add(nm)
            if not isinstance(trans,list) or not 2 <= len(trans) <= science_act_probe.MAX_EDGES:
                raise ValueError('INVALID_TRANSITIONS')
            entries=[]
            for tr in trans:
                if not isinstance(tr,dict) or set(tr)!={'source','target','rate'}:
                    raise ValueError('INVALID_TRANSITION')
                if tr['source'] not in em or tr['target'] not in em or tr['source']==tr['target']:
                    raise ValueError('INVALID_TRANSITION_END')
                rate=_num(tr['rate'],'transition_rate',1e-12,1e5)
                entries.append((tr['source'],tr['target'],rate))
            # Anchor observable only initializes the preexisting numerical CTMC solver.
            call={'models':[{'name':'researcher_model','states':names_i,
                'transitions':[{'source':s,'target':t,'rate':r,'group':'fixed'} for s,t,r in entries],
                'observables':[{'name':'anchor','bright_states':[names_i[0]]}]}],
                'sweep':{'rate_group':'','multipliers':[1.]}}
            solution=science_act_probe.evaluate(call)['models'][0]['results'][0]
            pi=solution['stationary_probabilities']
            avg=[sum(pi[s]*em[s][k] for s in names_i) for k in range(nm)]
            cov=[[sum(pi[s]*(em[s][i]-avg[i])*(em[s][j]-avg[j]) for s in names_i)
                for j in range(nm)] for i in range(nm)]
            def threshold_metrics(bright):
                occ=sum(pi[s] for s in bright)
                flux=sum(pi[s]*r for s,t,r in entries if s not in bright and t in bright)
                return {'occupancy':occ,'entry_flux':flux,
                    'mean_bright_duration':occ/flux if flux>1e-12 else None,
                    'mean_dark_duration':(1-occ)/flux if flux>1e-12 else None}
            thresholds=[]
            for threshold in ths:
                thresholds.append({'threshold':threshold,
                    'mode_metrics':[threshold_metrics({s for s in names_i if em[s][i]>threshold}) for i in range(nm)],
                    'any_mode':threshold_metrics({s for s in names_i if any(x>threshold for x in em[s])}),
                    'all_modes':threshold_metrics({s for s in names_i if all(x>threshold for x in em[s])})})
            results[c]={'stationary_probabilities':pi,'mean':avg,'covariance':cov,
                        'threshold_metrics':thresholds}
        output.append({'name':model['name'],'conditions':results})
    if len(set(names))!=len(names):raise ValueError('DUPLICATE_MODEL_NAMES')
    if len(n_modes)!=1:raise ValueError('OBSERVABLE_DIMENSIONS_DIFFER')
    comparisons=[]
    if len(output)==2:
        for c in conds:
            diffs=[]
            for a,b in zip(output[0]['conditions'][c]['threshold_metrics'],
                           output[1]['conditions'][c]['threshold_metrics']):
                pairs=list(zip(a['mode_metrics'],b['mode_metrics']))+[(a['any_mode'],b['any_mode']),(a['all_modes'],b['all_modes'])]
                diffs.append({'threshold':a['threshold'],
                    'max_abs_occupancy_difference':max(abs(x['occupancy']-y['occupancy']) for x,y in pairs),
                    'max_abs_entry_flux_difference':max(abs(x['entry_flux']-y['entry_flux']) for x,y in pairs)})
            comparisons.append({'condition':c,'threshold_differences':diffs})
    return {'tool':'FINITE_MARKOV_EMISSION_THRESHOLD_COMPARISON',
        'epistemic_status':'CONDITIONAL_CTMC_STATIONARY_RESULTS_NOT_EXPERIMENTAL_OBSERVATIONS',
        'limitations':'Assumes time-homogeneous CTMC, exact emissions, direct threshold state partition; not noisy/binned event detection or microscopic causal inference.',
        'models':output,'comparisons':comparisons}
