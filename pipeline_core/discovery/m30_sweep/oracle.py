"""M30 deterministic, synthetic, sealed evaluators. Never put this module into an LLM prompt.

NOT empirical discoveries, CausalDynamics instances, or real Pt/Ye observations.
The suite is purpose-built for evaluating feedback use under controlled conditions.
"""
from __future__ import annotations
import math
from collections import deque

MAX_NODES=47

def eval_expr(x,env, budget=None):
    if budget is None: budget=[MAX_NODES]
    budget[0]-=1
    if budget[0]<0: raise ValueError('EXPRESSION_TOO_LARGE')
    if isinstance(x,dict):
        if set(x)=={'const'} and type(x['const']) in (float,int):
            v=float(x['const'])
            if abs(v)>100:raise ValueError('CONSTANT_BOUND')
            return v
        if set(x)=={'var'} and x['var'] in env:return float(env[x['var']])
        if set(x)=={'op','args'}:
            op,args=x['op'],x['args']
            if not isinstance(args,list):raise ValueError('BAD_ARGS')
            if op in ('neg','exp','tanh') and len(args)==1:
                a=eval_expr(args[0],env,budget)
                return {'neg':lambda:-a,'exp':lambda:math.exp(max(-30,min(30,a))), 'tanh':lambda:math.tanh(a)}[op]()
            if op in ('+','-','*','/') and len(args)==2:
                a=eval_expr(args[0],env,budget);b=eval_expr(args[1],env,budget)
                return {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b if abs(b)>1e-8 else (_ for _ in ()).throw(ValueError('NEAR_ZERO_DIVISOR'))}[op]()
    raise ValueError('INVALID_AST')

def safe_calc(tree,env):
    v=eval_expr(tree,env)
    if not math.isfinite(v) or abs(v)>1e5: raise ValueError('NONFINITE_OR_EXPLOSIVE')
    return v

def sidon_diagnostics(cand,N):
    if not isinstance(cand,dict) or not isinstance(cand.get('set'),list):raise ValueError('EXPECTED_SET')
    vals=cand['set']
    if len(vals)>N or any(type(z)!=int or not 0<=z<N for z in vals):raise ValueError('OUTSIDE_UNIVERSE')
    if len(set(vals))!=len(vals):raise ValueError('DUPLICATED_ELEMENTS')
    vs=sorted(vals);seen={};pairs=[]
    for i in range(len(vs)):
        for j in range(i):
            d=vs[i]-vs[j]
            if d in seen and len(pairs)<8:pairs.append({'difference':d,'pairs':[list(seen[d]),[vs[j],vs[i]]]})
            seen[d]=(vs[j],vs[i])
    return {'valid':not pairs,'set_size':len(vs),'conflicts':pairs,'difference_count':len(seen)}

DFA_TARGETS={
 'dfa_mod3':{'transitions':[[0,1],[1,2],[2,0]],'accepting':[0]},
 'dfa_suffix01':{'transitions':[[1,0],[1,2],[1,0]],'accepting':[2]},
}

def dfa_check(cand):
    if not isinstance(cand,dict):raise ValueError('EXPECTED_DFA')
    tr=cand.get('transitions');acc=cand.get('accepting')
    if not isinstance(tr,list) or not 1<=len(tr)<=8:raise ValueError('DFA_STATE_COUNT')
    if any(not isinstance(row,list) or len(row)!=2 or any(type(v)!=int or not 0<=v<len(tr) for v in row) for row in tr):raise ValueError('DFA_TRANSITIONS')
    if not isinstance(acc,list) or any(type(v)!=int or not 0<=v<len(tr) for v in acc):raise ValueError('DFA_ACCEPTING')
    return {'transitions':tr,'accepting':sorted(set(acc))}

def accepts(dfa, word):
    d=dfa_check(dfa);state=0
    for c in word:state=d['transitions'][state][int(c)]
    return state in d['accepting']

def counterexample(cand,target):
    d=dfa_check(cand)
    q=deque([(0,0,'')]);seen={(0,0)}
    while q:
        i,j,w=q.popleft()
        if (i in d['accepting'])!=(j in target['accepting']):return w
        for a in (0,1):
            ii=d['transitions'][i][a];jj=target['transitions'][j][a]
            if (ii,jj) not in seen:seen.add((ii,jj));q.append((ii,jj,w+str(a)))
    return None

def target_formula(task,row):
    if task=='law_saturation':return 0.18+1.9*row['x']/(0.65+row['x'])
    if task=='law_inhibition':return 2.6*row['x']/(0.45+row['x'])/(1+0.8*row['z'])
    if task=='chem_blocking':return 2.3*row['c']*row['o']*(1-row['c'])
    if task=='chem_dualpath':return 1.3*row['c']*row['o'] + 0.5*row['c']*row['r']/(0.3+row['c'])
    raise ValueError('NO_FORMULA_TASK')

def numeric_rows(task,heldout=False):
    if task.startswith('law_'):
        n=12 if not heldout else 14
        return [{'x':round(0.12+(i+.3*(heldout))*2.5/(n-1),4),'z':round(((i*7+3+(2 if heldout else 0))%11)/5,4)} for i in range(n)]
    if task.startswith('chem_'):
        n=18 if not heldout else 20
        return [{'c':round(0.06+((i*7+(2 if heldout else 0))%n)/(n-1)*.83,4),
                 'o':round(0.1+((i*11+3+(1 if heldout else 0))%n)/(n-1)*.8,4),
                 'r':round(.05+((i*13+4+(3 if heldout else 0))%n)/(n-1)*.75,4)} for i in range(n)]
    raise ValueError('NOT_NUMERIC_TASK')

def _rhs(task,x,y,u):
    if task=='dyn_predator':return (0.7*x-0.4*x*y+0.15*u,0.22*x*y-0.5*y)
    if task=='dyn_resource':return (u-0.65*x-0.3*x*y,0.34*x*y-0.33*y)
    raise ValueError('NOT_DYNAMICS')

def _integrate(rhs,x0,y0,u):
    dt=0.04;x=float(x0);y=float(y0);out=[]
    for i in range(26):
        if i%5==0:out.append({'t':round(i*dt,3),'x':round(x,5),'y':round(y,5)})
        if i==25:break
        a,b=rhs(x,y,u); c,d=rhs(x+dt*a/2,y+dt*b/2,u)
        e,f=rhs(x+dt*c/2,y+dt*d/2,u);g,h=rhs(x+dt*e,y+dt*f,u)
        x+=dt*(a+2*c+2*e+g)/6;y+=dt*(b+2*d+2*f+h)/6
        if not math.isfinite(x+y) or abs(x)>1e4 or abs(y)>1e4: raise ValueError('UNSTABLE_SIMULATION')
    return out

def simulate_truth(task,protocol):return _integrate(lambda x,y,u:_rhs(task,x,y,u), protocol['x0'],protocol['y0'],protocol['u'])
def simulate_candidate(cand,protocol):
    if not isinstance(cand,dict) or set(cand)!={'dx','dy'}:raise ValueError('CANDIDATE_DX_DY_REQUIRED')
    return _integrate(lambda x,y,u:(safe_calc(cand['dx'],{'x':x,'y':y,'u':u}),safe_calc(cand['dy'],{'x':x,'y':y,'u':u})),protocol['x0'],protocol['y0'],protocol['u'])

def dyn_protocols(task,heldout=False):
    if heldout:
        return [{'x0':1.35,'y0':.8,'u':1.35},{'x0':.6,'y0':1.25,'u':.65},{'x0':1.9,'y0':.45,'u':1.65}]
    return [{'x0':.85,'y0':.75,'u':.25},{'x0':1.6,'y0':.55,'u':1.0}]

def _rmse_score(yhat,real):
    mse=sum((a-b)**2 for a,b in zip(yhat,real))/len(real)
    rms=math.sqrt(sum(z*z for z in real)/len(real))
    return max(0.0,min(1.0,1-math.sqrt(mse)/(0.1+rms))), math.sqrt(mse)

def final_score(task,cand):
    kind=task.split('_')[0]
    try:
        if kind=='sidon':
            N=int(task.split('_')[1]);res=sidon_diagnostics(cand,N)
            return {'score':round(min(1,res['set_size']/(1.8*math.sqrt(N))),6) if res['valid'] else 0,
                    'metric':'valid_construction_size_scaled','valid':res['valid'],'size':res['set_size'],'N':N,'conflicts':res['conflicts']}
        if kind=='dfa':
            target=DFA_TARGETS[task];d=dfa_check(cand);ce=counterexample(d,target)
            words=['']
            for k in range(1,10):words.extend(format(i,f'0{k}b') for i in range(2**k))
            hits=sum(accepts(d,w)==accepts(target,w) for w in words)
            return {'score':round(hits/len(words),6),'metric':'heldout_language_agreement_length_0_to_9', 'exact_equivalence':ce is None,'shortest_counterexample':ce,'states':len(d['transitions'])}
        if kind in ('law','chem'):
            rows=numeric_rows(task,True); yy=[target_formula(task,z) for z in rows]; yh=[safe_calc(cand,z) for z in rows];sc,rmse=_rmse_score(yh,yy)
            return {'score':round(sc,6),'metric':'heldout_normalized_rmse_score','rmse':round(rmse,6),'points':len(rows)}
        if kind=='dyn':
            if task not in ('dyn_predator','dyn_resource'):raise ValueError('UNKNOWN_DYN')
            real=[];yh=[]
            for prot in dyn_protocols(task,True):
                for a,b in zip(simulate_truth(task,prot),simulate_candidate(cand,prot)):
                    real.extend([a['x'],a['y']]);yh.extend([b['x'],b['y']])
            sc,rmse=_rmse_score(yh,real)
            return {'score':round(sc,6),'metric':'heldout_intervention_trajectory_score','rmse':round(rmse,6),'points':len(real)}
        raise ValueError('UNKNOWN_KIND')
    except Exception as e:
        return {'score':0.0,'metric':'invalid_candidate','error':str(e)[:240]}

def experiment(task,req,cand):
    if not isinstance(req,dict):return {'status':'INVALID_REQUEST','error':'expected dict'}
    try:
        kind=task.split('_')[0]
        if kind=='sidon':return {'status':'EXECUTED','result':sidon_diagnostics(cand,int(task.split('_')[1]))}
        if kind=='dfa':
            d=dfa_check(cand);ce=counterexample(d,DFA_TARGETS[task]);return {'status':'EXECUTED','result':{'equivalent':ce is None,'shortest_counterexample':ce,'correct_label':accepts(DFA_TARGETS[task],ce) if ce is not None else None}}
        if kind in ('law','chem'):
            fields=['x','z'] if kind=='law' else ['c','o','r'];point=req.get('point',{})
            if not isinstance(point,dict):raise ValueError('point must be dict')
            p={v:float(point.get(v,0.4)) for v in fields}
            if any(not 0<=z<=3 for z in p.values()):raise ValueError('POINT_OUT_OF_BOUNDS')
            truth=target_formula(task,p)
            try: predicted=safe_calc(cand,p)
            except Exception as e:predicted={'error':str(e)}
            return {'status':'EXECUTED','result':{'point':p,'observed_y':round(truth,6),'candidate_y':predicted}}
        if kind=='dyn':
            prot={'x0':float(req.get('x0',1)),'y0':float(req.get('y0',.7)),'u':float(req.get('u',1))}
            if any(not -0.25<=v<=2 for v in prot.values()):raise ValueError('PROTOCOL_OUT_OF_BOUNDS')
            observed=simulate_truth(task,prot)
            try:predicted=simulate_candidate(cand,prot)
            except Exception as e:predicted={'error':str(e)}
            return {'status':'EXECUTED','result':{'protocol':prot,'observed_trajectory':observed,'predicted_trajectory':predicted}}
        raise ValueError('UNKNOWN_TASK')
    except Exception as e:
        return {'status':'INVALID_REQUEST','error':str(e)[:240]}

def public_problem(task):
    kind=task.split('_')[0]
    rules={
     'dyn':'Find a two-variable continuous-time causal dynamical model dx/dt,dy/dt; measured x,y and control u. Controlled interventions on u and initial x,y are available in tool arms. Identify any nonlinearity or coupling that is needed. Output AST for dx and dy.',
     'law':'Find a predictive scientific relationship y=f(x,z); identify minimal functional structure, not merely curve-fitting a table. z may be irrelevant. Output expression AST.',
     'chem':'Stylized synthetic catalyst microkinetic rate law, NOT actual experimental catalyst data. Measured c=CO coverage, o=chemisorbed oxygen, r=oxide-like reservoir. Infer rate y=f(c,o,r). Output expression AST.',
     'sidon':'Find as large as possible A subset {0,...,N-1} with all positive pairwise differences distinct. Output {"set":[...]} with unique in-range integers. Exact validator available in tool arms.',
     'dfa':'Recover binary-language recognizer as deterministic finite automaton from membership examples. Alphabet 0/1; initial state 0. Output {"transitions":[[next_on_0,next_on_1],...],"accepting":[state_indices]}. Exact equivalence checking available in tool arms.'
    }
    if kind in ('law','chem'):
        rows=numeric_rows(task);obs=[{**z,'y':round(target_formula(task,z),6)} for z in rows]
    elif kind=='dyn':
        obs=[{'protocol':p,'trajectory':simulate_truth(task,p)} for p in dyn_protocols(task)]
    elif kind=='sidon':obs={'N':int(task.split('_')[1]),'example_valid_set':[0,1,3]}
    elif kind=='dfa':
        target=DFA_TARGETS[task]
        words=['']+[format(i,f'0{k}b') for k in range(1,4) for i in range(2**k)]
        obs=[{'word':w,'accepted':accepts(target,w)} for w in words]
    else:raise ValueError('UNKNOWN_TASK')
    return {'id':task,'family':kind,'scientific_problem':rules[kind], 'observed_data':obs,
            'candidate_format':('AST_DX_DY' if kind=='dyn' else 'EXPRESSION_AST' if kind in ('law','chem') else 'SIDON_SET' if kind=='sidon' else 'DFA'),
            'ast_grammar':{'const':'{"const":0.5}','var':'{"var":"x"} use only measured variable names','binary':'{"op":"+"|"-"|"*"|"/","args":[ast,ast]}','unary':'{"op":"neg"|"exp"|"tanh","args":[ast]}'} if kind in ('dyn','law','chem') else None,
            'experimental_intervention_note':('tool arms may choose point / protocol to reveal new oracle observations' if kind in ('dyn','law','chem') else 'tool arms may ask exact candidate validator'),
            'science_truth_status':'HIDDEN_SYNTHETIC_REFERENCE_NOT_EMPIRICAL'}

TASKS=['dyn_predator','dyn_resource','sidon_25','sidon_40','law_saturation','law_inhibition','dfa_mod3','dfa_suffix01','chem_blocking','chem_dualpath']
