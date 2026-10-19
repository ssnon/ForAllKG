"""M31 new DEVELOPMENT PILOT scientific instruments; never authoritative real-world discovery.

Tools expose only requested measurements/checks. Hidden formulas are not supplied to researcher
prompts. Each tool is deterministic and safe/data-only. The task family is fixed per episode.
"""
from __future__ import annotations
import math, json, hashlib
from fractions import Fraction

IDS=('G01','G02','I01','I02','P01','P02','X01','X02','R01','R02','E01','E02')
GROUPS={'G01':'OPEN_HYPOTHESIS','G02':'OPEN_HYPOTHESIS',
'I01':'MECHANISM_IDENTIFICATION','I02':'MECHANISM_IDENTIFICATION',
'P01':'MATHEMATICAL_PROOF','P02':'MATHEMATICAL_PROOF',
'X01':'COUNTEREXAMPLE_PIVOT','X02':'COUNTEREXAMPLE_PIVOT',
'R01':'CANDIDATE_RETENTION','R02':'CANDIDATE_RETENTION',
'E01':'EVIDENCE_SYNTHESIS','E02':'EVIDENCE_SYNTHESIS'}

TASKS={
'G01':('Two measured variables x,y decay under an adjustable external input u. Determine whether the measured trajectories require coupling, an unobserved state, or a nonlinear term. Design an independent intervention with numerical predictions and state what the data cannot identify.', 'You may measure instantaneous dx,dy and x,y after 0.3 time units at explicitly requested initial x0,y0 and constant u.'),
'G02':('A two-variable system shows different transient responses to excitation. Form at least two explicit mechanistic alternatives and propose a discriminating independent initial-state intervention; avoid claiming uniqueness from sparse observations.', 'You may measure instantaneous dx,dy and x,y after 0.3 time units at explicitly requested initial x0,y0 and constant u.'),
'I01':('An unknown reaction has substrate c and inhibitor i. Several rates are reported near a narrow region. Decide what can be inferred about inhibition and saturation, distinguish competing equations, and design an informative held-out measurement.', 'You may measure reaction rate r for an explicitly requested nonnegative c and i.'),
'I02':('A synthetic catalyst has rates depending on reagent c and added modifier i. Investigate whether modifiers act through available-site competition, catalytic activation, or an alternative. Design a quantitative intervention.', 'You may measure reaction rate r for an explicitly requested nonnegative c and i.'),
'P01':('For arbitrary invertible real 2x2 matrices A and B, does A^3=B^3 imply AB=BA? Either prove the implication or give an exact integer-matrix counterexample, and distinguish failure of one route from a general conclusion.', 'You may request an exact integer-matrix verification of A^3=B^3, invertibility, and AB=BA for submitted A,B.'),
'P02':('For idempotent real 2x2 matrices A and B satisfying tr(A)=tr(B), must AB=BA? Derive a proof or submit an exact integer-matrix counterexample. Address the significance of equal rank.', 'You may request an exact integer-matrix verification of idempotence, equal trace and AB=BA for submitted A,B.'),
'X01':('Consider the proposed general bound |x-y| <= sqrt(x^2+y^2) for arbitrary real x,y. Determine its maximal valid region or find an exact counterexample, then repair the statement with the weakest clearly stated assumption.', 'You may check the conjectured bound at explicitly requested real x,y.'),
'X02':('A researcher proposes (x+y)^2 <= 2(x^2+y^2) and claims that equality is possible only at x=y=0. Check both parts independently; if one fails, supply a stronger mathematically justified characterization.', 'You may check the inequality and claimed equality characterization at requested real x,y.'),
'R01':('Three competing functional explanations fit a narrow set of measurements. Develop several explicit candidate laws, select informative experiments, and retain a justified alternative if an initially preferred law fails. Avoid using fit on the initial observations as proof of mechanism.', 'You may measure a scalar response for explicitly requested x,z.'),
'R02':('Two restricted scientific response laws and a flexible phenomenological law all fit a small pilot region. Develop a portfolio, use discriminating interventions rather than training fit alone, and preserve the strongest defensible finding.', 'You may measure a scalar response for explicitly requested x,z.'),
'E01':('Three synthetic laboratory reports disagree on a concentration-dependent response. The reports differ in instruments, batches and calibration. Reconcile what can be inferred using provenance, rather than assuming one report must be correct.', 'You may retrieve specific independent calibration/raw entries by record_id.'),
'E02':('Two synthetic reports on a candidate transition disagree in trend and threshold. Determine whether apparent disagreement can be caused by measurement conventions, and propose a decisive replication. Do not invent metadata.', 'You may retrieve specific independent calibration/raw entries by record_id.'),
}

def finite(x, lo=-3.0, hi=4.0):
    if type(x) not in (int,float) or not math.isfinite(float(x)) or not lo<=float(x)<=hi:
        raise ValueError('REAL_FINITE_NUMERIC_VALUE_OUT_OF_DOMAIN')
    return float(x)

def _dynamic(t,x,y,u):
    if t=='G01':return (.72*u-.43*x-.26*x*y, .19*x-.36*y+.13*u)
    return (.55*u-.38*x-.21*y*y,.16*x*y-.31*y+.09*u)

def _observe_dynamic(t, inp):
    if set(inp)!={'x0','y0','u'}:raise ValueError('REQUIRED_FIELDS_x0_y0_u')
    x=finite(inp['x0'],0,2.5);y=finite(inp['y0'],0,2.5);u=finite(inp['u'],0,2.5)
    dx0,dy0=_dynamic(t,x,y,u)
    h=.01
    for _ in range(30):
        a,b=_dynamic(t,x,y,u)
        x+=h*a;y+=h*b
    return {'dx_initial':round(dx0,8),'dy_initial':round(dy0,8),'x_after_0p3':round(x,8),'y_after_0p3':round(y,8)}

def _kinetics(t,inp):
    if set(inp)!={'c','i'}:raise ValueError('REQUIRED_FIELDS_c_i')
    c=finite(inp['c'],0,3);i=finite(inp['i'],0,3)
    if t=='I01':r=1.8*c/(.48*(1+2.1*i)+c)
    else:r=1.45*c/(.32+c)/(1+.42*i)+.11*c*i/(1+c)
    return {'r':round(r,9)}

def mat(inp):
    if type(inp)!=list or len(inp)!=2 or any(type(row)!=list or len(row)!=2 for row in inp):raise ValueError('REQUIRES_EXACT_2X2_ARRAY')
    a=[]
    for row in inp:
        r=[]
        for v in row:
            if type(v)!=int or abs(v)>12:raise ValueError('MATRIX_ENTRIES_MUST_BE_INTEGERS_ABS_LE_12')
            r.append(v)
        a.append(r)
    return a

def mul(A,B):return [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def sq(A):return mul(A,A)
def det(A):return A[0][0]*A[1][1]-A[0][1]*A[1][0]
def _matrix(t,inp):
    if set(inp)!={'A','B'}:raise ValueError('REQUIRED_FIELDS_A_B')
    A=mat(inp['A']);B=mat(inp['B']);AB=mul(A,B);BA=mul(B,A)
    if t=='P01':
        premises=det(A)!=0 and det(B)!=0 and mul(sq(A),A)==mul(sq(B),B)
        condition={'invertible':det(A)!=0 and det(B)!=0,'equal_cubes':mul(sq(A),A)==mul(sq(B),B)}
    else:
        premises=sq(A)==A and sq(B)==B and A[0][0]+A[1][1]==B[0][0]+B[1][1]
        condition={'both_idempotent':sq(A)==A and sq(B)==B,'equal_trace':A[0][0]+A[1][1]==B[0][0]+B[1][1]}
    return {'premises_hold':premises,'conditions':condition,'commute':AB==BA,'counterexample_verified':premises and AB!=BA,'AB':AB,'BA':BA}

def _bound(t,inp):
    if set(inp)!={'x','y'}:raise ValueError('REQUIRED_FIELDS_x_y')
    x=finite(inp['x'],-3,3);y=finite(inp['y'],-3,3)
    if t=='X01':
        lhs=abs(x-y);rhs=math.hypot(x,y);viol=lhs>rhs+1e-10
        return {'lhs':round(lhs,9),'rhs':round(rhs,9),'bound_holds':not viol,'counterexample_verified':viol}
    lhs=(x+y)**2;rhs=2*(x*x+y*y)
    return {'lhs':round(lhs,9),'rhs':round(rhs,9),'inequality_holds':lhs<=rhs+1e-10,'equality_observed':abs(lhs-rhs)<=1e-10,'claimed_only_zero_equality':abs(x)<=1e-10 and abs(y)<=1e-10}

def _portfolio(t,inp):
    if set(inp)!={'x','z'}:raise ValueError('REQUIRED_FIELDS_x_z')
    x=finite(inp['x'],0,3);z=finite(inp['z'],0,3)
    r=(.7*x/(.4+x)+.28*z*z/(1+z*z)) if t=='R01' else (1.25*x/(.3+x)/(1+.6*z)+.05*z*x)
    return {'observed_response':round(r,9)}

_RAW={
 'E01':{
  'LAB_A_01':{'source':'A','raw_signal':21.0,'blank':1.0,'calibration_factor':0.05,'concentration':1.0},
  'LAB_A_02':{'source':'A','raw_signal':34.0,'blank':2.0,'calibration_factor':0.05,'concentration':2.0},
  'LAB_B_01':{'source':'B','raw_signal':10.4,'blank':0.4,'calibration_factor':0.1,'concentration':1.0},
  'LAB_B_02':{'source':'B','raw_signal':16.8,'blank':0.8,'calibration_factor':0.1,'concentration':2.0},
  'LAB_C_01':{'source':'C','raw_signal':13.0,'blank':3.0,'calibration_factor':0.1,'concentration':1.0},
  'LAB_C_02':{'source':'C','raw_signal':19.0,'blank':3.0,'calibration_factor':0.1,'concentration':2.0}},
 'E02':{
  'LAB_D_01':{'source':'D','raw_signal':120,'blank':20,'calibration_factor':0.01,'stimulus':0.5,'measurement_direction':'increasing'},
  'LAB_D_02':{'source':'D','raw_signal':155,'blank':20,'calibration_factor':0.01,'stimulus':1.0,'measurement_direction':'increasing'},
  'LAB_F_01':{'source':'F','raw_signal':58,'blank':8,'calibration_factor':0.02,'stimulus':0.5,'measurement_direction':'decreasing'},
  'LAB_F_02':{'source':'F','raw_signal':68,'blank':8,'calibration_factor':0.02,'stimulus':1.0,'measurement_direction':'decreasing'}}
}

def tool_spec(t):
    if t in ('G01','G02'):return 'OBSERVE inputs exactly {"x0":number,"y0":number,"u":number}, all in [0,2.5]. Response: initial derivatives and states after 0.3 time units.'
    if t in ('I01','I02'):return 'OBSERVE inputs exactly {"c":number,"i":number}, each in [0,3]. Response: measured rate.'
    if t in ('P01','P02'):return 'CHECK inputs exactly {"A":[[int,int],[int,int]],"B":[[int,int],[int,int]]}, integer entries abs <=12. Response: exact premise check and AB-vs-BA witness.'
    if t in ('X01','X02'):return 'CHECK inputs exactly {"x":number,"y":number}, each in [-3,3]. Response: evaluated inequality and counterexample checks.'
    if t in ('R01','R02'):return 'OBSERVE inputs exactly {"x":number,"z":number}, each in [0,3]. Response: response.'
    return 'LOOKUP inputs exactly {"record_id":string}. Available IDs: '+', '.join(sorted(_RAW[t]))

def experiment(task,request):
    # Tool gets a scientific action, not a scorer query. Max 4 instrument actions per episode enforced by caller.
    try:
        if task not in IDS:raise ValueError('UNKNOWN_TASK')
        if type(request)!=dict or set(request)!={'operation','inputs'}:raise ValueError('REQUIRES_EXACT_OPERATION_AND_INPUTS')
        op=request['operation'];inp=request['inputs']
        if type(inp)!=dict:raise ValueError('INPUTS_MUST_BE_OBJECT')
        must='CHECK' if task[0] in 'PX' else 'LOOKUP' if task[0]=='E' else 'OBSERVE'
        if op!=must:raise ValueError('INVALID_OPERATION_FOR_TASK:'+must)
        if task[0]=='G':result=_observe_dynamic(task,inp)
        elif task[0]=='I':result=_kinetics(task,inp)
        elif task[0]=='P':result=_matrix(task,inp)
        elif task[0]=='X':result=_bound(task,inp)
        elif task[0]=='R':result=_portfolio(task,inp)
        else:
            if set(inp)!={'record_id'} or type(inp['record_id'])!=str:raise ValueError('REQUIRED_record_id')
            if inp['record_id'] not in _RAW[task]:raise ValueError('UNKNOWN_RECORD_ID')
            result=_RAW[task][inp['record_id']]
        return {'status':'COMPUTED','request':request,'result':result,'source':'M31_SYNTHETIC_INSTRUMENT_NOT_LAB_MEASUREMENT'}
    except (ValueError,KeyError,TypeError,OverflowError,ZeroDivisionError) as ex:
        return {'status':'REJECTED_INVALID_REQUEST','request':request,'error':str(ex)[:350],'source':'M31_SYNTHETIC_INSTRUMENT_NOT_LAB_MEASUREMENT'}

def initial_data(t):
    if t[0]=='G':
        return [{'x0':x,'y0':y,'u':u,**_observe_dynamic(t,{'x0':x,'y0':y,'u':u})} for x,y,u in [(.5,.7,.4),(.8,.4,1.1)]]
    if t[0]=='I':
        return [{'c':c,'i':i,**_kinetics(t,{'c':c,'i':i})} for c,i in [(0.3,0),(1,0),(0.3,.15),(1,.15)]]
    if t[0]=='R':
        return [{'x':x,'z':z,**_portfolio(t,{'x':x,'z':z})} for x,z in [(.4,.0),(.6,.2),(.8,.0)]]
    if t[0]=='E':
        return [{'source':source,'summary_trend':trend} for source,trend in ([('LAB_A','increasing'),('LAB_B','nearly-flat'),('LAB_C','increasing')] if t=='E01' else [('LAB_D','high response'),('LAB_F','lower response')])]
    return []

def public_packet(t):
    if t not in IDS:raise ValueError(t)
    question,tool=TASKS[t]
    return {'schema_version':'m30r-generic-frozen-problem-v1','problem_id':'M31_PILOT_'+t,
            'problem_kind':'SYNTHETIC_CONTROLLED' if t[0] not in 'PX' else 'FORMAL_PROBLEM',
            'first_source_id':'EVID:PROBLEM',
            'scientific_question':question,
            'sources':{'pages':[{'id':'EVID:PROBLEM','title':'Synthetic M31 development pilot '+t,
              'text':json.dumps({'problem':question,'initial_observations':initial_data(t),
                                 'instrument_accessible_when_enabled':tool,
                                 'scope':'SYNTHETIC_OR_FORMAL_DEVELOPMENT_ONLY'},sort_keys=True,ensure_ascii=False),
              'source_kind':'FORMAL_SPECIFICATION' if t[0] in 'PX' else 'PRIMARY_OBSERVATION','provenance':'AUTHOR_PROVIDED_PRIMARY'}],
              'works':[]}}

def check_consistency(t,packet):
    return t in IDS and packet==public_packet(t)

if __name__=='__main__':
    print(json.dumps({t:public_packet(t) for t in IDS},indent=2))
