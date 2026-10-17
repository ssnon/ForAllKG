"""M24: one autonomous physical constraint proposal, deterministic log-linear feasibility.

An opt-in, add-only experiment, not a research manager. No inference of GNR-PVA-specific
bounds from perspective-review numeric statements. Scientifically unverified constraints
from the LLM are assumptions. Raw paid replies are written BEFORE any validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Literal

import numpy as np
from scipy.optimize import linprog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base
from scripts.discovery import run_m23_evidence_constrained_reasoning as m23

PROTOCOL='m24-evidence-feasible-domain-v1'
SOURCE_PAPER='literature:6e7e524ff6be84a37326'


def path(p):return Path(p).expanduser().resolve()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):base.write_new(Path(p),x)
def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(x):return hashlib.sha256(stable(x).encode()).hexdigest()


def _facts(anchors):
    """Narrow pre-registered extraction. NO free-text/LLM numeric assertions are trusted."""
    seen={a['id']:a for a in anchors}
    facts=[]
    if 'HOT:0' in seen:
        s=seen['HOT:0']['pdf_source_span']
        if re.search(r'1\s*[–-]\s*100\s*fs',s):
            facts.append({'fact_id':'HOT:DEPHASE_FS','anchor_id':'HOT:0',
                'symbol':'plasmon_dephasing_fs','meaning':'plasmon dephasing time (NOT trapped charge lifetime)',
                'unit':'fs','lower':1.,'upper':100.,'scope':'METAL_PLASMON_DEPHASING',
                'transfer_status':'REFERENCE_REPORTED_TIMESCALE_NOT_UNIVERSAL_YE_BOUND',
                'tier':'reported_context','verbatim_sha256':sha_file_bytes(s.encode())})
    if 'HOT:1' in seen:
        s=seen['HOT:1']['pdf_source_span']
        if re.search(r'typically\s*<\s*2\s*%',s,re.I):
            facts.append({'fact_id':'HOT:INDIRECT_ETA_TYPICAL','anchor_id':'HOT:1',
                'symbol':'eta_indirect','meaning':'hot electron injection efficiency via indirect pathway (NOT total charge trapping yield)',
                'unit':'fraction','lower':None,'upper':0.02,'scope':'INDIRECT_TRANSFER_TYPICAL_CASES',
                'transfer_status':'TYPICAL_EXAMPLE_NOT_A_STRICT_UPPER_BOUND_OR_YE_MEASUREMENT',
                'tier':'typical_scenario','verbatim_sha256':sha_file_bytes(s.encode())})
    return facts

def sha_file_bytes(b):return hashlib.sha256(b).hexdigest()


class Variable(BaseModel):
    model_config=ConfigDict(extra='forbid')
    symbol:str=Field(pattern=r'^[a-z][a-z0-9_]{0,35}$')
    meaning:str=Field(min_length=12,max_length=230)
    unit:str=Field(min_length=1,max_length=50)
    assumed_min:float|None=None
    assumed_max:float|None=None
    source_fact_id:str|None=None
    @model_validator(mode='after')
    def bounds(self):
        a,b=self.assumed_min,self.assumed_max
        if a is not None and (not math.isfinite(a) or a<=0):raise ValueError('LOG_VARIABLE_MIN_MUST_BE_POSITIVE')
        if b is not None and (not math.isfinite(b) or b<=0):raise ValueError('LOG_VARIABLE_MAX_MUST_BE_POSITIVE')
        if a is not None and b is not None and a>b:raise ValueError('BAD_DOMAIN')
        return self

class Constraint(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=5,max_length=90)
    powers:dict[str,float]=Field(min_length=1,max_length=8)
    relation:Literal['LE','GE']
    rhs:float=Field(gt=0)
    physical_justification:str=Field(min_length=25)
    status:Literal['MODEL_ASSUMPTION','HYPOTHETICAL_OBSERVATION']
    @model_validator(mode='after')
    def exponents(self):
        if not math.isfinite(self.rhs) or any(not math.isfinite(x) or abs(x)>6 or x==0 for x in self.powers.values()):
            raise ValueError('INVALID_MONOMIAL_CONSTRAINT')
        return self

class Target(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=5,max_length=110)
    powers:dict[str,float]=Field(min_length=1,max_length=8)
    positive_factor:float=Field(gt=0)
    units_as_assumed:str=Field(min_length=1)
    why_physically_relevant:str=Field(min_length=30)
    @model_validator(mode='after')
    def exponents(self):
        if not math.isfinite(self.positive_factor) or any(not math.isfinite(x) or abs(x)>6 or x==0 for x in self.powers.values()):
            raise ValueError('INVALID_MONOMIAL_TARGET')
        return self

class Proposal(BaseModel):
    model_config=ConfigDict(extra='forbid')
    selected_scientific_question:str=Field(min_length=30)
    independent_evidence_needed:str=Field(min_length=30)
    scientific_model_or_bridge:str=Field(min_length=40)
    scientific_scope:Literal['YE_GNR_PVA','GENERAL_PLASMONIC_CASE','OTHER_HYPOTHETICAL_CASE']
    variables:list[Variable]=Field(min_length=1,max_length=8)
    assumed_constraints:list[Constraint]=Field(default_factory=list,max_length=12)
    test_quantity:Target
    candidate_conclusion:str=Field(min_length=30)
    what_would_change_after_evidence:str=Field(min_length=30)
    unmeasured_quantities:list[str]=Field(min_length=1)
    alternative_explanation:str=Field(min_length=30)
    source_transfer_risks:list[str]=Field(min_length=1)
    @model_validator(mode='after')
    def refs(self):
        names=[v.symbol for v in self.variables]
        if len(names)!=len(set(names)):raise ValueError('DUPLICATE_VARIABLE')
        for item in [*self.assumed_constraints,self.test_quantity]:
            if not set(item.powers).issubset(names):raise ValueError('UNKNOWN_VARIABLE_IN_MONOMIAL')
        return self

SYS='''You are FAKG pursuing scientific understanding rather than generating another exploratory story.
Read the previous source-grounded research and strongest null. Independently choose one research
question and the physical quantity whose feasible range would matter. Encode a physically
motivated monomial constraint model that a deterministic log-linear solver can evaluate, e.g.
eta*phi*tau >= N_required via powers; or choose variables with a genuine independent source
bound. Do NOT assign arbitrary numerical values and represent them as source-reported facts.
The only source-reported numeric facts you may bind are the source_fact_ids in the supplied
catalog. If you bind a source_fact_id, copy that catalog entry's symbol, meaning,
and unit EXACTLY (character for character), and set source_fact_id to that fact_id.
If you cannot justify the binding, use source_fact_id=null and state the gap.
The plasmon 1-100 fs lifetime is NOT charge-trap residence time. Typical indirect
transfer <2% is NOT a strict universal upper bound or a GNR-PVA measurement. Other numeric
constants must be labeled MODEL_ASSUMPTION or HYPOTHETICAL_OBSERVATION, including units.
Use at most 8 positive variables, 12 monomial assumptions and ONE physically relevant target
for upper/lower feasible-bound optimization. This solver cannot handle sums/differences,
non-monomial dynamics or sign-changing variables; if your physics needs these, do not fake
an equivalent monomial: state the limitation clearly and choose a genuine restricted case.
Try to obtain a falsifiable necessary condition, or demonstrate why no defensible Ye-specific
constraint is yet available. Merely assigning different observables to rival models does NOT
constitute physics. Your result is a CANDIDATE; the computer will independently solve the
specified algebra but will not certify physical applicability. Return a single JSON object
exactly satisfying response_schema, no prefatory text.
'''


def prepare(m23_out,out):
    out=path(out);src=path(m23_out)
    if out.exists():raise FileExistsError('FRESH_OUTPUT_REQUIRED')
    state=m23.verify(src)
    if state['develop']!='READY' or state['challenge']!='READY':raise ValueError('M23_NOT_COMPLETE')
    ctx=read(src/'input.json');anchors=ctx['verified_SOURCE_EXCERPTS'];facts=_facts(anchors)
    if not facts:raise ValueError('NO_REGISTERED_SOURCE_NUMERIC_FACTS')
    origins={str(src/x):sha_file(src/x) for x in ('input.json','lock.json','develop/normalized.json','challenge/normalized.json')}
    packet={'research_subject':'Ye 2025 immobilized gold nanorod ensemble SERS blinking',
        'M23_autonomously_selected_question':read(src/'develop/normalized.json')['selected_question'],
        'M23_development_UNVERIFIED':read(src/'develop/normalized.json'),
        'M23_adversarial_review_UNVERIFIED':read(src/'challenge/normalized.json'),
        'M23_original_context':ctx,
        'registered_numeric_evidence_facts':facts,
        'critical_scope_note':'No Ye-specific charge yield, trapping lifetime, field-response calibration, or measured response amplitude has been registered as a direct numerical constraint. Reference facts cannot become hard Ye bounds.',
        'origin_hashes':origins,
        'input_epistemic_status':'UNVERIFIED_PRIOR_RESEARCH_AND_SOURCE_EXCERPTS_SEPARATED'}
    m23lock=read(src/'lock.json')
    out.mkdir(parents=True)
    try:
        write(out/'input.json',packet)
        write(out/'lock.json',{'schema':PROTOCOL,'runner_sha256':sha_file(__file__),
            'input_sha256':sha_file(out/'input.json'),
            'model':m23lock['model'],'temperature':m23lock['temperature'],
            'max_tokens':max(5500,m23lock['max_tokens']),'scientific_validation':False})
        verify(out)
    except BaseException:
        shutil.rmtree(out);raise
    return {'status':'M24_PREPARED_EXISTING_SOURCE_NO_PAID_CALL','registered_facts':len(facts),
            'original_research_unchanged':True,'paid_calls':0,'out':str(out)}


def ctxcheck(out):
    out=path(out)
    packet=read(out/'input.json');lock=read(out/'lock.json')
    if lock['schema']!=PROTOCOL or lock['runner_sha256']!=sha_file(__file__) or lock['input_sha256']!=sha_file(out/'input.json'):
        raise ValueError('M24_LOCK_OR_INPUT_CHANGED')
    for origin,h in packet['origin_hashes'].items():
        if sha_file(origin)!=h:raise ValueError('M24_ORIGINAL_RESEARCH_CHANGED')
    return out,packet,lock


def request(out):
    _,pkt,lock=ctxcheck(out)
    payload={'research_state_and_literature_evidence':{k:v for k,v in pkt.items() if k!='origin_hashes'},
             'numeric_source_fact_ids_and_scopes':pkt['registered_numeric_evidence_facts'],
             'response_schema':Proposal.model_json_schema()}
    return {'model':lock['model'],'temperature':lock['temperature'],
        'max_tokens':lock['max_tokens'],'messages':[{'role':'system','content':SYS},
        {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}


def parse(raw,pkt):
    obj=Proposal.model_validate_json(raw).model_dump(mode='json')
    facts={f['fact_id']:f for f in pkt['registered_numeric_evidence_facts']}
    used=[]
    for v in obj['variables']:
        ref=v['source_fact_id']
        if ref is None:continue
        if ref not in facts:raise ValueError('NONEXISTENT_SOURCE_FACT_ID')
        f=facts[ref]
        if v['symbol']!=f['symbol'] or v['unit']!=f['unit'] or v['meaning']!=f['meaning']:
            raise ValueError('SOURCE_NUMERIC_SEMANTIC_REBINDING_REFUSED')
        if ref in used:raise ValueError('DUPLICATED_SOURCE_FACT_BINDING')
        used.append(ref)
    return obj


def _row(powers,vs):return np.array([powers.get(v,0.) for v in vs],dtype=float)

def _solve(model,active_facts):
    variables=model['variables'];names=[x['symbol'] for x in variables]
    bounds=[(math.log(v['assumed_min']) if v['assumed_min'] is not None else None,
             math.log(v['assumed_max']) if v['assumed_max'] is not None else None) for v in variables]
    rows=[];rhs=[]
    def add(powers,rel,limit):
        ar=_row(powers,names)
        if rel=='GE':ar=-ar;limit=1/limit
        rows.append(ar);rhs.append(math.log(limit))
    for c in model['assumed_constraints']:add(c['powers'],c['relation'],c['rhs'])
    for v in variables:
        fid=v['source_fact_id']
        if fid not in active_facts:continue
        fact=active_facts[fid]
        if fact['lower'] is not None:add({v['symbol']:1},'GE',fact['lower'])
        if fact['upper'] is not None:add({v['symbol']:1},'LE',fact['upper'])
    A=np.array(rows) if rows else None;b=np.array(rhs) if rhs else None
    obj=_row(model['test_quantity']['powers'],names)
    # check feasibility first; SLSQP/noisy sampling is not used.
    feas=linprog(np.zeros(len(names)),A_ub=A,b_ub=b,bounds=bounds,method='highs')
    if feas.status==2:return {'status':'INFEASIBLE_ASSUMPTION_SET','scientific_validation':False}
    if feas.status!=0:return {'status':'SOLVER_UNCERTAIN','solver_code':int(feas.status),'scientific_validation':False}
    ends=[]
    for sign in (1,-1):
        fit=linprog(sign*obj,A_ub=A,b_ub=b,bounds=bounds,method='highs')
        if fit.status==3:ends.append(None);continue
        if fit.status!=0:return {'status':'SOLVER_UNCERTAIN','solver_code':int(fit.status),'scientific_validation':False}
        val=float(np.dot(obj,fit.x)+math.log(model['test_quantity']['positive_factor']))
        ends.append({'log_value':round(val,12),
                     'value':(math.exp(val) if -700<val<700 else None)})
    return {'status':'FEASIBLE_CONDITIONAL_LOG_LINEAR_DOMAIN',
            'lower_bound':ends[0],'upper_bound':ends[1],
            'proven_assuming_encoded_inequalities_only':True,
            'numerical_solver':'scipy.optimize.linprog(method=highs)',
            'scientific_validation':False}


def _tighter(base_res,other):
    if base_res['status']!='FEASIBLE_CONDITIONAL_LOG_LINEAR_DOMAIN' or other['status']!='FEASIBLE_CONDITIONAL_LOG_LINEAR_DOMAIN':return False
    for k in ('lower_bound','upper_bound'):
        a=base_res[k];b=other[k]
        if a is None and b is not None:return True
        if a and b:
            diff=b['log_value']-a['log_value']
            if (k=='lower_bound' and diff>1e-7) or (k=='upper_bound' and diff < -1e-7):return True
    return False


def evaluate(model,pkt):
    allfacts={f['fact_id']:f for f in pkt['registered_numeric_evidence_facts']}
    direct={k:v for k,v in allfacts.items() if v['tier']=='reported_context'}
    typical={k:v for k,v in allfacts.items() if v['tier']=='typical_scenario'}
    bas=_solve(model,{})
    context=_solve(model,direct)
    case=_solve(model,direct|typical)
    context_cut=_tighter(bas,context)
    typical_cut=_tighter(context,case)
    used=[v['source_fact_id'] for v in model['variables'] if v['source_fact_id']]
    return {
      'status':'M24_CONDITIONAL_FEASIBLE_DOMAIN_AUDITED_NO_CALL',
      'question':model['selected_scientific_question'],
      'test_quantity':model['test_quantity'],
      'model_assumption_only':bas,
      'with_review_reported_context':context,
      'with_typical_indirect_transfer_scenario':case,
      'source_context_changes_feasible_domain':context_cut,
      'typical_scenario_changes_feasible_domain':typical_cut,
      'used_source_fact_ids':used,
      'hard_ye_specific_source_constraints_available':False,
      'hard_ye_specific_source_constraints_added':0,
      'YE_case_feasible_domain_reduced_by_independent_measurement':False,
      'interpretation':'Physical/causal validity of proposed monomial relations NOT verified. Review-reported ranges are not case-specific; typical <2% is not strict. Differences from model-only domain are conditional reference/scenario comparisons, NOT Ye exclusions.',
      'algorithm_limits':'Positive monomial variables only; no sums, units proof, empirical calibration, or scientific-discovery certification.',
      'scientific_validation':False,'level_ii_certified':False}


def verify(out):
    out,pkt,_=ctxcheck(out);p=out/'propose'
    if not p.exists():return {'status':'M24_VERIFIED_NO_CALL','proposal':'NOT_STARTED','scientific_validation':False}
    req=request(out)
    if read(p/'request.json')!=req or read(p/'attempt.json')['request_sha256']!=sha(req):raise ValueError('M24_REQUEST_CHANGED')
    if not (p/'raw_response.json').exists():status='BILLING_UNCERTAIN_NO_RETRY'
    elif not (p/'normalized.json').exists():status='RAW_SAVED_RECOVER_NO_REPAY'
    else:
        model=parse(read(p/'raw_response.json')['raw_text'],pkt)
        if read(p/'normalized.json')!=model:raise ValueError('M24_NORMALIZED_CHANGED')
        if not (p/'feasibility.json').exists():status='RAW_SAVED_RECOVER_NO_REPAY'
        elif read(p/'feasibility.json')!=evaluate(model,pkt):raise ValueError('M24_FEASIBILITY_CHANGED')
        elif not (p/'receipt.json').exists():status='RAW_SAVED_RECOVER_NO_REPAY'
        else:
            exp={'request_sha256':sha(req),'raw_file_sha256':sha_file(p/'raw_response.json'),
                'normalized_file_sha256':sha_file(p/'normalized.json'),
                'feasibility_file_sha256':sha_file(p/'feasibility.json')}
            if read(p/'receipt.json')!=exp:raise ValueError('M24_RECEIPT_CHANGED')
            status='READY'
    return {'status':'M24_VERIFIED_NO_CALL','proposal':status,'scientific_validation':False}


def preview(out):
    s=verify(out)
    if s['proposal']!='NOT_STARTED':raise ValueError('ALREADY_STARTED_DO_NOT_REPAY')
    return {'status':'M24_PREVIEW_NO_CALL','next_call_is_paid':True,'request_sha256':sha(request(out))}


def recover(out):
    out,pkt,_=ctxcheck(out);p=out/'propose';rawfile=p/'raw_response.json'
    if not rawfile.is_file():raise RuntimeError('NO_SAVED_RAW_DO_NOT_INFER_BILLING')
    if read(p/'request.json')!=request(out):raise ValueError('SAVED_REQUEST_CHANGED')
    previous=sha_file(rawfile)
    model=parse(read(rawfile)['raw_text'],pkt)
    if not (p/'normalized.json').exists():write(p/'normalized.json',model)
    elif read(p/'normalized.json')!=model:raise ValueError('M24_MODEL_CONFLICT')
    res=evaluate(model,pkt)
    if not (p/'feasibility.json').exists():write(p/'feasibility.json',res)
    elif read(p/'feasibility.json')!=res:raise ValueError('M24_FEASIBILITY_CONFLICT')
    receipt={'request_sha256':sha(request(out)),'raw_file_sha256':previous,
             'normalized_file_sha256':sha_file(p/'normalized.json'),
             'feasibility_file_sha256':sha_file(p/'feasibility.json')}
    if not (p/'receipt.json').exists():write(p/'receipt.json',receipt)
    elif read(p/'receipt.json')!=receipt:raise ValueError('M24_RECEIPT_CONFLICT')
    if sha_file(rawfile)!=previous:raise ValueError('ORIGINAL_PAID_RAW_CHANGED')
    s=verify(out)
    return {'status':'M24_SAVED_FEASIBILITY_RECOVERED_NO_REPAY','proposal':s['proposal'],
            'scientific_validation':False,'new_paid_calls':0}


def propose(out,allow_paid=False,backend=None):
    preview(out)
    base._require_paid(backend,allow_paid,1)
    out=path(out);p=out/'propose';p.mkdir()
    req=request(out);write(p/'request.json',req)
    write(p/'attempt.json',{'request_sha256':sha(req),'status':'BILLING_UNKNOWN_UNTIL_RAW_SAVED'})
    raw,usage,rid=base._invoke(req,backend)
    write(p/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:res=recover(out)
    except Exception as exc:raise RuntimeError('M24_PAID_RAW_SAVED_DO_NOT_REPAY_USE_RECOVER') from exc
    return res|{'new_paid_calls':int(backend is None)}


def report(out):
    if verify(out)['proposal']!='READY':raise ValueError('M24_NOT_READY')
    out=path(out);f=read(out/'propose/feasibility.json')
    return {'status':'M24_CONSTRAINT_RESEARCH_RECORDED_NO_CALL',
        'autonomous_scientific_question':f['question'],
        'source_context_changes_feasible_domain':f['source_context_changes_feasible_domain'],
        'typical_scenario_changes_feasible_domain':f['typical_scenario_changes_feasible_domain'],
        'YE_case_feasible_domain_reduced_by_independent_measurement':False,
        'feasibility_results':f,
        'model_proposal_UNVERIFIED':read(out/'propose/normalized.json'),
        'science_and_novelty_not_certified':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','preview','propose','recover','verify','report'])
    ap.add_argument('--m23',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--allow-paid',action='store_true')
    args=ap.parse_args()
    if args.command=='prepare':
        if args.m23 is None:ap.error('prepare needs --m23')
        ans=prepare(args.m23,args.out)
    elif args.command=='preview':ans=preview(args.out)
    elif args.command=='propose':ans=propose(args.out,allow_paid=args.allow_paid)
    elif args.command=='recover':ans=recover(args.out)
    elif args.command=='verify':ans=verify(args.out)
    else:ans=report(args.out)
    print(json.dumps(ans,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
