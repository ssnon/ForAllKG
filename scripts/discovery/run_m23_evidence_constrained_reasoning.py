"""M23: FAKG source-grounded derivation + adversarial competing-mechanism test.

Add-only stand-alone scientific experiment, NOT a controller or a modification to M21/M22.
Read full PDF before trusting saved M22 anchor text. The two actual model calls are
separate and always save raw bytes BEFORE strict validation. One call per invocation.
Algebraic numeric checks are diagnostic, NOT certified scientific implications.
"""
from __future__ import annotations
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypdf import PdfReader
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base

SCHEMA='m23-evidence-constrained-reasoning-v1'
PAPER_ID='literature:6e7e524ff6be84a37326'


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canon(obj):return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def hashobj(obj):return sha256(canon(obj).encode()).hexdigest()
def put(p,x):base.write_new(Path(p),x)
def root(p):return Path(p).expanduser().resolve()

# PDF line-ending discretionary hyphens are formatting only; DO NOT remove en/em dashes,
# punctuation, digits or substitute scientific symbols. Removing whitespace also matches
# a missing word boundary, but MUST uniquely match a long consecutive source span.
def source_with_offsets(text):
    chars=[];positions=[];i=0
    while i<len(text):
        c=text[i]
        if c=='-' and i+1<len(text) and text[i+1]=='\n' and i>0 and text[i-1].isalnum():
            i+=2
            while i<len(text) and text[i] in ' \t':i+=1
            continue
        if not c.isspace():chars.append(c);positions.append(i)
        i+=1
    return ''.join(chars),positions


def locate_verbatim(text,claimed):
    needle,_=source_with_offsets(claimed)
    hay,offset=source_with_offsets(text)
    if len(needle)<45:raise ValueError('SHORT_SOURCE_ANCHOR_REQUIRES_HUMAN_REVIEW')
    pos=hay.find(needle)
    if pos<0:raise ValueError('CLAIMED_ANCHOR_NOT_FOUND_IN_ACTUAL_PDF')
    if hay.find(needle,pos+1)>=0:raise ValueError('AMBIGUOUS_ANCHOR_IN_ACTUAL_PDF')
    original=text[offset[pos]:offset[pos+len(needle)-1]+1]
    return {'exact_pdf_excerpt':original,'start_offset':offset[pos],
            'end_offset':offset[pos+len(needle)-1]+1}


def snapshot(m21,charge,pdf,thermal=None):
    m21=root(m21);charge=root(charge);pdf=root(pdf)
    required={'m21_step6':m21/'step_0006/normalized.json',
              'charge_raw':charge/'analyze/raw_response.json',
              'charge_evidence':charge/'pdf_evidence.json',
              'charge_plan':charge/'plan/normalized.json',
              'charge_pdf':pdf}
    if thermal is not None:
        thermal=root(thermal)
        required['thermal_analysis']=thermal/'analyze/normalized.json'
    for name,file in required.items():
        if not file.is_file():raise FileNotFoundError(f'{name}: {file}')
    evidence=load(required['charge_evidence'])
    if evidence.get('status')!='PDF_PAGES_RETRIEVED' or evidence.get('paper_id')!=PAPER_ID:
        raise ValueError('CHARGE_SOURCE_PAPER_OR_FETCH_STATE_INVALID')
    if evidence.get('pdf_sha256')!=digest(pdf):raise ValueError('PDF_BYTES_DO_NOT_MATCH_M22_SAVED_FETCH')
    pdf_reader=PdfReader(str(pdf))
    raw=load(required['charge_raw'])['raw_text']
    analysis=json.loads(raw)
    if not isinstance(analysis,dict) or not isinstance(analysis.get('anchors'),list) or not analysis['anchors']:
        raise ValueError('M22_RAW_ANALYSIS_INVALID')
    scan_pages={p['page'] for p in evidence.get('selected_pages',[])}
    anchors=[]
    for i,a in enumerate(analysis['anchors']):
        page=a['page']
        if not isinstance(page,int) or page not in scan_pages or page<1 or page>len(pdf_reader.pages):
            raise ValueError('ANCHOR_PAGE_OUTSIDE_SAVED_SEARCH')
        full=pdf_reader.pages[page-1].extract_text() or ''
        match=locate_verbatim(full,a['exact_source_span'])
        anchors.append({'id':f'HOT:{i}', 'paper_id':PAPER_ID,'pdf_page_in_part':page,
             'pdf_source_span':match['exact_pdf_excerpt'],
             'source_location':{'start':match['start_offset'],'end':match['end_offset']},
             'source_conditions_as_interpreted_BY_M22_UNVERIFIED':a.get('source_conditions'),
             'physical_relation_or_parameter_as_interpreted_BY_M22_UNVERIFIED':a.get('physical_relation_or_parameter'),
             'source_sha256':sha256(match['exact_pdf_excerpt'].encode()).hexdigest()})
    m21_step=load(required['m21_step6'])
    prior=load(required['charge_plan'])
    source_files={str(p):digest(p) for p in required.values()}
    thermal_data=load(required['thermal_analysis']) if thermal else None
    pkt={
      'research_subject':'Ye 2025 gold-nanorod PVA-embedded ensemble SERS blinking',
      'previous_scientific_execution_STEP6_UNVERIFIED':m21_step,
      'autonomous_M22_question':prior.get('research_question'),
      'autonomous_M22_missing_constraint':prior.get('missing_physical_constraint'),
      'M22_source_interpretation_UNVERIFIED':{k:analysis.get(k) for k in (
         'attempted_question','scientific_relation','conditions_and_domain',
         'what_is_not_supported','research_next_action','epistemic_status')},
      'verified_SOURCE_EXCERPTS':anchors,
      'earlier_photothermal_M22_interpretation_UNVERIFIED':(
        {k:thermal_data.get(k) for k in ('scientific_relation','conditions_and_domain','what_is_not_supported','calculation')} if thermal_data else None),
      'M22_calculation_status':'NO_DEFENSIBLE_NUMERIC_CALCULATION_SUBMITTED' if analysis.get('calculation') is None else 'HAS_UNVERIFIED_CALCULATION',
      'epistemic_rule':'Only verified_SOURCE_EXCERPTS constitute direct cited evidence. M21/M22 explanations and any model relations are hypotheses, not source-verified facts.',
      'source_file_hashes':source_files,
    }
    return pkt


class Premise(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['SOURCE','SOURCE_INTERPRETATION','MODEL_ASSUMPTION','YE_OBSERVATION','MATHEMATICAL']
    statement:str=Field(min_length=10)
    source_ids:list[str]=Field(default_factory=list)
    applicability_limit:str=Field(min_length=5)

class Probe(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['ARITHMETIC_IDENTITY','NUMERIC_COUNTEREXAMPLE']
    left_expression:str=Field(min_length=1,max_length=220)
    right_expression:str=Field(min_length=1,max_length=220)
    variables:dict[str,float]=Field(min_length=1,max_length=12)
    relation:Literal['EQ','LE','GE','NE']
    physical_meaning:str=Field(min_length=12)

class Develop(BaseModel):
    model_config=ConfigDict(extra='forbid')
    selected_question:str=Field(min_length=25)
    scientific_action:Literal['DEVELOP','REFUTE','RESTRICT','PIVOT','ABSTAIN']
    reason_for_selection:str=Field(min_length=35)
    premises:list[Premise]=Field(min_length=1,max_length=12)
    derivation_steps:list[str]=Field(min_length=1,max_length=12)
    result_kind:Literal['NECESSARY_CONDITION','BOUND','CONDITIONAL_PREDICTION','COUNTEREXAMPLE','IDENTIFIABILITY_LIMIT','NO_NEW_RESULT']
    explicit_result:str=Field(min_length=30)
    physical_variable_definitions:str=Field(min_length=20)
    strongest_competing_explanation:str=Field(min_length=40)
    testable_discriminator_or_obstruction:str=Field(min_length=40)
    unsupported_leaps:list[str]
    numeric_or_symbolic_probe:Probe|None=None
    what_changed_from_M21_and_M22:str=Field(min_length=40)
    epistemic_status:Literal['CONDITIONAL_MODEL_RESULT','SOURCE_CONSTRAINED_RESULT','INCONCLUSIVE']

class Challenge(BaseModel):
    model_config=ConfigDict(extra='forbid')
    strongest_null_mechanism:str=Field(min_length=40)
    attempted_counterexample_or_derivation:str=Field(min_length=50)
    constraints_shared_by_both_models:list[str]=Field(min_length=1)
    previous_result_survives:Literal['YES_CONDITIONALLY','NO','UNDETERMINED']
    survival_conditions:list[str]
    objection_or_repair:str=Field(min_length=30)
    new_research_information:str=Field(min_length=30)
    revised_falsification_or_measurement:str=Field(min_length=40)
    final_disposition:Literal['RETAIN_CONDITIONAL','REVISE','REJECT','ABSTAIN']
    evidence_ids_used:list[str]

DEVELOP_SYSTEM='''You are the scientific reasoning agent FAKG, continuing your *own* previous research.
Select the most promising question yourself (continuation, refutation or pivot are all allowed).
Do NOT merely summarize literature or propose future experiments. Try to derive a new explicit conditional
necessary condition, upper/lower bound, counterexample or identifiability obstruction, with steps and
independent physical assumptions. If that is impossible, report NO_NEW_RESULT and identify exactly why.
Cite only HOT:N IDs for directly sourced facts. DO NOT copy passages into citations or introduce imaginary
source coefficients. Before applying a reported number, distinguish its actual material, process, measurement and
range of validity from the mechanism being researched. No source is a universal law by default.
Strong nulls must be given physically permissible observables, not (0,0,0) by fiat.
An optional arithmetic probe is for *testing* a result, NOT proof and not a forced target.
Output ONE JSON object matching response_schema exactly, no preamble.
'''
CHALLENGE_SYSTEM='''You are the next scientific research pass in FAKG, critically challenging the earlier derivation.
Your mission is not to approve the hypothesis. Construct the strongest physically allowed competing
mechanism or countermodel compatible with the same source evidence, and attempt to defeat the claimed
new result. Do not assume the thermal/structural null has zero signed optical or molecular response.
Separate algebraic validity, independent source support, and causal identifiability.
If insufficient, abstain/revise. This is not a novelty or discovery certificate.
Use anchor IDs for source claims; do not rewrite purported verbatim quotations.
Output ONE JSON object matching response_schema exactly.
'''


def calc_expr(src,vars):
    if len(src)>220:raise ValueError('EXPRESSION_OVERLONG')
    tree=ast.parse(src,mode='eval')
    def f(node):
        if isinstance(node,ast.Expression):return f(node.body)
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)) and not isinstance(node.value,bool):
            if abs(node.value)>1e12:raise ValueError('CONSTANT_TOO_LARGE')
            return float(node.value)
        if isinstance(node,ast.Name):return float(vars[node.id])
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return (1 if isinstance(node.op,ast.UAdd) else -1)*f(node.operand)
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            l,r=f(node.left),f(node.right)
            if isinstance(node.op,ast.Add):v=l+r
            elif isinstance(node.op,ast.Sub):v=l-r
            elif isinstance(node.op,ast.Mult):v=l*r
            elif isinstance(node.op,ast.Div):v=l/r
            else:
                if abs(r)>8:raise ValueError('POWER_EXPONENT_TOO_LARGE')
                v=l**r
            if isinstance(v,complex) or abs(v)>1e100:raise ValueError('UNSTABLE_ARITHMETIC')
            return v
        raise ValueError('UNSUPPORTED_EXPRESSION_TOKEN')
    return f(tree)


def evaluate(probe):
    if probe is None:return {'status':'NO_ARITHMETIC_PROBE_SUPPLIED','scientific_validation':False}
    try:
        p=Probe.model_validate(probe)
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,31}',x) for x in p.variables):
            raise ValueError('INVALID_SYMBOL_NAME')
        left=calc_expr(p.left_expression,p.variables);right=calc_expr(p.right_expression,p.variables)
        tolerance=1e-9*max(1,abs(left),abs(right));d=left-right
        passed={'EQ':abs(d)<=tolerance,'NE':abs(d)>tolerance,'LE':d<=tolerance,'GE':d>=-tolerance}[p.relation]
        return {'status':'DIAGNOSTIC_NUMERIC_PROBE_PASSED' if passed else 'DIAGNOSTIC_NUMERIC_PROBE_FAILED',
                'lhs':left,'rhs':right,'relation':p.relation,
                'applies_only_to_declared_values':True,'scientific_validation':False,
                'warning':'A single numeric test does not prove a symbolic bound, physical causality, or novelty.'}
    except Exception as e:
        return {'status':'PROBE_REJECTED','error':type(e).__name__+': '+str(e)[:140],
                'scientific_validation':False}


def context_ok(out):
    out=root(out)
    lock=load(out/'lock.json')
    if lock['schema']!=SCHEMA:raise ValueError('M23_SCHEMA_CHANGED')
    if digest(Path(__file__))!=lock['runner_sha256']:
        raise ValueError('M23_RUNNER_CHANGED')
    if digest(out/'input.json')!=lock['input_sha256']:
        raise ValueError('M23_FROZEN_INPUT_CHANGED')
    ctx=load(out/'input.json')
    for loc,expected in ctx['source_file_hashes'].items():
        if digest(loc)!=expected:raise ValueError('M23_ORIGINAL_SOURCE_CHANGED:'+loc)
    return out,lock,ctx


def prepare(m21,charge,pdf,out,thermal=None):
    out=root(out)
    if out.exists():raise FileExistsError('FRESH_OUTPUT_REQUIRED')
    ctx=snapshot(m21,charge,pdf,thermal)
    model_source=root(charge)/'lock.json'
    model_lock=load(model_source)
    if not all(k in model_lock for k in ('model','temperature','max_tokens')):
        raise ValueError('M22_MODEL_LOCK_INCOMPATIBLE')
    out.mkdir(parents=True)
    try:
        put(out/'input.json',ctx)
        put(out/'lock.json',{'schema':SCHEMA,'runner_sha256':digest(__file__),
            'input_sha256':digest(out/'input.json'),'model':model_lock['model'],
            'temperature':model_lock['temperature'],'max_tokens':model_lock['max_tokens'],
            'scientific_validation':False})
        verify(out)
    except BaseException:
        shutil.rmtree(out);raise
    return {'status':'M23_PREPARED_SOURCE_ANCHORED_NO_CALL','out':str(out),
            'verified_pdf_anchors':len(ctx['verified_SOURCE_EXCERPTS']),
            'science_source_unmodified':True,'paid_calls':0}


def request(out,phase):
    _,lock,ctx=context_ok(out)
    if phase=='develop':
        payload={'source_context':{k:v for k,v in ctx.items() if k!='source_file_hashes'},
                 'response_schema':Develop.model_json_schema(),
                 'purpose':'Derive a genuinely informative physical/scientific constraint using independently identified evidence and explicitly stated assumptions.'}
        system=DEVELOP_SYSTEM
    elif phase=='challenge':
        prior=load(root(out)/'develop/normalized.json')
        computed=load(root(out)/'develop/numeric_probe.json')
        payload={'source_context':{k:v for k,v in ctx.items() if k!='source_file_hashes'},
                 'prior_scientific_derivation_UNVERIFIED':prior,
                 'prior_numeric_diagnostic_NOT_PROOF':computed,
                 'response_schema':Challenge.model_json_schema()}
        system=CHALLENGE_SYSTEM
    else:raise ValueError('UNKNOWN_PHASE')
    return {'model':lock['model'],'temperature':lock['temperature'],
        'max_tokens':lock['max_tokens'],
        'messages':[{'role':'system','content':system},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False,sort_keys=True)}]}


def _validate_sources(phase,normalized,ctx):
    valid={a['id'] for a in ctx['verified_SOURCE_EXCERPTS']}
    refs=[]
    if phase=='develop':
        for p in normalized['premises']:
            for x in p['source_ids']:
                if x not in valid:raise ValueError('UNREGISTERED_SOURCE_ANCHOR')
            if p['kind']=='SOURCE' and not p['source_ids']:
                raise ValueError('SOURCE_PREMISE_MISSING_VERIFIED_ANCHOR_ID')
            if p['kind']!='SOURCE' and p['source_ids'] and p['kind'] not in ('SOURCE_INTERPRETATION',):
                raise ValueError('NON_SOURCE_PREMISE_MISREPRESENTS_SOURCE')
            refs+=p['source_ids']
        if normalized['epistemic_status']=='SOURCE_CONSTRAINED_RESULT' and not refs:
            raise ValueError('SOURCE_CONSTRAINED_WITHOUT_SOURCE_PREMISE')
    else:
        refs=normalized['evidence_ids_used']
        if any(x not in valid for x in refs):raise ValueError('UNREGISTERED_CHALLENGE_EVIDENCE')


def parse(phase,raw,ctx):
    spec=Develop if phase=='develop' else Challenge
    obj=spec.model_validate_json(raw).model_dump(mode='json')
    _validate_sources(phase,obj,ctx)
    return obj


def verify(out):
    out,lock,ctx=context_ok(out)
    statuses={}
    for phase in ('develop','challenge'):
        p=out/phase
        if not p.exists():statuses[phase]='NOT_STARTED';continue
        req=request(out,phase) if phase=='develop' or statuses.get('develop')=='READY' else None
        if req is None:raise ValueError('CHALLENGE_WITHOUT_COMPLETED_DEVELOP')
        if load(p/'request.json')!=req or load(p/'attempt.json')['request_sha256']!=hashobj(req):
            raise ValueError('SAVED_REQUEST_CHANGED')
        if not (p/'raw_response.json').exists():statuses[phase]='BILLING_UNCERTAIN_NO_RETRY';break
        if not (p/'normalized.json').exists():statuses[phase]='RAW_SAVED_RECOVER_NO_REPAY';break
        norm=parse(phase,load(p/'raw_response.json')['raw_text'],ctx)
        if load(p/'normalized.json')!=norm:raise ValueError('M23_NORMALIZED_SCIENCE_CHANGED')
        if phase=='develop' and load(p/'numeric_probe.json')!=evaluate(norm['numeric_or_symbolic_probe']):
            raise ValueError('M23_NUMERIC_CHECK_CHANGED')
        expected={'phase':phase,'request_sha256':hashobj(req),
                  'raw_sha256':digest(p/'raw_response.json'),
                  'normalized_sha256':digest(p/'normalized.json'),
                  'probe_sha256':digest(p/'numeric_probe.json') if phase=='develop' else None}
        if load(p/'receipt.json')!=expected:raise ValueError('M23_RECEIPT_CHANGED')
        statuses[phase]='READY'
    return {'status':'M23_VERIFIED_NO_CALL',**statuses,
            'scientific_validation':False,'new_discovery_certified':False}


def preview(out,phase):
    state=verify(out)
    if state.get(phase)!='NOT_STARTED':raise ValueError('PHASE_ALREADY_STARTED_USE_RECOVER_OR_INSPECT')
    if phase=='challenge' and state['develop']!='READY':raise ValueError('DEVELOP_NOT_READY')
    req=request(out,phase)
    return {'status':'M23_PREVIEW_NO_CALL','phase':phase,'next_call_is_paid':True,
            'request_sha256':hashobj(req)}


def recover(out,phase):
    out,_,ctx=context_ok(out)
    p=out/phase
    if not (p/'raw_response.json').is_file():
        raise RuntimeError('NO_SAVED_RAW_DO_NOT_INFER_CALL_STATUS')
    req=request(out,phase)
    if load(p/'request.json')!=req:raise ValueError('SAVED_REQUEST_CHANGED')
    rawhash=digest(p/'raw_response.json')
    norm=parse(phase,load(p/'raw_response.json')['raw_text'],ctx)
    if not (p/'normalized.json').exists():put(p/'normalized.json',norm)
    elif load(p/'normalized.json')!=norm:raise ValueError('M23_NORMALIZED_CONFLICT')
    if phase=='develop':
        check=evaluate(norm['numeric_or_symbolic_probe'])
        if not (p/'numeric_probe.json').exists():put(p/'numeric_probe.json',check)
        elif load(p/'numeric_probe.json')!=check:raise ValueError('M23_PROBE_CONFLICT')
    receipt={'phase':phase,'request_sha256':hashobj(req),'raw_sha256':rawhash,
             'normalized_sha256':digest(p/'normalized.json'),
             'probe_sha256':digest(p/'numeric_probe.json') if phase=='develop' else None}
    if not (p/'receipt.json').exists():put(p/'receipt.json',receipt)
    elif load(p/'receipt.json')!=receipt:raise ValueError('M23_RECEIPT_CONFLICT')
    if digest(p/'raw_response.json')!=rawhash:raise ValueError('M23_PAID_RAW_CHANGED')
    state=verify(out)
    return {'status':'M23_SAVED_SCIENCE_RECOVERED_NO_PAID_CALL','phase':phase,
            'verified_status':state[phase],'scientific_validation':False,
            'new_paid_calls_this_command':0}


def execute(out,phase,allow_paid=False,backend=None):
    prev=preview(out,phase)
    base._require_paid(backend,allow_paid,1)
    out=root(out);p=out/phase;p.mkdir()
    req=request(out,phase)
    put(p/'request.json',req)
    put(p/'attempt.json',{'request_sha256':hashobj(req),
        'status':'BILLING_UNKNOWN_UNTIL_RAW_RESPONSE_SAVED'})
    raw,usage,rid=base._invoke(req,backend)
    put(p/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:res=recover(out,phase)
    except Exception as exc:raise RuntimeError('M23_PAID_RAW_SAVED_DO_NOT_REPAY_USE_RECOVER') from exc
    return res|{'new_paid_calls_this_command':int(backend is None)}


def report(out):
    state=verify(out)
    if state.get('develop')!='READY':raise ValueError('M23_DEVELOP_NOT_READY')
    out=root(out);develop=load(out/'develop/normalized.json');probe=load(out/'develop/numeric_probe.json')
    challenged=state.get('challenge')=='READY'
    ch=load(out/'challenge/normalized.json') if challenged else None
    return {'status':'M23_SCIENTIFIC_AUDIT_READY_NO_CALL' if challenged else 'M23_DEVELOP_AWAITING_CHALLENGE',
        'autonomously_selected_question':develop['selected_question'],
        'source_anchor_ids':[a['id'] for a in load(out/'input.json')['verified_SOURCE_EXCERPTS']],
        'claimed_result_kind':develop['result_kind'],
        'claimed_scientific_result_UNVERIFIED':develop['explicit_result'],
        'what_changed_from_prior_UNVERIFIED':develop['what_changed_from_M21_and_M22'],
        'numeric_diagnostic_NOT_PROOF':probe,
        'competing_model_review_UNVERIFIED':ch,
        'scientific_validation':False,'level_ii_certified':False,
        'independent_scientific_review_required':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','preview','develop','challenge','recover','verify','report'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--m21-work',type=Path)
    ap.add_argument('--charge',type=Path)
    ap.add_argument('--charge-pdf',type=Path)
    ap.add_argument('--thermal',type=Path)
    ap.add_argument('--phase',choices=['develop','challenge'])
    ap.add_argument('--allow-paid',action='store_true')
    a=ap.parse_args()
    if a.command=='prepare':
        if not all((a.m21_work,a.charge,a.charge_pdf)):
            ap.error('prepare needs --m21-work --charge --charge-pdf')
        ans=prepare(a.m21_work,a.charge,a.charge_pdf,a.out,a.thermal)
    elif a.command=='preview':
        if not a.phase:ap.error('preview needs --phase')
        ans=preview(a.out,a.phase)
    elif a.command in ('develop','challenge'):ans=execute(a.out,a.command,allow_paid=a.allow_paid)
    elif a.command=='recover':
        if not a.phase:ap.error('recover needs --phase')
        ans=recover(a.out,a.phase)
    elif a.command=='verify':ans=verify(a.out)
    else:ans=report(a.out)
    print(json.dumps(ans,indent=2,ensure_ascii=False,sort_keys=True))

if __name__=='__main__':main()
