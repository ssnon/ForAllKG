#!/usr/bin/env python3
"""M31 V241 shared structured-response transport. No hypothesis rewriting, no retries.

Uses json_schema format; local Pydantic validation and native state rules remain authoritative.
The OpenRouter provider may reject unsupported schema features: treat that as a recorded
operational failure, never silently fall back or retry a paid action.
"""
from __future__ import annotations
import copy
from .research_work_v12_contracts import ResearchDecision,ReasonWorkContent

PROTOCOL='M31_V241_STRUCTURED_OUTPUT_NO_SCIENCE_REPAIR'

def _format(name,schema):
    return {'type':'json_schema','json_schema':{'name':name,'strict':False,'schema':schema}}

def _require_all(properties):
    return {'type':'object','properties':copy.deepcopy(properties),
            'required':list(properties),'additionalProperties':False}

def decision_output_format(state,manifest):
    base=ResearchDecision.model_json_schema()
    prop=base['properties']
    pending=state.get('pending_work')
    if pending is not None and pending['status']!='EXECUTED':
        raise ValueError('M31_V241_CANNOT_DECIDE_BEFORE_WORK')
    interpretation = ({'$ref':'#/$defs/WorkInterpretation'} if pending is not None else {'type':'null'})
    # A syntactic decision is not a scientific endorsement. All candidate citations
    # and epistemic dispositions still pass the exact frozen native validators.
    end_reasons=['BUDGET_EXHAUSTED','NO_TRACTABLE_NEXT_WORK','QUESTION_RESOLVED','RESOURCE_LIMIT','OTHER']
    if state['decision_count']+1 < manifest['max_decisions']:
        end_reasons.remove('BUDGET_EXHAUSTED')
    branches=[]
    for disp in ('ABSTAIN','PROVISIONAL_RESEARCH_CANDIDATE'):
        if disp=='PROVISIONAL_RESEARCH_CANDIDATE':
            interpreted={i['work_id'] for i in state.get('interpretations',[])}
            supported={w['work_id'] for w in state.get('work_history',[])
                       if w['work_kind']=='REASON' and w['execution_status']=='COMPLETED'}
            # pending interpreted by current decision can support this END
            if pending and pending['status']=='EXECUTED' and pending['request']['work_kind']=='REASON':
                interpreted.add(pending['work_id']);supported.add(pending['work_id'])
            if not interpreted.intersection(supported):continue
        p=copy.deepcopy(prop)
        p['action']={'const':'END'}
        p['work_request']={'type':'null'}
        p['interpretation']=interpretation
        p['idea_update']={'type':'null'}
        p['research_end_reason']={'type':'string','enum':end_reasons}
        p['final_disposition']={'const':disp}
        p['final_justification']={'type':'string','minLength':1}
        p['provisional_candidate']=({'type':'null'} if disp=='ABSTAIN'
                                    else {'$ref':'#/$defs/ProvisionalCandidateSnapshot'})
        branches.append(_require_all(p))
    # Non-END variants enforce action/work-kind pairing and prohibit END-only fields.
    for action in ('READ_EVIDENCE','EXPLORE','CHALLENGE','PIVOT','DEEPEN'):
        p=copy.deepcopy(prop)
        p['action']={'const':action}
        p['interpretation']=interpretation
        req=copy.deepcopy(base['$defs']['ResearchWorkRequest'])
        req['properties']['work_kind']={'const':('READ_FROZEN_SOURCE' if action=='READ_EVIDENCE' else 'REASON')}
        p['work_request']=req
        for key in ('final_disposition','final_justification','provisional_candidate','research_end_reason'):
            p[key]={'type':'null'}
        # ResearchIdea mutations retain the original separate Pydantic and state contract.
        branches.append(_require_all(p))
    schema={'type':'object','properties':{'payload':{'anyOf':branches}},
            'required':['payload'],'additionalProperties':False,'$defs':base.get('$defs',{})}
    return _format('m31_native_decision_v241',schema)

def plain_output_format(*,tool_available,final):
    s=_require_all({
       'research_action':{'type':'string','enum':['EXPLORE','CONSTRUCT','DERIVE','CHALLENGE','DEVELOP','REVISE','PIVOT','RETURN']},
       'current_claim':{'type':'string'},'premises':{'type':'array','items':{'type':'string'}},
       'reasoning_steps':{'type':'array','items':{'type':'string'}},
       'candidate_artifact':{'anyOf':[{'type':'object'},{'type':'null'}]},
       'counterexample_attempt':{'type':['string','null']},
       'uncertainties':{'type':'array','items':{'type':'string'}},
       'next_action':({'const':'RETURN'} if final else {'enum':['CONTINUE','RETURN']}),
       'why_next_action':{'type':'string'},
       'probe_request':({'anyOf':[{'type':'object'},{'type':'null'}]} if tool_available else {'type':'null'}),
       'prospective_probe_prediction':({'type':['string','null']} if tool_available else {'type':'null'})})
    return _format('m31_plain_step_v241',s)

def step_output_format(source_schema,*,n,tool_available):
    s=copy.deepcopy(source_schema)
    s['additionalProperties']=False
    s['required']=list(s['properties'])
    if n==3:s['properties']['next_action']={'const':'RETURN'}
    if not tool_available:
        s['properties']['probe_request']={'type':'null'}
        s['properties']['prospective_probe_prediction']={'type':'null'}
    return _format('m31_native_work_v241',s)

def reason_output_format():
    return _format('m31_native_reason_commit_v241',ReasonWorkContent.model_json_schema())
