#!/usr/bin/env python3
"""Strong plain 12-turn scientific researcher, non-FAKG baseline. Offline safety and immutable receipts.

Not a scientific verifier and not the FAKG controller. Paid calls require explicit flag and env.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, pathlib, sys
from typing import Any
P=pathlib.Path
PROTOCOL='m31-plain-multistep-pilot-v241'
ACTIONS={'EXPLORE','CONSTRUCT','DERIVE','CHALLENGE','DEVELOP','REVISE','PIVOT','RETURN'}
MAX=12
SRC=P(__file__).resolve()
from pipeline_core.discovery.scientific_research_control import m31_pilot_oracle_v22 as oracle
from pipeline_core.discovery.scientific_research_control.m31_output_contract_v241 import plain_output_format
def canon(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(v):return hashlib.sha256(canon(v)).hexdigest()
def hfile(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def read(p):return json.loads(P(p).read_text(encoding='utf-8'))
def put(p,v):
 p=P(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def parse(raw):
 x=json.loads(raw)
 expected={'research_action','current_claim','premises','reasoning_steps','candidate_artifact','counterexample_attempt','uncertainties','next_action','why_next_action','probe_request','prospective_probe_prediction'}
 if not isinstance(x,dict) or set(x)!=expected:raise ValueError('M31_PLAIN_SCHEMA_KEYS_REQUIRED:'+repr(sorted(expected-set(x) if isinstance(x,dict) else expected)))
 if x['research_action'] not in ACTIONS or x['next_action'] not in ('CONTINUE','RETURN'):raise ValueError('BAD_ACTION')
 for k in ('current_claim','why_next_action'):
  if not isinstance(x[k],str) or len(x[k])<15:raise ValueError('BAD_SCIENCE_TEXT:'+k)
 for k in ('premises','reasoning_steps','uncertainties'):
  if not isinstance(x[k],list) or not all(isinstance(v,str) for v in x[k]):raise ValueError('BAD_LIST:'+k)
 if x['probe_request'] is not None:
  if type(x['probe_request'])!=dict or x['next_action']!='CONTINUE' or type(x['prospective_probe_prediction'])!=str or len(x['prospective_probe_prediction'])<16:raise ValueError('PROBE_REQUIRES_PREDICTION_AND_CONTINUATION')
 elif x['prospective_probe_prediction'] is not None:raise ValueError('PREDICTION_WITHOUT_PROBE')
 if x['candidate_artifact'] is not None and not isinstance(x['candidate_artifact'],dict):raise ValueError('CANDIDATE_ARTIFACT_MUST_BE_OBJECT_OR_NULL')
 if x['counterexample_attempt'] is not None and not isinstance(x['counterexample_attempt'],str):raise ValueError('BAD_COUNTEREXAMPLE')
 return x

def sys_prompt(n,tool_enabled=False,tool_desc=''):
 prompt="""You are an autonomous scientific researcher. You have a scientific question and the research notes from your preceding steps. Choose the next intellectual research action autonomously. You may create several competing hypotheses, derive mathematical consequences, search for counterexamples, return to earlier candidates, revise or abandon a flawed approach, and preserve your strongest rigorous result. You may RETURN when further work is not valuable. A provisional candidate or a valid negative result is acceptable. Do not claim experimental validation or originality without evidence. Be creative in exploration but conservative about established facts. You have a total budget of 12 model calls, with one JSON research step per call; previous steps persist. Output EXACTLY ONE JSON object with keys research_action (EXPLORE/CONSTRUCT/DERIVE/CHALLENGE/DEVELOP/REVISE/PIVOT/RETURN), current_claim (string, at least 15 chars), premises (string list), reasoning_steps (string list), candidate_artifact (structured JSON object or null; optional formal witness or multiple candidate models), counterexample_attempt (string or null), uncertainties (string list), next_action (CONTINUE/RETURN), why_next_action (string >=15 chars), probe_request (object or null), prospective_probe_prediction (string or null). Do not fabricate performed measurements. When a probe is requested, next_action must be CONTINUE to actually interpret the returned evidence."""
 if tool_enabled and n<MAX:
  prompt+=' OPTIONAL SYNTHETIC OR EXACT MATHEMATICAL INSTRUMENT: request a single tool action with probe_request as {"operation":...,"inputs":{...}}, fully explicit values. The tool will return the requested computed measurement to your following research step, not hidden answers. State prospective_probe_prediction before requesting, or set both nullable tool fields to null. Instrument specification: '+tool_desc
 else:
  prompt+=' No executable instrument in this step. Set probe_request:null and prospective_probe_prediction:null.'
 if n==MAX:prompt+=' This is the final allowed call; next_action must be RETURN.'
 return prompt

def request(lock,packet,rows,n):
 return {'response_format':plain_output_format(tool_available=lock['tool_enabled'] and n<MAX,final=n==MAX),'provider':{'require_parameters':True},'model':lock['model'],'temperature':lock['temperature'],'max_tokens':lock['max_tokens'],
         'messages':[{'role':'system','content':sys_prompt(n,lock['tool_enabled'],oracle.tool_spec(lock['tool_task']) if lock['tool_enabled'] else '')},{'role':'user','content':json.dumps({'scientific_question':packet['scientific_question'],'source_inventory':packet['sources'],'previous_research_steps':[{'normalized':r['normalized'],'instrument_result':r['instrument_result']} for r in rows],'next_step':n,'max_steps':MAX,'instrument_available':lock['tool_enabled'] and n<MAX,'allowed_to_end_early':True,'response_is_unverified_science':True},sort_keys=True,ensure_ascii=False)}]}

def snapshot(out):
 out=P(out).resolve();lock=read(out/'LOCK.json');packet=read(out/'problem.json')
 if lock['protocol']!=PROTOCOL or hfile(out/'problem.json')!=lock['packet_sha256'] or hfile(SRC)!=lock['runner_sha256'] or hfile(oracle.__file__)!=lock['oracle_sha256']:raise ValueError('LOCK_OR_SOURCE_TAMPERED')
 if not oracle.check_consistency(lock['tool_task'],packet):raise ValueError('PROBLEM_SOURCE_CHANGED')
 rows=[];pending=None
 for i in range(1,MAX+1):
  step=out/'steps'/f'step_{i:04d}'
  if not step.exists():continue
  if i!=len(rows)+1 or (rows and rows[-1]['normalized']['next_action']=='RETURN'):raise ValueError('NONCONTIGUOUS_STEPS')
  req=request(lock,packet,rows,i)
  if read(step/'request.json')!=req:raise ValueError('REQUEST_REPLAY_CHANGED')
  a=read(step/'attempt.json')
  if a['request_sha256']!=sha(req) or a['previous_receipt_sha256']!=(sha(rows[-1]['receipt']) if rows else None):raise ValueError('CHAIN_CHANGED')
  if not (step/'raw_response.json').exists() or not (step/'normalized.json').exists() or not (step/'receipt.json').exists() or not (step/'instrument_result.json').exists():
   pending=step;break
  norm=parse(read(step/'raw_response.json')['raw_text'])
  if norm['next_action']!='RETURN' and i==MAX:raise ValueError('FINAL_RESEARCH_STEP_MUST_RETURN')
  if read(step/'normalized.json')!=norm:raise ValueError('NORMALIZED_TAMPERED')
  if norm['probe_request'] is not None and not lock['tool_enabled']:raise ValueError('NO_TOOL_AVAILABLE')
  result=({'status':'REJECTED_TOOL_BUDGET','error':'M31_PILOT_MAX_4_PROBE_REQUESTS'} if sum(r['normalized']['probe_request'] is not None for r in rows)>=4 else oracle.experiment(lock['tool_task'],norm['probe_request'])) if norm['probe_request'] is not None else None
  if read(step/'instrument_result.json')!={'instrument_result':result}:raise ValueError('TOOL_RESULT_TAMPERED')
  receipt={'number':i,'request_sha256':sha(req),'raw_sha256':hfile(step/'raw_response.json'),'normalized_sha256':hfile(step/'normalized.json'),'instrument_sha256':hfile(step/'instrument_result.json'),'previous_receipt_sha256':sha(rows[-1]['receipt']) if rows else None}
  if read(step/'receipt.json')!=receipt:raise ValueError('RECEIPT_TAMPERED')
  rows.append({'normalized':norm,'instrument_result':result,'receipt':receipt})
 if any((out/'steps'/f'step_{i:04d}').exists() for i in range(len(rows)+2,MAX+1)):raise ValueError('UNEXPECTED_LATER_STEP')
 return lock,packet,rows,pending

def prepare(args):
 out=P(args.out).expanduser().resolve();p=P(args.packet).resolve()
 if out.exists():raise FileExistsError('NO_OVERWRITE: '+str(out))
 packet=read(p)
 task=packet.get('problem_id','').removeprefix('M31_PILOT_')
 if not oracle.check_consistency(task,packet):raise ValueError('INVALID_PUBLIC_PACKET_OR_RESEARCH_PROBLEM')
 out.mkdir(parents=True);put(out/'problem.json',packet)
 lock={'protocol':PROTOCOL,'packet_sha256':hfile(out/'problem.json'),'model':args.model,'max_tokens':4500,'temperature':0.5,'max_calls':MAX,'runner_sha256':hfile(SRC),'oracle_sha256':hfile(oracle.__file__),'tool_task':task,'tool_enabled':bool(args.with_tool),'scientific_validation':False}
 put(out/'LOCK.json',lock);snapshot(out)
 return {'status':'M31_PLAIN_PREPARED_NO_CALL','run':str(out),'max_calls':MAX,'model':args.model}

def preview(out):
 lock,p,rows,pending=snapshot(out)
 if pending:return {'status':'PAUSED_SAVED_ATTEMPT_RECOVER_OR_INSPECT_NO_RETRY','next':'RECOVER','steps':len(rows)}
 if rows and rows[-1]['normalized']['next_action']=='RETURN':return {'status':'COMPLETE_NO_CALL','next':'COMPLETE','steps':len(rows)}
 if len(rows)>=MAX:return {'status':'BUDGET_EXHAUSTED','next':'NONE','steps':len(rows)}
 return {'status':'M31_PLAIN_VERIFIED_NO_CALL','next':'STEP_MODEL_CALL','steps':len(rows),'next_step':len(rows)+1}

def recover(out):
 out=P(out);lock,p,rows,pending=snapshot(out)
 if not pending:raise ValueError('NO_PENDING_ATTEMPT')
 if not (pending/'raw_response.json').exists():raise RuntimeError('RAW_NOT_SAVED_BILLING_UNCERTAIN_DO_NOT_REPEAT')
 norm=parse(read(pending/'raw_response.json')['raw_text'])
 if len(rows)+1==MAX and norm['next_action']!='RETURN':raise ValueError('FINAL_STEP_INVALID_SAVED_RAW_DO_NOT_RETRY')
 if norm['probe_request'] is not None and not lock['tool_enabled']:raise ValueError('TOOL_NOT_ENABLED')
 # Budget excess is a documented rejected probe, not an irreversible invalid raw response.
 result=({'status':'REJECTED_TOOL_BUDGET','error':'M31_PILOT_MAX_4_PROBE_REQUESTS'} if sum(r['normalized']['probe_request'] is not None for r in rows)>=4 else oracle.experiment(lock['tool_task'],norm['probe_request'])) if norm['probe_request'] is not None else None
 if not (pending/'normalized.json').exists():put(pending/'normalized.json',norm)
 if not (pending/'instrument_result.json').exists():put(pending/'instrument_result.json',{'instrument_result':result})
 receipt={'number':len(rows)+1,'request_sha256':read(pending/'attempt.json')['request_sha256'],'raw_sha256':hfile(pending/'raw_response.json'),'normalized_sha256':hfile(pending/'normalized.json'),'instrument_sha256':hfile(pending/'instrument_result.json'),'previous_receipt_sha256':sha(rows[-1]['receipt']) if rows else None}
 if not (pending/'receipt.json').exists():put(pending/'receipt.json',receipt)
 snapshot(out)
 return {'status':'M31_PLAIN_RECOVERED_NO_CALL','next_action':norm['next_action'],'steps':len(rows)+1}

def step(out,allow_paid=False,backend=None):
 out=P(out);info=preview(out)
 if info['next']!='STEP_MODEL_CALL':raise ValueError('PAID_CALL_NOT_ADMISSIBLE:'+info['next'])
 if backend is None:
  if not allow_paid or os.getenv('FAKG_RESEARCH_ALLOW_PAID')!='YES':raise PermissionError('REQUIRES_EXPLICIT_ALLOW_PAID_AND_ENV')
  if not os.getenv('OPENROUTER_API_KEY'):raise EnvironmentError('OPENROUTER_API_KEY_NOT_SET')
 lock,packet,rows,_=snapshot(out);n=len(rows)+1;req=request(lock,packet,rows,n)
 s=out/'steps'/f'step_{n:04d}';s.mkdir(parents=True,exist_ok=False)
 put(s/'request.json',req);put(s/'attempt.json',{'request_sha256':sha(req),'previous_receipt_sha256':sha(rows[-1]['receipt']) if rows else None,'status':'BILLING_UNKNOWN_UNTIL_RAW_SAVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 if backend is None:
  from pipeline_core.discovery.scientific_research_control import m30r_generic_v12_v241 as native
  raw,usage,rid=native._invoke(req,None)
 else:raw,usage,rid=backend(req)
 put(s/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
 try:result=recover(out)
 except Exception as ex:raise RuntimeError('RAW_SAVED_REVIEW_WITH_RECOVER_NO_AUTO_RETRY') from ex
 result['new_paid_calls_this_command']=int(backend is None)
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('prepare');a.add_argument('--packet',required=True);a.add_argument('--out',required=True);a.add_argument('--model',default='openai/gpt-5.6-luna');a.add_argument('--with-tool',action='store_true')
 for name in ('preview','verify','step','recover'):
  p=sub.add_parser(name);p.add_argument('--out',required=True)
  if name=='step':p.add_argument('--allow-paid',action='store_true')
 args=ap.parse_args()
 if args.cmd=='prepare':result=prepare(args)
 elif args.cmd=='step':result=step(args.out,args.allow_paid)
 elif args.cmd=='recover':result=recover(args.out)
 elif args.cmd=='verify':
  lock,packet,rows,pending=snapshot(args.out)
  result={'status':'M31_PLAIN_REPLAY_VERIFIED_NO_CALL','tool_results':sum(r['instrument_result'] is not None for r in rows),'steps':len(rows),'pending_attempt':bool(pending),'finished':bool(rows and rows[-1]['normalized']['next_action']=='RETURN')}
 else:result=preview(args.out)
 print(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
