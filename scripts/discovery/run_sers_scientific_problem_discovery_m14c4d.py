#!/usr/bin/env python3
"""M14-C4D: open research-problem prospecting, not an empirical novelty certificate.

harvest: public literature lookups only, with explicit network opt-in.
prepare: offline source screening, anchored briefs, frozen prompt.
generate-paid: EXACTLY ONE opt-in paid model attempt, immutable intent/receipt.
validate: offline evidence-grounding and candidate evaluation.
verify: offline chain-of-custody replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from pipeline_core.discovery import sers_scientific_problem_discovery_m14c4d as core

FRESH='M14C4D_FRESH_SOURCE_SNAPSHOT.json'
PLAN='M14C4D_DISCOVERY_PLAN.json';DOCS='M14C4D_SOURCE_INVENTORY.json'
BRIEFS='M14C4D_CROSS_SOURCE_OPPORTUNITIES.json';PROMPT='M14C4D_GENERATION_PROMPT.json'
INTENT='M14C4D_GENERATION_INTENT.json';RECEIPT='M14C4D_GENERATION_RAW_RECEIPT.json'
CANDIDATES='M14C4D_OPEN_RESEARCH_PROBLEM_CANDIDATES.json'
AUDIT='M14C4D_CANDIDATE_AUDIT.json';REVIEW='M14C4D_INDEPENDENT_REVIEW_TEMPLATE.json'
QUERIES=(
 'surface enhanced Raman nanoscale hotspot fluctuation quantitative stability measurement',
 'operando electrochemical SERS adsorption site selectivity spectral dynamics',
 'time resolved tip enhanced Raman spatial resolution chemical reaction intermediates',
 'surface enhanced Raman interfacial water isotope adsorption ionic environment',
 'SERS transferability batch effects optical enhancement independent calibration',
 'plasmon enhanced Raman nanogap vibrational relaxation independent optical controls',
 'surface enhanced Raman nanoparticle heterogeneity single molecule orientation dynamics',
 'surface enhanced Raman temperature charge transfer molecular junction control',
)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_bytes())
def freeze(path,val):
 path=Path(path);body=core.canonical(val)
 if path.exists():
  if path.read_bytes()!=body:raise ValueError('M14C4D_FROZEN_ARTIFACT_DIFFERENT: '+str(path))
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(body)

def _config_sources(m14_root,c4c_actions,fresh_snapshot):
 base=Path(m14_root)
 required=[base/'common_discovery/M14_SOURCE_SNAPSHOT.json',base/'M14_PROVISIONAL_PROBLEMS.json']
 if c4c_actions is not None:required.append(Path(c4c_actions))
 if fresh_snapshot is not None:required.append(Path(fresh_snapshot))
 for p in required:
  if not p.is_file():raise FileNotFoundError('M14C4D_SOURCE_MISSING: '+str(p))
 return {'source_files':[{ 'path':str(p.resolve()),'sha256':digest(p)} for p in required],
         'm14_root':str(base.resolve()),'empirical_actions':str(Path(c4c_actions).resolve()) if c4c_actions else None,
         'fresh_snapshot':str(Path(fresh_snapshot).resolve()) if fresh_snapshot else None}

def _input(plan):
 for src in plan['source_files']:
  p=Path(src['path'])
  if not p.is_file() or digest(p)!=src['sha256']:raise ValueError('M14C4D_SOURCE_HASH_OR_MISSING: '+str(p))
 raw=load(Path(plan['m14_root'])/'common_discovery/M14_SOURCE_SNAPSHOT.json')
 if plan['fresh_snapshot']:
  fresh=load(plan['fresh_snapshot'])
  # union is a source inventory, NOT evidence of scientific independence
  raw={'works':raw['works']+fresh['works']}
 docs,excluded=core.documents_from_snapshot(raw)
 if len(docs)<8:raise ValueError('M14C4D_INSUFFICIENT_ABSTRACT_EVIDENCE')
 if plan['empirical_actions']:
  emp=core.empirical_from_actions(load(plan['empirical_actions']))
  if emp:docs.append(emp)
 return docs,excluded

def harvest(out,allow_network,query_file=None):
 if not allow_network:raise ValueError('Network retrieval requires --allow-network')
 from pipeline_core.discovery.sers_autonomous_novelty_m12 import lookup
 from pipeline_core.discovery.sers_novelty_search_m14 import unique_works
 queries=list(QUERIES) if query_file is None else load(query_file)
 if not isinstance(queries,list) or not 3<=len(queries)<=12 or any(not isinstance(q,str) or not 12<=len(q)<=250 for q in queries):
  raise ValueError('Require 3-12 bounded search queries')
 if len(set(queries))!=len(queries):raise ValueError('Duplicate search queries')
 out.mkdir(parents=True,exist_ok=True)
 qplan={'queries':queries,'providers':['openalex','europepmc'],
        'query_file_sha256':digest(query_file) if query_file else None,'novelty_certified':False}
 freeze(out/'M14C4D_HARVEST_CONFIG.json',qplan)
 receipts=[]
 for q in queries:
  for prov in ('openalex','europepmc'):
   stem=prov+'_'+core.sha(core.canonical([prov,q]))[:16]
   p=out/'literature'/(stem+'_SUCCESS.json')
   if p.exists():
    r=load(p)
    if r.get('provider')!=prov or r.get('query')!=q or r.get('success') is not True:
     raise ValueError('Invalid saved literature success: '+str(p))
   else:
    try:r=lookup(prov,q,key=os.environ.get('OPENALEX_API_KEY',''))
    except Exception as exc:
     err=out/'literature'/(stem+'_ERROR.json')
     freeze(err,{'provider':prov,'query':q,'error':str(exc),'missing_source_does_not_establish_gap':True})
     raise RuntimeError('Literature source failure, no gap inference: '+str(err)) from exc
    if r.get('provider')!=prov or r.get('query')!=q or r.get('success') is not True:
     raise ValueError('Malformed live literature response, no gap inference')
    freeze(p,r)
   receipts.append(r)
 snapshot=unique_works(receipts,limit=80)
 if len(snapshot['works'])<8:raise ValueError('M14C4D_FRESH_RETRIEVAL_HAS_FEWER_THAN_EIGHT_PUBLICATIONS; no novelty inference')
 freeze(out/FRESH,snapshot)
 return {'status':'M14C4D_FRESH_PUBLIC_ABSTRACTS_FROZEN','source_works':len(snapshot['works']),
         'queries':len(queries),'search_receipts':len(receipts),'paid_model_calls':0,
         'full_text_verified':False,'novelty_certified':False}

def prepare(out,m14_root,c4c_actions=None,fresh_snapshot=None,desired_candidates=5):
 if not 2<=desired_candidates<=8:raise ValueError('Allowed candidate count 2-8')
 if (out/PLAN).exists():raise FileExistsError('M14C4D frozen plan already exists. Use inspect/verify, not prepare again.')
 config=_config_sources(m14_root,c4c_actions,fresh_snapshot)
 h=load(Path(m14_root)/'M14_PROVISIONAL_PROBLEMS.json')
 titles=[q['scientific_question'] for q in h['questions']]
 plan={'schema':core.SCHEMA,'status':'M14C4D_PROBLEM_PROSPECTING_PREREGISTERED',
   'research_scope':'SERS and neighboring spectroscopic/chemical mechanism research (question selection not restricted to M14 cards)',
   'original_m14_problems_only_for_overlap_comparison':True,
   'historical_question_titles':titles,'desired_candidates':desired_candidates,
   'source_files':config['source_files'],'m14_root':config['m14_root'],
   'empirical_actions':config['empirical_actions'],'fresh_snapshot':config['fresh_snapshot'],
   'source_role':'Exact abstract excerpts are literature leads, not measured contradiction or guaranteed scientific gaps',
   'empirical_evidence_role':'Previously published predictive secondary reanalysis, not independent causal identification',
   'core_code_sha256':digest(core.__file__),'runner_code_sha256':digest(__file__),
   'novelty_certified':False,'paid_calls_authorized_by_plan':False}
 docs,excluded=_input(plan)
 briefs=core.build_opportunities(docs,max_pairs=12)
 briefs=core.add_empirical_bridge(briefs,docs)
 if len(briefs)<3:raise ValueError('M14C4D_NOT_ENOUGH_INDEPENDENT_SOURCE_BRIEFS')
 evidence={'schema':core.SCHEMA,'documents':docs,'excluded':excluded,
           'source_type_counts':dict(__import__('collections').Counter(d['kind'] for d in docs)),
           'novelty_certified':False}
 prompt=core.make_prompt(plan,docs,briefs)
 out.mkdir(parents=True,exist_ok=True)
 for name,data in [(PLAN,plan),(DOCS,evidence),(BRIEFS,briefs),(PROMPT,prompt)]:freeze(out/name,data)
 return {'status':'M14C4D_OPEN_DISCOVERY_PROMPT_PREPARED','literature_documents':len(docs),
         'excluded_source_rows':len(excluded),'cross_source_leads':len(briefs),
         'historical_questions_as_whitelist':False,'max_candidate_budget':desired_candidates,
         'paid_calls':0,'scientific_novelty_certified':False}

def context(out):
 plan=load(out/PLAN)
 if digest(core.__file__)!=plan['core_code_sha256'] or digest(__file__)!=plan['runner_code_sha256']:
  raise ValueError('M14C4D_EXECUTABLE_HASH_DIFFERS_FROM_FROZEN_PLAN')
 docs,excluded=_input(plan)
 exp={'schema':core.SCHEMA,'documents':docs,'excluded':excluded,
           'source_type_counts':dict(__import__('collections').Counter(d['kind'] for d in docs)),
           'novelty_certified':False}
 briefs=core.add_empirical_bridge(core.build_opportunities(docs,max_pairs=12),docs)
 expected={DOCS:exp,BRIEFS:briefs,PROMPT:core.make_prompt(plan,docs,briefs)}
 for name,value in expected.items():
  if (out/name).read_bytes()!=core.canonical(value):raise ValueError('M14C4D_REPLAY_MISMATCH: '+name)
 return plan,docs,briefs,expected[PROMPT]

def inspect(out):
 plan,docs,briefs,prompt=context(out)
 return {'status':'M14C4D_EVIDENCE_GROUNDED_OPEN_QUESTION_PROSPECTUS',
         'published_abstracts':sum(d['kind']=='published_abstract' for d in docs),
         'secondary_empirical_sources':sum(d['kind']=='published_data_secondary_analysis' for d in docs),
         'cross_source_leads':len(briefs),
         'source_ids_per_brief':[b['source_ids'] for b in briefs],
         'historical_card_whitelist':False,'scientific_gaps_proven':0,
         'prompt_sha256':core.sha(core.canonical(prompt)),'paid_calls':0}

def paid(out,model,api_key_env,base_url,max_tokens,allow_paid,authorize_one_call):
 if not (allow_paid and authorize_one_call):raise ValueError('Paid generation requires both --allow-paid and --i-authorize-one-model-call')
 if not model:raise ValueError('Provide explicit --model (or OPENROUTER_AGENT_MODEL)')
 plan,docs,briefs,prompt=context(out)
 intent=out/INTENT;receipt=out/RECEIPT
 if receipt.exists():
  saved=load(receipt)
  if saved.get('prompt_sha256')!=core.sha(core.canonical(prompt)) or saved.get('model')!=model:
   raise ValueError('Existing paid receipt does not match model or frozen prompt; no recall')
  return {'status':'M14C4D_EXISTING_PAID_RECEIPT_REUSE_ONLY','paid_calls':0,
          'next_step':'validate; DO NOT re-run provider'}
 if intent.exists():raise RuntimeError('M14C4D_PAID_INTENT_EXISTS_NO_AUTOMATIC_RETRY')
 key=os.environ.get(api_key_env)
 if not key:raise RuntimeError('Missing provider secret env: '+api_key_env)
 try:from openai import OpenAI
 except ImportError as e:raise RuntimeError('Requires openai Python package') from e
 ph=core.sha(core.canonical(prompt))
 attempt={'schema':core.SCHEMA,'model':model,'base_url':base_url,'prompt_sha256':ph,
          'status':'PAID_ATTEMPT_STARTED_NO_AUTO_RETRY','max_tokens':max_tokens}
 freeze(intent,attempt)
 try:
  client=OpenAI(api_key=key,base_url=base_url,max_retries=0,timeout=300)
  rsp=client.chat.completions.create(model=model,
      messages=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user']}],
      temperature=0.7,max_tokens=max_tokens)
 except Exception as e:
  raise RuntimeError('Provider may have charged. Intent retained, NO automatic second call: '+str(e)) from e
 result={'schema':core.SCHEMA,'model':model,'prompt_sha256':ph,
         'finish_reason':rsp.choices[0].finish_reason,'raw':rsp.choices[0].message.content or '',
         'usage':rsp.usage.model_dump(mode='json') if rsp.usage else None,
         'new_scientific_discovery_certified':False}
 freeze(receipt,result)
 if result['finish_reason']!='stop':raise RuntimeError('Saved incomplete raw; no automatic paid recall')
 return {'status':'M14C4D_ONE_PAID_PROSPECTING_RECEIPT_SAVED',
         'new_paid_calls':1,'next_step':'validate','scientific_novelty_certified':False}

def _strict_json(data):
 def no_duplicates(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('Duplicate JSON key: '+k)
   out[k]=v
  return out
 def no_finite(s):raise ValueError('Invalid JSON constant: '+s)
 return json.loads(data,object_pairs_hook=no_duplicates,parse_constant=no_finite)

def validate(out):
 plan,docs,briefs,prompt=context(out)
 receipt=load(out/RECEIPT)
 if receipt.get('prompt_sha256')!=core.sha(core.canonical(prompt)) or receipt.get('finish_reason')!='stop':
  raise ValueError('M14C4D_RAW_RECEIPT_INVALID_OR_DIFFERENT_PROMPT_NO_RECALL')
 data=_strict_json(receipt['raw'])
 cs=core.validate_submission(data,docs,plan['desired_candidates'])
 ranked=core.summarize(data,plan,docs)
 ranked['receipt_sha256']=digest(out/RECEIPT)
 ranked['prompt_sha256']=receipt['prompt_sha256']
 ranked['source_inventory_sha256']=digest(out/DOCS)
 ranked['discovered_scientific_gap_certified']=False
 ranked['data_anchoring_is_not_the_same_as_scientific_novelty']=True
 audit={'schema':core.SCHEMA,'status':'M14C4D_SOURCE_ANCHORS_VALIDATED_SCIENTIFIC_REVIEW_PENDING',
        'receipt_sha256':digest(out/RECEIPT),'candidate_file_sha256':core.sha(core.canonical(ranked)),
        'candidate_count':len(cs),'all_anchors_exact_excerpts':True,
        'independent_fulltext_prior_art_review_done':False,
        'new_empirical_mechanism_established':False,'scientific_novelty_certified':False}
 review={'schema':core.SCHEMA,'status':'M14C4D_INDEPENDENT_SCIENCE_REVIEW_TO_COMPLETE',
         'not_pre_filled_as_certified':True,'candidates':[{ 'candidate_id':c['candidate_id'],
           'question':c['research_question'], 'source_ids':[a['source_id'] for a in c['evidence_anchors']],
           'assessment':{'source_relevance':'NOT_REVIEWED','null_model_falsifiability':'NOT_REVIEWED',
            'independent_measurement_feasibility':'NOT_REVIEWED','fulltext_prior_art_distinctness':'NOT_REVIEWED',
            'recommendation':'NOT_REVIEWED'}} for c in ranked['candidates']]}
 for name,content in [(CANDIDATES,ranked),(AUDIT,audit),(REVIEW,review)]:freeze(out/name,content)
 return {'status':ranked['status'],'validated_candidate_questions':len(cs),
         'independent_scientific_review_required':True,'paid_calls_during_validation':0,
         'novelty_certified':False}

def verify(out):
 plan,docs,briefs,prompt=context(out)
 if not (out/RECEIPT).exists():return {'status':'M14C4D_PREPARED_ONLY_NO_GENERATION_YET','paid_calls':0}
 receipt=load(out/RECEIPT)
 if receipt.get('prompt_sha256')!=core.sha(core.canonical(prompt)):raise ValueError('M14C4D_RECEIPT_PROMPT_MISMATCH')
 if not (out/CANDIDATES).exists():
  return {'status':'M14C4D_PAID_RECEIPT_PRESENT_VALIDATION_PENDING' if receipt.get('finish_reason')=='stop' else 'M14C4D_TRUNCATED_PAID_RECEIPT_REQUIRES_MANUAL_REVIEW', 'paid_calls':0}
 before={n:digest(out/n) for n in (RECEIPT,CANDIDATES,AUDIT,REVIEW)}
 result=validate(out)
 after={n:digest(out/n) for n in before}
 if before!=after:raise ValueError('M14C4D_VERIFY_MUTATED_SAVED_ARTIFACT')
 return {'status':'M14C4D_REPLAY_VERIFIED_OPEN_PROSPECTS_ONLY',
         'validated_candidate_questions':result['validated_candidate_questions'],
         'original_receipt_unchanged':True,'novelty_certified':False,'new_paid_calls':0}

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 commands=parser.add_subparsers(dest='cmd',required=True)
 a=commands.add_parser('harvest');a.add_argument('--out-dir',type=Path,required=True)
 a.add_argument('--allow-network',action='store_true');a.add_argument('--queries-file',type=Path)
 a=commands.add_parser('prepare');a.add_argument('--out-dir',type=Path,required=True)
 a.add_argument('--m14-root',type=Path,required=True);a.add_argument('--c4c-actions',type=Path)
 a.add_argument('--fresh-snapshot',type=Path);a.add_argument('--desired-candidates',type=int,default=5)
 for cmd in ('inspect','generate-paid','validate','verify'):
  a=commands.add_parser(cmd);a.add_argument('--out-dir',type=Path,required=True)
  if cmd=='generate-paid':
   a.add_argument('--model',default=os.environ.get('OPENROUTER_AGENT_MODEL'))
   a.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
   a.add_argument('--base-url',default='https://openrouter.ai/api/v1')
   a.add_argument('--max-tokens',type=int,default=9000)
   a.add_argument('--allow-paid',action='store_true');a.add_argument('--i-authorize-one-model-call',action='store_true')
 args=parser.parse_args(argv)
 if args.cmd=='harvest':out=harvest(args.out_dir,args.allow_network,args.queries_file)
 elif args.cmd=='prepare':out=prepare(args.out_dir,args.m14_root,args.c4c_actions,args.fresh_snapshot,args.desired_candidates)
 elif args.cmd=='inspect':out=inspect(args.out_dir)
 elif args.cmd=='generate-paid':out=paid(args.out_dir,args.model,args.api_key_env,args.base_url,args.max_tokens,args.allow_paid,args.i_authorize_one_model_call)
 elif args.cmd=='validate':out=validate(args.out_dir)
 else:out=verify(args.out_dir)
 print(json.dumps(out,ensure_ascii=False,indent=2));return out

if __name__=='__main__':main()
