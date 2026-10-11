#!/usr/bin/env python3
"""Offline M14 saved-physics candidate-ID triage and conservative replay.

No network/model calls. Refuses order-only mapping. Source receipts remain immutable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import frozen, canon

AUDIT_LEAF='M14_PHYSICS_IDENTITY_RECOVERY.json'

def load(p:Path):return json.loads(p.read_text(encoding='utf-8'))
def file_sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def data_sha(d):return hashlib.sha256(canon(d).encode('utf-8')).hexdigest()
def raw_sha(raw:str):return hashlib.sha256(raw.encode('utf-8')).hexdigest()
def emit(p:Path,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')

def locate(plan:Path,receipt_sha:str):
    frozen(load(plan))
    if not re.fullmatch(r'[a-f0-9]{64}',receipt_sha):raise ValueError('Exact 64-char receipt SHA-256 required')
    root=plan.parent
    hits=[]
    for p in root.glob('r*/g*/*/physics_RAW_RECEIPT.json'):
        if file_sha(p)==receipt_sha:hits.append(p.parent)
    if len(hits)!=1:raise RuntimeError('Expected exactly one matching physics receipt, found '+str(len(hits)))
    folder=hits[0]
    original=load(folder/'physics_RAW_RECEIPT.json')
    from scripts.discovery.run_sers_novelty_search_m14 import _strict_json
    review=_strict_json(original['raw'])
    cs=load(folder/'M14_GENERATION_PARSED.json')['candidates']
    if not isinstance(review,dict) or review.get('novelty_certified') is not False:
        raise ValueError('Original physics object invalid or asserts novelty')
    rs=review.get('reviews')
    if not isinstance(rs,list) or len(rs)!=len(cs) or len(cs)!=2 or not all(isinstance(r,dict) for r in rs):
        raise ValueError('Expected two physics reviews for two candidate records; manual external adjudication required')
    return folder,original,review,cs

def normal_form(val, expected:list[str]):
    if not isinstance(val,str):return None
    compact=val.strip().strip('`\'" ').strip()
    # Either full exact canonical token, a full 20-hex hash without prefix,
    # or a unique full 20-hex hash with m14 case variation. No fuzzy similarity.
    matches=[]
    for e in expected:
        token=e.split(':',1)[1]
        if compact==e or compact.lower()==e.lower() or compact.lower()==token.lower():matches.append(e)
    return matches[0] if len(matches)==1 else None

def inspect(plan:Path,sha:str)->dict:
    folder,receipt,doc,cs=locate(plan,sha)
    wanted=[x['candidate_id'] for x in cs]
    suggestions=[normal_form(r.get('candidate_id'),wanted) for r in doc['reviews']]
    auto=(all(suggestions) and len(set(suggestions))==len(wanted) and set(suggestions)==set(wanted))
    status={
       'status':'M14_PHYSICS_IDENTITIES_INSPECTED_NO_MODEL_CALLS',
       'folder':str(folder),'original_receipt_sha256':sha,
       'expected_candidates':[{'candidate_id':c['candidate_id'],'lane':c.get('lane'),
                'title':c.get('title'),'claim_atom':c.get('claim_atom'),
                'mechanism':c.get('mechanism'),'strongest_null':c.get('strongest_null')}
           for c in cs],
       'actual_review_entries':[{'review_index':ix,'candidate_id':r.get('candidate_id'),
                'physical_status':r.get('physical_status'),
                'critical_flaw':r.get('critical_flaw'),
                'counterexample_or_derivation':r.get('counterexample_or_derivation'),
                'discriminating_experiment_or_why_impossible':r.get('discriminating_experiment_or_why_impossible'),
                'proposed_canonical_id_when_token_exact':suggestions[ix]}
            for ix,r in enumerate(doc['reviews'])],
       'token_normalization_unambiguous':auto,
       'only_token_normalization_eligible_for_auto_recovery':auto,
       'original_raw_unchanged':True,'provider_calls':0,
       'warning':'Review index is NOT scientific identity evidence; never map by position alone.'}
    return status

def build_audit(folder:Path,doc:dict,cs:list[dict],mapping:list[dict],kind:str,plan_id:str):
    orig=load(folder/'physics_RAW_RECEIPT.json')
    if (folder/'M14_PHYSICS_CRITIQUE.json').exists():raise RuntimeError('Physics already parsed; refuse identity recovery')
    wanted={c['candidate_id'] for c in cs}
    if len(mapping)!=len(cs) or {x['review_index'] for x in mapping}!=set(range(len(cs))):
        raise ValueError('Every original review must have one mapping by explicit review_index')
    if {x['expected_candidate_id'] for x in mapping}!=wanted:
        raise ValueError('Mapped target IDs must be exactly the two frozen generator IDs')
    aligned=json.loads(canon(doc))
    for m in mapping:
        i=m['review_index']
        if m['source_review_sha256']!=data_sha(doc['reviews'][i]):
            raise ValueError('Source review content hash mismatch')
        if m['original_candidate_id']!=doc['reviews'][i].get('candidate_id'):
            raise ValueError('Original ID changed since inspection')
        aligned['reviews'][i]['candidate_id']=m['expected_candidate_id']
        if kind=='TOKEN_EXACT' and normal_form(m['original_candidate_id'],list(wanted))!=m['expected_candidate_id']:
            raise ValueError('Non-exact token mapping must require independent human science check')
    if aligned.get('novelty_certified') is not False:raise ValueError('Authoritative novelty claim prohibited')
    from scripts.discovery.run_sers_novelty_search_m14 import physics_validated
    parsed,other=physics_validated(canon(aligned),cs)
    return {
        'schema':'m14-physics-identity-replay-v1',
        'status':'M14_ORIGINAL_PHYSICS_REVIEW_IDENTITIES_MAPPED_NOT_NEW_SCIENCE',
        'kind':kind,'plan_id':plan_id,'original_receipt_sha256':file_sha(folder/'physics_RAW_RECEIPT.json'),
        'original_raw_sha256':raw_sha(orig['raw']),'original_reviews_sha256':data_sha(doc['reviews']),
        'mapping':mapping,'canonical_review_sha256':data_sha(aligned),
        'strict_parsed_sha256':data_sha(parsed),
        'science_changed_beyond_candidate_id':False,
        'model_calls_for_recovery':0,'novelty_certified':False,
        'human_scientific_match_required':kind=='MANUAL_SCIENTIFIC_ALIGNMENT',
        'the_model_review_is_not_empirical_truth':True,
    }

def verified_replay(folder:Path,raw:str,candidates:list[dict]):
    audit=load(folder/AUDIT_LEAF)
    if audit.get('schema')!='m14-physics-identity-replay-v1' or audit.get('novelty_certified') is not False or audit.get('model_calls_for_recovery')!=0:
        raise RuntimeError('Invalid replay audit schema/authority')
    root=folder.parent.parent.parent
    if audit.get('plan_id')!=frozen(load(root/'M14_PLAN.json'))['plan_id']:
        raise RuntimeError('Replay plan identity mismatch')
    if audit.get('original_receipt_sha256')!=file_sha(folder/'physics_RAW_RECEIPT.json') or audit.get('original_raw_sha256')!=raw_sha(raw):
        raise RuntimeError('Original physics raw receipt drift')
    from scripts.discovery.run_sers_novelty_search_m14 import _strict_json
    doc=_strict_json(raw)
    if audit.get('original_reviews_sha256')!=data_sha(doc['reviews']):raise RuntimeError('Original review drift')
    reconstructed=build_audit_reconstruction(folder,doc,candidates,audit)
    if reconstructed!=audit:raise RuntimeError('Audited identity replay is not reproducible')
    normalized=json.loads(canon(doc))
    for m in audit['mapping']:
        normalized['reviews'][m['review_index']]['candidate_id']=m['expected_candidate_id']
    return canon(normalized),{'audit_sha256':file_sha(folder/AUDIT_LEAF),
        'original_receipt_sha256':audit['original_receipt_sha256'],
        'kind':audit['kind'],'scientific_content_unchanged':True,
        'candidate_identity_mapping_not_truth_certification':True}

def build_audit_reconstruction(folder,doc,cs,audit):
    # Replay uses exact same deterministic mapping, but does not write files.
    kind=audit.get('kind')
    if kind not in ('TOKEN_EXACT','MANUAL_SCIENTIFIC_ALIGNMENT'):raise RuntimeError('Unknown mapping kind')
    if (folder/'M14_PHYSICS_CRITIQUE.json').exists():
        # A replay after the parsed physics artifact exists is legitimate.
        pass
    # Avoid using the no-parsed guard from build_audit during replay.
    wanted={c['candidate_id'] for c in cs};mp=audit.get('mapping')
    if not isinstance(mp,list) or len(mp)!=len(cs):raise RuntimeError('Invalid mapping')
    if {m.get('review_index') for m in mp}!=set(range(len(cs))):raise RuntimeError('Index mismatch')
    if {m.get('expected_candidate_id') for m in mp}!=wanted:raise RuntimeError('Mapped IDs do not match generator')
    aligned=json.loads(canon(doc))
    for m in mp:
        i=m['review_index'];old=doc['reviews'][i]
        if m.get('source_review_sha256')!=data_sha(old) or m.get('original_candidate_id')!=old.get('candidate_id'):
            raise RuntimeError('Original critic content mismatch')
        if kind=='TOKEN_EXACT' and normal_form(m['original_candidate_id'],list(wanted))!=m['expected_candidate_id']:
            raise RuntimeError('False token-only mapping')
        if kind=='MANUAL_SCIENTIFIC_ALIGNMENT' and m.get('human_confirmed_scientific_match') is not True:
            raise RuntimeError('Human science mapping approval missing')
        aligned['reviews'][i]['candidate_id']=m['expected_candidate_id']
    from scripts.discovery.run_sers_novelty_search_m14 import physics_validated
    parsed,_=physics_validated(canon(aligned),cs)
    return {
        'schema':'m14-physics-identity-replay-v1','status':'M14_ORIGINAL_PHYSICS_REVIEW_IDENTITIES_MAPPED_NOT_NEW_SCIENCE',
        'kind':kind,'plan_id':audit['plan_id'],'original_receipt_sha256':file_sha(folder/'physics_RAW_RECEIPT.json'),
        'original_raw_sha256':raw_sha(load(folder/'physics_RAW_RECEIPT.json')['raw']),
        'original_reviews_sha256':data_sha(doc['reviews']),'mapping':mp,
        'canonical_review_sha256':data_sha(aligned),'strict_parsed_sha256':data_sha(parsed),
        'science_changed_beyond_candidate_id':False,'model_calls_for_recovery':0,
        'novelty_certified':False,'human_scientific_match_required':kind=='MANUAL_SCIENTIFIC_ALIGNMENT',
        'the_model_review_is_not_empirical_truth':True}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    q=sub.add_parser('inspect');q.add_argument('--plan',type=Path,required=True);q.add_argument('--receipt-sha',required=True)
    q=sub.add_parser('recover-token-only');q.add_argument('--plan',type=Path,required=True);q.add_argument('--receipt-sha',required=True)
    q=sub.add_parser('make-manual-template');q.add_argument('--plan',type=Path,required=True);q.add_argument('--receipt-sha',required=True)
    q=sub.add_parser('recover-reviewed');q.add_argument('--plan',type=Path,required=True);q.add_argument('--receipt-sha',required=True)
    q.add_argument('--mapping-file',type=Path,required=True)
    q.add_argument('--i-reviewed-the-scientific-identity-of-each-critique',action='store_true')
    args=ap.parse_args(argv)
    folder,receipt,doc,cs=locate(args.plan,args.receipt_sha)
    if args.command=='inspect':print(json.dumps(inspect(args.plan,args.receipt_sha),indent=2,ensure_ascii=False));return
    wanted=[c['candidate_id'] for c in cs]
    if args.command=='make-manual-template':
        out=folder/'M14_PHYSICS_ID_MAPPING_MANUAL_TO_FILL.json'
        template={'original_receipt_sha256':args.receipt_sha,
           'critical_instruction':'Verify physics review text matches each candidate claim, not review order. Set expected_candidate_id and human_confirmed_scientific_match true ONLY after inspection.',
           'mappings':[{'review_index':i,'original_candidate_id':r.get('candidate_id'),
               'source_review_sha256':data_sha(r),'expected_candidate_id':None,
               'human_confirmed_scientific_match':False} for i,r in enumerate(doc['reviews'])]}
        emit(out,template);print('M14_MANUAL_MAPPING_TEMPLATE='+str(out));return
    if args.command=='recover-token-only':
        ids=[normal_form(r.get('candidate_id'),wanted) for r in doc['reviews']]
        if not all(ids) or set(ids)!=set(wanted):
            raise RuntimeError('Cannot prove candidate identities via exact 20-hex token. Inspect science and use reviewed mapping; NO automatic positional mapping.')
        mapping=[{'review_index':i,'original_candidate_id':r.get('candidate_id'),
                 'source_review_sha256':data_sha(r),'expected_candidate_id':ids[i]} for i,r in enumerate(doc['reviews'])]
        kind='TOKEN_EXACT'
    else:
        if not args.i_reviewed_the_scientific_identity_of_each_critique:
            raise RuntimeError('Explicit human scientific identity verification required')
        provided=load(args.mapping_file)
        if provided.get('original_receipt_sha256')!=args.receipt_sha:raise RuntimeError('Manual file not anchored to exact original receipt')
        mapping=provided.get('mappings')
        if not isinstance(mapping,list) or any(m.get('human_confirmed_scientific_match') is not True for m in mapping):
            raise RuntimeError('All review-to-claim matches must be individually confirmed by human')
        kind='MANUAL_SCIENTIFIC_ALIGNMENT'
    audit=build_audit(folder,doc,cs,mapping,kind,frozen(load(args.plan))['plan_id'])
    # Validate the exact deterministic reconstruction BEFORE creating an audit file.
    expected=build_audit_reconstruction(folder,doc,cs,audit)
    if expected!=audit:raise RuntimeError('Proposed audit cannot be deterministically replayed')
    emit(folder/AUDIT_LEAF,audit)
    replayed,proof=verified_replay(folder,receipt['raw'],cs)
    print(json.dumps({'status':'M14_PHYSICS_IDENTITY_RECOVERY_SAVED_NO_MODEL_CALLS',
        'kind':kind,'replayed_strict_parser':True,
        'audit_path':str(folder/AUDIT_LEAF),'original_receipt_unchanged':True,
        'new_model_calls':0,'novelty_certified':False,
        'next_step':'Resume run-paid; do not regenerate existing physics review'},indent=2))
if __name__=='__main__':main()
