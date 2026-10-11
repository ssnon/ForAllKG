#!/usr/bin/env python3
"""Offline, additive M14-C4D evidence-anchor forensics and REVIEWED projection.

NEVER mutates the paid receipt, frozen sources, frozen prompt, original C4D code,
or original pending validation state. No model/network import or calls.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pipeline_core.discovery import sers_scientific_problem_discovery_m14c4d as core
from scripts.discovery import run_sers_scientific_problem_discovery_m14c4d as original

TEMPLATE = 'M14C4D_ANCHOR_REVIEW_TEMPLATE.json'
REVIEWED = 'M14C4D_ANCHOR_REVIEWED.json'
RECOVERED = 'M14C4D_REVIEWED_OPEN_RESEARCH_PROBLEM_CANDIDATES.json'
AUDIT = 'M14C4D_REVIEWED_ANCHOR_RECOVERY_AUDIT.json'
INDEPENDENT_REVIEW = 'M14C4D_RECOVERED_INDEPENDENT_SCIENCE_REVIEW_TEMPLATE.json'


def _validated_raw(out):
    plan, docs, _, prompt = original.context(out)  # enforces frozen source, code and prompt hashes
    rp=out/original.RECEIPT
    receipt=original.load(rp)
    if receipt.get('prompt_sha256') != core.sha(core.canonical(prompt)) or receipt.get('finish_reason') != 'stop':
        raise ValueError('M14C4D_SAVED_PAID_RECEIPT_PROMPT_OR_FINISH_INVALID_NO_RECALL')
    data=original._strict_json(receipt['raw'])
    if not isinstance(data,dict) or set(data)!={'candidates'} or not isinstance(data['candidates'],list):
        raise ValueError('M14C4D_INVALID_RAW_CANDIDATE_STRUCTURE')
    if not 2 <= len(data['candidates']) <= plan['desired_candidates']:
        raise ValueError('M14C4D_RAW_CANDIDATE_COUNT_INVALID')
    return plan, docs, receipt, data, original.digest(rp)


def _anchor_status(a, idx):
    if not isinstance(a,dict) or set(a)!={'source_id','exact_quote'}:
        return {'valid':False,'reason':'MALFORMED_ANCHOR','source_id':None,'kind':None}
    sid=a['source_id']; quote=a['exact_quote']
    doc=idx.get(sid) if isinstance(sid,str) else None
    if doc is None:
        return {'valid':False,'reason':'UNKNOWN_SOURCE_ID','source_id':sid,'kind':None}
    if not isinstance(quote,str):
        return {'valid':False,'reason':'NON_STRING_QUOTE','source_id':sid,'kind':doc['kind']}
    if len(quote)<18:
        reason='QUOTE_TOO_SHORT'
    elif len(quote)>250:
        reason='QUOTE_TOO_LONG'
    elif quote not in doc['excerpt']:
        reason='NOT_EXACTLY_IN_FROZEN_EXCERPT'
    else:
        reason='VALID'
    return {'valid':reason=='VALID','reason':reason,'source_id':sid,'kind':doc['kind'],
            'quote_len':len(quote),'quote':quote,
            'exactly_in_full_source_abstract':quote in doc['abstract'],
            'excerpt_preview':doc['excerpt'][:360],
            'excerpt_len':len(doc['excerpt'])}


def _diagnosis(out):
    plan,docs,receipt,data,receipt_sha=_validated_raw(out)
    idx={d['source_id']:d for d in docs}
    projected=copy.deepcopy(data)
    candidates=[]; drops=[]; blocking=[]
    for ci,ca in enumerate(data['candidates']):
        anchors=ca.get('evidence_anchors') if isinstance(ca,dict) else None
        if not isinstance(anchors,list):
            blocking.append(f'candidate_{ci}: malformed anchor list');
            candidates.append({'candidate_index':ci,'error':'malformed anchor list'});continue
        diagnostics=[]; selected=[]
        for ai,a in enumerate(anchors):
            s=_anchor_status(a,idx)
            diagnostics.append({'anchor_index':ai,**s})
            if s['valid']:
                selected.append(a)
            elif s['kind']=='published_data_secondary_analysis' and s['reason'] in ('NOT_EXACTLY_IN_FROZEN_EXCERPT','QUOTE_TOO_LONG','QUOTE_TOO_SHORT'):
                drops.append({'candidate_index':ci,'anchor_index':ai,'source_id':s['source_id'],
                              'original_quote_sha256':core.sha(core.canonical(s.get('quote'))),
                              'reason':s['reason']})
                # User can review removal of the optional secondary evidence anchor only.
            else:
                blocking.append(f'candidate_{ci}_anchor_{ai}:{s["reason"]}')
                selected.append(a)
        projected['candidates'][ci]['evidence_anchors']=selected
        published_valid={core.norm(idx[s['source_id']]['title']) for s in diagnostics
                         if s['valid'] and s['kind']=='published_abstract'}
        candidates.append({'candidate_index':ci,
          'research_question':ca.get('research_question') if isinstance(ca,dict) else None,
          'hypothesis':ca.get('hypothesis') if isinstance(ca,dict) else None,
          'original_anchor_count':len(anchors),'valid_independent_published_papers':len(published_valid),
          'anchors':diagnostics})
    if not drops:blocking.append('NO_ELIGIBLE_SECONDARY_EVIDENCE_ANCHOR_TO_REVIEW')
    for ci,ca in enumerate(candidates):
        if ca.get('valid_independent_published_papers',0)<2:
            blocking.append(f'candidate_{ci}:FEWER_THAN_TWO_VALID_INDEPENDENT_PUBLISHED_PAPERS')
    if not blocking:
        try:core.validate_submission(projected,docs,plan['desired_candidates'])
        except Exception as exc:blocking.append('FURTHER_ORIGINAL_VALIDATOR_FAILURE:'+str(exc))
    status=('M14C4D_OPTIONAL_SECONDARY_ANCHOR_REVIEW_POSSIBLE' if not blocking
            else 'M14C4D_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
    return {'status':status,'original_receipt_sha256':receipt_sha,
         'original_prompt_sha256':receipt['prompt_sha256'],
         'source_inventory_sha256':original.digest(out/original.DOCS),
         'original_model_calls_already_saved':1,'new_model_calls':0,
         'original_receipt_unchanged':True,'original_science_verified':False,
         'proposed_optional_secondary_anchor_omissions':drops,
         'blocking_reasons':blocking,'candidates':candidates}, projected, plan, docs


def inspect(out):
    result, *_=_diagnosis(out)
    return result


def _template_from(diag):
    return {'schema':'m14c4d-optional-secondary-anchor-review-v1',
            'receipt_sha256':diag['original_receipt_sha256'],
            'source_inventory_sha256':diag['source_inventory_sha256'],
            'proposed_omissions':diag['proposed_optional_secondary_anchor_omissions'],
            'reviewed_scientific_dependence_on_each_omitted_secondary_anchor':False,
            'authorized_optional_secondary_anchor_omission':False,
            'reviewer_reason':'',
            'scientific_novelty_certified':False,
            'the_original_model_claims_and_predictions_remain_unchanged':True}


def prepare_review(out):
    diag,*_=_diagnosis(out)
    if diag['blocking_reasons']:raise ValueError('M14C4D_REVIEW_TEMPLATE_BLOCKED: '+ '; '.join(diag['blocking_reasons']))
    path=out/TEMPLATE
    original.freeze(path,_template_from(diag))
    return {'status':'M14C4D_OPTIONAL_ANCHOR_REVIEW_TEMPLATE_CREATED',
       'review_template':str(path),'review_file_to_create':str(out/REVIEWED),
       'omission_count':len(diag['proposed_optional_secondary_anchor_omissions']),
       'scientific_review_still_required':True,'new_paid_calls':0,'raw_receipt_unchanged':True}


def _approved(out):
    diag,projected,plan,docs=_diagnosis(out)
    if diag['blocking_reasons']:
        raise ValueError('M14C4D_REVIEWED_PROJECTION_BLOCKED: '+ '; '.join(diag['blocking_reasons']))
    if not (out/TEMPLATE).is_file() or original.load(out/TEMPLATE)!=_template_from(diag):
        raise ValueError('M14C4D_FROZEN_REVIEW_TEMPLATE_CHANGED')
    review=original.load(out/REVIEWED)
    expected=_template_from(diag)
    expected['reviewed_scientific_dependence_on_each_omitted_secondary_anchor']=True
    expected['authorized_optional_secondary_anchor_omission']=True
    if not isinstance(review.get('reviewer_reason'),str) or len(review['reviewer_reason'].strip())<60:
        raise ValueError('M14C4D_EXPLICIT_SCIENTIFIC_REVIEW_REASON_REQUIRED_60_CHARS')
    expected['reviewer_reason']=review['reviewer_reason']
    if review!=expected:raise ValueError('M14C4D_REVIEW_FILE_MUST_MATCH_FROZEN_TEMPLATE_EXCEPT_EXPLICIT_REVIEW_FIELDS')
    core.validate_submission(projected,docs,plan['desired_candidates'])
    return diag,projected,plan,docs,review


def _outputs(out):
    diag,projection,plan,docs,review=_approved(out)
    results=core.summarize(projection,plan,docs)
    # The original raw result was rejected; this is explicitly a reviewer-approved projection.
    results.update({'status':'M14C4D_REVIEWED_PROJECTION_FROM_SAVED_PAID_RESPONSE_NOT_ORIGINAL_VALIDATION',
       'original_paid_receipt_sha256':diag['original_receipt_sha256'],
       'original_prompt_sha256':diag['original_prompt_sha256'],
       'source_inventory_sha256':diag['source_inventory_sha256'],
       'anchor_omission_count':len(diag['proposed_optional_secondary_anchor_omissions']),
       'exact_quote_validation_on_remaining_anchors':True,
       'reviewed_projection_not_a_model_regeneration':True,
       'scientific_novelty_certified':False,'independent_scientific_review_completed':False})
    audit={'schema':'m14c4d-optional-secondary-anchor-review-v1',
       'status':'M14C4D_REVIEWED_PROJECTION_ANCHOR_AUDIT_ONLY',
       'original_paid_receipt_sha256':diag['original_receipt_sha256'],
       'original_prompt_sha256':diag['original_prompt_sha256'],
       'source_inventory_sha256':diag['source_inventory_sha256'],
       'approved_review_sha256':core.sha(core.canonical(review)),
       'proposed_omissions':diag['proposed_optional_secondary_anchor_omissions'],
       'reviewer_reason':review['reviewer_reason'],
       'recovered_candidate_sha256':core.sha(core.canonical(results)),
       'original_submission_validated':False,
       'published_papers_remain_independent':True,
       'original_scientific_claims_unmodified':True,
       'scientific_novelty_certified':False,'independent_fulltext_prior_art_review_done':False,
       'raw_receipt_unchanged':True,'new_model_calls':0}
    independent={'schema':'m14c4d-optional-secondary-anchor-review-v1',
       'status':'M14C4D_INDEPENDENT_SCIENCE_REVIEW_PENDING',
       'candidates':[{'candidate_id':c['candidate_id'],
                      'question':c['research_question'],
                      'source_ids':[a['source_id'] for a in c['evidence_anchors']],
                      'source_relevance':'NOT_REVIEWED',
                      'physics_and_null':'NOT_REVIEWED',
                      'fulltext_prior_art':'NOT_REVIEWED',
                      'recommendation':'NOT_REVIEWED'} for c in results['candidates']]}
    return {RECOVERED:results,AUDIT:audit,INDEPENDENT_REVIEW:independent}


def recover_reviewed(out,authorize):
    if not authorize:raise ValueError('M14C4D_EXPLICIT_HUMAN_REVIEW_FLAG_REQUIRED')
    outputs=_outputs(out)
    for n in outputs:
        if (out/n).exists() and (out/n).read_bytes()!=core.canonical(outputs[n]):
            raise ValueError('M14C4D_EXISTING_RECOVERY_OUTPUT_DIFFERS: '+n)
    for n,data in outputs.items():original.freeze(out/n,data)
    return {'status':'M14C4D_REVIEWED_OPTIONAL_ANCHOR_PROJECTION_SAVED',
      'candidate_count':outputs[RECOVERED]['proposed_count'],
      'omitted_optional_anchors':outputs[RECOVERED]['anchor_omission_count'],
      'original_submission_validated':False,
      'independent_science_review_required':True,
      'recovered_candidate_file':str(out/RECOVERED),
      'new_model_calls':0,'original_raw_receipt_unchanged':True}


def verify_recovered(out):
    expected=_outputs(out)
    for n,record in expected.items():
        p=out/n
        if not p.is_file() or p.read_bytes()!=core.canonical(record):
            raise ValueError('M14C4D_REVIEWED_REPLAY_MISMATCH: '+n)
    return {'status':'M14C4D_REVIEWED_PROJECTION_REPLAY_VERIFIED',
     'original_submission_validated':False,
     'original_raw_receipt_unchanged':True,'new_model_calls':0,
     'scientific_novelty_certified':False,
     'candidate_count':expected[RECOVERED]['proposed_count']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    cmd=p.add_subparsers(dest='cmd',required=True)
    for action in ('inspect','prepare-review','recover-reviewed','verify-recovered'):
        sub=cmd.add_parser(action);sub.add_argument('--out-dir',type=Path,required=True)
        if action=='recover-reviewed':sub.add_argument('--i-reviewed-omitted-evidence',action='store_true')
    args=p.parse_args(argv)
    if args.cmd=='inspect':result=inspect(args.out_dir)
    elif args.cmd=='prepare-review':result=prepare_review(args.out_dir)
    elif args.cmd=='recover-reviewed':result=recover_reviewed(args.out_dir,args.i_reviewed_omitted_evidence)
    else:result=verify_recovered(args.out_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
