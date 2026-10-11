"""M14-C2 offline recovery from saved paid receipt: family LABEL ONLY.

NEVER modifies provider receipt, intent, original contract, original scientific hypothesis.
Requires independent human scientific-content alignment; no automatic retrial.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import sha as m14sha
from pipeline_core.discovery.sers_adaptive_policy_m14b import normalize
from scripts.discovery.run_sers_evidence_successor_m14c2 import (
    CONTRACT, RAW, _load_contract, _validate_submission,
)

RECEIPT = 'm14c2_successor_RAW_RECEIPT.json'
INTENT = 'm14c2_successor_INTENT.json'
PROMPT = 'M14C2_GENERATION_PROMPT.json'
TEMPLATE = 'M14C2_FAMILY_ALIGNMENT_TO_REVIEW.json'
AUDIT = 'M14C2_FAMILY_ALIGNMENT_RECOVERY_AUDIT.json'

def _sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()

def _canon(value)->bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')

def _read(path:Path):
    return json.loads(path.read_bytes())

def _write_once(path:Path,data:bytes):
    with path.open('xb') as f: f.write(data)

def _decode_response(raw:str):
    if not isinstance(raw,str): raise ValueError('Receipt raw must be a string')
    trimmed=raw.strip()
    if trimmed.startswith('```json'):trimmed=trimmed[7:].strip()
    elif trimmed.startswith('```'):trimmed=trimmed[3:].strip()
    if trimmed.endswith('```'): trimmed=trimmed[:-3].strip()
    result=json.loads(trimmed)
    if not isinstance(result,dict):raise ValueError('Expected JSON object')
    return result

def _ctx(out:Path):
    if not out.is_dir(): raise FileNotFoundError('M14-C2 output folder does not exist')
    # Do not access any API/client provider. Receipts are read-only.
    contract_path=out/CONTRACT
    receipt_path=out/RECEIPT
    prompt_path=out/PROMPT
    contract=_load_contract(out)
    receipt=_read(receipt_path)
    prompt=_read(prompt_path)
    if receipt.get('finish_reason')!='stop': raise ValueError('Provider response was incomplete; scientific recovery forbidden')
    if receipt.get('prompt_sha256')!=m14sha(prompt):raise ValueError('Raw receipt is not bound to the frozen generation prompt')
    if (out/INTENT).is_file():
        intent=_read(out/INTENT)
        if intent.get('prompt_sha256')!=receipt['prompt_sha256'] or intent.get('model')!=receipt.get('model'):
            raise ValueError('Intent and response receipt mismatch')
    submission=_decode_response(receipt.get('raw'))
    successor=submission.get('successor') if isinstance(submission,dict) else None
    if not isinstance(successor,dict):raise ValueError('Original response lacks a successor object')
    old_family=successor.get('family')
    if not isinstance(old_family,str) or not old_family.strip():raise ValueError('Original successor lacks a nonempty family label')
    problem_id=successor.get('problem_id')
    cards=contract['lanes']['MECHANISM_SEARCH']['alternative_cards']
    matches=[p for p in cards if p.get('problem_id')==problem_id]
    if len(matches)!=1: raise ValueError('Original response did not select exactly one anchored alternate problem card')
    selected=matches[0]
    canonical=selected['domain_family']
    if not isinstance(canonical,str) or not canonical.strip():raise ValueError('Invalid anchor family')
    if normalize(old_family)==normalize(canonical):
        raise ValueError('This is not a family-label mismatch; use original verify command')
    if any(normalize(c['domain_family'])==normalize(old_family) for c in cards if c['problem_id']!=problem_id):
        raise ValueError('Original family matches another card; conflicting card identities need new scientific review')
    patched=copy.deepcopy(submission)
    patched['successor']['family']=canonical
    # Fail-closed: the *only* alteration is top-level family; every other gate must pass.
    test=_validate_submission(contract,patched)
    return {'contract':contract,'submission':submission,'patched':patched,'selected':selected,
       'test':test,'receipt_sha256':_sha(receipt_path.read_bytes()),
       'contract_sha256':_sha(contract_path.read_bytes()),
       'prompt_sha256':_sha(prompt_path.read_bytes()),
       'old_family':old_family,'canonical_family':canonical,
       'source':receipt_path}

def inspection(out:Path):
    try:
        c=_ctx(out)
    except Exception as ex:
        return {'status':'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR',
                'reason':str(ex),'new_model_calls':0,'original_receipt_unchanged':True}
    sig=c['submission']['successor'].get('mechanism_signature',{})
    selected=c['selected']
    return {'status':'M14C2_FAMILY_LABEL_REVIEW_POSSIBLE_NO_MODEL_CALLS',
            'original_receipt_sha256':c['receipt_sha256'],
            'contract_sha256':c['contract_sha256'],
            'successor_problem_id':c['submission']['successor']['problem_id'],
            'model_family':c['old_family'],
            'anchored_card_domain_family':c['canonical_family'],
            'anchored_card_scientific_question':{k:selected[k] for k in (
                'research_question','provisional_question','problem_statement','open_question','description',
                'problem','scientific_gap','contradiction','provisional_problem',
                'scientific_question','title','critical_unknown','competing_explanations',
                'differential_observable','why_still_uncertain') if k in selected},
            'anchored_card_source_anchors':selected.get('source_anchors',[])[:6],
            'model_causal_driver':sig.get('causal_driver'),
            'model_core_assumption':sig.get('core_assumption'),
            'model_mechanism_family_id':sig.get('mechanism_family_id'),
            'model_strongest_null':sig.get('strongest_null'),
            'model_predicted_observable':sig.get('predicted_observable'),
            'model_independent_intervention':sig.get('independent_intervention'),
            'model_falsification':sig.get('falsification'),
            'model_conditional_predictions':c['submission']['successor'].get('conditional_predictions'),
            'model_independent_observables':c['submission']['successor'].get('independent_observables'),
            'family_label_only_replacement_passes_other_structural_gates':True,
            'scientific_semantic_match_certified':False,
            'warning':'Structural validity is not scientific alignment. Do NOT recover until scientific content matches selected anchored card.',
            'new_model_calls':0,'original_receipt_unchanged':True}

def make_review(out:Path):
    ctx=_ctx(out)
    dest=out/TEMPLATE
    content={'schema':'m14c2-family-human-alignment-review-v1',
        'original_receipt_sha256':ctx['receipt_sha256'],
        'contract_sha256':ctx['contract_sha256'],
        'selected_problem_id':ctx['submission']['successor']['problem_id'],
        'model_family_exact':ctx['old_family'],
        'corrected_family_exact':ctx['canonical_family'],
        'scientific_alignment_confirmed':False,
        'alignment_reason':'',
        'independent_review_of_causal_driver_and_problem':False,
        'scientific_novelty_certified':False}
    if dest.exists():
        if _read(dest)!=content:raise FileExistsError('Existing review template differs. Never overwrite.')
    else:_write_once(dest,_canon(content))
    return {'status':'M14C2_SCIENTIFIC_FAMILY_REVIEW_TEMPLATE_PREPARED','path':str(dest),
            'new_model_calls':0,'original_receipt_unchanged':True}

def recover(out:Path,review_file:Path,approved:bool):
    if not approved:raise ValueError('Explicit human scientific review confirmation flag required')
    ctx=_ctx(out)
    review=_read(review_file)
    expected={'schema':'m14c2-family-human-alignment-review-v1',
        'original_receipt_sha256':ctx['receipt_sha256'],'contract_sha256':ctx['contract_sha256'],
        'selected_problem_id':ctx['submission']['successor']['problem_id'],
        'model_family_exact':ctx['old_family'],'corrected_family_exact':ctx['canonical_family'],
        'scientific_novelty_certified':False}
    for k,v in expected.items():
        if review.get(k)!=v: raise ValueError('Review is not bound to exact receipt/contract/card: '+k)
    if review.get('scientific_alignment_confirmed') is not True or review.get('independent_review_of_causal_driver_and_problem') is not True:
        raise ValueError('Both scientific alignment confirmations required')
    reason=review.get('alignment_reason')
    if not isinstance(reason,str) or len(reason.strip())<35:
        raise ValueError('Specific scientific alignment justification (>=35 chars) required')
    if (out/RAW).exists() or (out/AUDIT).exists():
        raise FileExistsError('Recovery output or audit already present: do not overwrite')
    # Confirm no other field changes. Important before freezing anything.
    orig=copy.deepcopy(ctx['patched']);orig['successor']['family']=ctx['old_family']
    if orig!=ctx['submission']:raise RuntimeError('Non-family scientific content changed: forbidden')
    result=ctx['test']
    if result.get('status')!='M14C2_ONE_SUCCESSOR_STRUCTURALLY_VALIDATED':
        raise ValueError('Original C2 strict validator rejected candidate after label-only repair')
    corrected=_canon(ctx['patched'])
    audit={'schema':'m14c2-family-alignment-recovery-audit-v1',
        'status':'M14C2_REVIEWED_FAMILY_LABEL_RECOVERED_NO_MODEL_CALLS',
        'original_raw_receipt_sha256':ctx['receipt_sha256'],
        'frozen_contract_sha256':ctx['contract_sha256'],
        'review_sha256':_sha(review_file.read_bytes()),
        'selected_problem_id':ctx['submission']['successor']['problem_id'],
        'original_model_family':ctx['old_family'],'corrected_card_family':ctx['canonical_family'],
        'only_changed_scientific_field':'successor.family',
        'scientific_content_preserved':True,
        'human_review_alignment_reason':reason,
        'submission_sha256':_sha(corrected),
        'strict_original_c2_validation_replayed':True,
        'scientific_novelty_certified':False,
        'provider_calls':0,
        'original_receipt_unchanged':True}
    # Audit first: if interrupted, manual inspection required; no paid recall.
    _write_once(out/AUDIT,_canon(audit))
    _write_once(out/RAW,corrected)
    return {'status':audit['status'], 'replayed_strict_C2_validator':True,
            'original_raw_receipt_sha256':ctx['receipt_sha256'],
            'repaired_submission_sha256':audit['submission_sha256'],
            'audit_path':str(out/AUDIT),'new_model_calls':0,
            'original_receipt_unchanged':True,
            'scientific_novelty_certified':False,
            'next_step':'Run original M14-C2 verify command; never repeat generate-paid'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('inspect','make-review-template','recover-reviewed'):
        a=sub.add_parser(name)
        a.add_argument('--out-dir',type=Path,required=True)
        if name=='recover-reviewed':
            a.add_argument('--review-file',type=Path,required=True)
            a.add_argument('--i-reviewed-problem-and-mechanism',action='store_true')
    args=p.parse_args(argv)
    if args.cmd=='inspect':v=inspection(args.out_dir)
    elif args.cmd=='make-review-template':v=make_review(args.out_dir)
    else:v=recover(args.out_dir,args.review_file,args.i_reviewed_problem_and_mechanism)
    print(json.dumps(v,indent=2,ensure_ascii=False))
    return v

if __name__=='__main__':main()
