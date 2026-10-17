"""M22 saved-analysis recovery without any LLM call.

Only evidence-identity transformations are admissible: the exact original substring
on another selected page, or the unique PDF substring matching after whitespace
collapse and typographic (NOT mathematical) ligature expansion. All scientific
interpretation, numbers, formulas, PDF evidence and original raw remain intact.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from unittest.mock import patch

from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m
from scripts.discovery import run_m22_plan_terms_recovery as plan_recovery

PROTOCOL = 'm22-saved-analysis-source-span-fork-v1'
AUDIT = 'm22_source_span_recovery_audit.json'
ORIGINAL_PARSE = m.Analysis.model_validate_json
LIGATURES = {'\ufb00':'ff','\ufb01':'fi','\ufb02':'fl','\ufb03':'ffi','\ufb04':'ffl', '\ufb05':'st','\ufb06':'st'}


def fhash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def shash(value):
    return sha256(value.encode('utf-8')).hexdigest()


def canonical_with_offsets(value):
    """No case-folding, spelling correction, math normalization or punctuation edits."""
    norm=[];offsets=[]
    for index, c in enumerate(value):
        if c.isspace():
            if norm and norm[-1] == ' ':continue
            expansion=' '
        else:expansion=LIGATURES.get(c,c)
        for ch in expansion:
            norm.append(ch);offsets.append(index)
    while norm and norm[0]==' ':
        norm.pop(0);offsets.pop(0)
    while norm and norm[-1]==' ':
        norm.pop();offsets.pop()
    return ''.join(norm),offsets


def exact_selected_span(span,pages):
    """Return unique (page, PDF-verbatim span, match_type) or None.

    Reconstructed slices are contiguous regions from the SAVED PDF page text;
    only whitespace and well-known typographic ligatures can differ.
    """
    matches=[]
    target,_=canonical_with_offsets(span)
    if len(target)<18:return None
    for page,text in pages.items():
        if span in text:
            if text.count(span)==1:
                matches.append((page,span,'literal'))
            continue
        canon,offsets=canonical_with_offsets(text)
        pos=canon.find(target)
        if pos >= 0:
            second=canon.find(target,pos+1)
            if second >= 0:continue # Repeated span is ambiguous even on same page.
            # Index map maps normalized characters to verbatim PDF characters.
            raw_span=text[offsets[pos]:offsets[pos+len(target)-1]+1]
            if len(raw_span)>1000 or len(raw_span)<18:continue
            matches.append((page,raw_span,'whitespace_or_typographic_ligature'))
    return matches[0] if len(matches)==1 else None


def closest_context(span,pages):
    """Non-authoritative diagnostics only. NEVER used as a recovery decision."""
    wanted,_=canonical_with_offsets(span)
    candidates=[]
    for page,text in pages.items():
        norm,_=canonical_with_offsets(text)
        size=max(60,min(len(wanted),500))
        for start in range(0,max(1,len(norm)-size+1),max(25,size//3)):
            candidate=norm[start:start+size]
            ratio=SequenceMatcher(None,wanted[:500],candidate,autojunk=False).ratio()
            candidates.append((ratio,page,candidate[:240]))
    if not candidates:return None
    ratio,page,preview=max(candidates)
    return {'page':page,'similarity_for_diagnosis_only':round(ratio,3),'page_text_preview':preview}


def analyze_saved(out):
    out=Path(out).expanduser().resolve()
    with plan_scope(out):
        state=m.verify(out)
        if not (state['plan']=='READY' and state['pdf']=='PDF_PAGES_RETRIEVED' and
                state['analysis']=='RAW_SAVED_RECOVER_NO_RETRY'):
            raise ValueError('REQUIRES_ORIGINAL_SAVED_UNCOMMITTED_ANALYSIS_AND_RETRIEVED_PDF')
    raw=m.read(out/'analyze/raw_response.json')['raw_text']
    obj=ORIGINAL_PARSE(raw).model_dump(mode='json')
    scan=m.read(out/'pdf_evidence.json')
    pages={page['page']:page['text'] for page in scan['selected_pages']}
    corrected=copy.deepcopy(obj);changes=[];blocked=[]
    for idx,anchor in enumerate(obj['anchors']):
        before=anchor['exact_source_span'];page=anchor['page']
        if page in pages and before in pages[page]:continue
        match=exact_selected_span(before,pages)
        if not match:
            blocked.append({'index':idx,'claimed_page':page,'claimed_span':before[:300],
                            'reason':'NO_UNIQUE_EXACT_OR_WHITESPACE_LIGATURE_EQUIVALENT_ON_SAVED_PAGES',
                            'closest_candidate_NOT_EVIDENCE':closest_context(before,pages)})
            continue
        truepage,span,how=match
        corrected['anchors'][idx]['page']=truepage
        corrected['anchors'][idx]['exact_source_span']=span
        changes.append({'index':idx,'original_page':page,'verbatim_page':truepage,
                        'original_span':before,'verbatim_span':span,
                        'match_method':how})
    if not changes and not blocked:raise ValueError('NO_ANCHOR_MISMATCH_FOUND')
    if not blocked:
        canonical=m.Analysis.model_validate(corrected).model_dump(mode='json')
        # Full original validators, including reported numeric literal requirements.
        m._validate_anchors(canonical,scan)
        m._execute(canonical)
        for key in obj:
            if key!='anchors' and canonical[key]!=obj[key]:
                raise ValueError('NON_EVIDENCE_SCIENCE_FIELD_CHANGED')
        for old,new in zip(obj['anchors'],canonical['anchors']):
            for key in old:
                if key not in ('page','exact_source_span') and old[key]!=new[key]:
                    raise ValueError('SCIENTIFIC_ANCHOR_INTERPRETATION_CHANGED')
    else:canonical=None
    return state,obj,canonical,changes,blocked


@contextmanager
def plan_scope(out):
    audit,canonical=plan_recovery.audit_check(out)
    with plan_recovery.scoped_saved_raw(audit['original']['raw_text_sha256'],canonical):
        yield


@contextmanager
def both_scopes(out,raw_hash,corrected):
    original=m.Analysis.model_validate_json
    def interpret(cls,raw,*args,**kwargs):
        comp=raw.decode('utf-8') if isinstance(raw,bytes) else raw
        if isinstance(comp,str) and shash(comp)==raw_hash:
            # Require original response still conforms to strict Analysis contract.
            ORIGINAL_PARSE(raw,*args,**kwargs)
            return cls.model_validate(corrected)
        return original(raw,*args,**kwargs)
    with plan_scope(out),patch.object(m.Analysis,'model_validate_json',classmethod(interpret)):
        yield


def inspect(source):
    _,_,_,changes,blocked=analyze_saved(source)
    return {'status':('M22_VERBATIM_RECOVERY_ELIGIBLE_NO_CALL' if not blocked else
                      'M22_UNVERIFIED_QUOTES_NEED_MANUAL_REVIEW'),
            'saved_paid_response_preserved':True,'new_paid_calls_required':False,
            'safe_matches':changes,'unresolved_anchors':blocked,
            'important':'Fuzzy context is diagnostic only, never substituted for source evidence.'}


def source_digests(source):
    files=['plan/raw_response.json','plan/normalized.json','analyze/raw_response.json',
           'analyze/request.json','lock.json','pdf_evidence.json','pdf_receipt.json',
           plan_recovery.AUDIT]
    return {f:fhash(source/f) for f in files}


def fork(source,target):
    source=Path(source).expanduser().resolve();target=Path(target).expanduser().resolve()
    if target.exists() or target==source or source in target.parents or target in source.parents:
        raise ValueError('FRESH_DISJOINT_OUTPUT_REQUIRED')
    _,raw,corrected,changes,blocked=analyze_saved(source)
    if blocked:
        raise ValueError('UNMATCHED_SOURCE_QUOTES_CANNOT_BE_AUTO_REPAIRED_USE_INSPECT')
    original_hashes=source_digests(source)
    raw_text=m.read(source/'analyze/raw_response.json')['raw_text']
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,target)
    try:
        with both_scopes(target,shash(raw_text),corrected):
            result=m.recover(target,'analyze')
        audit={'protocol':PROTOCOL,'source':str(source),'source_file_hashes':original_hashes,
               'original_raw_text_sha256':shash(raw_text),
               'canonical_science_sha256':m.sha(corrected),
               'canonical_file_sha256':fhash(target/'analyze/normalized.json'),
               'changes':changes,'scientific_validation':False,'paid_calls':0,
               'policy':'Only unique exact or whitespace/ligature-equivalent saved PDF excerpts, with page relocation if uniquely matched; no semantic or numerical edits.'}
        m.put(target/AUDIT,audit)
        verify(target)
        if source_digests(source)!=original_hashes:
            raise ValueError('SOURCE_RUN_CHANGED_DURING_FORK')
    except BaseException:
        shutil.rmtree(target);raise
    return {'status':'M22_SAVED_ANALYSIS_RECOVERED_NO_PAID_CALL','target':str(target),
            'anchors_corrected':len(changes),'calculation_status':m.read(target/'analyze/calculation_result.json')['status'],
            'source_untouched':True,'scientific_validation':False,'paid_calls':0}


def check_audit(out):
    out=Path(out).expanduser().resolve()
    audit=m.read(out/AUDIT)
    if audit.get('protocol')!=PROTOCOL or audit.get('scientific_validation') is not False:
        raise ValueError('M22_ANALYSIS_RECOVERY_AUDIT_INVALID')
    src=Path(audit['source']).expanduser().resolve()
    if out==src or src in out.parents or out in src.parents:
        raise ValueError('AUDIT_SOURCE_TARGET_OVERLAP')
    if source_digests(src)!=audit['source_file_hashes']:
        raise ValueError('M22_SOURCE_ORIGINAL_CHANGED')
    _,_,canonical,changes,blocked=analyze_saved(src)
    if blocked or m.sha(canonical)!=audit['canonical_science_sha256'] or changes!=audit['changes']:
        raise ValueError('M22_RECOVERY_CANONICAL_CHANGED')
    if fhash(out/'analyze/raw_response.json')!=audit['source_file_hashes']['analyze/raw_response.json']:
        raise ValueError('M22_FORK_PAID_RAW_CHANGED')
    if fhash(out/'pdf_evidence.json')!=audit['source_file_hashes']['pdf_evidence.json']:
        raise ValueError('M22_FORK_PDF_SOURCE_CHANGED')
    if fhash(out/'analyze/normalized.json')!=audit['canonical_file_sha256']:
        raise ValueError('M22_FORK_NORMALIZED_CHANGED')
    return audit,canonical


def verify(out):
    audit,corrected=check_audit(out)
    with both_scopes(out,audit['original_raw_text_sha256'],corrected):
        return m.verify(out)


def report(out):
    audit,corrected=check_audit(out)
    with both_scopes(out,audit['original_raw_text_sha256'],corrected):
        return m.report(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inspect','fork','verify','report'])
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='inspect':
        if args.source is None:p.error('inspect requires --source')
        result=inspect(args.source)
    elif args.command=='fork':
        if args.source is None or args.out is None:p.error('fork needs --source and --out')
        result=fork(args.source,args.out)
    else:
        if args.out is None:p.error('verify/report requires --out')
        result=verify(args.out) if args.command=='verify' else report(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
