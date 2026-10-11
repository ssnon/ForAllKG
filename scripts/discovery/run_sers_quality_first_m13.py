#!/usr/bin/env python3
"""Opt-in M13 quality-first matched scientific loop vs no-feedback control."""
from __future__ import annotations
import argparse
import json
import os
import re
import hashlib
import time
import zipfile
from pathlib import Path
from pipeline_core.discovery.sers_quality_first_m13 import (
    ARMS, checked, prepare, read, create, file_sha, sh, generate_prompt,
    parse_generation, compress_evidence, physics_prompt, literature_prompt,
    parse_physics, parse_literature, parse_literature_with_quarantine, science_feedback, blinded_rows, blind_csv, score_completed_blind, diagnose_generation_receipt, replacement_generation_prompt, verify_new_claims_not_literal_rescue)


def saved_model(root:Path,rel:str,prompt:dict,model:str,call):
    target=root/rel
    intent=target.with_name(target.name+'_INTENT.json')
    rawfile=target.with_name(target.name+'_RAW_RECEIPT.json')
    p_hash=sh(prompt)
    if rawfile.exists():
        r=read(rawfile)
        if r.get('prompt_sha')!=p_hash or r.get('model')!=model:
            raise RuntimeError('M13 stored receipt prompt/model mismatch; do not re-call: '+str(rawfile))
        if r.get('finish_reason')!='stop':raise RuntimeError('Saved incomplete response; do not recall '+str(rawfile))
        return r['raw'],False
    if intent.exists():raise RuntimeError('Indeterminate attempted paid call: '+str(intent)+'; no auto-retry')
    create(intent,{'status':'PROVIDER_ATTEMPT_UNKNOWN_UNTIL_RAW_SAVED',
      'prompt_sha':p_hash,'model':model})
    try: response=call(model,prompt)
    except Exception as e:
        raise RuntimeError('M13 provider error with indeterminate intent preserved: '+str(e)) from e
    create(rawfile,{'prompt_sha':p_hash,'model':model,'raw':response['text'],
      'finish_reason':response['finish_reason'],'usage':response.get('usage'),
      'novelty_certified':False,'authoritative_science':False})
    if response['finish_reason']!='stop':raise RuntimeError('M13 model raw preserved but finish_reason incomplete; do not auto-retry')
    return response['text'],True


def parse_or_save(p:Path,v:dict):
    if p.exists():
        if read(p)!=v:raise RuntimeError('Existing parsed file different: '+str(p))
    else:create(p,v)


def bibliographic_search(root:Path,arm:str,g:int,gen:dict,lookup,rep:int=1):
    results=[]
    for c in gen['candidates']:
        for ix,q in enumerate(c['prior_art_queries']):
            for provider in ('openalex','europepmc'):
                directory=root/f'r{rep:02}'/f'g{g:02}'/arm/'literature'
                key=sh({'candidate':c['candidate_id'],'provider':provider,'query':q})[:20]
                path=directory/f'{ix+1}_{c["lane"]}_{provider}_{key}_SUCCESS.json'
                if path.exists(): r=read(path)
                else:
                    try: r=lookup(provider,q)
                    except Exception as e:
                        failures=sorted(directory.glob(f'{ix+1}_{c["lane"]}_{provider}_{key}_FAIL_*.json'))
                        failpath=directory/f'{ix+1}_{c["lane"]}_{provider}_{key}_FAIL_{len(failures)+1:03}.json'
                        create(failpath,{'error_type':type(e).__name__,'detail':str(e),
                             'provider':provider,'query':q,'new_provider_calls':0})
                        raise RuntimeError('Scholarly search failed; no novelty assumption, resume once API recovers: '+str(failpath)) from e
                    if r.get('success') is not True or r.get('provider')!=provider or r.get('query')!=q:
                        raise RuntimeError('Literature provider result incomplete; do not proceed')
                    create(path,r)
                results.append(r)
    if not results:return {'works':[], 'work_ids':[],'search_queries':[],'absence_proves_novelty':False}
    e=compress_evidence(results)
    path=root/f'r{rep:02}'/f'g{g:02}'/arm/'literature'/'M13_EVIDENCE_FOR_REVIEW.json'
    parse_or_save(path,e)
    return e



# --- M13 one-defect syntax-only recovery: original paid receipt remains immutable ---
SYNTAX_RECOVERY_BASENAME = 'M13_GENERATION_SYNTAX_RECOVERY.json'


def _single_stray_array_bracket_after_strongest_null(raw: str) -> tuple[str, dict]:
    """Repair only: <string field strongest_null> newline `],` newline <title>.

    A repaired candidate must still pass the unmodified strict M13 scientific parser.
    Never guess missing scientific text, truncate, change quotes, or fix multiple defects.
    """
    unwrapped = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    try:
        json.loads(unwrapped)
    except json.JSONDecodeError as source_error:
        source_error_text = str(source_error)
    else:
        raise ValueError('Source raw already valid JSON; syntax recovery prohibited')
    # Match exactly a JSON string value (including escaped characters) followed by
    # a stray array closing delimiter, and the next literal property `title`.
    pattern = re.compile(r'(?P<value>^[ \t]*"strongest_null"[ \t]*:[ \t]*"(?:\\.|[^"\\])*")(?P<newline>\r?\n)[ \t]*\],[ \t]*\r?\n(?P<title>[ \t]*"title"[ \t]*:)', re.M)
    hits=list(pattern.finditer(unwrapped))
    if len(hits)!=1:
        raise ValueError('Expected exactly one known stray-array-close defect; found '+str(len(hits)))
    hit=hits[0]
    fixed=unwrapped[:hit.start()] + hit.group('value') + ',' + hit.group('newline') + hit.group('title') + unwrapped[hit.end():]
    # Important: a single regex patch must make the ENTIRE payload parseable.
    obj=json.loads(fixed)
    if not isinstance(obj,dict):raise ValueError('Fixed result not a JSON object')
    return fixed, {'rule':'ONE_STRAY_CLOSING_ARRAY_BRACKET_BETWEEN_STRONGEST_NULL_AND_TITLE',
        'original_parser_error':source_error_text,
        'replaced_segment_sha256':hashlib.sha256(hit.group(0).encode('utf-8')).hexdigest(),
        'before_excerpt':hit.group(0)[-240:],
        'after_excerpt':(hit.group('value')+','+hit.group('newline')+hit.group('title'))[-240:],
        'corrected_raw_sha256':hashlib.sha256(fixed.encode('utf-8')).hexdigest(),
        'semantic_content_edited':False}


def _validated_saved_syntax_repair(path: Path, plan: dict, rep: int, g: int, arm: str, original_raw: str) -> tuple[str, dict]:
    audited=read(path/SYNTAX_RECOVERY_BASENAME)
    original_receipt=path/'generation_RAW_RECEIPT.json'
    if audited.get('source_receipt_sha256') != file_sha(original_receipt):
        raise RuntimeError('M13_SYNTAX_RECOVERY_SOURCE_RECEIPT_SHA_DRIFT')
    if audited.get('original_raw_sha256') != hashlib.sha256(original_raw.encode('utf-8')).hexdigest():
        raise RuntimeError('M13_SYNTAX_RECOVERY_ORIGINAL_RAW_SHA_DRIFT')
    if (audited.get('replicate'),audited.get('generation'),audited.get('arm')) != (rep,g,arm):
        raise RuntimeError('M13_SYNTAX_RECOVERY_WRONG_COHORT')
    if audited.get('plan_id') != plan['plan_id'] or audited.get('novelty_certified') is not False:
        raise RuntimeError('M13_SYNTAX_RECOVERY_PLAN_OR_AUTHORITY_DRIFT')
    fixed,mechanical=_single_stray_array_bracket_after_strongest_null(original_raw)
    if audited.get('repair') != mechanical or audited.get('corrected_raw') != fixed:
        raise RuntimeError('M13_SYNTAX_RECOVERY_PATCH_DRIFT')
    # Strict scientific validation is repeated at consumption time.
    parse_generation(fixed,g,plan,rep,arm)
    return fixed,audited


def cmd_recover_syntax(args):
    plan=checked(read(args.plan));root=args.plan.parent
    rep,g,arm=args.replicate,args.generation,args.arm
    if arm not in ARMS or not (1 <= rep <= plan['design']['replicates']) or not (1 <= g <= plan['design']['generations']):
        raise ValueError('Recovery cohort outside frozen M13 plan')
    path=root/f'r{rep:02}'/f'g{g:02}'/arm
    origin=path/'generation_RAW_RECEIPT.json'
    if not origin.is_file():raise FileNotFoundError(str(origin))
    if (path/'generation_REPLACEMENT_RAW_RECEIPT.json').exists():
        raise RuntimeError('New paid replacement already exists here; cannot relabel lineage as original syntax-only')
    if (path/'M13_GENERATION_PARSED.json').exists():
        raise RuntimeError('Generation already parsed; no syntax recovery necessary')
    receipt=read(origin)
    if receipt.get('finish_reason')!='stop':raise RuntimeError('Incomplete model output; cannot syntax-repair')
    raw=receipt['raw']
    repaired,details=_single_stray_array_bracket_after_strongest_null(raw)
    checked_generation=parse_generation(repaired,g,plan,rep,arm)
    audit={
      'status':'M13_ORIGINAL_MODEL_RESPONSE_SYNTAX_ONLY_REPAIRED',
      'plan_id':plan['plan_id'],'replicate':rep,'generation':g,'arm':arm,
      'source_receipt_sha256':file_sha(origin),
      'original_raw_sha256':hashlib.sha256(raw.encode('utf-8')).hexdigest(),
      'corrected_raw':repaired,'repair':details,
      'candidates_strictly_validated':len(checked_generation['candidates']),
      'raw_receipt_immutable':True,'new_paid_calls':0,
      'new_scientific_ideas_generated':0,'novelty_certified':False}
    parse_or_save(path/SYNTAX_RECOVERY_BASENAME,audit)
    print(json.dumps({'status':audit['status'],'replicate':rep,'generation':g,'arm':arm,
        'candidate_count':len(checked_generation['candidates']),
        'source_receipt_sha256':audit['source_receipt_sha256'],
        'syntax_recovery_audit':str(path/SYNTAX_RECOVERY_BASENAME),
        'new_paid_calls':0,'novelty_certified':False},ensure_ascii=False,indent=2))


# --- M13 saved Physics JSON LaTeX escaping recovery; no LLM call ---
PHYSICS_SYNTAX_RECOVERY_BASENAME = 'M13_PHYSICS_JSON_SYNTAX_RECOVERY.json'
# A single backslash followed by one of these longer TeX commands is a
# literal TeX control word, EVEN when its first letter is a JSON escape
# (\\t, \\n, \\r, \\b or \\f). Without this test the JSON decoder
# silently corrupts physical equations, e.g. \\theta becomes TAB+heta.
_LATEX_CONTROLS = frozenset(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega', 'frac', 'dfrac', 'tfrac', 'sqrt', 'sum', 'prod', 'int', 'iint', 'iiint', 'oint', 'lim', 'exp', 'log', 'ln', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'max', 'min', 'sup', 'inf', 'partial', 'nabla', 'cdot', 'times', 'pm', 'mp', 'le', 'leq', 'ge', 'geq', 'neq', 'approx', 'sim', 'propto', 'infty', 'to', 'mapsto', 'rightarrow', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'begin', 'end', 'text', 'textrm', 'textbf', 'mathrm', 'mathbf', 'mathit', 'mathcal', 'mathbb', 'mathsf', 'operatorname', 'overline', 'underline', 'hat', 'widehat', 'tilde', 'widetilde', 'bar', 'vec', 'dot', 'ddot', 'underbrace', 'overbrace', 'displaystyle', 'scriptstyle', 'scriptletter', 'boldsymbol', 'bm', 'boxed', 'tag', 'label', 'ref', 'eqref', 'hbar', 'ell', 'lVert', 'rVert', 'vert', 'Vert', 'langle', 'rangle', 'ldots', 'cdots', 'vdots', 'ddots', 'quad', 'qquad', 'thinspace', 'hspace', 'vspace', 'color', 'over', 'setminus', 'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap', 'land', 'lor', 'neg', 'forall', 'exists', 'in', 'notin', 'equiv', 'congruent', 'perp', 'parallel', 'otimes', 'oplus', 'infty', 'aleph', 'Re', 'Im', 'det', 'tr', 'Tr', 'E', 'Var', 'Cov', 'Pr', 'independent', 'sumlimits', 'nolimits', 'leftarrow', 'longrightarrow', 'longleftarrow', 'longleftrightarrow', 'Longrightarrow', 'phantom', 'smash', 'not'])


def _repair_physics_json_latex_only(raw: str) -> tuple[str,dict]:
    """Escape literal LaTeX backslashes in JSON string values, never alter prose.

    Repairs only backslash encoding; fail closed on ambiguous JSON escapes.
    All repaired physics rows are subsequently validated by parse_physics.
    """
    try:
        json.loads(raw)
    except json.JSONDecodeError as original_error:
        error_before=str(original_error)
        if 'Invalid \\escape' not in error_before:
            raise ValueError('Only an invalid JSON backslash escape can be recovered: '+error_before)
    else:
        raise ValueError('Physics raw already valid JSON, no syntax repair needed')
    edits=[]
    inside=False
    i=0
    result=[]
    while i<len(raw):
        ch=raw[i]
        if ch=='"':
            inside=not inside
            result.append(ch)
            i+=1
            continue
        if inside and ch=='\\':
            if i+1>=len(raw):
                raise ValueError('Dangling terminal backslash; manual review required')
            nxt=raw[i+1]
            if nxt in ('\\','"','/'):
                result.append(raw[i:i+2]);i+=2;continue
            if nxt=='u' and re.match(r'^[0-9a-fA-F]{4}$', raw[i+2:i+6]):
                result.append(raw[i:i+6]);i+=6;continue
            word_match=re.match(r'[A-Za-z]+',raw[i+1:])
            word=word_match.group(0) if word_match else ''
            is_latex=word in _LATEX_CONTROLS
            if nxt in 'bfnrt' and word and len(word)>1 and not is_latex:
                # Could be an intentional JSON newline + text, or a TeX
                # command (e.g. \\newmacro); cannot guess silently.
                raise ValueError('Ambiguous JSON escape vs TeX command at offset '+str(i)+': '+repr(raw[i:i+36]))
            if (nxt in 'bfnrt' and not is_latex) or (nxt=='u' and word=='u'):
                result.append(raw[i:i+2]);i+=2;continue
            # For invalid JSON escapes, we only insert one extra backslash.
            # This represents exactly the literal slash authored by the model.
            reason=('KNOWN_TEX_COMMAND' if is_latex else 'INVALID_JSON_ESCAPE_LITERAL_BACKSLASH')
            if not is_latex and nxt=='u' and len(word)>1:
                # \\unknown is a literal TeX command, not valid JSON unicode.
                reason='INVALID_JSON_UNICODE_ESCAPE_LITERAL_BACKSLASH'
            if nxt=='u' and not word_match and not re.match(r'^[0-9a-fA-F]{4}$',raw[i+2:i+6]):
                reason='INVALID_JSON_UNICODE_ESCAPE_LITERAL_BACKSLASH'
            if nxt=='u' and re.match(r'^[0-9a-fA-F]{4}$',raw[i+2:i+6]):
                raise ValueError('Ambiguous existing JSON Unicode escape')
            result.append('\\\\')
            result.append(raw[i+1])
            edits.append({'offset':i,'kind':reason,
                          'literal_before':raw[i:i+min(30,len(raw)-i)]})
            i+=2
            continue
        result.append(ch)
        i+=1
    if inside:
        raise ValueError('Unterminated JSON string, not a simple TeX escape issue')
    fixed=''.join(result)
    parsed=json.loads(fixed)
    if not isinstance(parsed,dict):raise ValueError('Recovered JSON must be object')
    if not edits:raise ValueError('No narrowly justified literal-backslash edits found')
    return fixed, {'rule':'ONLY_DOUBLE_LITERAL_BACKSLASH_FOR_INVALID_OR_KNOWN_TEX_COMMANDS',
        'source_parser_error':error_before,'edits':edits,'edit_count':len(edits),
        'corrected_raw_sha256':hashlib.sha256(fixed.encode()).hexdigest(),
        'scientific_prose_modified':False,'scientific_formulas_modified':False,
        'original_backslash_literal_preserved':True}



# The reviewer sometimes combines two official categories in one unrecognized
# literal. This exact compound entails NON_IDENTIFIABLE (its decisive blocking
# defect), not validity or novelty. Preserve the secondary known-physics caveat
# verbatim in an audit instead of inventing a new physical conclusion.
_M13_EXACT_COMPOSITE_PHYSICS_STATUS = 'KNOWN_PHYSICS_PLAUSIBLE_BUT_NON_IDENTIFIABLE'


def _normalize_exact_composite_physics_status(escaped_json: str, candidates: list[dict]) -> tuple[str, dict | None]:
    """Normalize only one observed composite enum, with no other semantic changes.

    Both literal content and model response survive in the original saved receipt;
    only the reviewer status enum becomes canonical for the strict M13 parser.
    """
    original = json.loads(escaped_json)
    if not isinstance(original, dict) or not isinstance(original.get('reviews'), list):
        raise ValueError('Physics JSON structure invalid; status recovery prohibited')
    import copy
    normalized = copy.deepcopy(original)
    mappings = []
    for i, item in enumerate(normalized['reviews']):
        if not isinstance(item, dict):
            raise ValueError('Physics review row not an object')
        source_status = item.get('physical_status')
        if source_status == _M13_EXACT_COMPOSITE_PHYSICS_STATUS:
            item['physical_status'] = 'NON_IDENTIFIABLE'
            mappings.append({
                'row_index': i,
                'candidate_id': item.get('candidate_id'),
                'original_physical_status': source_status,
                'canonical_physical_status': 'NON_IDENTIFIABLE',
                'secondary_caveat_retained_in_source': 'KNOWN_PHYSICS_PLAUSIBLE',
                'reason': 'Explicit NON_IDENTIFIABLE in source composite is the decisive physical-identifiability limitation; no knownness or novelty certified.'
            })
    if not mappings:
        return escaped_json, None
    if original.get('novelty_certified') is not False:
        raise ValueError('Novelty claim in Physics source; enum repair prohibited')
    expected = {c['candidate_id'] for c in candidates}
    if len(normalized['reviews']) != len(expected) or {x.get('candidate_id') for x in normalized['reviews']} != expected:
        raise ValueError('Physics candidate IDs do not match generation; enum repair prohibited')
    # Require that ALL changes consist of the explicitly authorized enum rewrite.
    checked_copy = copy.deepcopy(normalized)
    for change in mappings:
        checked_copy['reviews'][change['row_index']]['physical_status'] = change['original_physical_status']
    if checked_copy != original:
        raise RuntimeError('Non-enum scientific content changed; repair prohibited')
    serialized = json.dumps(normalized, ensure_ascii=False, indent=2)
    parse_physics(serialized, candidates)  # Still enforces all scientific fields.
    return serialized, {
        'rule': 'ONLY_EXACT_KNOWN_PHYSICS_PLAUSIBLE_BUT_NON_IDENTIFIABLE_TO_NON_IDENTIFIABLE',
        'count': len(mappings),
        'mappings': mappings,
        'pre_normalization_json_sha256': hashlib.sha256(escaped_json.encode('utf-8')).hexdigest(),
        'post_normalization_json_sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest(),
        'all_non_status_json_values_unchanged': True,
        'novelty_certified': False,
        'new_model_calls': 0
    }


def _validated_saved_physics_repair(path: Path, original_raw: str, candidates: list[dict], plan: dict, rep: int, g: int, arm: str) -> tuple[str,dict]:
    audit=read(path/PHYSICS_SYNTAX_RECOVERY_BASENAME)
    receipt=path/'physics_RAW_RECEIPT.json'
    if audit.get('plan_id')!=plan['plan_id'] or (audit.get('replicate'),audit.get('generation'),audit.get('arm'))!=(rep,g,arm):
        raise RuntimeError('M13_PHYSICS_RECOVERY_COHORT_OR_PLAN_DRIFT')
    if audit.get('source_receipt_sha256')!=file_sha(receipt):
        raise RuntimeError('M13_PHYSICS_RECEIPT_SHA_DRIFT')
    if audit.get('original_raw_sha256')!=hashlib.sha256(original_raw.encode()).hexdigest():
        raise RuntimeError('M13_PHYSICS_RAW_SHA_DRIFT')
    fixed,mechanical=_repair_physics_json_latex_only(original_raw)
    # Backwards compatibility: already-audited syntax-only repairs from R3
    # have no status_normalization. Their bytes/hash must remain unchanged.
    if 'status_normalization' in audit:
        normalized, status_detail=_normalize_exact_composite_physics_status(fixed,candidates)
        if status_detail is None or status_detail!=audit['status_normalization']:
            raise RuntimeError('M13_PHYSICS_ENUM_MAPPING_AUDIT_DRIFT')
        fixed=normalized
    if audit.get('repair')!=mechanical or audit.get('corrected_raw')!=fixed:
        raise RuntimeError('M13_PHYSICS_AUDIT_DRIFT')
    parsed=parse_physics(fixed,candidates)
    if audit.get('parsed_physics_sha256')!=sh(parsed) or audit.get('novelty_certified') is not False:
        raise RuntimeError('M13_PHYSICS_PARSED_DRIFT')
    return fixed,audit


def cmd_recover_physics_json(args):
    p=checked(read(args.plan));root=args.plan.parent
    rep,g,arm=args.replicate,args.generation,args.arm
    if arm not in ARMS or not 1<=rep<=p['design']['replicates'] or not 1<=g<=p['design']['generations']:
        raise ValueError('Invalid M13 cohort')
    path=root/f'r{rep:02}'/f'g{g:02}'/arm
    receipt=path/'physics_RAW_RECEIPT.json'
    if not receipt.is_file():raise FileNotFoundError('Saved physics receipt missing: '+str(receipt))
    if (path/'M13_PHYSICS_CRITIQUE.json').exists():
        raise RuntimeError('Physics critique already parsed; repair prohibited')
    original=read(receipt)
    if original.get('finish_reason')!='stop':
        raise RuntimeError('Physics model finish_reason not stop; syntax recovery prohibited')
    gen=read(path/'M13_GENERATION_PARSED.json')
    repaired,details=_repair_physics_json_latex_only(original['raw'])
    canonical,status_detail=_normalize_exact_composite_physics_status(repaired,gen['candidates'])
    parsed=parse_physics(canonical,gen['candidates'])
    audit={'status':('M13_ORIGINAL_PHYSICS_JSON_AND_EXACT_COMPOSITE_ENUM_REPAIRED'
                     if status_detail is not None else 'M13_ORIGINAL_PHYSICS_REVIEW_JSON_ESCAPES_REPAIRED'),
        'plan_id':p['plan_id'],'replicate':rep,'generation':g,'arm':arm,
        'source_receipt_sha256':file_sha(receipt),
        'original_raw_sha256':hashlib.sha256(original['raw'].encode()).hexdigest(),
        'corrected_raw':canonical,'repair':details,'parsed_physics_sha256':sh(parsed),
        'raw_receipt_immutable':True,'new_paid_calls':0,'novelty_certified':False}
    if status_detail is not None:
        audit['status_normalization']=status_detail
    output=path/PHYSICS_SYNTAX_RECOVERY_BASENAME
    parse_or_save(output,audit)
    print(json.dumps({'status':audit['status'],'replicate':rep,'generation':g,
        'arm':arm,'edit_count':details['edit_count'],
        'exact_composite_status_mapping_count':0 if status_detail is None else status_detail['count'],
        'source_receipt_sha256':audit['source_receipt_sha256'],
        'new_paid_calls':0,'novelty_certified':False,
        'syntax_recovery_audit':str(output)},indent=2,ensure_ascii=False))


def run(plan:dict,root:Path,max_new_calls:int|None,call,lookup,model_gen:str,model_phy:str,model_lit:str):
    checked(plan)
    if read(root/'M13_PLAN.json')!=plan:raise RuntimeError('M13 frozen plan changed')
    if max_new_calls is not None and max_new_calls<0:raise ValueError('Negative new-call cap')
    calls=0
    rounds=[]
    for rep in range(1,plan['design']['replicates']+1):
        for g in range(1,plan['design']['generations']+1):
            # Alternating arm order to reduce treatment/order bias.
            order=ARMS if g%2 else tuple(reversed(ARMS))
            current={}
            for arm in order:
                path=root/f'r{rep:02}'/f'g{g:02}'/arm
                last_feedback=[]
                if arm=='FEEDBACK':
                    for prev in range(1,g):
                        fpath=root/f'r{rep:02}'/f'g{prev:02}'/arm/'M13_SCIENTIFIC_FEEDBACK.json'
                        if not fpath.is_file():raise RuntimeError('Missing predecessor feedback')
                        last_feedback.append(read(fpath))
                prompt=generate_prompt(plan,arm,g,last_feedback,rep)
                generation_raw=path/'generation_RAW_RECEIPT.json'
                if not generation_raw.exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M13_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'generation'}
                raw,created=saved_model(root,str(path.relative_to(root))+'/generation',prompt,model_gen,call)
                calls+=int(created)
                # Reuse a narrowly audited, deterministic syntax-only correction,
                # NEVER mutate an original paid response or recast it as a new hypothesis.
                syntax_audit=path/SYNTAX_RECOVERY_BASENAME
                try:
                    if syntax_audit.is_file():
                        repaired,syntax_receipt=_validated_saved_syntax_repair(path,plan,rep,g,arm,raw)
                        gen=parse_generation(repaired,g,plan,rep,arm)
                        gen['provenance']={
                          'generation_source':'ORIGINAL_PAID_MODEL_RAW_SYNTAX_ONLY_RESTORED',
                          'original_raw_receipt_sha256':syntax_receipt['source_receipt_sha256'],
                          'recovered_json_sha256':syntax_receipt['repair']['corrected_raw_sha256'],
                          'original_generation_scientific_content_preserved':True,
                          'model_paid_replacement_attempts':0,
                          'scientific_claim_is_not_certified':True}
                        parse_or_save(path/'M13_GENERATION_SYNTAX_RECOVERY_LINEAGE.json',gen['provenance'])
                    else:
                        gen=parse_generation(raw,g,plan,rep,arm)
                except ValueError as exc:
                    if syntax_audit.is_file():
                        raise RuntimeError('Audited syntax recovery failed the strict scientific validator; do not regenerate automatically') from exc
                    defect=diagnose_generation_receipt(raw,g,plan,rep,arm)
                    defect['strict_validator_error']=str(exc)
                    defect['source_raw_receipt_sha256']=file_sha(path/'generation_RAW_RECEIPT.json')
                    parse_or_save(path/'M13_GENERATION_SCIENCE_INVALID.json',defect)
                    resample_prompt=replacement_generation_prompt(plan,arm,g,last_feedback,rep,raw,defect)
                    resample_receipt=path/'generation_REPLACEMENT_RAW_RECEIPT.json'
                    if not resample_receipt.exists():
                        return {'status':'M13_INVALID_GENERATION_NEW_HYPOTHESIS_REQUIRED',
                         'replicate':rep,'generation':g,'arm':arm,'stage':'generation',
                         'new_model_calls':calls,'existing_paid_receipt_preserved':True,
                         'repair_must_be_explicitly_authorized':True,
                         'original_candidate_science_not_promoted':True,
                         'science_defect_audit':str(path/'M13_GENERATION_SCIENCE_INVALID.json')}
                    replacement_raw,created=saved_model(root,
                        str(path.relative_to(root))+'/generation_REPLACEMENT',
                        resample_prompt,model_gen,call)
                    calls+=int(created)
                    try:
                        gen=parse_generation(replacement_raw,g,plan,rep,arm)
                        verify_new_claims_not_literal_rescue(raw,gen)
                    except ValueError as repair_exc:
                        parse_or_save(path/'M13_GENERATION_REPLACEMENT_INVALID.json',{
                            'status':'M13_REPLACEMENT_SCIENCE_INVALID_DO_NOT_RECALL',
                            'reason':str(repair_exc),
                            'raw_receipt_sha256':file_sha(resample_receipt),
                            'novelty_certified':False})
                        return {'status':'M13_REPLACEMENT_INVALID_NEEDS_INDEPENDENT_REVIEW',
                            'replicate':rep,'generation':g,'arm':arm,
                            'new_model_calls':calls,'original_and_replacement_raw_retained':True}
                    gen['provenance']={
                        'generation_source':'NEW_REPLACEMENT_GENERATION_NOT_ORIGINAL_SALVAGE',
                        'invalid_original_receipt_sha256':file_sha(path/'generation_RAW_RECEIPT.json'),
                        'replacement_raw_receipt_sha256':file_sha(resample_receipt),
                        'original_generation_valid':False,
                        'model_paid_replacement_attempts':1,
                        'scientific_claim_is_not_certified':True}
                    parse_or_save(path/'M13_GENERATION_REPLACEMENT_LINEAGE.json',gen['provenance'])
                parse_or_save(path/'M13_GENERATION_PARSED.json',gen)
                if not gen['candidates']:
                    # Both-arm budget fairness cannot be claimed when one arm abstains.
                    parse_or_save(path/'M13_ABSTENTION.json',{'reason':gen['abstained_reason'],'replicate':rep,'generation':g,'arm':arm})
                    return {'status':'M13_ABSTAINED_ARM_FAIRNESS_INCOMPLETE','replicate':rep,'generation':g,'arm':arm,'new_model_calls':calls}
                ev=bibliographic_search(root,arm,g,gen,lookup,rep)
                if not ev['work_ids']:
                    raise RuntimeError('No scholarly works retrieved; never treat this as novelty')
                phy_prompt=physics_prompt(gen['candidates'])
                physics_raw=path/'physics_RAW_RECEIPT.json'
                if not physics_raw.exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M13_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'physics'}
                raw,created=saved_model(root,str(path.relative_to(root))+'/physics',phy_prompt,model_phy,call)
                calls+=int(created)
                physics_recovery=path/PHYSICS_SYNTAX_RECOVERY_BASENAME
                if physics_recovery.is_file():
                    raw,_=_validated_saved_physics_repair(path,raw,gen['candidates'],plan,rep,g,arm)
                try:
                    phy=parse_physics(raw,gen['candidates'])
                except json.JSONDecodeError as exc:
                    status={'status':'M13_SAVED_PHYSICS_JSON_INVALID_RECOVERY_NEEDED',
                        'replicate':rep,'generation':g,'arm':arm,
                        'source_receipt_sha256':file_sha(physics_raw),
                        'parser_error':str(exc),'new_paid_calls':calls,
                        'existing_physics_receipt_preserved':True,
                        'novelty_certified':False}
                    parse_or_save(path/'M13_PHYSICS_JSON_INVALID.json',status)
                    return status
                parse_or_save(path/'M13_PHYSICS_CRITIQUE.json',phy)
                lit_prompt=literature_prompt(gen['candidates'],ev)
                lit_raw=path/'literature_RAW_RECEIPT.json'
                if not lit_raw.exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M13_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'literature'}
                raw,created=saved_model(root,str(path.relative_to(root))+'/literature',lit_prompt,model_lit,call)
                calls+=int(created)
                lit,quarantine=parse_literature_with_quarantine(raw,gen['candidates'],ev)
                if quarantine is not None:
                    quarantine={**quarantine,
                        'source_raw_receipt_sha256':file_sha(path/'literature_RAW_RECEIPT.json'),
                        'retrieved_evidence_sha256':sh(ev),
                        'sanitized_review_sha256':sh(lit)}
                    parse_or_save(path/'M13_LITERATURE_CITATION_QUARANTINE.json',quarantine)
                    print('M13_LITERATURE_CITATIONS_QUARANTINED='+str(path/'M13_LITERATURE_CITATION_QUARANTINE.json'))
                parse_or_save(path/'M13_LITERATURE_CRITIQUE.json',lit)
                f=science_feedback(gen,phy,lit)
                parse_or_save(path/'M13_SCIENTIFIC_FEEDBACK.json',f)
                current[arm]={'replicate':rep,'arm':arm,'generation':gen,'physics':phy,'literature':lit,
                              'retrieved_work_count':len(ev['work_ids'])}
            # Enforce BOTH arms completed BEFORE any next generation is allowed.
            if set(current)!=set(ARMS):raise RuntimeError('Matched-arm generation unfinished')
            rounds.extend(current[a] for a in ARMS)
    # Comparison requires actual independent blind review, not model's unverified grade.
    b=blinded_rows(rounds,plan)
    parse_or_save(root/'M13_ARM_KEY_PRIVATE.json',{
       row['blind_id']:{'arm':run['arm'],'replicate':run['replicate'],'generation':run['generation']['generation'],
       'candidate_id':c['candidate_id']}
       for run in rounds for c in run['generation']['candidates']
       for row in b if row['blind_id']=='BLIND_'+sh({'plan':plan['plan_id'],'idea':c['candidate_id']})[:16]})
    parse_or_save(root/'M13_BLINDED_CANDIDATES.json',{'candidates':b,
      'prior_art_search_completed':True,'arm_assignment_exposed':False})
    worksheet=root/'M13_BLIND_REVIEW_TO_FILL.csv'
    if worksheet.exists():
        if worksheet.read_bytes()!=blind_csv(b).encode('utf-8'):raise RuntimeError('Blinded worksheet edited, refusing overwrite')
    else:create(worksheet,blind_csv(b))
    resample_counts={arm:len(list(root.glob('r*/g*/'+arm+'/generation_REPLACEMENT_RAW_RECEIPT.json'))) for arm in ARMS}
    resample_pairs=[{'replicate':rep,'generation':generation,
         'feedback_replacement':(root/f'r{rep:02}'/f'g{generation:02}'/'FEEDBACK'/'generation_REPLACEMENT_RAW_RECEIPT.json').exists(),
         'control_replacement':(root/f'r{rep:02}'/f'g{generation:02}'/'CONTROL'/'generation_REPLACEMENT_RAW_RECEIPT.json').exists()}
         for rep in range(1,plan['design']['replicates']+1)
         for generation in range(1,plan['design']['generations']+1)]
    realized_call_parity=all(row['feedback_replacement']==row['control_replacement'] for row in resample_pairs)
    summary={'status':'M13_MATCHED_ARMS_COMPLETE_SCIENCE_UNREVIEWED' if realized_call_parity
             else 'M13_ARMS_COMPLETE_REALIZED_CALLS_UNBALANCED_DESCRIPTIVE_ONLY',
     'generations':plan['design']['generations'],'replicates':plan['design']['replicates'], 'candidates':len(b),
     'matched_generation_and_review_calls_planned':6*plan['design']['generations']*plan['design']['replicates'],
     'novelty_certified':False,
     'feedback_outperforms_control':None,
     'replacement_generation_calls_by_arm':resample_counts,
     'replacement_attempts_by_paired_round':resample_pairs,
     'equal_realized_model_calls_between_arms':realized_call_parity,
     'strict_matched_compute_budget_interpretation':realized_call_parity,
     'human_blind_review_required':True,
     'notable_limits':['Multiple independent search replicates but no experimental SERS validation',
       'Same or different reviewer models are not independent human experts',
       'Anchored abstract excerpts are not full-paper verification']}
    parse_or_save(root/'M13_BENCHMARK_SUMMARY.json',summary)
    return {**summary,'new_model_calls_this_invocation':calls}


def _openrouter(args):
    from openai import OpenAI
    client=OpenAI(api_key=os.environ[args.api_key_env],base_url=args.base_url,timeout=300,max_retries=0)
    def _call(model,prompt):
        response=client.chat.completions.create(model=model,
          messages=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user']}],
          temperature=args.temperature,max_tokens=args.max_output_tokens)
        return {'text':response.choices[0].message.content or '',
          'finish_reason':response.choices[0].finish_reason,
          'usage':response.usage.model_dump(mode='json') if response.usage else None}
    return _call


def _source_lookup(provider,q):
    # Uses M12's actual retriever from user's installed local ForAllKG.
    from pipeline_core.discovery.sers_autonomous_novelty_m12 import lookup
    return lookup(provider,q,key=os.getenv('OPENALEX_API_KEY',''))


def cmd_prepare(args):
    plan=prepare(args.m12_packet,args.m12_review,args.generations,args.queries_per_candidate,args.replicates)
    create(args.out_dir/'M13_PLAN.json',plan)
    print(json.dumps({'status':'M13_PREPARED_NO_MODEL_CALLS','plan_id':plan['plan_id'],
      'out_dir':str(args.out_dir),'quality_first':True,
      'expected_matched_model_calls':plan['design']['comparative_call_budget'],
      'generations':plan['design']['generations'],'replicates':plan['design']['replicates'],'novelty_certified':False},indent=2))


def cmd_preflight(args):
    plan=checked(read(args.plan))
    found=[]
    for provider in ('openalex','europepmc'):
        try:
            result=_source_lookup(provider,'surface enhanced Raman spectroscopy kinetics')
            ok=result.get('success') is True and bool(result.get('works'))
            found.append({'provider':provider,'success':ok,
              'sample_titles':[w.get('title') for w in result.get('works',[])[:2]]})
        except Exception as e:
            found.append({'provider':provider,'success':False,'error':f'{type(e).__name__}: {e}'})
    status={'status':'M13_LITERATURE_PREFLIGHT_PASS' if all(x['success'] for x in found) else 'M13_LITERATURE_PREFLIGHT_FAILED',
      'plan_id':plan['plan_id'],'providers':found,'new_model_calls':0,'novelty_certified':False}
    print(json.dumps(status,indent=2,ensure_ascii=False))
    if status['status']!='M13_LITERATURE_PREFLIGHT_PASS':
        raise RuntimeError('External scholarly retrieval not reachable; do not spend on model calls')
    path=args.plan.parent/'M13_LITERATURE_PREFLIGHT_PASS.json'
    if path.exists():
        if read(path).get('plan_id')!=plan['plan_id']:
            raise RuntimeError('Existing literature preflight belongs to another plan')
    else:create(path,status)


def cmd_run(args):
    if not (args.allow_paid and args.i_authorize_model_calls):
        raise ValueError('Explicit paid model authorization flags needed')
    if not os.getenv(args.api_key_env):raise ValueError('Missing API key variable')
    p=checked(read(args.plan));root=args.plan.parent
    pre=root/'M13_LITERATURE_PREFLIGHT_PASS.json'
    if not pre.exists() or read(pre).get('plan_id')!=p['plan_id'] or read(pre).get('status')!='M13_LITERATURE_PREFLIGHT_PASS':
        raise RuntimeError('Run preflight-literature (free) before paid experiment')
    lock={'generator_model':args.generator_model,'physics_model':args.physics_model,
        'literature_model':args.literature_model,'base_url':args.base_url,
        'temperature':args.temperature,'max_output_tokens':args.max_output_tokens}
    f=root/'M13_MODEL_LOCK.json'
    if f.exists():
        if read(f)!=lock:raise RuntimeError('Model/protocol changed during matched experiment')
    else:create(f,lock)
    result=run(p,root,args.max_new_calls,_openrouter(args),_source_lookup,
        args.generator_model,args.physics_model,args.literature_model)
    print(json.dumps(result,ensure_ascii=False,indent=2))



def cmd_recover_saved(args):
    """Process all currently saved receipts, but NEVER spend or fetch new works."""
    p=checked(read(args.plan));root=args.plan.parent
    def no_model(model,prompt):
        raise AssertionError('M13 RECOVERY MUST NEVER CALL A MODEL')
    def no_search(provider,query):
        raise RuntimeError('M13 offline recovery missing required cached bibliographic work; '
                           'do not infer literature claims. Provider='+provider)
    lock=read(root/'M13_MODEL_LOCK.json')
    result=run(p,root,0,no_model,no_search,
       lock['generator_model'],lock['physics_model'],lock['literature_model'])
    print(json.dumps({**result,'recovery_no_model_calls':True,
       'recovery_offline_only':True},ensure_ascii=False,indent=2))


def cmd_inspect_generation(args):
    p=checked(read(args.plan));root=args.plan.parent
    rows=[]
    for receipt in sorted(root.glob('r*/g*/*/generation_RAW_RECEIPT.json')):
        arm=receipt.parent.name;g=int(receipt.parent.parent.name[1:]);rep=int(receipt.parent.parent.parent.name[1:])
        obj=read(receipt)
        try:
            generation=parse_generation(obj['raw'],g,p,rep,arm)
            valid=True;problem=[]
        except (ValueError,KeyError,TypeError) as e:
            valid=False
            diag=diagnose_generation_receipt(obj.get('raw',''),g,p,rep,arm)
            problem=diag.get('issues',[])
            if not problem:problem=[type(e).__name__+':'+str(e)]
        rows.append({'replicate':rep,'generation':g,'arm':arm,
            'valid_scientific_generation':valid,
            'unresolved_issues':problem,
            'replacement_saved':(receipt.parent/'generation_REPLACEMENT_RAW_RECEIPT.json').is_file(),
            'raw_receipt_sha256':file_sha(receipt)})
    print(json.dumps({'status':'M13_SAVED_GENERATION_DIAGNOSTICS_NO_PROVIDER_CALLS',
       'saved_generation_receipts':len(rows),
       'invalid_generations':sum(not x['valid_scientific_generation'] for x in rows),
       'rows':rows,'new_model_calls':0,'novelty_certified':False},indent=2,ensure_ascii=False))


def cmd_replacement_paid(args):
    if not (args.allow_paid and args.i_authorize_model_calls):
        raise RuntimeError('Explicit authorization required for one NEW paid scientific idea generation')
    p=checked(read(args.plan));root=args.plan.parent
    lock=read(root/'M13_MODEL_LOCK.json')
    if lock.get('generator_model')!=args.generator_model or lock.get('max_output_tokens')!=args.max_output_tokens or lock.get('temperature')!=args.temperature or lock.get('base_url')!=args.base_url:
        raise RuntimeError('Generator parameters differ from original frozen matched protocol')
    rep,g,arm=args.replicate,args.generation,args.arm
    if arm not in ARMS or not 1<=rep<=p['design']['replicates'] or not 1<=g<=p['design']['generations']:
        raise ValueError('Invalid frozen cohort identity')
    path=root/f'r{rep:02}'/f'g{g:02}'/arm
    receipt=path/'generation_RAW_RECEIPT.json'
    if not receipt.is_file():raise FileNotFoundError('No ORIGINAL paid raw receipt to regenerate: '+str(receipt))
    original=read(receipt)
    prior=[]
    if arm=='FEEDBACK':
        for i in range(1,g):
            prior.append(read(root/f'r{rep:02}'/f'g{i:02}'/arm/'M13_SCIENTIFIC_FEEDBACK.json'))
    # Respect original lineage; do not create a new version when original is already valid.
    try:
        parse_generation(original['raw'],g,p,rep,arm)
    except ValueError:pass
    else:raise RuntimeError('Original generation was valid: regeneration prohibited')
    diag=diagnose_generation_receipt(original['raw'],g,p,rep,arm)
    prompt=replacement_generation_prompt(p,arm,g,prior,rep,original['raw'],diag)
    path_out=path/'generation_REPLACEMENT_RAW_RECEIPT.json'
    if not path_out.is_file():
        if not os.getenv(args.api_key_env):raise RuntimeError('Missing API key')
    # The first call may be paid. The second with saved receipt is free.
    def no_call_if_cached(model,prompt):
        raise AssertionError('Cached scientific replacement must never call model')
    sender=no_call_if_cached if path_out.exists() else _openrouter(args)
    raw,created=saved_model(root,str(path.relative_to(root))+'/generation_REPLACEMENT',
        prompt,args.generator_model,sender)
    try:
        new_candidate=parse_generation(raw,g,p,rep,arm)
        verify_new_claims_not_literal_rescue(original['raw'],new_candidate)
    except ValueError as exc:
        parse_or_save(path/'M13_GENERATION_REPLACEMENT_INVALID.json',{
            'status':'M13_REPLACEMENT_SCIENCE_INVALID_DO_NOT_RECALL',
            'reason':str(exc),'raw_receipt_sha256':file_sha(path_out),
            'novelty_certified':False})
        print(json.dumps({'status':'M13_REPLACEMENT_INVALID_MANUAL_SCIENCE_REVIEW_NEEDED',
           'new_model_calls':int(created),'receipt_saved':True,'novelty_certified':False},indent=2))
        return
    print(json.dumps({'status':'M13_NEW_REPLACEMENT_HYPOTHESES_VALID_SAVED_NOT_OLD_CLAIM_RESCUED',
        'source_invalid_raw_sha256':file_sha(receipt),
        'replacement_raw_sha256':file_sha(path_out),
        'new_model_calls':int(created),
        'candidate_count':len(new_candidate['candidates']),
        'requires_original_workflow_resume':True,'novelty_certified':False},indent=2))


def cmd_status(args):
    p=checked(read(args.plan));root=args.plan.parent
    rows=[]
    for rep in range(1,p['design']['replicates']+1):
      for g in range(1,p['design']['generations']+1):
        for arm in ARMS:
            f=root/f'r{rep:02}'/f'g{g:02}'/arm
            rows.append({'replicate':rep,'generation':g,'arm':arm,
                'generated':(f/'M13_GENERATION_PARSED.json').exists(),
                'original_generation_raw_saved':(f/'generation_RAW_RECEIPT.json').exists(),
                'science_invalid_generation_audit':(f/'M13_GENERATION_SCIENCE_INVALID.json').exists(),
                'replacement_raw_saved':(f/'generation_REPLACEMENT_RAW_RECEIPT.json').exists(),
                'physics_reviewed':(f/'M13_PHYSICS_CRITIQUE.json').exists(),
                'literature_reviewed':(f/'M13_LITERATURE_CRITIQUE.json').exists(),
                'feedback_written':(f/'M13_SCIENTIFIC_FEEDBACK.json').exists()})
    print(json.dumps({'status':'M13_STATUS_NO_MODEL_CALLS','round_statuses':rows},indent=2))


def cmd_export(args):
    root=args.plan.parent
    if not (root/'M13_BENCHMARK_SUMMARY.json').exists():
        raise RuntimeError('Matched comparison incomplete')
    if args.blind_zip.exists() or args.audit_zip.exists():raise FileExistsError('Export target exists')
    args.blind_zip.parent.mkdir(parents=True,exist_ok=True)
    args.audit_zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.blind_zip,'x',zipfile.ZIP_DEFLATED) as z:
        for leaf in ('M13_BLINDED_CANDIDATES.json','M13_BLIND_REVIEW_TO_FILL.csv'):
            z.write(root/leaf,leaf)
    with zipfile.ZipFile(args.audit_zip,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f not in (args.blind_zip,args.audit_zip) and '_INTENT' not in f.name:
                z.write(f,f.relative_to(root).as_posix())
    print('M13_BLIND_REVIEW_PACKET='+str(args.blind_zip))
    print('M13_FULL_AUDIT_PACKET='+str(args.audit_zip))
    print('WARNING: submit blind packet to independent reviewer BEFORE opening private arm key.')


def cmd_analyze(args):
    root=args.plan.parent
    p=checked(read(args.plan))
    if not (root/'M13_BENCHMARK_SUMMARY.json').exists():raise RuntimeError('Not completed')
    scores=score_completed_blind(args.filled_review.read_text(encoding='utf-8-sig'),
           read(root/'M13_ARM_KEY_PRIVATE.json'),p)
    benchmark=read(root/'M13_BENCHMARK_SUMMARY.json')
    if not benchmark.get('strict_matched_compute_budget_interpretation',True):
        scores['status']='M13_EXPLORATORY_BLIND_SCORES_UNBALANCED_GENERATION_ATTEMPTS'
        scores['strict_feedback_effect_identification']=False
        scores['warning']='Unequal generation attempts within at least one paired round; do not claim equal-compute causal effect.'
        scores['feedback_improvement_established']=False
    create(args.output,scores)
    print(json.dumps({'status':scores['status'],
           'paired_rounds':len(scores['paired_rounds']),
           'mean_paired_quality_delta':scores['aggregate_mean_paired_quality_delta'],
           'mean_eligible_delta':scores['aggregate_mean_eligible_delta'],
           'science_novelty_certified':False},indent=2))


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    sub=pa.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--m12-packet',type=Path,required=True)
    p.add_argument('--m12-review',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--generations',type=int,default=4)
    p.add_argument('--replicates',type=int,default=3)
    p.add_argument('--queries-per-candidate',type=int,default=4)
    p.set_defaults(func=cmd_prepare)
    p=sub.add_parser('preflight-literature');p.add_argument('--plan',type=Path,required=True);p.set_defaults(func=cmd_preflight)
    p=sub.add_parser('run-paid');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--generator-model',default='openai/gpt-5.6-luna')
    p.add_argument('--physics-model',default='openai/gpt-5.6-luna')
    p.add_argument('--literature-model',default='openai/gpt-5.6-luna')
    p.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--max-output-tokens',type=int,default=9000)
    p.add_argument('--max-new-calls',type=int,default=None,
        help='Optional per-invocation safety guard; omit to run all frozen protocol calls')
    p.add_argument('--allow-paid',action='store_true')
    p.add_argument('--i-authorize-model-calls',action='store_true')
    p.set_defaults(func=cmd_run)
    p=sub.add_parser('recover-saved');p.add_argument('--plan',type=Path,required=True);p.set_defaults(func=cmd_recover_saved)
    p=sub.add_parser('inspect-generations');p.add_argument('--plan',type=Path,required=True);p.set_defaults(func=cmd_inspect_generation)
    p=sub.add_parser('recover-json-syntax');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--replicate',type=int,required=True)
    p.add_argument('--generation',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.set_defaults(func=cmd_recover_syntax)
    p=sub.add_parser('recover-physics-json');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--replicate',type=int,required=True)
    p.add_argument('--generation',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.set_defaults(func=cmd_recover_physics_json)
    p=sub.add_parser('replace-invalid-generation-paid');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--replicate',type=int,required=True)
    p.add_argument('--generation',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--generator-model',default='openai/gpt-5.6-luna')
    p.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--max-output-tokens',type=int,default=9000)
    p.add_argument('--allow-paid',action='store_true')
    p.add_argument('--i-authorize-model-calls',action='store_true')
    p.set_defaults(func=cmd_replacement_paid)
    p=sub.add_parser('status');p.add_argument('--plan',type=Path,required=True);p.set_defaults(func=cmd_status)
    p=sub.add_parser('export');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--blind-zip',type=Path,required=True);p.add_argument('--audit-zip',type=Path,required=True)
    p.set_defaults(func=cmd_export)
    p=sub.add_parser('analyze-blinded');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--filled-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.set_defaults(func=cmd_analyze)
    a=pa.parse_args();a.func(a)
if __name__=='__main__':main()
