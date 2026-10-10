"""M7 paid-response salvage: never erase receipts, never retry without authorization.

Offline salvage recovers only completed top-level JSON properties. An explicit,
one-request provider operation may complete missing properties if required fields
were cut off. This is not a scientific quality or novelty evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING
import json
import os
from pathlib import Path
from typing import Any
import sys

from pipeline_core.discovery.scientific_development_m7 import ScientificProgramDraft, digest
from scripts.discovery.run_scientific_development_m7 import _read_plan, read, write_new

FIELDS = ScientificProgramDraft.__dataclass_fields__
REQUIRED = tuple(k for k, f in FIELDS.items() if f.default is MISSING and f.default_factory is MISSING)
AUTHORITY = ('novelty_certified', 'scientific_truth_authority', 'evidence_claim_authority')


def _prefix_object(raw: str) -> tuple[dict[str, Any], str | None]:
    """Read only complete key/value tokens; do not auto-close incomplete strings.

    Supports an unclosed outer ``{\"program\": { ...`` envelope. Discards the
    last incomplete field rather than fabricating a value for it.
    """
    decoder = json.JSONDecoder()
    t = raw.lstrip()
    if not t.startswith('{'):
        raise ValueError('Raw response does not start with a JSON object')
    pos = 1
    partial: dict[str, Any] = {}
    while True:
        while pos < len(t) and t[pos].isspace(): pos += 1
        if pos >= len(t): return partial, '<end-of-response>'
        if t[pos] == '}': return partial, None
        try:
            key, end = decoder.raw_decode(t, pos)
        except ValueError:
            return partial, '<unfinished-property-name>'
        if not isinstance(key, str):
            raise ValueError('Non-string JSON property name')
        if key in partial:
            raise ValueError('Duplicate JSON property cannot be salvaged safely: '+key)
        pos = end
        while pos < len(t) and t[pos].isspace(): pos += 1
        if pos >= len(t) or t[pos] != ':': return partial, key
        pos += 1
        while pos < len(t) and t[pos].isspace(): pos += 1
        if key == 'program' and t[pos:pos+1] == '{':
            if partial: raise ValueError('Mixed envelope properties not supported')
            return _prefix_object(t[pos:])
        try:
            value, end = decoder.raw_decode(t, pos)
        except ValueError:
            return partial, key
        partial[key] = value
        pos = end
        while pos < len(t) and t[pos].isspace(): pos += 1
        if pos >= len(t): return partial, '<end-of-response>'
        if t[pos] == '}': return partial, None
        if t[pos] != ',': raise ValueError('Malformed JSON separator after '+key)
        pos += 1


def _load_saved(plan: dict, root: Path, label: str) -> tuple[dict, dict, dict, str | None]:
    task = next((t for t in plan['tasks'] if t['label'] == label), None)
    if task is None: raise ValueError('Unknown frozen task label: '+label)
    path = root/'receipts'/(label+'.json')
    if not path.is_file(): raise FileNotFoundError('No saved original receipt: '+str(path))
    receipt = read(path)
    if receipt.get('task_sha') != digest(task) or receipt.get('label') != label:
        raise ValueError('Original receipt / frozen task identity mismatch')
    raw = receipt.get('raw_response')
    if not isinstance(raw, str): raise ValueError('Receipt has no textual provider response')
    try:
        full = json.loads(raw)
    except json.JSONDecodeError:
        partial, incomplete = _prefix_object(raw)
    else:
        if isinstance(full, dict) and isinstance(full.get('program'), dict): full = full['program']
        if not isinstance(full, dict): raise ValueError('Provider response is not an object')
        partial, incomplete = full, None
    extras = set(partial) - set(FIELDS)
    if extras: raise ValueError('Unknown fields in provider prefix: '+str(sorted(extras)))
    for attr, expected in [('idea_label',label),('parent_idea_id',task['parent_idea_id']),('source_role',task['role'])]:
        if attr in partial and partial[attr] != expected:
            raise ValueError('Provider prefix identity differs from frozen task: '+attr)
    if any(partial.get(k, False) is not False for k in AUTHORITY):
        raise ValueError('Provider prefix sets scientific claim authority')
    return task, receipt, partial, incomplete


def _build_science(label: str, task: dict, complete: dict, *, salvaged: bool) -> ScientificProgramDraft:
    final = dict(complete)
    for field, value in [('idea_label',label),('parent_idea_id',task['parent_idea_id']),('source_role',task['role'])]:
        if field in final and final[field] != value: raise ValueError('Recovered identity mismatch: '+field)
        final[field] = value
    for flag in AUTHORITY:
        if final.get(flag,False) is not False: raise ValueError('Recovered authority escalation')
        final[flag] = False
    if salvaged:
        final['source_of_draft'] = 'MODEL_SPECULATIVE_TRUNCATED_RECEIPT_SALVAGE'
    return ScientificProgramDraft.parse(final)


def inspect(plan_path: Path) -> list[dict]:
    plan = _read_plan(plan_path); root=plan_path.parent
    out=[]
    for task in plan['tasks']:
        label=task['label']
        parsed = root/'parsed'/(label+'.json')
        receipt = root/'receipts'/(label+'.json')
        row={'label':label,'original_receipt_exists':receipt.is_file(),
             'parsed_program_exists':parsed.is_file(),'recommended_action':'NONE'}
        if receipt.is_file():
            try:
                _, record, fields, incomplete = _load_saved(plan,root,label)
                row.update(finish_reason=record.get('finish_reason'),response_chars=len(record['raw_response']),
                           usage=record.get('usage'),complete_field_count=len(fields),
                           missing_required_fields=[k for k in REQUIRED if k not in fields],
                           truncated_field=incomplete)
                if parsed.is_file():
                    _build_science(label,task,read(parsed),salvaged=False)
                    row['recommended_action']='ALREADY_PARSED_NO_NEW_CALL'
                elif all(k in fields for k in REQUIRED):
                    row['recommended_action']='OFFLINE_SALVAGE_POSSIBLE_NO_NEW_CALL'
                else:
                    row['recommended_action']='EXPLICIT_RECOVER_PAID_OPTIONAL_AFTER_REVIEW'
            except Exception as exc:
                row.update(recommended_action='MANUAL_INSPECTION_REQUIRED',diagnostic=f'{type(exc).__name__}: {exc}')
        out.append(row)
    return out


def salvage(plan_path: Path, label: str) -> dict:
    plan=_read_plan(plan_path);root=plan_path.parent
    task, receipt, fields, incomplete=_load_saved(plan,root,label)
    parsed=root/'parsed'/(label+'.json')
    if parsed.exists():
        _build_science(label,task,read(parsed),salvaged=False)
        return {'label':label,'status':'ALREADY_PARSED','new_provider_calls':0}
    missing=[k for k in REQUIRED if k not in fields]
    if missing:
        return {'label':label,'status':'NEEDS_EXPLICIT_PAID_COMPLETION',
                'missing_required_fields':missing,'new_provider_calls':0}
    prog=_build_science(label,task,fields,salvaged=incomplete is not None)
    write_new(parsed,prog.to_dict())
    write_new(root/'recovery_manifests'/(label+'.json'),{
        'status':'OFFLINE_COMPLETED_FIELD_SALVAGE_REVIEW_REQUIRED',
        'source_original_receipt_sha':digest(receipt), 'source_finish_reason':receipt.get('finish_reason'),
        'source_truncated_field':incomplete, 'complete_original_fields':sorted(fields),
        'omitted_incomplete_or_absent_optional_fields':sorted(set(FIELDS)-set(fields)),
        'no_missing_required_scientific_fields':True,
        'automatic_scientific_quality_authority':False,'new_provider_calls':0,
    })
    return {'label':label,'status':'OFFLINE_SALVAGED_PROGRAM_REVIEW_REQUIRED',
            'source_truncated_field':incomplete,'new_provider_calls':0}


def paid_complete(plan_path: Path, label: str, model: str, api_key_env: str,
                  base_url: str | None, max_output_tokens: int,
                  allow_paid: bool, authorize: bool) -> dict:
    if not (allow_paid and authorize):
        raise ValueError('Paid completion requires BOTH explicit authorization flags')
    if not 800 <= max_output_tokens <= 5000: raise ValueError('token limit 800..5000')
    plan=_read_plan(plan_path);root=plan_path.parent
    task, original, partial, incomplete=_load_saved(plan,root,label)
    target=root/'parsed'/(label+'.json')
    if target.exists(): raise FileExistsError('Already parsed; do not spend again: '+str(target))
    missing=[k for k in REQUIRED if k not in partial]
    if not missing:
        raise ValueError('Can salvage offline without an API call; use recover-offline instead')
    if len(partial)<2:
        raise ValueError('Provider prefix too incomplete for reliable low-budget repair; manual review required')
    pending=root/'recovery_attempts'/(label+'.json')
    recovery_receipt=root/'recovery_receipts'/(label+'.json')
    if pending.exists() or recovery_receipt.exists():
        raise FileExistsError('Recovery previously attempted; no automatic re-payment: '+str(pending))
    token=os.getenv(api_key_env)
    if not token: raise RuntimeError('Missing provider key environment variable: '+api_key_env)
    missing_fields=[name for name in FIELDS if name not in partial and name not in AUTHORITY
                    and name!='source_of_draft']
    # Original completed fields are preserved byte-for-byte in their parsed JSON values.
    messages=[
        {'role':'system','content':(
            'You are completing an UNVERIFIED scientific research-program JSON that was truncated due to an output limit. '
            'Do not rewrite supplied completed values. Return a FLAT JSON object containing ONLY requested missing fields. '
            'Use precise but COMPACT science (strings 1-3 sentences; arrays 1-3 concise entries). '
            'Distinguish theoretical null results from novel predictions and explicitly flag unproven assumptions. '
            'Do not claim scientific novelty, empirical proof, or citations. No markdown. '
            'Never add scientific authority flags.')},
        {'role':'user','content':json.dumps({
            'frozen_task':task,
            'already_completed_fields_to_keep_unchanged':partial,
            'required_missing_json_keys':missing_fields,
            'requested_output': 'Only a JSON object with exactly the required_missing_json_keys; preserve identity and existing science',
            'original_unfinished_key':incomplete,
            'max_words_per_new_string':65,
        },ensure_ascii=False)}]
    write_new(pending,{
        'label':label,'task_sha':digest(task),'original_receipt_sha':digest(original),
        'model':model,'request_hash':digest(messages), 'new_paid_call_authorized':True,
        'new_paid_call_attempted':True,'sdk_retries':0})
    # No provider import or network call before the authorization/receipt guards.
    from openai import OpenAI
    client=OpenAI(api_key=token,base_url=base_url or 'https://openrouter.ai/api/v1',max_retries=0,timeout=180.0)
    try:
        reply=client.chat.completions.create(model=model,messages=messages,temperature=0.0,
                  max_tokens=max_output_tokens,response_format={'type':'json_object'})
    except Exception as exc:
        raise RuntimeError('Paid completion attempt may have incurred a charge; inspect recovery_attempts, do NOT rerun automatically') from exc
    choice=reply.choices[0]
    raw=choice.message.content or ''
    receipt={'label':label,'source_original_receipt_sha':digest(original),'task_sha':digest(task),
             'provider_response_id':reply.id,'model':model,'finish_reason':choice.finish_reason,
             'raw_response':raw,'new_api_calls_for_this_receipt':1,
             'usage':reply.usage.model_dump(mode='json') if reply.usage else None}
    write_new(recovery_receipt,receipt)
    try:
        supplied=json.loads(raw)
        if not isinstance(supplied,dict): raise ValueError('Completion response must be a JSON object')
        keys=set(supplied)
        if keys != set(missing_fields):
            raise ValueError('Missing/extra completed keys; expected '+repr(missing_fields)+', got '+repr(sorted(keys)))
        merged={**partial,**supplied}
        prog=_build_science(label,task,merged,salvaged=True)
        write_new(target,prog.to_dict())
        write_new(root/'recovery_manifests'/(label+'.json'),{
            'status':'PAID_MISSING_FIELD_COMPLETION_REVIEW_REQUIRED',
            'original_receipt_sha':digest(original),'recovery_receipt_sha':digest(receipt),
            'original_completed_fields':sorted(partial),'new_fields':missing_fields,
            'source_truncated_field':incomplete,'new_provider_calls':1,
            'model_generated_science_not_experimental_evidence':True,
            'automatic_scientific_quality_authority':False,
        })
    except Exception as exc:
        raise RuntimeError('Paid completion response saved, but structured recovery failed. No repeat without manual review: '+str(exc)) from exc
    return {'label':label,'status':'PAID_FIELD_COMPLETION_SAVED_REVIEW_REQUIRED',
            'new_provider_calls':1,'parsed_program':str(target),'original_receipt_preserved':True}


def main() -> None:
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest='command',required=True)
    for command in ('inspect','recover-offline','recover-paid'):
        c=sub.add_parser(command)
        c.add_argument('--plan',type=Path,required=True)
        if command != 'inspect': c.add_argument('--label',required=True)
        if command == 'recover-paid':
            c.add_argument('--model',default='openai/gpt-5.6-luna')
            c.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
            c.add_argument('--base-url')
            c.add_argument('--max-output-tokens',type=int,default=2200)
            c.add_argument('--allow-paid',action='store_true')
            c.add_argument('--i-authorize-new-llm-calls',action='store_true')
    args=cli.parse_args()
    if args.command=='inspect': value={'status':'M7_RECEIPT_INSPECTION_OFFLINE','rows':inspect(args.plan),'new_provider_calls':0}
    elif args.command=='recover-offline':value=salvage(args.plan,args.label)
    else:value=paid_complete(args.plan,args.label,args.model,args.api_key_env,args.base_url,
                            args.max_output_tokens,args.allow_paid,args.i_authorize_new_llm_calls)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
