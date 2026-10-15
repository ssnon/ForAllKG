#!/usr/bin/env python3
"""CASE01: bounded, opt-in scientific adversarial feedback pilot.

Reuses existing FAKG M7 ScientificProgramDraft/create_model_prompt/challenge_program.
It DOES NOT execute native E2E/SIS, certify novelty, or modify existing files.
Four paid calls maximum: two reviews (standard / scientific-adversarial), followed
by one new research-program proposal conditioned on each review. Identical parent
and source in both arms. Scientific authority remains with external review.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from datetime import datetime, timezone
from typing import Any

SCHEMA = 'case01-m7-level2-feedback-pilot-v1'
ARMS = ('STANDARD_FEEDBACK', 'SCIENTIFIC_ADVERSARIAL_FEEDBACK')
STEPS = ('REVIEW_STANDARD_FEEDBACK', 'REVIEW_SCIENTIFIC_ADVERSARIAL_FEEDBACK',
         'REVISE_STANDARD_FEEDBACK', 'REVISE_SCIENTIFIC_ADVERSARIAL_FEEDBACK')
SOURCE_ARM = 'YE_FULLTEXT_FIGURES'
FIGURES = (3, 4, 5, 6)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def native():
    from pipeline_core.discovery.scientific_development_m7 import (
        ScientificProgramDraft, create_model_prompt, challenge_program,
    )
    from scripts.discovery import run_case01_m7_source_bridge as bridge
    return ScientificProgramDraft, create_model_prompt, challenge_program, bridge


def source_snapshot(previous: Path, kit: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ScientificProgramDraft, _, _, bridge = native()
    oldkit, _ = bridge.check_frozen(previous)
    if oldkit.resolve() != kit.resolve():
        raise ValueError('KIT_PATH_MISMATCH_WITH_FROZEN_ORIGINAL_BRIDGE')
    parent_path = previous / 'parsed' / (SOURCE_ARM + '.json')
    result_path = previous / 'responses' / (SOURCE_ARM + '.json')
    if not parent_path.is_file() or not result_path.is_file():
        raise FileNotFoundError('MISSING_PREVIOUS_NATIVE_M7_ADAPTER_RESULT')
    previous_response = load(result_path)
    if previous_response.get('finish_reason') != 'stop':
        raise ValueError('PARENT_RESPONSE_NOT_FINISHED')
    parent = ScientificProgramDraft.parse(load(parent_path))
    task = bridge.build_task(kit, SOURCE_ARM)
    if (parent.idea_label != task['label'] or
        parent.parent_idea_id != task['parent_idea_id'] or
        parent.source_role != task['role']):
        raise ValueError('PARENT_NATIVE_M7_IDENTITY_MISMATCH')
    original = parent.to_dict()
    if original.get('novelty_certified') or original.get('scientific_truth_authority') or original.get('evidence_claim_authority'):
        raise ValueError('FALSE_SCIENTIFIC_AUTHORITY_REQUIRED')
    evidence = bridge.build_evidence(kit, SOURCE_ARM)
    manifest = {
        'prior_run_manifest_sha256': sha_file(previous / 'RUN_MANIFEST.json'),
        'prior_response_sha256': sha_file(result_path),
        'parent_file_sha256': sha_file(parent_path),
        'parent_program_sha256': sha_obj(original),
        'native_M7_sha256': bridge.native_m7_module_sha(),
        'bridge_script_sha256': sha_file(Path(bridge.__file__)),
        'kit_source_sha256': bridge.verify_frozen(kit)['file_sha256'],
        'figure_sha256': {str(pg): sha_file(kit / f'source/Ye_page_{pg:02d}.png') for pg in FIGURES},
    }
    return original, evidence, {'task': task, 'hashes': manifest}


def reviewer_messages(parent: dict[str, Any], evidence: dict[str, Any], arm: str) -> list[dict[str, str]]:
    if arm not in ARMS:
        raise ValueError('UNKNOWN_ARM')
    common = ("You are an independent scientific research reviewer. You have no authority to "
              "certify novelty or scientific truth. Treat the existing research program as an "
              "UNVERIFIED proposal, not as source evidence. Respect the supplied paper; do not "
              "invent experiments, parameters, or references. You must NOT be told any earlier "
              "human/AI scientific assessment. Return one JSON object with keys "
              "'diagnosis' (2-6 concise strings), 'repair_priorities' (2-6 concise strings), "
              "'abstain_boundary' (one string), and 'overall_judgment' (one string). "
              "Keep output within ~1500-2600 tokens. Do not assert Level II certification.\n")
    if arm == 'STANDARD_FEEDBACK':
        difference = ("Perform a competent conventional manuscript-development review: "
                      "identify communication gaps, vague assumptions, missing control measurements, "
                      "incomplete specification, possible alternatives, and clear ways to make the "
                      "proposal more testable. Consider plausibility. Do not assume an adversarial "
                      "mathematical-countermodel task; offer balanced actionable research feedback.")
    else:
        difference = ("Act as a scientific falsification adversary. Do not reward formalism. "
                      "For each central proposed mechanism/prediction, check mathematical entailment "
                      "from its written state equations and boundary conditions. If invalid, derive "
                      "a counterexample or missing necessary assumption. Construct at least one "
                      "physically constrained, NON-strawman known-physics competitor that reproduces "
                      "the claimed key observable WITHOUT the proposed cause. Distinguish strong "
                      "observational equivalence from a genuinely discriminating intervention. "
                      "Seek a new causal mechanism or a tighter conditional signature if available; "
                      "otherwise recommend principled abstention. Prior-art uncertainty must be explicit. "
                      "Do not insert any particular candidate mechanism or presolved objection.")
    data = {'frozen_prior_unverified_M7_program': parent,
            'source_evidence_identical_in_both_arms': evidence,
            'request': difference}
    return [{'role': 'system', 'content': common + '\n' + difference},
            {'role': 'user', 'content': json.dumps(data, ensure_ascii=False, sort_keys=True)}]


def revision_messages(parent: dict[str, Any], evidence: dict[str, Any],
                      reviewer: dict[str, Any], task: dict[str, Any], arm: str) -> list[dict[str, str]]:
    _, create_model_prompt, _, _ = native()
    messages = create_model_prompt(task)
    # Existing native M7 generation contract remains the schema and authority boundary.
    messages[0]['content'] += (
        '\nTHIS IS A RESEARCH-CONTINUATION EXPERIMENT, NOT A NOVELTY CERTIFICATION. '
        'Improve the actual scientific mechanism, not only rhetoric. Check whether its '
        'distinguishing prediction truly follows from written equations, and whether '
        'physically constrained existing alternatives can reproduce it. You may explicitly '
        'abandon the parent mechanism and propose a genuinely different causal model, or '
        'state that no discovery-worthy differentiator was obtained. '
        'Keep output a COMPLETE compact JSON object following the original exact schema. '
        'Prefer under 3500 output tokens while preserving equations, assumptions and falsifier. '
        'Never invent supporting prior-art citations or measured results. '
        'Keep authority flags false. No predetermined physical answer is required.')
    data = {
        'frozen_original_parent_program_search_only': parent,
        'ye_fulltext_and_retrieved_abstracts_source_evidence': evidence,
        'independent_reviewer_feedback_search_only': reviewer,
        'revision_lane': 'SCIENTIFIC_PROGRAM_REVISION',
        'required_scientific_outcome': ('A revised mechanistic research program with a defensible '
                                        'differential prediction, or an explicit failure to obtain one. '
                                        'Either is an acceptable scientific result.'),
    }
    messages[1]['content'] += '\n\nFROZEN SOURCE AND REVIEW FEEDBACK:\n' + json.dumps(data, ensure_ascii=False, sort_keys=True)
    return messages


def _validate_review(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict) or set(parsed) != {'diagnosis', 'repair_priorities', 'abstain_boundary', 'overall_judgment'}:
        raise ValueError('REVIEWER_SCHEMA_INVALID')
    for k in ('diagnosis', 'repair_priorities'):
        if not isinstance(parsed[k], list) or not (2 <= len(parsed[k]) <= 8) or not all(isinstance(v, str) and v.strip() for v in parsed[k]):
            raise ValueError('REVIEWER_LIST_INVALID_' + k)
    for k in ('abstain_boundary', 'overall_judgment'):
        if not isinstance(parsed[k], str) or not parsed[k].strip():
            raise ValueError('REVIEWER_TEXT_INVALID_' + k)
    return parsed


def prepare(previous: Path, kit: Path, out: Path) -> dict[str, Any]:
    previous, kit, out = previous.resolve(), kit.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS_NO_OVERWRITE')
    if out == previous or previous in out.parents or out in previous.parents or out == kit or kit in out.parents or out in kit.parents:
        raise ValueError('OUTPUT_MUST_BE_SEPARATED_FROM_ALL_INPUTS')
    parent, evidence, info = source_snapshot(previous, kit)
    save_new(out / 'inputs' / 'parent.json', parent)
    save_new(out / 'inputs' / 'evidence.json', evidence)
    save_new(out / 'inputs' / 'task.json', info['task'])
    hashes = info['hashes']
    for arm in ARMS:
        save_new(out / 'prompts' / ('REVIEW_' + arm + '.json'),
                 {'messages': reviewer_messages(parent, evidence, arm)})
    # Token-budget parity is by identical model/max_tokens for matched phases;
    # advice content deliberately differs. Critic call adds no new literature provider.
    schedule = list(ARMS)
    secrets.SystemRandom().shuffle(schedule)
    schedule = ['REVIEW_' + a for a in schedule] + ['REVISE_' + a for a in schedule]
    manifest = {'schema_version': SCHEMA, 'status': 'PREPARED_NO_PAID_CALLS',
        'source_output_dir': str(previous), 'frozen_kit_dir': str(kit),
        'source_hashes': hashes, 'parent_research_program_sha256': sha_obj(parent),
        'frozen_input_sha256': {name: sha_file(out / 'inputs' / name) for name in ('parent.json','evidence.json','task.json')},
        'review_prompt_sha256': {arm: sha_file(out / 'prompts' / ('REVIEW_' + arm + '.json')) for arm in ARMS},
        'schedule': schedule, 'max_new_paid_calls': 4, 'new_paid_calls_at_preparation': 0,
        'same_original_parent_and_fulltext_in_both_arms': True,
        'figure_pages_for_all_calls': list(FIGURES),
        'actual_M7_core_reused': True, 'official_FAKG_E2E': False,
        'official_SIS_scheduler_executed': False, 'canonical_graph_mutated': False,
        'scientific_truth_certified': False, 'novelty_certified': False,
        'no_prior_human_science_review_in_prompts': True,
        'blinding': 'Only revised programs shared; source-arm assignment private until review frozen'}
    save_new(out / 'RUN_MANIFEST.json', manifest)
    return {'status': manifest['status'], 'out': str(out),
            'paid_schedule': schedule, 'max_new_calls': 4,
            'parent': parent['idea_label'], 'source_verified': True}


def verify(out: Path) -> dict[str, Any]:
    m = load(out / 'RUN_MANIFEST.json')
    if m.get('schema_version') != SCHEMA or m.get('status') != 'PREPARED_NO_PAID_CALLS':
        raise ValueError('MANIFEST_SCHEMA_STATUS_INVALID')
    parent, evidence, info = source_snapshot(Path(m['source_output_dir']), Path(m['frozen_kit_dir']))
    if info['hashes'] != m['source_hashes']:
        raise ValueError('FROZEN_UPSTREAM_SOURCE_DRIFT')
    if sha_obj(parent) != m['parent_research_program_sha256']:
        raise ValueError('PARENT_HASH_DRIFT')
    for n,h in m['frozen_input_sha256'].items():
        if sha_file(out / 'inputs' / n) != h:
            raise ValueError('FROZEN_INPUT_FILE_CHANGED_' + n)
    if load(out / 'inputs' / 'parent.json') != parent or load(out / 'inputs' / 'evidence.json') != evidence:
        raise ValueError('FROZEN_PARENT_OR_EVIDENCE_MISMATCH')
    for arm,h in m['review_prompt_sha256'].items():
        if sha_file(out / 'prompts' / ('REVIEW_' + arm + '.json')) != h:
            raise ValueError('REVIEW_PROMPT_CHANGED')
    expected = ['REVIEW_' + a for a in ARMS] + ['REVISE_' + a for a in ARMS]
    if sorted(m['schedule']) != sorted(expected) or len(m['schedule']) != 4:
        raise ValueError('INVALID_EXPERIMENT_CALL_PLAN')
    return m


def step_name(out: Path) -> str | None:
    m = verify(out)
    for step in m['schedule']:
        receipt = out / 'receipts' / (step + '.json')
        attempt = out / 'attempts' / (step + '.json')
        parsed = out / ('parsed_reviews' if step.startswith('REVIEW_') else 'parsed_revisions') / (step + '.json')
        if receipt.exists():
            if load(receipt).get('finish_reason') != 'stop' or not parsed.is_file():
                raise RuntimeError('PRIOR_PAID_RESULT_NOT_VALID_NO_AUTORETRY: ' + step)
            continue
        if attempt.exists():
            raise RuntimeError('PAID_ATTEMPT_UNRESOLVED_NO_AUTORETRY: ' + step)
        return step
    return None


def _attached_images(messages: list[dict[str, Any]], kit: Path) -> list[dict[str, Any]]:
    # Image inputs are symmetric for each scientific arm and each phase.
    messages = json.loads(json.dumps(messages))
    tail = [{'type': 'text', 'text': messages[-1]['content']}]
    for pg in FIGURES:
        data = (kit / f'source/Ye_page_{pg:02d}.png').read_bytes()
        tail.append({'type': 'text', 'text': f'Ye 2025 PRB, original PDF page {pg}'})
        tail.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}})
    messages[-1]['content'] = tail
    return messages


def run_one(out: Path, model: str, api_key_env: str, base_url: str,
            review_tokens: int, revision_tokens: int, temperature: float) -> dict[str, Any]:
    m = verify(out)
    step = step_name(out)
    if step is None:
        return {'status': 'ALL_FOUR_CALLS_ALREADY_SAVED', 'new_calls': 0}
    if not os.getenv(api_key_env):
        raise PermissionError('MISSING_' + api_key_env)
    if not (1800 <= review_tokens <= 4500 and 3500 <= revision_tokens <= 10000):
        raise ValueError('PAID_OUTPUT_TOKEN_CAP_OUTSIDE_BOUND')
    if not (0 <= temperature <= 1):
        raise ValueError('INVALID_TEMPERATURE')
    config_path = out / 'PROVIDER_CONFIG.json'
    requested_config = {'model':model,'base_url':base_url,'api_key_env':api_key_env,
                        'temperature':temperature,'review_tokens':review_tokens,
                        'revision_tokens':revision_tokens}
    if config_path.is_file():
        if load(config_path) != requested_config:
            raise ValueError('PAID_PROVIDER_SETTINGS_CHANGED_FROM_FIRST_CALL')
    else:
        save_new(config_path, requested_config)
    attempt = out / 'attempts' / (step + '.json')
    receipt = out / 'receipts' / (step + '.json')
    if attempt.exists() or receipt.exists():
        raise RuntimeError('INCOMPLETE_OLD_PAID_ATTEMPT_NO_AUTORETRY: ' + step)
    arm = step[len('REVIEW_'):] if step.startswith('REVIEW_') else step[len('REVISE_'):]
    if arm not in ARMS:
        raise ValueError('INVALID_ARM_AT_EXECUTION')
    if step.startswith('REVIEW_'):
        messages = load(out / 'prompts' / (step + '.json'))['messages']
        tokens = review_tokens
    else:
        rfile = out / 'parsed_reviews' / ('REVIEW_' + arm + '.json')
        if not rfile.exists():
            raise RuntimeError('NEED_PARSED_REVIEW_BEFORE_REVISION')
        reviewer = _validate_review(load(rfile))
        messages = revision_messages(load(out / 'inputs' / 'parent.json'),
            load(out / 'inputs' / 'evidence.json'), reviewer,
            load(out / 'inputs' / 'task.json'), arm)
        tokens = revision_tokens
    messages = _attached_images(messages, Path(m['frozen_kit_dir']))
    prompt_hash = sha_obj(messages)
    # Marker BEFORE call: if transport fails, do not silently pay twice.
    save_new(attempt, {'state':'ATTEMPT_STARTED_NO_AUTORETRY', 'step':step,
                       'model':model,'temperature':temperature,'max_tokens':tokens,
                       'prompt_sha256':prompt_hash,
                       'utc_started_at':datetime.now(timezone.utc).isoformat()})
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[api_key_env],base_url=base_url,max_retries=0,timeout=300)
    response = client.chat.completions.create(model=model, messages=messages,
             response_format={'type':'json_object'},temperature=temperature,max_tokens=tokens)
    choice = response.choices[0]
    raw = choice.message.content or ''
    save_new(receipt, {'status': 'PAID_RESPONSE_SAVED_NOT_SCIENCE_CERTIFIED',
                       'step': step, 'arm': arm, 'model': model,
                       'finish_reason': choice.finish_reason, 'raw_content': raw,
                       'provider_response': response.model_dump(mode='json'),
                       'prompt_sha256': prompt_hash})
    if choice.finish_reason != 'stop':
        raise RuntimeError(f'PAID_RECEIPT_INCOMPLETE_NO_RETRY: {step} finish_reason={choice.finish_reason}')
    try:
        item = json.loads(raw)
        if step.startswith('REVIEW_'):
            parsed = _validate_review(item)
            save_new(out / 'parsed_reviews' / (step+'.json'), parsed)
        else:
            ScientificProgramDraft, _, challenge_program, _ = native()
            if isinstance(item, dict) and isinstance(item.get('program'),dict):
                item = item['program']
            prog = ScientificProgramDraft.parse(item)
            task = load(out / 'inputs' / 'task.json')
            if (prog.idea_label != task['label'] or prog.parent_idea_id != task['parent_idea_id']
                or prog.source_role != task['role']):
                raise ValueError('REVISED_PROGRAM_IDENTITY_MISMATCH')
            save_new(out / 'parsed_revisions' / (step+'.json'), prog.to_dict())
            save_new(out / 'structural_feedback' / (step+'.json'), challenge_program(prog))
    except Exception as ex:
        raise RuntimeError(f'RESPONSE_SAVED_BUT_INVALID_NO_AUTORETRY: {step}: {ex}') from ex
    return {'status':'ONE_PAID_STEP_SAVED','step':step,'finish_reason':choice.finish_reason,
            'remaining':sum(not (out / 'receipts' / (s+'.json')).is_file() for s in STEPS)}


def make_blind(out: Path) -> dict[str, Any]:
    m = verify(out)
    if (out / 'BLIND_REVIEW_PUBLIC.json').exists() or (out / 'BLIND_ASSIGNMENT_PRIVATE.json').exists():
        raise FileExistsError('BLIND_PACKET_ALREADY_EXISTS')
    rows=[]
    ScientificProgramDraft,_,_,_ = native()
    parent = load(out / 'inputs' / 'parent.json')
    for arm in ARMS:
        step = 'REVISE_' + arm
        receipt = load(out / 'receipts' / (step + '.json'))
        if receipt['finish_reason'] != 'stop':
            raise ValueError('INCOMPLETE_PAID_RESULT_' + step)
        p=ScientificProgramDraft.parse(load(out / 'parsed_revisions' / (step+'.json')))
        item=p.to_dict()
        for key in ('idea_label','parent_idea_id','source_role','source_of_draft'):
            item.pop(key,None)
        rows.append((arm,item))
    secrets.SystemRandom().shuffle(rows)
    public={'schema_version':SCHEMA,'status':'LEVEL2_BLIND_REVIEW_READY',
        'official_FAKG_E2E':False, 'source': 'historical native FAKG portfolio + Ye fulltext via M7 opt-in bridge',
        'same_parent_and_sources_in_both_arms':True,
        'parent_unverified':{k:v for k,v in parent.items() if k not in ('source_of_draft',)},
        'items':[{'id':f'B{i+1:02d}', 'program':p} for i,(_,p) in enumerate(rows)],
        'review_rubric': ('Grade mathematical entailment and physical feasibility; construct concrete '
          'strong null countermodels; identify an exclusive prospective signature; check prior art, '
          'independent intervention and falsifier; do not reward length or formalism. '
          'Explicitly abstain if no novel distinguishable prediction survives.'),
        'scientific_novelty_certified':False}
    save_new(out / 'BLIND_REVIEW_PUBLIC.json',public)
    save_new(out / 'BLIND_ASSIGNMENT_PRIVATE.json',{f'B{i+1:02d}':arm for i,(arm,_) in enumerate(rows)})
    return {'status':'LEVEL2_BLIND_REVIEW_READY', 'public_path':str(out/'BLIND_REVIEW_PUBLIC.json'),
            'private_mapping_exposed':False,'new_paid_calls':0}


def report(out: Path) -> dict[str, Any]:
    m=verify(out)
    statuses=[]
    for step in m['schedule']:
        receipt=out/'receipts'/(step+'.json')
        attempt=out/'attempts'/(step+'.json')
        parsed=out/('parsed_reviews' if step.startswith('REVIEW_') else 'parsed_revisions')/(step+'.json')
        raw=load(receipt) if receipt.exists() else None
        if raw:
            state='COMPLETE' if raw.get('finish_reason')=='stop' and parsed.is_file() else 'SAVED_NEEDS_MANUAL_REVIEW'
        else:
            state='ATTEMPT_UNRESOLVED' if attempt.exists() else 'PENDING'
        statuses.append({'step':step,'status':state,
                        'finish_reason':raw.get('finish_reason') if raw else None})
    try: next_step=step_name(out)
    except RuntimeError as ex: next_step='BLOCKED: '+str(ex)
    return {'status':'PILOT_EXECUTION_REPORT_NOT_SCIENCE_CERTIFIED','steps':statuses,
            'calls_recorded':sum(x['status'] in ('COMPLETE','SAVED_NEEDS_MANUAL_REVIEW') for x in statuses),
            'next_step':next_step}


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--previous-out',type=Path,required=True);p.add_argument('--kit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for op in ('verify','status','blind'):
        p=sub.add_parser(op);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run-paid');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='openai/gpt-5.6-luna');p.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    p.add_argument('--review-tokens',type=int,default=3200);p.add_argument('--revision-tokens',type=int,default=8000)
    p.add_argument('--temperature',type=float,default=0.2)
    p.add_argument('--allow-paid',action='store_true');p.add_argument('--i-authorize-new-llm-calls',action='store_true')
    a=ap.parse_args()
    try:
        if a.cmd=='prepare': result=prepare(a.previous_out,a.kit,a.out)
        elif a.cmd=='verify': result={'status':'FROZEN_INPUTS_VERIFIED_NO_CALLS','max_new_calls':verify(a.out)['max_new_paid_calls']}
        elif a.cmd=='status': result=report(a.out)
        elif a.cmd=='blind': result=make_blind(a.out)
        else:
            if not (a.allow_paid and a.i_authorize_new_llm_calls and os.getenv('CASE01_ALLOW_PAID')=='YES'):
                raise PermissionError('PAID_CALL_REQUIRES_APPROVAL_FLAGS_AND_CASE01_ALLOW_PAID=YES')
            result=run_one(a.out,a.model,a.api_key_env,a.base_url,a.review_tokens,a.revision_tokens,a.temperature)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except Exception as ex:
        print(f'{type(ex).__name__}: {ex}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
