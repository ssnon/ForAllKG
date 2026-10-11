#!/usr/bin/env python3
"""M14 opt-in recovery of an indeterminate generation provider attempt.

Never deletes or rewrites the original intent or any prior model receipt.
An explicit second call may incur a second charge for one scientific generation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable

from pipeline_core.discovery.sers_novelty_search_m14 import candidate_prompt, frozen, sha
from scripts.discovery.run_sers_novelty_search_m14 import (
    create, digest, load, source_call,
)


def inspect(plan_file: Path, replicate: int, generation: int, arm: str, model: str) -> tuple[dict, dict, Path]:
    """Inspect and cryptographically verify a single pending original intent, offline."""
    plan = frozen(load(plan_file))
    if load(plan_file.parent / 'M14_PLAN.json') != plan:
        raise RuntimeError('M14 plan file mismatch')
    d = plan['design']
    if not (1 <= replicate <= d['replicates'] and 1 <= generation <= d['generations']):
        raise ValueError('Round outside frozen plan')
    if arm not in ('ADAPTIVE', 'FIXED'):
        raise ValueError('Unknown experiment arm')
    problems = load(plan_file.parent / 'M14_PROVISIONAL_PROBLEMS.json')
    if problems.get('novelty_certified') is not False or len(problems.get('questions', [])) != 4:
        raise RuntimeError('Provisional problems have invalid status')
    history = []
    for g in range(1, generation):
        prev = plan_file.parent / f'r{replicate:02d}' / f'g{g:02d}' / arm / 'M14_GENERATION_RESULT.json'
        if not prev.is_file():
            raise RuntimeError('Missing previous generation history')
        history.append(load(prev))
    prompt = candidate_prompt(plan, problems, replicate, generation, arm, history)
    folder = plan_file.parent / f'r{replicate:02d}' / f'g{generation:02d}' / arm
    intent_path = folder / 'generation_INTENT.json'
    receipt_path = folder / 'generation_RAW_RECEIPT.json'
    retry_intent_path = folder / 'generation_RETRY_INTENT.json'
    if not intent_path.exists():
        raise RuntimeError('Original generation intent does not exist; this recovery is not applicable')
    intent = load(intent_path)
    expected_hash = sha(prompt)
    if (intent.get('status') != 'PROVIDER_ATTEMPT_STARTED_NO_AUTO_RETRY' or
        intent.get('model') != model or intent.get('prompt_sha256') != expected_hash):
        raise RuntimeError('Original intent does not match exact frozen prompt/model; cannot retry')
    status = {
        'status': 'M14_INDETERMINATE_GENERATION_INSPECTED_NO_PROVIDER_CALLS',
        'replicate': replicate, 'generation': generation, 'arm': arm,
        'prompt_sha256': expected_hash,
        'original_intent_sha256': digest(intent_path),
        'original_intent_exists': True, 'original_receipt_exists': receipt_path.exists(),
        'explicit_retry_intent_exists': retry_intent_path.exists(),
        'original_request_billable_status': 'UNKNOWN',
        'automatic_retry_safe': False,
        'novelty_certified': False,
    }
    return status, prompt, folder


def perform_explicit_retry(plan_file: Path, replicate: int, generation: int,
                           arm: str, model: str, call: Callable, *,
                           accept_possible_duplicate: bool, confirm_not_running: bool) -> dict:
    if not (accept_possible_duplicate and confirm_not_running):
        raise RuntimeError('Explicit duplicate-charge acknowledgement and prior-call-stopped confirmation required')
    status, prompt, folder = inspect(plan_file, replicate, generation, arm, model)
    if status['original_receipt_exists']:
        raise RuntimeError('Original receipt already present; no retry allowed')
    if status['explicit_retry_intent_exists']:
        raise RuntimeError('Retry intent already present; never auto repeat an indeterminate retry')
    if (folder / 'M14_GENERATION_PARSED.json').exists() or (folder / 'M14_GENERATION_RESULT.json').exists():
        raise RuntimeError('Downstream science artifacts exist; refuse to change generation')
    # This explicit record is created BEFORE invoking the potentially billable provider.
    # If a process dies, the second attempt is itself protected from any duplicate retry.
    audit = {
        'status': 'M14_EXPLICIT_RETRY_OF_INDETERMINATE_PROVIDER_REQUEST',
        'original_intent_sha256': status['original_intent_sha256'],
        'original_intent_path': str(folder / 'generation_INTENT.json'),
        'prompt_sha256': status['prompt_sha256'],
        'model': model, 'unknown_if_original_call_was_billed': True,
        'user_acknowledged_possible_duplicate_charge': True,
        'user_confirmed_original_process_not_running': True,
        'never_certifies_novelty': True,
    }
    create(folder / 'generation_RETRY_INTENT.json', audit)
    try:
        response = call(model, prompt)
    except Exception as exc:
        raise RuntimeError('Explicit retry provider result indeterminate; retry intent retained; DO NOT RECALL: '+str(exc)) from exc
    create(folder / 'generation_RAW_RECEIPT.json', {
        'prompt_sha256': status['prompt_sha256'],
        'model': model, 'raw': response['text'],
        'finish_reason': response['finish_reason'], 'usage': response.get('usage'),
        'novelty_certified': False,
        'explicit_retry_of_unknown_original_charge': True,
    })
    create(folder / 'M14_INDETERMINATE_RETRY_AUDIT.json', {
        **audit, 'status':'M14_EXPLICIT_RETRY_RECEIPT_SAVED',
        'retry_intent_sha256':digest(folder/'generation_RETRY_INTENT.json'),
        'source_receipt_sha256':digest(folder/'generation_RAW_RECEIPT.json'),
        'actual_original_provider_charge_remains_unknown':True,
    })
    if response['finish_reason'] != 'stop':
        raise RuntimeError('New receipt incomplete, saved for manual examination; no automatic recall')
    return {
        'status':'M14_EXPLICIT_RETRY_RECEIPT_SAVED_ORIGINAL_INTENT_PRESERVED',
        'new_model_calls_this_invocation':1,
        'original_request_billing':'UNKNOWN',
        'total_charge_parity_certifiable':False,
        'original_intent_unchanged':status['original_intent_sha256']==digest(folder/'generation_INTENT.json'),
        'retry_receipt_sha256':digest(folder/'generation_RAW_RECEIPT.json'),
        'next_step':'Resume original M14 run-paid; do NOT repeat explicit retry',
        'novelty_certified':False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'retry-paid'):
        q = sub.add_parser(name)
        q.add_argument('--plan', type=Path, required=True)
        q.add_argument('--replicate', type=int, required=True)
        q.add_argument('--generation', type=int, required=True)
        q.add_argument('--arm', choices=('ADAPTIVE','FIXED'), required=True)
        q.add_argument('--generator-model', default='openai/gpt-5.6-luna')
        if name == 'retry-paid':
            q.add_argument('--api-key-env', default='OPENROUTER_API_KEY')
            q.add_argument('--base-url', default='https://openrouter.ai/api/v1')
            q.add_argument('--temperature', type=float, default=0.7)
            q.add_argument('--max-output-tokens', type=int, default=9000)
            q.add_argument('--allow-paid', action='store_true')
            q.add_argument('--i-authorize-model-calls', action='store_true')
            q.add_argument('--i-accept-possible-duplicate-charge', action='store_true')
            q.add_argument('--i-confirm-original-call-stopped', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'inspect':
        out, _, _ = inspect(args.plan, args.replicate, args.generation, args.arm, args.generator_model)
    else:
        if not (args.allow_paid and args.i_authorize_model_calls):
            raise RuntimeError('Explicit paid model authorization required')
        # Provider factory may load credentials; this function alone does not call it.
        fn = source_call(args)
        out = perform_explicit_retry(args.plan, args.replicate, args.generation,
            args.arm, args.generator_model, fn,
            accept_possible_duplicate=args.i_accept_possible_duplicate_charge,
            confirm_not_running=args.i_confirm_original_call_stopped)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
