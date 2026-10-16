"""Add-only compatibility runner for one frozen dual-JSON paid ResearchStep.

The locked native research executor and its source digest are never changed.
The original raw_response.json is never changed. The parser compatibility is
process-local, used for verify/recover/next/finish/commit-native on this fork.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.scientific_research_control import scientific_claim_development as claim
from pipeline_core.discovery.scientific_research_control import w08_science_act_continuation as lineage
from pipeline_core.discovery.scientific_research_control import research_work_v18_feedback_runner as native

EXPECTED_RAW_SHA256 = 'e37a5659b675bea1255852de0daf5fa968af963da1e463348636db2cb3b162a6'
EXPECTED_WORK = 'work_0005_bc6cce75a01a'
RECOVERY_VERSION = 'claim-step2-dual-json-compat-v1'


def _objects_exactly_two(raw: str):
    """Strictly parse exactly two consecutive JSON objects; reject other suffixes."""
    if not isinstance(raw, str):
        raise ValueError('RAW_NOT_TEXT')
    dec = json.JSONDecoder()
    a, end_a = dec.raw_decode(raw)
    at_b = end_a
    while at_b < len(raw) and raw[at_b].isspace():
        at_b += 1
    if at_b == end_a:
        raise ValueError('MISSING_JSON_SEPARATOR')
    b, end_b = dec.raw_decode(raw, at_b)
    if raw[end_b:].strip():
        raise ValueError('MORE_THAN_TWO_JSON_OBJECTS')
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError('NOT_TWO_JSON_OBJECTS')
    return a, b


def canonical_step(raw: str) -> dict:
    """Only joins an inert probe and valid continuation. No scientific text editing."""
    probe, response = _objects_exactly_two(raw)
    if set(probe) != {'kind', 'payload'} or probe['kind'] not in ('ctmc_threshold',):
        raise ValueError('UNEXPECTED_PROBE_SHAPE')
    if not isinstance(probe['payload'], dict):
        raise ValueError('UNEXPECTED_PROBE_PAYLOAD')
    if response.get('probe_request') is not None:
        raise ValueError('RESPONSE_ALREADY_HAS_PROBE')
    if response.get('operation') != 'TEST' or response.get('next_action') != 'CONTINUE':
        raise ValueError('NOT_A_TEST_CONTINUATION')
    if response.get('claim_maturity') == 'COMPUTATION_CHECKED':
        raise ValueError('UNJUSTIFIED_COMPUTATION_CHECKED_BEFORE_PROBE')
    # Require it to be a genuinely valid ResearchStep before combining them.
    original = claim.ResearchStep.model_validate(response)
    if original.probe_request is not None:
        raise ValueError('PROBE_CONFLICT')
    combined = dict(response)
    combined['probe_request'] = probe
    claim.ResearchStep.model_validate(combined)
    return combined


# Patching ONLY this Python process. The on-disk locked module is unchanged.
_ORIGINAL_PARSE = claim.ResearchStep.model_validate_json


def _compatible_parse(cls, json_data, **kwargs):
    try:
        return _ORIGINAL_PARSE(json_data, **kwargs)
    except ValueError:
        if isinstance(json_data, bytes):
            json_data = json_data.decode('utf-8')
        obj = canonical_step(json_data)
        return claim.ResearchStep.model_validate(obj)


def _activate():
    claim.ResearchStep.model_validate_json = classmethod(_compatible_parse)


def _guard(out: Path):
    path = out / 'claim_work' / 'step_0002' / 'raw_response.json'
    if not path.is_file():
        raise FileNotFoundError('SAVED_STEP2_RAW_NOT_FOUND')
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != EXPECTED_RAW_SHA256:
        raise ValueError(f'SAVED_RAW_SHA256_MISMATCH: {sha}')
    raw = json.loads(data)['raw_text']
    combined = canonical_step(raw)
    lock = json.loads((out / 'claim_work' / 'lock.json').read_text('utf-8'))
    if lock.get('work_id') != EXPECTED_WORK:
        raise ValueError('WORK_ID_NOT_EXPECTED')
    if not (out / 'claim_work' / 'step_0001' / 'receipt.json').is_file():
        raise ValueError('STEP1_NOT_COMMITTED')
    return path, data, raw, combined


def _audit(out: Path, verified: dict) -> dict:
    path, data, raw, combined = _guard(out)
    step = path.parent
    if verified.get('claim_work', {}).get('steps', 0) < 2:
        raise ValueError('STEP2_NOT_RECOVERED')
    normalized = json.loads((step/'normalized.json').read_text('utf-8'))
    if normalized != claim.ResearchStep.model_validate(combined).model_dump(mode='json'):
        raise ValueError('DERIVED_NORMALIZED_DIFFERS')
    instrument = json.loads((step/'instrument_result.json').read_text('utf-8'))
    if instrument != {'instrument_result':claim._check_probe(combined['probe_request'])}:
        raise ValueError('DERIVED_INSTRUMENT_DIFFERS')
    report = {
        'protocol': RECOVERY_VERSION,
        'original_raw_response_sha256': hashlib.sha256(data).hexdigest(),
        'raw_response_unchanged': True,
        'recovery_rule': 'first JSON kind/payload set as probe_request on second valid TEST CONTINUE ResearchStep',
        'first_json_kind': combined['probe_request']['kind'],
        'scientific_text_modified': False,
        'native_source_modified': False,
        'normalized_sha256': hashlib.sha256((step/'normalized.json').read_bytes()).hexdigest(),
        'instrument_result_sha256': hashlib.sha256((step/'instrument_result.json').read_bytes()).hexdigest(),
    }
    report_path = step/'dual_json_recovery_audit.json'
    if report_path.is_file():
        if json.loads(report_path.read_text('utf-8')) != report:
            raise ValueError('RECOVERY_AUDIT_CHANGED')
    else:
        # Audit is additive metadata not part of the locked event receipts.
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n','utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('recover', 'verify', 'inspect', 'preview', 'next', 'finish', 'commit-native'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--allow-paid', action='store_true')
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    _activate()
    _guard(out)
    if args.command == 'recover':
        before = claim.verify(out)
        if before['claim_work']['incomplete']:
            result = claim.recover(out)
        else:
            result = {'status': 'ALREADY_RECOVERED_NO_CALL'}
        after = claim.verify(out)
        audit = _audit(out, after)
        result['audit'] = audit
    elif args.command == 'verify':
        result = claim.verify(out)
        if result['claim_work']['steps'] >= 2:
            result['recovery_audit'] = _audit(out,result)
    elif args.command == 'inspect':
        result = claim.verify(out)
        if result['claim_work']['steps'] >= 2:
            result['recovery_audit'] = _audit(out,result)
            res = json.loads((out/'claim_work'/'step_0002'/'instrument_result.json').read_text('utf-8'))
            result['step2'] = {
                'operation': 'TEST',
                'next_action': 'CONTINUE',
                'probe_kind': 'ctmc_threshold',
                'instrument_status': (res['instrument_result'] or {}).get('status'),
                'original_raw_sha256': EXPECTED_RAW_SHA256,
            }
    elif args.command == 'preview':
        result = claim.preview(out)
    elif args.command == 'next':
        result = claim.next_step(out, allow_paid=args.allow_paid)
    elif args.command == 'finish':
        result = claim.finish(out)
    else:
        result = claim.commit_native(out)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
