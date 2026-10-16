"""M19-E4B target-only control: pin finalized E4, add one isolated targeted review.

External auditor's *target selection* is disclosed as a prompted intervention;
no external proof, numerical counterexample, self-arm or evidence-arm response
is supplied to the model. This is not autonomous discovery or continuation.
"""
from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import m19_e4_independent_audit as e4
from . import research_work_v12_stable_runner as base

SCHEMA = 'm19-e4b-target-only-control-v1'
REVIEW_KIND = 'TARGETED_CLAIM_REVIEW_NO_EXTERNAL_EVIDENCE'
ARM = 'target'
MAX_PAID = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def guard(out: Path) -> Path:
    root = (repo_root() / 'experiments' / 'm19_e4b' / 'runs').resolve()
    p = out.expanduser().resolve()
    if p == root or root not in p.parents:
        raise ValueError('OUTSIDE_REPO_M19_E4B_RUNS')
    return p


def source_guard(e4_out: Path) -> Path:
    root = (repo_root() / 'experiments' / 'm19_e4' / 'runs').resolve()
    p = e4_out.expanduser().resolve(strict=True)
    if p == root or root not in p.parents:
        raise ValueError('SOURCE_OUTSIDE_M19_E4_RUNS')
    return p


def _pin_files(source: Path) -> dict[str, str]:
    # Pin complete original E4 trial, including both arms and the external audit.
    # This pin is metadata only; the target model NEVER receives these files.
    result = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('E4_SOURCE_SYMLINK_DISALLOWED')
        if path.is_file():
            rel = path.relative_to(source).as_posix()
            result[rel] = base.file_sha(path)
    if not result:
        raise ValueError('E4_SOURCE_EMPTY')
    required = {
        'lock.json', 'frozen_final_product.json', 'frozen_deep_work_trace.json',
        'external_audit.json', 'final_report.json',
        'arms/self/request.json', 'arms/self/normalized.json', 'arms/self/record.json',
        'arms/evidence/request.json', 'arms/evidence/normalized.json',
        'arms/evidence/record.json',
    }
    if not required.issubset(result):
        raise ValueError('E4_SOURCE_REQUIRED_FILES_MISSING')
    return result


def _require_final_e4(source: Path) -> dict:
    v = e4.verify(source)
    if not v['finalized'] or v['arms'] != {'self': 'COMPLETE', 'evidence': 'COMPLETE'}:
        raise ValueError('EXPECTED_FINALIZED_E4_BOTH_ARMS')
    return v


def prepare(*, e4_out: Path, out: Path) -> dict:
    dest = guard(out)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('E4B_ALREADY_EXISTS_NO_OVERWRITE')
    source = source_guard(e4_out)
    verified = _require_final_e4(source)
    e4_lock = base.read(source / 'lock.json')
    frozen_product = base.read(source / 'frozen_final_product.json')
    if not any(e4.TARGET_SENTENCE in s for s in frozen_product['content']['reasoning_or_operations']):
        raise ValueError('TARGET_NOT_IN_ORIGINAL_RESEARCH')
    lock = {
        'schema': SCHEMA,
        'e4_source_path': str(source),
        'e4_files_sha256': _pin_files(source),
        'source_work_id': frozen_product['work_id'],
        'source_d4_sha256': e4_lock['source_d4_sha256'],
        'source_original_return': 'RETURN',
        'source_final_report_sha256': base.file_sha(source / 'final_report.json'),
        'source_e4_audit_sha256': verified['audit_sha256'],
        'e4_runner_sha256': base.file_sha(Path(e4.__file__)),
        'base_runner_sha256': base.file_sha(Path(base.__file__)),
        'target_runner_sha256': base.file_sha(Path(__file__)),
        'model': e4_lock['model'],
        'max_new_paid_attempts': MAX_PAID,
        'target_sentence': e4.TARGET_SENTENCE,
        'target_selection_origin': 'EXTERNAL_AUDITOR_SELECTED_CLAIM_NO_PROOF_PROVIDED',
        'external_evidence_in_model_prompt': False,
        'self_or_evidence_outcomes_in_model_prompt': False,
        'autonomous_continuation': False,
        'scientific_validation': False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.preparing')
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError('STALE_E4B_PREPARATION_REQUIRES_REVIEW')
    tmp.mkdir(mode=0o700)
    try:
        base.write_new(tmp / 'lock.json', lock)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    verify(dest)
    return {'status': 'E4B_PREPARED_PINNED_FINAL_E4_NO_CALL', 'out': str(dest),
            'model': lock['model'], 'source_work_id': lock['source_work_id'],
            'source_e4_finalized': True, 'new_paid_calls': 0,
            'scientific_validation': False}


def _request(out: Path) -> dict:
    lock = base.read(out / 'lock.json')
    e4_out = Path(lock['e4_source_path'])
    # Base this on the exact *self-arm* request, not the evidence request.
    # Only the control intervention below differs from the original self prompt.
    req = copy.deepcopy(e4._request(e4_out, 'self'))
    payload = json.loads(req['messages'][1]['content'])
    if 'specific_claim_for_review' in payload or 'externally_supplied_deterministic_audit_not_self_discovered' in payload:
        raise ValueError('SOURCE_SELF_PROMPT_CONTAINS_AUDIT_LEAK')
    if 'self_selection_rule' not in payload:
        raise ValueError('SOURCE_SELF_PROMPT_SHAPE_CHANGED')
    del payload['self_selection_rule']
    payload['specific_claim_for_review'] = lock['target_sentence']
    payload['review_kind'] = REVIEW_KIND
    req['role'] = 'm19_e4b_target_only_review'
    req['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # Only these two differences from original self user payload are authorized.
    baseline = json.loads(e4._request(e4_out, 'self')['messages'][1]['content'])
    changed = copy.deepcopy(baseline)
    changed.pop('self_selection_rule')
    changed['specific_claim_for_review'] = lock['target_sentence']
    changed['review_kind'] = REVIEW_KIND
    if payload != changed:
        raise ValueError('TARGET_PROMPT_CONTENT_MISMATCH')
    if 'json' not in ' '.join(m['content'].lower() for m in req['messages']):
        raise ValueError('JSON_MODE_NOT_READY')
    if req['model'] != lock['model'] or req['temperature'] != 0 or req['max_tokens'] != 6000:
        raise ValueError('MODEL_OR_BUDGET_CHANGED')
    return req


def _parse(raw_text: str) -> dict:
    # Identical scientific output contract to both original E4 arms.
    return e4._parse(raw_text)


def _record(out: Path) -> dict:
    ad = out / 'target'
    req = base.read(ad / 'request.json')
    if req != _request(out):
        raise ValueError('TARGET_REQUEST_REPLAY_MISMATCH')
    attempt = base.read(ad / 'attempt.json')
    if attempt['phase'] != ARM or attempt['request_sha256'] != base.sha(req):
        raise ValueError('TARGET_ATTEMPT_INVALID')
    raw = base.read(ad / 'raw_response.json')
    normalized = _parse(raw['raw_text'])
    if base.read(ad / 'normalized.json') != normalized:
        raise ValueError('TARGET_NORMALIZED_REPLAY_MISMATCH')
    result = {
        'phase': ARM,
        'request_sha256': base.file_sha(ad / 'request.json'),
        'attempt_sha256': base.file_sha(ad / 'attempt.json'),
        'raw_sha256': base.file_sha(ad / 'raw_response.json'),
        'normalized_sha256': base.file_sha(ad / 'normalized.json'),
        'response_id': raw.get('response_id'),
        'usage': raw.get('usage'),
        'review_kind': REVIEW_KIND,
        'specific_claim_provided': True,
        'external_audit_proof_provided': False,
        'existing_self_or_evidence_results_provided': False,
        'autonomous_continuation': False,
        'scientific_validation': False,
    }
    if base.read(ad / 'record.json') != result:
        raise ValueError('TARGET_RECORD_REPLAY_MISMATCH')
    return result


def _report(out: Path) -> dict:
    lock = base.read(out / 'lock.json')
    e4_out = Path(lock['e4_source_path'])
    original = base.read(e4_out / 'final_report.json')
    target = base.read(out / 'target' / 'normalized.json')
    rec = base.file_sha(out / 'target' / 'record.json')
    comparisons = {}
    for arm in ('self', 'evidence'):
        norm = base.read(e4_out / 'arms' / arm / 'normalized.json')
        comparisons[arm] = {
            'disposition': norm['disposition'],
            'earlier_return_was_scientifically_sufficient': norm['earlier_return_was_scientifically_sufficient'],
            'new_testable_consequence_present': bool(norm['new_testable_consequence']),
            'normalized_source_sha256': base.file_sha(e4_out / 'arms' / arm / 'normalized.json'),
        }
    comparisons['target'] = {
        'disposition': target['disposition'],
        'earlier_return_was_scientifically_sufficient': target['earlier_return_was_scientifically_sufficient'],
        'new_testable_consequence_present': bool(target['new_testable_consequence']),
        'record_sha256': rec,
    }
    return {
        'schema': SCHEMA, 'source_work_id': lock['source_work_id'],
        'original_e4_final_report_sha256': base.file_sha(e4_out / 'final_report.json'),
        'original_e4_audit_sha256': lock['source_e4_audit_sha256'],
        'source_e4_report_not_modified': True,
        'three_arm_comparison': comparisons,
        'intervention_design': {
            'self': 'FULL_ORIGINAL_WITHOUT_TARGET_OR_EXTERNAL_PROOF',
            'target': 'FULL_ORIGINAL_PLUS_SPECIFIC_CLAIM_NO_EXTERNAL_PROOF',
            'evidence': 'FULL_ORIGINAL_PLUS_CLAIM_AND_EXTERNAL_COUNTERMODEL',
            'target_selected_by': 'EXTERNAL_AUDITOR_NOT_MODEL',
            'all_posthoc_reviews': True,
            'one_observation_per_arm': True,
        },
        'source_e4_auto_scientific_judgment': original['auto_scientific_judgment'],
        'auto_scientific_judgment': 'NOT_EVALUATED_REQUIRES_INDEPENDENT_MATHEMATICAL_REVIEW',
        'no_autonomous_continuation_credit': True,
        'audit_credit': 'EXTERNAL_NOT_FAKG',
        'native_episode_modified': False,
        'scientific_validation': False,
    }


def verify(out: Path) -> dict:
    out = guard(out)
    lock = base.read(out / 'lock.json')
    if (lock.get('schema') != SCHEMA or lock.get('max_new_paid_attempts') != 1 or
            lock.get('scientific_validation') is not False or
            lock.get('external_evidence_in_model_prompt') is not False or
            lock.get('self_or_evidence_outcomes_in_model_prompt') is not False or
            lock.get('autonomous_continuation') is not False or
            lock.get('target_sentence') != e4.TARGET_SENTENCE or
            lock.get('target_selection_origin') != 'EXTERNAL_AUDITOR_SELECTED_CLAIM_NO_PROOF_PROVIDED'):
        raise ValueError('BAD_E4B_LOCK')
    for k, mod in [('e4_runner_sha256', e4), ('base_runner_sha256', base),
                   ('target_runner_sha256', __import__(__name__, fromlist=['*']))]:
        if base.file_sha(Path(mod.__file__)) != lock[k]:
            raise ValueError('PINNED_SOURCE_CHANGED_' + k)
    source = source_guard(Path(lock['e4_source_path']))
    v = _require_final_e4(source)
    if v['audit_sha256'] != lock['source_e4_audit_sha256']:
        raise ValueError('E4_AUDIT_CHANGED')
    actual_files = _pin_files(source)
    if actual_files != lock['e4_files_sha256']:
        raise ValueError('ORIGINAL_E4_FILES_CHANGED')
    if base.file_sha(source / 'final_report.json') != lock['source_final_report_sha256']:
        raise ValueError('ORIGINAL_E4_FINAL_CHANGED')
    src_lock = base.read(source / 'lock.json')
    if lock['model'] != src_lock['model'] or lock['source_work_id'] != src_lock['source_work_id'] or lock['source_d4_sha256'] != src_lock['source_d4_sha256']:
        raise ValueError('E4_SOURCE_LOCK_MISMATCH')
    target_dir = out / ARM
    status = 'NOT_STARTED'
    if target_dir.exists():
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise ValueError('TARGET_PATH_INVALID')
        req = base.read(target_dir / 'request.json')
        if req != _request(out):
            raise ValueError('TARGET_PROMPT_TAMPERED')
        att = base.read(target_dir / 'attempt.json')
        if att['phase'] != ARM or att['request_sha256'] != base.sha(req):
            raise ValueError('TARGET_ATTEMPT_TAMPERED')
        if not (target_dir / 'raw_response.json').is_file():
            if any((target_dir / f).exists() for f in ('normalized.json', 'record.json')):
                raise ValueError('TARGET_NO_RAW_BUT_DOWNSTREAM_EXISTS')
            status = 'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY'
        elif not (target_dir / 'record.json').is_file():
            status = 'RAW_SAVED_OFFLINE_RECOVER_REQUIRED'
        else:
            _record(out)
            status = 'COMPLETE'
    report_file = out / 'final_report.json'
    if report_file.exists():
        if status != 'COMPLETE':
            raise ValueError('REPORT_WITHOUT_TARGET')
        if base.read(report_file) != _report(out):
            raise ValueError('FINAL_REPORT_TAMPERED')
    return {
        'status': 'VERIFIED_NO_CALL', 'source_work_id': lock['source_work_id'],
        'original_e4_pinned': True, 'original_e4_finalized': True,
        'target_status': status, 'attempt_count': int(status != 'NOT_STARTED'),
        'finalized': report_file.exists(), 'model': lock['model'],
        'target_specific_claim_only': True,
        'external_audit_evidence_in_model_prompt': False,
        'self_evidence_outputs_in_model_prompt': False,
        'source_original_return_unchanged': True,
        'scientific_validation': False, 'paid_calls_this_command': 0,
    }


def preview(out: Path) -> dict:
    out = guard(out)
    v = verify(out)
    if v['finalized'] or v['target_status'] != 'NOT_STARTED':
        raise RuntimeError('TARGET_ALREADY_ATTEMPTED_OR_FINALIZED_NO_REPEAT')
    req = _request(out)
    return {'status': 'PREVIEW_NO_CALL', 'phase': ARM, 'model': req['model'],
            'request_sha256': base.sha(req), 'json_mode_ready': True,
            'external_audit_evidence_in_model_prompt': False,
            'specific_claim_in_model_prompt': True,
            'new_paid_calls': 0}


def _commit_saved_raw(out: Path) -> dict:
    ad = out / ARM
    norm = _parse(base.read(ad / 'raw_response.json')['raw_text'])
    normalized = ad / 'normalized.json'
    if normalized.exists():
        if base.read(normalized) != norm:
            raise ValueError('EXISTING_NORMALIZED_MISMATCH')
    else:
        base.write_new(normalized, norm)
    raw = base.read(ad / 'raw_response.json')
    record = {
        'phase': ARM, 'request_sha256': base.file_sha(ad / 'request.json'),
        'attempt_sha256': base.file_sha(ad / 'attempt.json'),
        'raw_sha256': base.file_sha(ad / 'raw_response.json'),
        'normalized_sha256': base.file_sha(normalized),
        'response_id': raw.get('response_id'), 'usage': raw.get('usage'),
        'review_kind': REVIEW_KIND, 'specific_claim_provided': True,
        'external_audit_proof_provided': False,
        'existing_self_or_evidence_results_provided': False,
        'autonomous_continuation': False, 'scientific_validation': False,
    }
    base.write_new(ad / 'record.json', record)
    verify(out)
    return {'status': 'SAVED_RAW_COMMITTED_OFFLINE_NO_CALL', 'phase': ARM,
            'disposition': norm['disposition'], 'scientific_validation': False}


def recover(out: Path) -> dict:
    out = guard(out)
    v = verify(out)
    if v['target_status'] == 'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY':
        raise RuntimeError('UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
    if v['target_status'] != 'RAW_SAVED_OFFLINE_RECOVER_REQUIRED':
        raise RuntimeError('NO_RECOVERABLE_RAW')
    return _commit_saved_raw(out)


def next_step(out: Path, *, allow_paid: bool = False, max_new_paid_attempts: int = 1, backend=None) -> dict:
    out = guard(out)
    if max_new_paid_attempts != 1:
        raise ValueError('MAX_NEW_PAID_ATTEMPTS_MUST_EQUAL_ONE')
    preview(out)
    base._require_paid(backend, allow_paid, 1)
    req = _request(out)
    ad = out / ARM
    if ad.exists() or ad.is_symlink():
        raise FileExistsError('TARGET_ATTEMPT_EXISTS_NO_REPEAT')
    ad.mkdir(parents=True, exist_ok=False)
    base.write_new(ad / 'request.json', req)
    base.write_new(ad / 'attempt.json', {
        'phase': ARM, 'request_sha256': base.sha(req),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'STARTED_UNKNOWN_BILLING_UNTIL_RESPONSE_SAVED',
    })
    # If provider fails, retain unknown-billing record. Never retry.
    raw_text, usage, response_id = base._invoke(req, backend)
    base.write_new(ad / 'raw_response.json', {
        'raw_text': raw_text, 'usage': usage, 'response_id': response_id,
    })
    try:
        result = _commit_saved_raw(out)
    except Exception as exc:
        raise RuntimeError('RAW_SAVED_OFFLINE_RECOVER_REQUIRED: ' + str(ad)) from exc
    return {
        'status': 'TARGET_COMMITTED', 'phase': ARM,
        'disposition': result['disposition'],
        'new_paid_calls_this_command': int(backend is None),
        'scientific_validation': False,
    }


def finalize(out: Path) -> dict:
    out = guard(out)
    v = verify(out)
    if v['finalized']:
        raise RuntimeError('E4B_ALREADY_FINALIZED')
    if v['target_status'] != 'COMPLETE':
        raise RuntimeError('TARGET_MUST_BE_COMPLETE')
    base.write_new(out / 'final_report.json', _report(out))
    verify(out)
    return {'status': 'E4B_FINALIZED_NO_CALL', 'scientific_validation': False,
            'source_original_unchanged': True,
            'no_autonomous_continuation_credit': True}


def inspect(out: Path) -> dict:
    return verify(out)
