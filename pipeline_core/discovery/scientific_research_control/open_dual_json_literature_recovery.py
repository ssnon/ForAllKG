"""No-retry recovery for saved two-object literature tool + scientific EXPLORE.

Preserves the original immutable paid raw response and the scientific content.
Only combines an explicitly saved tool envelope with a valid LiteratureStep
that has no independent/conflicting tool. Also preserves the earlier single-JSON
EXPLORE/tool compatibility. Never patches locked engine source on disk.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from . import scientific_literature_work as lit
from . import open_literature_research as work
from . import open_explore_literature_compat as prior
from . import research_work_v12_stable_runner as base

PROTOCOL = 'open-dual-json-literature-recovery-v1'
ALLOW_KINDS = {'literature_search', 'literature_read'}
# Capture original parser; no self-reference during a temporary monkeypatch.
PREVIOUS_PARSE = prior.parse_compatible


def decode_envelope(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept exactly two JSON objects separated only by whitespace."""
    if not isinstance(raw, str):
        raise ValueError('RAW_TEXT_NOT_STRING')
    decoder = json.JSONDecoder()
    text = raw.lstrip()
    try:
        first, end = decoder.raw_decode(text)
        rest = text[end:].lstrip()
        second, end2 = decoder.raw_decode(rest)
    except (ValueError, TypeError) as exc:
        raise ValueError('NOT_TWO_COMPLETE_JSON_OBJECTS') from exc
    if rest[end2:].strip():
        raise ValueError('TRAILING_NON_JSON_CONTENT')
    if not isinstance(first, dict) or set(first) != {'kind', 'payload'}:
        raise ValueError('FIRST_OBJECT_NOT_TOOL_ENVELOPE')
    if first.get('kind') not in ALLOW_KINDS or not isinstance(first.get('payload'), dict):
        raise ValueError('UNSUPPORTED_TOOL_ENVELOPE')
    if not isinstance(second, dict):
        raise ValueError('SECOND_OBJECT_NOT_RESEARCH_STEP')
    if second.get('tool_request') not in (None, first):
        raise ValueError('CONFLICTING_TOOL_REQUESTS')
    return first, second


def parse_saved(raw: str) -> dict[str, Any]:
    """Strict first, then the narrow dual-object recovery; never rewrite science."""
    try:
        return PREVIOUS_PARSE(raw)
    except ValueError as original:
        try:
            tool, second = decode_envelope(raw)
            merged = dict(second)
            merged['tool_request'] = tool
            if merged.get('next_action') != 'CONTINUE':
                raise ValueError('TOOL_REQUIRES_CONTINUE')
            # Uses original Pydantic fields and previous EXPLORE compatibility.
            return PREVIOUS_PARSE(json.dumps(merged, ensure_ascii=False))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'INVALID_SAVED_DUAL_JSON_LITERATURE_STEP: {exc}') from original


def _is_dual(raw: str) -> bool:
    try:
        decode_envelope(raw)
    except ValueError:
        return False
    return True


@contextmanager
def recovery_scope():
    previous = lit._parse
    lit._parse = parse_saved
    try:
        yield
    finally:
        lit._parse = previous


def _path(out: Path, arm: str, step: int) -> Path:
    return work.arm_dir(out, arm) / f'step_{step:04d}'


def _raw(p: Path) -> str:
    content = base.read(p/'raw_response.json')
    if not isinstance(content, dict) or not isinstance(content.get('raw_text'), str):
        raise ValueError('MISSING_SAVED_RAW_TEXT')
    return content['raw_text']


def _audit(out: Path, v: dict) -> None:
    """Verify exact historical and new audit claims without extra side effects."""
    for arm, arm_state in v['arms'].items():
        for n in range(1, arm_state['steps'] + 1):
            p = _path(out, arm, n)
            raw = _raw(p)
            normalized = parse_saved(raw)
            old = p/'explore_tool_compat_audit.json'
            if old.is_file():
                expect_old = {
                    'protocol': prior.COMPAT_PROTOCOL,
                    'raw_sha256': work.hfile(p/'raw_response.json'),
                    'operation_preserved': normalized['operation'],
                    'tool_kind': (normalized['tool_request'] or {}).get('kind'),
                    'normalized_sha256': work.hfile(p/'normalized.json'),
                    'tool_result_sha256': work.hfile(p/'tool_result.json'),
                    'scientific_text_modified': False,
                    'existing_engine_files_modified': False,
                }
                if base.read(old) != expect_old:
                    raise ValueError('PREVIOUS_COMPAT_AUDIT_CHANGED')
            new = p/'dual_json_recovery_audit.json'
            if new.is_file():
                expect_new = _audit_payload(p, raw, normalized)
                if not _is_dual(raw) or base.read(new) != expect_new:
                    raise ValueError('DUAL_JSON_RECOVERY_AUDIT_CHANGED')


def _audit_payload(p: Path, raw: str, normalized: dict) -> dict:
    first, second = decode_envelope(raw)
    return {
        'protocol': PROTOCOL,
        'raw_sha256': work.hfile(p/'raw_response.json'),
        'normalized_sha256': work.hfile(p/'normalized.json'),
        'tool_result_sha256': work.hfile(p/'tool_result.json'),
        'tool_envelope_kind': first['kind'],
        'second_object_operation': second.get('operation'),
        'recovered_operation': normalized['operation'],
        'scientific_text_modified': False,
        'raw_response_unchanged': True,
        'existing_engine_files_modified': False,
    }


def verify(out: Path) -> dict:
    with recovery_scope():
        v = work.verify(out)
        _audit(out, v)
        return v


def inspect(out: Path, arm: str) -> dict:
    v = verify(out)
    s = v['arms'][arm]
    report = {'status': 'INSPECT_NO_CALL', 'arm': arm,
              'completed_steps': s['steps'], 'incomplete': s['incomplete']}
    if not s['incomplete']:
        return report
    p = Path(s['incomplete'])
    raw_file = p/'raw_response.json'
    if not raw_file.is_file():
        return report | {'status': 'RAW_RESPONSE_NOT_SAVED_DO_NOT_RETRY'}
    raw = _raw(p)
    parsed = parse_saved(raw)
    first = decode_envelope(raw)[0] if _is_dual(raw) else parsed.get('tool_request')
    target = (first or {}).get('payload', {}).get('paper_id')
    known = []
    with recovery_scope():
        completed, _ = work._read_rows(out, arm)
        if target:
            known = [p for p in lit._known_papers(completed) if p['paper_id'] == target]
    return report | {
        'status': 'SAVED_RESPONSE_VALID_NO_CALL',
        'saved_raw_sha256': work.hfile(raw_file),
        'dual_json_envelope': _is_dual(raw),
        'operation': parsed['operation'],
        'next_action': parsed['next_action'],
        'tool_kind': (first or {}).get('kind'),
        'read_paper_id': target,
        'read_target_in_previous_search': len(known) == 1 if target else None,
        'read_target_title': known[0]['title'] if len(known)==1 else None,
        'scientific_text_modified': False,
    }


def recover(out: Path, arm: str) -> dict:
    v = verify(out)
    incomplete = v['arms'][arm]['incomplete']
    if not incomplete:
        raise ValueError('NO_INCOMPLETE_STEP')
    p = Path(incomplete)
    raw_file = p/'raw_response.json'
    if not raw_file.is_file():
        raise ValueError('NO_SAVED_RAW_RESPONSE_DO_NOT_RETRY')
    before = work.hfile(raw_file)
    raw = _raw(p)
    normalized = parse_saved(raw)
    is_dual = _is_dual(raw)
    with recovery_scope():
        result = work.recover(out, arm)
    if work.hfile(raw_file) != before:
        raise ValueError('ORIGINAL_PAID_RAW_MUTATED')
    if is_dual:
        audit = _audit_payload(p, raw, normalized)
        ap = p/'dual_json_recovery_audit.json'
        if not ap.exists():
            work.put(ap, audit)
        elif base.read(ap) != audit:
            raise ValueError('DUAL_JSON_RECOVERY_AUDIT_CHANGED')
    verify(out)
    return result | {'compatibility_protocol': PROTOCOL if is_dual else prior.COMPAT_PROTOCOL,
                     'raw_response_unchanged': True}


def preview(out: Path, arm: str) -> dict:
    with recovery_scope():
        return work.preview(out, arm)


def next_step(out: Path, arm: str, *, allow_paid: bool = False) -> dict:
    with recovery_scope():
        return work.next_step(out, arm, allow_paid=allow_paid)


def finish(out: Path, arm: str) -> dict:
    with recovery_scope():
        return work.finish(out, arm)
