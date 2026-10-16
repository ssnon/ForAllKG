"""No-new-LLM-call compatibility for EXACTLY two identical tool envelopes + a LiteratureStep.

Three-object recovery, not a generic permissive JSON parser. Existing paid raw and
prior engines stay unchanged; only one literature tool dispatch occurs. The
previous two-object and single-object compatibility paths remain available.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from . import open_dual_json_literature_recovery as previous
from . import open_literature_research as work
from . import research_work_v12_stable_runner as base

PROTOCOL = 'open-repeated-literature-tool-recovery-v1'
ORIGINAL_DECODE = previous.decode_envelope
AUDIT_NAME = 'repeated_tool_recovery_audit.json'


def _parse_objects(raw: str) -> list[dict[str, Any]]:
    if not isinstance(raw, str):
        raise ValueError('RAW_TEXT_NOT_STRING')
    dec = json.JSONDecoder()
    pos = 0
    objects: list[dict[str, Any]] = []
    while True:
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos == len(raw):
            break
        if len(objects) >= 3:
            raise ValueError('MORE_THAN_THREE_OBJECTS_OR_TRAILING_TEXT')
        try:
            obj, end = dec.raw_decode(raw, pos)
        except (ValueError, TypeError) as exc:
            raise ValueError('NON_JSON_OR_TRAILING_TEXT') from exc
        if not isinstance(obj, dict):
            raise ValueError('OBJECT_NOT_DICTIONARY')
        objects.append(obj)
        pos = end
    return objects


def _repeated_only(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = _parse_objects(raw)
    if len(objects) != 3:
        raise ValueError('NOT_THREE_JSON_OBJECTS')
    first, duplicate, final = objects
    if set(first) != {'kind', 'payload'} or first.get('kind') != 'literature_read':
        raise ValueError('FIRST_NOT_LITERATURE_READ_ENVELOPE')
    if not isinstance(first.get('payload'), dict):
        raise ValueError('INVALID_TOOL_PAYLOAD')
    if duplicate != first or set(duplicate) != {'kind', 'payload'}:
        raise ValueError('DIFFERING_PRELIMINARY_TOOL_REQUESTS')
    if final.get('tool_request') != first:
        raise ValueError('FINAL_TOOL_REQUEST_DIFFERENT_FROM_PRELIMINARY')
    # Strictly validate the scientific response using the *existing* EXPLORE
    # compatibility. This does not modify science or allow malformed fields.
    previous.PREVIOUS_PARSE(json.dumps(final, ensure_ascii=False))
    return first, final


def decode_envelope(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return ORIGINAL_DECODE(raw)
    except ValueError:
        return _repeated_only(raw)


@contextmanager
def scope():
    before = previous.decode_envelope
    previous.decode_envelope = decode_envelope
    try:
        yield
    finally:
        previous.decode_envelope = before


def _target(out: Path, arm: str, n: int) -> Path:
    return work.arm_dir(out, arm) / f'step_{n:04d}'


def _triple_info(path: Path) -> dict[str, Any] | None:
    raw = base.read(path / 'raw_response.json')['raw_text']
    try:
        tool, final = _repeated_only(raw)
    except ValueError:
        return None
    return {
        'tool': tool, 'final': final,
        'raw_sha256': work.hfile(path / 'raw_response.json'),
        'tool_sha256': work.sha(tool),
        'scientific_step_sha256': work.sha(final),
    }


def _audit_value(path: Path, info: dict[str, Any]) -> dict[str, Any]:
    return {
        'protocol': PROTOCOL,
        'original_paid_raw_sha256': info['raw_sha256'],
        'repeated_tool_envelopes': 2,
        'identical_repetitions_verified': True,
        'final_tool_request_matches': True,
        'tool_kind': info['tool']['kind'],
        'paper_id': info['tool']['payload']['paper_id'],
        'requested_depth': info['tool']['payload']['depth'],
        'tool_sha256': info['tool_sha256'],
        'scientific_step_sha256': info['scientific_step_sha256'],
        'normalized_sha256': work.hfile(path / 'normalized.json'),
        'tool_result_sha256': work.hfile(path / 'tool_result.json'),
        'raw_response_unchanged': True,
        'scientific_text_modified': False,
        'existing_engine_files_modified': False,
    }


def verify(out: Path) -> dict:
    with scope():
        state = previous.verify(out)
        for arm, arm_state in state['arms'].items():
            for n in range(1, arm_state['steps'] + 1):
                p = _target(out, arm, n)
                info = _triple_info(p)
                if info:
                    audit = p / AUDIT_NAME
                    if not audit.is_file() or base.read(audit) != _audit_value(p, info):
                        raise ValueError('REPEATED_TOOL_RECOVERY_AUDIT_MISSING_OR_CHANGED')
        return state


def inspect(out: Path, arm: str) -> dict:
    with scope():
        result = previous.inspect(out, arm)
        incomplete = result.get('incomplete')
        if not incomplete or not (Path(incomplete) / 'raw_response.json').is_file():
            return result
        info = _triple_info(Path(incomplete))
        if info:
            result.update({
                'three_json_objects': True,
                'duplicate_tool_envelopes_identical': True,
                'matches_final_tool_request': True,
                'requested_depth': info['tool']['payload']['depth'],
                'paid_raw_sha256': info['raw_sha256'],
                'only_one_literature_read_will_be_dispatched': True,
            })
        return result


def recover(out: Path, arm: str) -> dict:
    report = inspect(out, arm)
    incomplete = report.get('incomplete')
    if not incomplete:
        raise ValueError('NO_INCOMPLETE_SAVED_STEP')
    p = Path(incomplete)
    info = _triple_info(p)
    if not info:
        raise ValueError('REQUIRES_TWO_IDENTICAL_ENVELOPES_AND_MATCHING_FINAL_JSON')
    if report.get('read_target_in_previous_search') is not True:
        raise ValueError('READ_TARGET_NOT_IN_PREVIOUS_SEARCH_DO_NOT_EXECUTE')
    if arm != 'literature_enabled':
        raise ValueError('NO_LITERATURE_IN_CONTROL_ARM')
    before = work.hfile(p / 'raw_response.json')
    with scope():
        result = previous.recover(out, arm)
        if work.hfile(p / 'raw_response.json') != before:
            raise ValueError('PAID_RAW_RESPONSE_CHANGED')
        # Original engine dispatches a single tool_request from the normalized
        # step. The two redundant preambles are never dispatched separately.
        value = _audit_value(p, info)
        audit = p / AUDIT_NAME
        if audit.exists():
            if base.read(audit) != value:
                raise ValueError('EXISTING_REPEATED_AUDIT_DIFFERS')
        else:
            work.put(audit, value)
    verify(out)
    return result | {
        'compatibility_protocol': PROTOCOL,
        'duplicate_envelopes_collapsed_for_execution': True,
        'literature_reads_dispatched': 1,
        'original_paid_raw_sha256': before,
        'raw_response_unchanged': True,
    }


def preview(out: Path, arm: str) -> dict:
    verify(out)
    with scope():
        return previous.preview(out, arm)


def next_step(out: Path, arm: str, *, allow_paid: bool = False) -> dict:
    verify(out)
    with scope():
        return previous.next_step(out, arm, allow_paid=allow_paid)


def finish(out: Path, arm: str) -> dict:
    verify(out)
    with scope():
        return previous.finish(out, arm)
