"""Read-only source compatibility for autonomous EXPLORE + literature tool request.

Original episode raw response, installed source modules and lock are never mutated.
The original LiteratureStep schema imposes a purely administrative operation==
INVESTIGATE constraint on valid literature requests; science may legitimately
EXPLORE using a literature tool. This adapter validates ALL other fields with
that schema, preserves EXPLORE in the normalized research record and dispatches
exactly the original requested tool via the existing engine.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from . import open_literature_research as open_work
from . import scientific_literature_work as literature
from . import research_work_v12_stable_runner as base

COMPAT_PROTOCOL = 'open-explore-literature-tool-compat-v1'
ORIGINAL_PARSE = literature._parse
ALLOW_OPERATIONS = {'EXPLORE'}
ALLOW_TOOLS = {'literature_search', 'literature_read'}


def parse_compatible(raw: str) -> dict:
    """Prefer native strict parser. Precisely admit one extra valid combination."""
    try:
        return ORIGINAL_PARSE(raw)
    except ValueError as original_error:
        try:
            obj = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise original_error
        if not isinstance(obj, dict) or obj.get('operation') not in ALLOW_OPERATIONS:
            raise original_error
        tool = obj.get('tool_request')
        if (
            not isinstance(tool, dict)
            or set(tool) != {'kind', 'payload'}
            or tool.get('kind') not in ALLOW_TOOLS
            or not isinstance(tool.get('payload'), dict)
            or obj.get('next_action') != 'CONTINUE'
        ):
            raise original_error
        # Validate all fields with unchanged strict Pydantic schema. Only the
        # enumerated tool/operation coupling is bypassed for validation.
        surrogate = dict(obj)
        surrogate['operation'] = 'INVESTIGATE'
        accepted = literature.LiteratureStep.model_validate(surrogate).model_dump(mode='json')
        accepted['operation'] = obj['operation']
        return accepted


@contextmanager
def compatibility_scope():
    previous = literature._parse
    literature._parse = parse_compatible
    try:
        yield
    finally:
        literature._parse = previous


def _target(out: Path, arm: str, step: int = 1) -> Path:
    return open_work.arm_dir(out, arm) / f'step_{step:04d}'


def inspect(out: Path, arm: str) -> dict[str, Any]:
    with compatibility_scope():
        status = open_work.verify(out)
        states = status['arms']
        if arm not in states:
            raise ValueError('UNKNOWN_ARM')
        s = states[arm]
        step_no = s['steps'] + 1 if s['incomplete'] else s['steps']
        rawpath = _target(out, arm, step_no) / 'raw_response.json' if step_no else None
        report: dict[str, Any] = {
            'status': 'INSPECT_NO_CALL',
            'arm': arm,
            'completed_steps': s['steps'],
            'incomplete': s['incomplete'],
            'last_action': s['last_action'],
        }
        if rawpath and rawpath.is_file():
            raw_obj = base.read(rawpath)
            raw = raw_obj['raw_text']
            parsed = parse_compatible(raw)
            report.update({
                'saved_raw_sha256': open_work.hfile(rawpath),
                'step': step_no,
                'operation': parsed['operation'],
                'next_action': parsed['next_action'],
                'tool_kind': (parsed['tool_request'] or {}).get('kind'),
                'parser_compatibility': parsed['operation'] == 'EXPLORE' and parsed['tool_request'] is not None,
            })
        return report


def verify(out: Path) -> dict:
    with compatibility_scope():
        status = open_work.verify(out)
        # Additional audit is optional for earlier completed steps, but when
        # present it must match the immutable raw/normalized/tool receipts.
        for arm, state in status['arms'].items():
            for step in range(1, state['steps'] + 1):
                p = _target(out, arm, step)
                audit_path = p/'explore_tool_compat_audit.json'
                if not audit_path.exists():
                    continue
                saved = base.read(audit_path)
                parsed = parse_compatible(base.read(p/'raw_response.json')['raw_text'])
                expected = {
                    'protocol': COMPAT_PROTOCOL,
                    'raw_sha256': open_work.hfile(p/'raw_response.json'),
                    'operation_preserved': parsed['operation'],
                    'tool_kind': (parsed['tool_request'] or {}).get('kind'),
                    'normalized_sha256': open_work.hfile(p/'normalized.json'),
                    'tool_result_sha256': open_work.hfile(p/'tool_result.json'),
                    'scientific_text_modified': False,
                    'existing_engine_files_modified': False,
                }
                if saved != expected:
                    raise ValueError('COMPATIBILITY_AUDIT_CHANGED')
        return status


def preview(out: Path, arm: str) -> dict:
    with compatibility_scope():
        return open_work.preview(out, arm)


def recover(out: Path, arm: str) -> dict:
    # No additional LLM invocation, no rewriting of original raw.
    with compatibility_scope():
        v = open_work.verify(out)
        incomplete = v['arms'][arm]['incomplete']
        if not incomplete:
            raise ValueError('NO_INCOMPLETE_STEP')
        p = Path(incomplete)
        before = open_work.hfile(p / 'raw_response.json')
        parsed = parse_compatible(base.read(p / 'raw_response.json')['raw_text'])
        result = open_work.recover(out, arm)
        after = open_work.hfile(p / 'raw_response.json')
        if before != after:
            raise ValueError('RAW_RESPONSE_MUTATED')
        if parsed['operation'] == 'EXPLORE' and parsed['tool_request'] is not None:
            audit = {
                'protocol': COMPAT_PROTOCOL,
                'raw_sha256': before,
                'operation_preserved': parsed['operation'],
                'tool_kind': parsed['tool_request']['kind'],
                'normalized_sha256': open_work.hfile(p/'normalized.json'),
                'tool_result_sha256': open_work.hfile(p/'tool_result.json'),
                'scientific_text_modified': False,
                'existing_engine_files_modified': False,
            }
            apath = p/'explore_tool_compat_audit.json'
            if not apath.exists():
                open_work.put(apath, audit)
            elif base.read(apath) != audit:
                raise ValueError('COMPATIBILITY_AUDIT_CHANGED')
        verify(out)
        return result | {'compatibility_protocol': COMPAT_PROTOCOL,
                         'raw_response_unchanged': True}


def next_step(out: Path, arm: str, *, allow_paid: bool = False) -> dict:
    with compatibility_scope():
        return open_work.next_step(out, arm, allow_paid=allow_paid)


def finish(out: Path, arm: str) -> dict:
    with compatibility_scope():
        return open_work.finish(out, arm)
