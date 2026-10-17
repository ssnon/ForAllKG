"""No-repay, original-preserving recovery for saved M22 plan search hints >80 chars.

Only the navigation field pdf_search_terms is shortened deterministically. Research
questions, source IDs, citations, physical claims and every other field are retained.
The original paid JSON is never rewritten. Use this compatibility CLI for the fork.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import shutil

from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m

PROTOCOL = 'm22-overlong-pdf-search-hint-fork-v1'
AUDIT = 'm22_plan_terms_recovery_audit.json'
RAW = 'plan/raw_response.json'
ORIGINAL_JSON = m.Plan.model_validate_json


def file_sha(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def str_sha(s: str) -> str:
    return sha256(s.encode('utf-8')).hexdigest()


def _shorten(hint: str) -> str:
    """Retain leading complete words of one overlong navigation hint.

    No fabricated search terms, no rewriting of scientific claims, no silent removal.
    """
    hint = hint.strip()
    if 3 <= len(hint) <= 80:
        return hint
    if len(hint) < 3:
        raise ValueError('UNDERLENGTH_OR_BLANK_SEARCH_TERM_NOT_AUTORECOVERED')
    prefix = hint[:80]
    # Permit a term ending at the 80th character only at a word boundary.
    if len(hint) > 80 and not hint[80].isspace() and not hint[79].isspace():
        if ' ' not in prefix:
            raise ValueError('LONG_UNSPLITTABLE_SEARCH_TERM_NOT_AUTORECOVERED')
        prefix = prefix.rsplit(' ', 1)[0]
    prefix = prefix.strip()
    if len(prefix) < 3:
        raise ValueError('LONG_UNSPLITTABLE_SEARCH_TERM_NOT_AUTORECOVERED')
    return prefix


def diagnosis(raw: str, papers: list[dict]) -> tuple[dict, dict]:
    """Require exactly this defect; all other strict plan validation must pass."""
    original = ORIGINAL_JSON(raw).model_dump(mode='json')
    ids = {p['paper_id'] for p in papers}
    ids_selected = original['ranked_paper_ids']
    if len(set(ids_selected)) != len(ids_selected) or not set(ids_selected) <= ids:
        raise ValueError('INVALID_OR_UNRETRIEVED_PAPER_ID_NO_REPAIR')
    hints = original['pdf_search_terms']
    if not any(isinstance(h, str) and len(h.strip()) > 80 for h in hints):
        raise ValueError('NO_OVERLONG_SEARCH_HINT_IN_SAVED_RAW')
    if any(not isinstance(h, str) or len(h.strip()) < 3 for h in hints):
        raise ValueError('OTHER_INVALID_SEARCH_HINT_NOT_AUTORECOVERED')
    normalized = original.copy()
    normalized['pdf_search_terms'] = [_shorten(h) for h in hints]
    if normalized['pdf_search_terms'] == hints:
        raise ValueError('NO_SEARCH_HINT_CHANGE')
    # Preserve exact plan fields other than PDF navigation hints.
    assert all(original[k] == normalized[k] for k in original if k != 'pdf_search_terms')
    validated = m.Plan.model_validate(normalized).model_dump(mode='json')
    if not all(3 <= len(h) <= 80 for h in validated['pdf_search_terms']):
        raise ValueError('REPAIRED_HINT_STILL_INVALID')
    audit_fields = [{'index': i, 'original_length': len(before.strip()),
                     'shortened_length': len(after), 'original_hint': before,
                     'new_hint': after, 'modified': before != after}
                    for i, (before, after) in enumerate(zip(hints, validated['pdf_search_terms']))]
    return validated, {'terms': audit_fields, 'chosen_paper_ids': ids_selected,
                       'unchanged_science_fields': [k for k in original if k != 'pdf_search_terms']}


@contextmanager
def scoped_saved_raw(raw_text_hash: str, canonical_plan: dict):
    """Only exact known paid text is interpreted as the explicitly recovered plan."""
    original = m.Plan.model_validate_json

    def compatible(cls, raw, *args, **kwargs):
        # All later/nonmatching responses use the original strict parser.
        if isinstance(raw, bytes):
            try: raw_cmp = raw.decode('utf-8')
            except UnicodeDecodeError: return original(raw, *args, **kwargs)
        else: raw_cmp = raw
        if isinstance(raw_cmp, str) and str_sha(raw_cmp) == raw_text_hash:
            # Validate full original as required. Fail if newly changed.
            ORIGINAL_JSON(raw, *args, **kwargs)
            return cls.model_validate(canonical_plan)
        return original(raw, *args, **kwargs)

    with patch.object(m.Plan, 'model_validate_json', classmethod(compatible)):
        yield


def source_inspect(source: Path) -> tuple[dict, dict, dict]:
    source = Path(source).expanduser().resolve()
    status = m.verify(source)
    if status != {'status': 'M22_VERIFIED_NO_CALL', 'plan': 'RAW_SAVED_RECOVER_NO_RETRY',
                  'pdf': 'NOT_STARTED', 'analysis': 'NOT_STARTED', 'scientific_validation': False}:
        raise ValueError('SOURCE_NOT_EXACTLY_RAW_SAVED_PLAN_WITHOUT_PDF')
    if (source/'plan/normalized.json').exists():
        raise ValueError('SOURCE_NORMALIZED_ALREADY_PRESENT')
    raw = m.read(source/RAW)['raw_text']
    papers = m.read(source/'source_papers.json')
    canonical, changes = diagnosis(raw, papers)
    return canonical, changes, {'raw_text_sha256': str_sha(raw),
            'raw_file_sha256': file_sha(source/RAW),
            'source_lock_sha256': file_sha(source/'lock.json'),
            'source_request_sha256': file_sha(source/'plan/request.json'),
            'source_papers_sha256': file_sha(source/'source_papers.json'),
            'source_status': status}


def inspect(source: Path) -> dict:
    canonical, changes, digests = source_inspect(source)
    return {'status': 'M22_OVERLONG_PLAN_TERMS_ELIGIBLE_NO_CALL',
            'source_untouched': True, 'requires_paid_call': False,
            'paper_ids': changes['chosen_paper_ids'],
            'search_terms': changes['terms'],
            'all_other_plan_fields_unchanged': True,
            'raw_sha256': digests['raw_file_sha256']}


def audit_check(out: Path) -> tuple[dict, dict]:
    out = Path(out).expanduser().resolve()
    audit = m.read(out/AUDIT)
    if audit.get('protocol') != PROTOCOL or audit.get('scientific_validation') is not False:
        raise ValueError('UNSUPPORTED_RECOVERY_AUDIT')
    source = Path(audit['source']).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('FORK_PATH_IS_NOT_INDEPENDENT')
    canonical, changes, digests = source_inspect(source)
    if audit.get('original') != digests or audit.get('change') != changes:
        raise ValueError('ORIGINAL_OR_RECOVERY_DESCRIPTION_CHANGED')
    if file_sha(out/RAW) != digests['raw_file_sha256']:
        raise ValueError('FORK_PAID_RAW_CHANGED')
    if file_sha(out/'lock.json') != digests['source_lock_sha256']:
        raise ValueError('FORK_LOCK_CHANGED')
    if file_sha(out/'plan/request.json') != digests['source_request_sha256']:
        raise ValueError('FORK_REQUEST_CHANGED')
    if file_sha(out/'source_papers.json') != digests['source_papers_sha256']:
        raise ValueError('FORK_PAPERS_CHANGED')
    if m.read(out/'plan/normalized.json') != canonical:
        raise ValueError('FORK_SCIENTIFIC_PLAN_CHANGED')
    if file_sha(out/'plan/normalized.json') != audit['canonical_file_sha256']:
        raise ValueError('FORK_NORMALIZATION_BYTES_CHANGED')
    return audit, canonical


def fork(source: Path, target: Path) -> dict:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if target.exists():
        raise FileExistsError('FRESH_FORK_TARGET_REQUIRED')
    if target == source or target in source.parents or source in target.parents:
        raise ValueError('FORK_PATHS_MUST_BE_DISJOINT')
    canonical, changes, digests = source_inspect(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=False)
    try:
        with scoped_saved_raw(digests['raw_text_sha256'], canonical):
            result = m.recover(target, 'plan')
        audit = {'protocol': PROTOCOL, 'source': str(source), 'original': digests,
                 'change': changes, 'canonical_file_sha256': file_sha(target/'plan/normalized.json'),
                 'scientific_validation': False, 'paid_calls_during_recovery': 0,
                 'original_paid_response_untouched': True,
                 'scope': 'Only overlong pdf_search_terms are shortened to complete-word prefixes; original terms remain in audit.'}
        m.put(target/AUDIT, audit)
        audit_check(target)
        with scoped_saved_raw(digests['raw_text_sha256'], canonical):
            state = m.verify(target)
        if state['plan'] != 'READY':
            raise ValueError('FORK_NOT_READY')
        if file_sha(source/RAW) != digests['raw_file_sha256']:
            raise ValueError('PAID_RAW_CHANGED')
    except BaseException:
        shutil.rmtree(target)
        raise
    return {'status': 'M22_PLAN_RECOVERED_NO_PAID_CALL', 'fork': str(target),
            'plan': state['plan'], 'source_untouched': True,
            'changed_field': 'pdf_search_terms',
            'changed_hints': sum(h['modified'] for h in changes['terms']),
            'original_terms_preserved_in_audit': True,
            'no_scientific_claim_added': True, 'paid_calls': 0}


def compatible(command: str, out: Path, *, allow_paid: bool = False, **kwargs):
    out = Path(out).expanduser().resolve()
    audit, canonical = audit_check(out)
    with scoped_saved_raw(audit['original']['raw_text_sha256'], canonical):
        if command == 'verify': return m.verify(out)
        if command == 'fetch': return m.fetch(out, local_pdf=kwargs.get('pdf_file'),paper_id=kwargs.get('paper_id'))
        if command == 'preview': return m.preview(out, kwargs['phase'])
        if command == 'analyze': return m.paid(out, 'analyze', allow_paid=allow_paid)
        if command == 'recover': return m.recover(out, 'analyze')
        if command == 'report': return m.report(out)
        if command == 'fork-fetch': return m.fork_after_failed_fetch(out, kwargs['fork_out'])
    raise ValueError('UNKNOWN_COMMAND')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inspect','fork','verify','fetch','preview','analyze','recover','report','fork-fetch'))
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--phase', choices=('analyze',))
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--pdf-file', type=Path)
    p.add_argument('--paper-id')
    p.add_argument('--fork-out', type=Path, help='New run after unsuccessful OA PDF fetch')
    args = p.parse_args()
    if args.command == 'inspect':
        if not args.source: p.error('inspect needs --source')
        result = inspect(args.source)
    elif args.command == 'fork':
        if not args.source or not args.out:p.error('fork needs --source and --out')
        result = fork(args.source, args.out)
    else:
        if not args.out:p.error('command needs --out')
        if args.command == 'preview' and args.phase != 'analyze':p.error('preview needs --phase analyze')
        result = compatible(args.command,args.out,allow_paid=args.allow_paid,pdf_file=args.pdf_file,paper_id=args.paper_id,phase=args.phase,fork_out=args.fork_out)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':main()
