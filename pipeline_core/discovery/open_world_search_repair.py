"""Optional, outcome-blind open-world literature search repair.

Converts the *frozen* task/gap description to compact bibliographic queries;
keeps no-abstract but topically relevant titles in a triage queue, not scientific
claims. No API or LLM calls, no hidden source knowledge, no forced DOI lookup.
"""
from __future__ import annotations

import re
from collections import Counter

_STOP = frozenset("""a an the and or of in on to with from into through for by about as at that this these those what which who why how do does did can could may might should would we they their our its it if when where while is are was were be been being have has had provide provided supplied packet source grounded evidence direct establish establishes established insufficient determine support supports relation relations relate relates relationship mechanisms mechanism produce produces resulting result results research question scientific study proposed suggest identify compare discuss investigate consider observations observe observed explain explaining explanations important potentially dependent dependence effect effects alter changes changing change using rather than newer novel new missing gap no not are unknown known different same possible current frozen paper works authors claim claims determine need needed supports report reported packet pathways like """.split())

_REWRITE_ONLY_STOP = _STOP | frozenset({'near', 'individual', 'dense', 'intermittent', 'about', 'consequently', 'current', 'directly', 'therefore', 'across', 'behavior', 'behaviour', 'first', 'second', 'third', 'data', 'references', 'expected', 'actually', 'selected', 'already', 'including'})

def _terms(s: str) -> list[str]:
    return re.findall(r'[A-Za-z][A-Za-z0-9-]*', str(s))


def focus_terms(question: str) -> tuple[str, ...]:
    """Extract up to 2 domain/phenomenon anchors from *input alone*.

    An explicitly parenthesized acronym is a strong domain anchor; for a
    research question such as '... (SERS) blinking?', use the word following
    that acronym as a phenomenon anchor. No paper identifiers are introduced.
    """
    q = str(question)
    a = re.search(r'\(([A-Z][A-Z0-9]{1,8})\)\s+([A-Za-z][A-Za-z-]{3,})', q)
    if a:
        return (a.group(1).lower(), a.group(2).lower().rstrip('s'))
    acronyms = [t.lower() for t in re.findall(r'\b[A-Z]{2,9}\b', q)]
    significant = [t.lower() for t in _terms(q.split('?')[0]) if len(t)>3 and t.lower() not in _STOP]
    counts = Counter(significant)
    common = sorted(set(significant), key=lambda x:(-counts[x],-len(x),significant.index(x)))
    if acronyms:
        return tuple(dict.fromkeys([acronyms[0], *common[:1]]))
    if len(common)>=2:
        return tuple(common[-2:])
    return tuple(common)


def compact_query(seed_text: str, question: str, *, question_seed: bool) -> str:
    """No model knowledge or retrieved outcomes influence search terms."""
    anchors = list(focus_terms(question))
    q = str(question).split('?')[0]
    source = q if question_seed else str(seed_text)
    words = [t.lower() for t in _terms(source)]
    unique = [w for w in dict.fromkeys(words) if w not in _REWRITE_ONLY_STOP and len(w)>2]
    # Explicitly make room for core terms even if long preliminary clauses exist.
    extra = [w for w in unique if w not in anchors and not w.isdecimal()]
    # Start broadly with the topic; gap queries supply bounded modifiers.
    keyword_text = ' '.join(anchors + ([] if question_seed else extra[:3]))
    if len(_terms(keyword_text)) < 2:
        return str(seed_text)
    return keyword_text[:120]


def _aliases(text: str) -> set[str]:
    """Small lexical harmonization; never use synonyms to infer causal evidence."""
    t = ' ' + re.sub(r'[^a-z0-9]+', ' ', str(text).lower()) + ' '
    s = set(t.split())
    if 'sers' in s or 'serrs' in s or ('surface' in s and 'raman' in s):
        s.add('sers')
    return s


def relevance_score(title: str, abstract: str | None, question: str) -> int:
    """Title-heavy rank. Relevance is NOT scientific evidence or novelty."""
    anchors = focus_terms(question)
    if not anchors:
        return 0
    name = _aliases(title)
    evidence = _aliases(abstract or '')
    found = sum(term in name or term in evidence for term in anchors)
    title_matches = sum(term in name for term in anchors)
    # Both subject and phenomenon required if two distinct anchors exist.
    if len(anchors)>=2 and found < 2:
        return 0
    if found == 0:
        return 0
    secondary = set(_terms(question))
    secondary = {t.lower() for t in secondary if len(t)>5 and t.lower() not in _STOP}
    secondary -= set(anchors)
    return 4*found + 2*title_matches + min(3, len(secondary & (name|evidence)))


def is_relevant(score: int, question: str) -> bool:
    return score >= (8 if len(focus_terms(question))>=2 else 4)
