from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import LiteratureRecord
from .selection_plan import LiteratureSelectionPlan


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalized(text: str | None) -> str:
    return _NORMALIZE_RE.sub(" ", (text or "").lower()).strip()


def _normalized_term(term: str) -> str:
    return _normalized(term)


def _term_hits(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    hits = []
    for term in terms:
        needle = _normalized_term(term)
        if needle and needle in text:
            hits.append(term)
    return tuple(sorted(set(hits)))


@dataclass(frozen=True)
class CandidateAssessment:
    paper_id: str
    eligible: bool
    total_score: float
    bucket_scores: dict[str, float]
    best_bucket: str | None
    exclusion_reasons: tuple[str, ...]
    context_hits: tuple[str, ...]
    mechanism_hits: tuple[str, ...]
    bucket_hits: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["exclusion_reasons"] = list(self.exclusion_reasons)
        value["context_hits"] = list(self.context_hits)
        value["mechanism_hits"] = list(self.mechanism_hits)
        value["bucket_hits"] = {
            key: list(items) for key, items in self.bucket_hits.items()
        }
        return value


def assess_candidate(
    record: LiteratureRecord,
    plan: LiteratureSelectionPlan,
) -> CandidateAssessment:
    title = _normalized(record.title)
    abstract = _normalized(record.abstract)
    combined = f"{title} {abstract}".strip()
    metadata = record.metadata or {}

    exclusions: list[str] = []
    raw_abstract = (record.abstract or "").strip()
    if not raw_abstract:
        exclusions.append("missing_abstract")
    elif len(raw_abstract) < plan.min_abstract_chars:
        exclusions.append("abstract_too_short")
    elif (
        plan.max_abstract_chars is not None
        and len(raw_abstract) > plan.max_abstract_chars
    ):
        exclusions.append("abstract_length_outlier")
    if bool(metadata.get("is_retracted")):
        exclusions.append("retracted")
    if bool(metadata.get("is_paratext")):
        exclusions.append("paratext")

    language = str(metadata.get("language") or "").strip().lower()
    if language and plan.allowed_languages and language not in plan.allowed_languages:
        exclusions.append("language_not_allowed")

    title_context = _term_hits(title, plan.global_context_terms)
    abstract_context = _term_hits(abstract, plan.global_context_terms)
    context_hits = tuple(sorted(set(title_context) | set(abstract_context)))
    if not context_hits:
        exclusions.append("missing_catalysis_context")

    title_mechanism = _term_hits(title, plan.global_mechanism_terms)
    abstract_mechanism = _term_hits(abstract, plan.global_mechanism_terms)
    mechanism_hits = tuple(sorted(set(title_mechanism) | set(abstract_mechanism)))

    bucket_scores: dict[str, float] = {}
    bucket_hits: dict[str, tuple[str, ...]] = {}
    for rule in plan.bucket_rules:
        title_hits = _term_hits(title, rule.keywords)
        abstract_hits = _term_hits(abstract, rule.keywords)
        hits = tuple(sorted(set(title_hits) | set(abstract_hits)))
        discovery_bonus = 4.0 if rule.bucket_id in record.mechanism_buckets else 0.0
        score = discovery_bonus + (3.0 * len(title_hits)) + (1.0 * len(abstract_hits))
        bucket_scores[rule.bucket_id] = score
        bucket_hits[rule.bucket_id] = hits

    context_score = (2.0 * len(title_context)) + (1.0 * len(abstract_context))
    mechanism_score = (1.5 * len(title_mechanism)) + (0.5 * len(abstract_mechanism))
    best_bucket = None
    best_bucket_score = 0.0
    for bucket_id, score in bucket_scores.items():
        if score > best_bucket_score:
            best_bucket = bucket_id
            best_bucket_score = score

    total_score = context_score + mechanism_score + best_bucket_score
    if total_score < plan.min_total_score:
        exclusions.append("below_min_total_score")
    if best_bucket is None:
        exclusions.append("no_mechanism_bucket_signal")

    return CandidateAssessment(
        paper_id=record.paper_id,
        eligible=not exclusions,
        total_score=round(total_score, 4),
        bucket_scores={key: round(value, 4) for key, value in bucket_scores.items()},
        best_bucket=best_bucket,
        exclusion_reasons=tuple(sorted(set(exclusions))),
        context_hits=context_hits,
        mechanism_hits=mechanism_hits,
        bucket_hits=bucket_hits,
    )
