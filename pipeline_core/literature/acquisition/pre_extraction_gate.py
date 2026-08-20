from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.literature.acquisition.contracts import AcquisitionProfile, SelectedCorpusWork
from pipeline_core.literature.catalog_contracts import CatalogWork


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


IdentityStatus = Literal[
    "verified",
    "weak_match",
    "mismatch",
    "unverifiable",
]
SuitabilityStatus = Literal[
    "suitable",
    "manual_review",
    "unsuitable",
    "unavailable",
]


class IdentityGatePolicy(StrictModel):
    front_matter_chars: int = Field(default=12000, ge=1000)
    doi_scan_chars: int = Field(default=12000, ge=1000)
    verified_title_token_f1: float = Field(default=0.78, ge=0.0, le=1.0)
    weak_title_token_f1: float = Field(default=0.55, ge=0.0, le=1.0)
    min_title_token_count: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _title_threshold_order(self) -> "IdentityGatePolicy":
        if self.weak_title_token_f1 > self.verified_title_token_f1:
            raise ValueError(
                "weak_title_token_f1 must be <= verified_title_token_f1"
            )
        return self


class SuitabilityGatePolicy(StrictModel):
    min_main_markdown_chars: int = Field(default=1500, ge=1)
    min_suitable_axes: int = Field(default=1, ge=1)
    min_axis_indicator_hits_per_axis: int = Field(default=1, ge=1)
    min_relation_context_blocks_per_axis: int = Field(default=1, ge=1)
    relation_signal_terms: list[str] = Field(min_length=1)


class PreExtractionGatePolicy(StrictModel):
    schema_version: Literal[
        "pre-extraction-gate-policy-v1"
    ] = "pre-extraction-gate-policy-v1"
    policy_id: str
    identity: IdentityGatePolicy = Field(default_factory=IdentityGatePolicy)
    suitability: SuitabilityGatePolicy
    allowed_identity_statuses: list[IdentityStatus] = Field(
        default_factory=lambda: ["verified"]
    )
    allowed_suitability_statuses: list[SuitabilityStatus] = Field(
        default_factory=lambda: ["suitable"]
    )


class BibliographicIdentityAssessment(StrictModel):
    status: IdentityStatus
    method: Literal[
        "doi_exact",
        "title_verified",
        "title_weak",
        "doi_conflict",
        "insufficient_metadata",
    ]
    expected_doi: str | None = None
    observed_dois: list[str] = Field(default_factory=list)
    best_title_token_f1: float = 0.0
    expected_title_token_count: int = 0
    reasons: list[str] = Field(default_factory=list)


class FullTextSuitabilityAssessment(StrictModel):
    status: SuitabilityStatus
    selected_axes: list[str] = Field(default_factory=list)
    axis_indicator_hits_by_axis: dict[str, list[str]] = Field(
        default_factory=dict
    )
    relation_context_blocks_by_axis: dict[str, int] = Field(
        default_factory=dict
    )
    suitable_axes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PreExtractionGateAssessment(StrictModel):
    schema_version: Literal[
        "pre-extraction-gate-assessment-v1"
    ] = "pre-extraction-gate-assessment-v1"
    policy_id: str
    acquisition_profile_id: str
    paper_id: str
    work_id: str
    title: str
    doi: str | None = None
    identity: BibliographicIdentityAssessment
    suitability: FullTextSuitabilityAssessment
    auto_extraction_allowed: bool
    scientific_result_direction_inferred: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False


class PreExtractionGateReport(StrictModel):
    schema_version: Literal[
        "pre-extraction-gate-report-v1"
    ] = "pre-extraction-gate-report-v1"
    policy_id: str
    acquisition_profile_id: str
    evaluated_paper_count: int
    auto_extraction_ready_count: int
    blocked_paper_count: int
    identity_status_counts: dict[str, int] = Field(default_factory=dict)
    suitability_status_counts: dict[str, int] = Field(default_factory=dict)
    blocked_paper_ids: list[str] = Field(default_factory=list)
    source_catalog_id: str | None = None
    source_m4_materialization_id: str | None = None
    source_m4_ready_paper_count: int = 0
    source_m4_dir: str | None = None
    input_config_path: str | None = None
    output_config_path: str | None = None
    extraction_plan_path: str | None = None
    llm_calls_performed: Literal[False] = False
    scientific_result_inference_performed: Literal[False] = False
    positive_evidence_promotion_performed: Literal[False] = False


_DOI_RE = re.compile(
    r"(?i)(?:https?://(?:dx\.)?doi\.org/|doi\s*[:=]\s*)?"
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)"
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}


def load_pre_extraction_gate_policy(path: Path) -> PreExtractionGatePolicy:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Pre-extraction gate policy must be a mapping: {path}")
    return PreExtractionGatePolicy.model_validate(loaded)


def _normalize_doi(value: str | None) -> str | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*[:=]\s*", "", text)
    text = text.strip().rstrip(".,;:)]}")
    return text or None


def _observed_dois(text: str, *, max_chars: int) -> list[str]:
    front = text[:max_chars]
    values = []
    for match in _DOI_RE.finditer(front):
        normalized = _normalize_doi(match.group(1))
        if normalized:
            values.append(normalized)
    return sorted(set(values))


def _title_tokens(value: str) -> list[str]:
    tokens = [token.casefold() for token in _WORD_RE.findall(value)]
    return [token for token in tokens if token not in _STOPWORDS]


def _token_f1(expected: list[str], observed: list[str]) -> float:
    if not expected or not observed:
        return 0.0
    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    overlap = sum(
        min(expected_counts[token], observed_counts[token])
        for token in expected_counts
    )
    if overlap == 0:
        return 0.0
    precision = overlap / sum(observed_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall)


def _front_matter_title_score(
    expected_title: str,
    markdown: str,
    *,
    max_chars: int,
) -> tuple[float, int]:
    expected = _title_tokens(expected_title)
    if not expected:
        return 0.0, 0

    lines = [
        line.strip()
        for line in markdown[:max_chars].splitlines()
        if line.strip()
    ]
    candidates: list[str] = []
    for index in range(min(len(lines), 80)):
        candidates.append(lines[index])
        if index + 1 < len(lines):
            candidates.append(lines[index] + " " + lines[index + 1])
        if index + 2 < len(lines):
            candidates.append(
                lines[index] + " " + lines[index + 1] + " " + lines[index + 2]
            )

    score = max(
        (_token_f1(expected, _title_tokens(candidate)) for candidate in candidates),
        default=0.0,
    )
    return score, len(expected)


def assess_bibliographic_identity(
    *,
    work: CatalogWork,
    main_markdown: str,
    policy: IdentityGatePolicy,
) -> BibliographicIdentityAssessment:
    expected_doi = _normalize_doi(work.doi)
    observed_dois = _observed_dois(
        main_markdown,
        max_chars=policy.doi_scan_chars,
    )
    title_score, title_token_count = _front_matter_title_score(
        work.title,
        main_markdown,
        max_chars=policy.front_matter_chars,
    )

    if expected_doi and expected_doi in observed_dois:
        return BibliographicIdentityAssessment(
            status="verified",
            method="doi_exact",
            expected_doi=expected_doi,
            observed_dois=observed_dois,
            best_title_token_f1=round(title_score, 6),
            expected_title_token_count=title_token_count,
            reasons=["expected_doi_found_in_front_matter"],
        )

    enough_title = title_token_count >= policy.min_title_token_count
    if enough_title and title_score >= policy.verified_title_token_f1:
        reasons = ["strong_title_match"]
        if expected_doi and observed_dois and expected_doi not in observed_dois:
            reasons.append("different_doi_observed_but_title_match_is_strong")
        return BibliographicIdentityAssessment(
            status="verified",
            method="title_verified",
            expected_doi=expected_doi,
            observed_dois=observed_dois,
            best_title_token_f1=round(title_score, 6),
            expected_title_token_count=title_token_count,
            reasons=reasons,
        )

    if (
        expected_doi
        and observed_dois
        and expected_doi not in observed_dois
        and title_score < policy.weak_title_token_f1
    ):
        return BibliographicIdentityAssessment(
            status="mismatch",
            method="doi_conflict",
            expected_doi=expected_doi,
            observed_dois=observed_dois,
            best_title_token_f1=round(title_score, 6),
            expected_title_token_count=title_token_count,
            reasons=[
                "front_matter_contains_different_doi",
                "title_match_below_weak_threshold",
            ],
        )

    if enough_title and title_score >= policy.weak_title_token_f1:
        return BibliographicIdentityAssessment(
            status="weak_match",
            method="title_weak",
            expected_doi=expected_doi,
            observed_dois=observed_dois,
            best_title_token_f1=round(title_score, 6),
            expected_title_token_count=title_token_count,
            reasons=["title_match_requires_manual_review"],
        )

    return BibliographicIdentityAssessment(
        status="unverifiable",
        method="insufficient_metadata",
        expected_doi=expected_doi,
        observed_dois=observed_dois,
        best_title_token_f1=round(title_score, 6),
        expected_title_token_count=title_token_count,
        reasons=["no_exact_doi_or_sufficient_title_match"],
    )


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.casefold()))


def _term_present(normalized_text: str, term: str) -> bool:
    target = _normalize_text(term)
    if not target:
        return False
    parts = target.split()
    if len(parts) == 1 and len(parts[0]) <= 3:
        return parts[0] in set(normalized_text.split())
    return target in normalized_text


def _matching_terms(normalized_text: str, terms: list[str]) -> list[str]:
    return sorted(
        {term for term in terms if _term_present(normalized_text, term)},
        key=lambda value: value.casefold(),
    )


def _context_blocks(
    markdown: str,
    *,
    expected_title: str,
) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n+", markdown)
    title_tokens = _title_tokens(expected_title)
    blocks = []
    for index, block in enumerate(raw_blocks):
        cleaned = _normalize_text(block)
        if not cleaned:
            continue
        block_tokens = _title_tokens(cleaned)
        # A title heading must not by itself satisfy the scientific-suitability
        # gate. Marker output normally places it in the first few blocks.
        title_like = (
            index < 5
            and title_tokens
            and len(block_tokens) <= len(title_tokens) + 10
            and _token_f1(title_tokens, block_tokens) >= 0.72
        )
        if title_like:
            continue
        blocks.append(cleaned)
    return blocks


def assess_fulltext_suitability(
    *,
    selected_work: SelectedCorpusWork,
    acquisition_profile: AcquisitionProfile,
    main_markdown: str,
    policy: SuitabilityGatePolicy,
) -> FullTextSuitabilityAssessment:
    selected_axes = sorted(set(selected_work.matched_axes))
    axis_map = {axis.axis_id: axis for axis in acquisition_profile.axes}
    unknown_axes = sorted(set(selected_axes) - set(axis_map))
    if unknown_axes:
        raise ValueError(
            "Selected work contains axes absent from acquisition profile: "
            + ", ".join(unknown_axes)
        )

    if not main_markdown.strip():
        return FullTextSuitabilityAssessment(
            status="unavailable",
            selected_axes=selected_axes,
            reasons=["main_markdown_unavailable"],
        )

    if len(main_markdown.strip()) < policy.min_main_markdown_chars:
        return FullTextSuitabilityAssessment(
            status="manual_review",
            selected_axes=selected_axes,
            reasons=["main_markdown_below_minimum_length"],
        )

    if not selected_axes:
        return FullTextSuitabilityAssessment(
            status="manual_review",
            selected_axes=[],
            reasons=["no_evidence_grounded_selected_axis"],
        )

    blocks = _context_blocks(
        main_markdown,
        expected_title=selected_work.title,
    )
    normalized_full = " ".join(blocks)
    relation_terms = policy.relation_signal_terms

    hits_by_axis: dict[str, list[str]] = {}
    context_counts: dict[str, int] = {}
    suitable_axes: list[str] = []

    for axis_id in selected_axes:
        indicators = list(axis_map[axis_id].indicators)
        hits = _matching_terms(normalized_full, indicators)
        hits_by_axis[axis_id] = hits

        context_count = 0
        if len(hits) >= policy.min_axis_indicator_hits_per_axis:
            for block in blocks:
                if not _matching_terms(block, hits):
                    continue
                if _matching_terms(block, relation_terms):
                    context_count += 1
        context_counts[axis_id] = context_count

        if (
            len(hits) >= policy.min_axis_indicator_hits_per_axis
            and context_count >= policy.min_relation_context_blocks_per_axis
        ):
            suitable_axes.append(axis_id)

    if len(suitable_axes) >= policy.min_suitable_axes:
        return FullTextSuitabilityAssessment(
            status="suitable",
            selected_axes=selected_axes,
            axis_indicator_hits_by_axis=hits_by_axis,
            relation_context_blocks_by_axis=context_counts,
            suitable_axes=sorted(suitable_axes),
            reasons=["axis_and_relation_context_confirmed_in_fulltext"],
        )

    any_axis_hits = any(hits_by_axis.values())
    if any_axis_hits:
        return FullTextSuitabilityAssessment(
            status="manual_review",
            selected_axes=selected_axes,
            axis_indicator_hits_by_axis=hits_by_axis,
            relation_context_blocks_by_axis=context_counts,
            suitable_axes=[],
            reasons=["axis_terms_present_without_sufficient_relation_context"],
        )

    return FullTextSuitabilityAssessment(
        status="unsuitable",
        selected_axes=selected_axes,
        axis_indicator_hits_by_axis=hits_by_axis,
        relation_context_blocks_by_axis=context_counts,
        suitable_axes=[],
        reasons=["selected_axis_terms_absent_from_fulltext"],
    )


def assess_pre_extraction_gate(
    *,
    paper_id: str,
    work: CatalogWork,
    selected_work: SelectedCorpusWork,
    acquisition_profile: AcquisitionProfile,
    main_markdown: str,
    policy: PreExtractionGatePolicy,
) -> PreExtractionGateAssessment:
    if selected_work.work_id != work.work_id:
        raise ValueError("Selected work/catalog work mismatch")

    identity = assess_bibliographic_identity(
        work=work,
        main_markdown=main_markdown,
        policy=policy.identity,
    )
    suitability = assess_fulltext_suitability(
        selected_work=selected_work,
        acquisition_profile=acquisition_profile,
        main_markdown=main_markdown,
        policy=policy.suitability,
    )
    allowed = (
        identity.status in set(policy.allowed_identity_statuses)
        and suitability.status in set(policy.allowed_suitability_statuses)
    )
    return PreExtractionGateAssessment(
        policy_id=policy.policy_id,
        acquisition_profile_id=acquisition_profile.profile_id,
        paper_id=paper_id,
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        identity=identity,
        suitability=suitability,
        auto_extraction_allowed=allowed,
        scientific_result_direction_inferred=False,
        positive_evidence_promotion_performed=False,
    )


def build_pre_extraction_gate_report(
    *,
    assessments: list[PreExtractionGateAssessment],
    policy: PreExtractionGatePolicy,
    acquisition_profile: AcquisitionProfile,
) -> PreExtractionGateReport:
    identity_counts = Counter(row.identity.status for row in assessments)
    suitability_counts = Counter(row.suitability.status for row in assessments)
    blocked = sorted(
        row.paper_id for row in assessments if not row.auto_extraction_allowed
    )
    return PreExtractionGateReport(
        policy_id=policy.policy_id,
        acquisition_profile_id=acquisition_profile.profile_id,
        evaluated_paper_count=len(assessments),
        auto_extraction_ready_count=sum(
            row.auto_extraction_allowed for row in assessments
        ),
        blocked_paper_count=len(blocked),
        identity_status_counts=dict(sorted(identity_counts.items())),
        suitability_status_counts=dict(sorted(suitability_counts.items())),
        blocked_paper_ids=blocked,
        llm_calls_performed=False,
        scientific_result_inference_performed=False,
        positive_evidence_promotion_performed=False,
    )
