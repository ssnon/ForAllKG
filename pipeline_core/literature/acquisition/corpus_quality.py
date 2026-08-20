from __future__ import annotations

import re
from collections import Counter

from pipeline_core.literature.acquisition.candidate_selection import select_candidates
from pipeline_core.literature.acquisition.contracts import AcquisitionProfile, CandidateAssessment, CorpusSelectionReport, SelectedCorpusWork
from pipeline_core.literature.acquisition.quality_contracts import CorpusQualityAssessment, CorpusQualityGateReport, CorpusQualityPolicy
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def _norm(value: str | None) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9α-ω가-힣@+._-]+", " ", text)
    return " ".join(text.split())


def _term_matches(text: str, terms: list[str]) -> list[str]:
    normalized = _norm(text)
    tokens = set(normalized.split())
    hits = []
    for term in terms:
        target = _norm(term)
        if not target:
            continue
        if len(target.split()) == 1 and len(target) <= 3:
            matched = target in tokens
        else:
            matched = target in normalized
        if matched:
            hits.append(term)
    return sorted(set(hits), key=str.casefold)


def _regex_hits(text: str, patterns: list[str]) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.I)
    ]


def assess_corpus_quality(
    *,
    work: CatalogWork,
    upstream: CandidateAssessment,
    policy: CorpusQualityPolicy,
    originally_selected: bool,
) -> CorpusQualityAssessment:
    title = str(work.title or "")
    abstract = str(work.abstract or "")
    reasons: list[str] = []

    if upstream.eligibility_status != "eligible":
        return CorpusQualityAssessment(
            work_id=work.work_id,
            title=work.title,
            doi=work.doi,
            status="exclude",
            reasons=[
                f"upstream_m2_status:{upstream.eligibility_status}"
            ],
            original_m2_eligibility_status=upstream.eligibility_status,
            originally_selected=originally_selected,
        )

    hard_patterns = _regex_hits(
        title,
        policy.hard_exclude_title_patterns,
    )
    if hard_patterns:
        reasons.extend(
            f"hard_exclude_title_pattern:{pattern}"
            for pattern in hard_patterns
        )

    observed_types = {
        _norm(value)
        for value in work.publication_types
        if _norm(value)
    }
    hard_types = {
        _norm(value)
        for value in policy.hard_exclude_publication_types
        if _norm(value)
    }
    hard_type_hits = sorted(observed_types & hard_types)
    reasons.extend(
        f"hard_exclude_publication_type:{value}"
        for value in hard_type_hits
    )

    primary_title = _term_matches(
        title,
        policy.primary_topic_terms,
    )
    primary_abstract = _term_matches(
        abstract,
        policy.primary_topic_terms,
    )
    title_context = _term_matches(
        title,
        policy.title_context_terms,
    )

    if policy.require_primary_topic_signal:
        if not primary_title and not primary_abstract:
            reasons.append("missing_primary_topic_signal")

    title_grounded = bool(primary_title)
    if not title_grounded and primary_abstract:
        title_grounded = (
            len(title_context)
            >= policy.min_title_context_matches_without_primary_topic
        )
    if policy.require_title_grounding and not title_grounded:
        reasons.append("weak_primary_topic_title_grounding")

    hard_reason = any(
        reason.startswith("hard_exclude_")
        or reason == "missing_primary_topic_signal"
        for reason in reasons
    )
    if hard_reason:
        status = "exclude"
    else:
        manual_patterns = _regex_hits(
            title,
            policy.manual_review_title_patterns,
        )
        manual_reasons = [
            f"manual_review_title_pattern:{pattern}"
            for pattern in manual_patterns
        ]

        manual_types = {
            _norm(value)
            for value in policy.manual_review_publication_types
            if _norm(value)
        }
        manual_type_hits = sorted(observed_types & manual_types)
        manual_reasons.extend(
            f"manual_review_publication_type:{value}"
            for value in manual_type_hits
        )

        doi = _norm(work.doi)
        for prefix in policy.manual_review_doi_prefixes:
            if doi.startswith(_norm(prefix)):
                manual_reasons.append(
                    f"manual_review_doi_prefix:{prefix}"
                )

        venue = _norm(work.venue)
        for term in policy.manual_review_venue_terms:
            if _norm(term) and _norm(term) in venue:
                manual_reasons.append(
                    f"manual_review_venue_term:{term}"
                )

        if "weak_primary_topic_title_grounding" in reasons:
            manual_reasons.append(
                "manual_review_weak_primary_topic_title_grounding"
            )
            reasons = [
                reason
                for reason in reasons
                if reason != "weak_primary_topic_title_grounding"
            ]

        reasons.extend(manual_reasons)
        status = "manual_review" if manual_reasons else "pass"

    return CorpusQualityAssessment(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        status=status,
        reasons=sorted(set(reasons)),
        matched_primary_topic_terms_title=primary_title,
        matched_primary_topic_terms_abstract=primary_abstract,
        matched_title_context_terms=title_context,
        original_m2_eligibility_status=upstream.eligibility_status,
        originally_selected=originally_selected,
    )


def apply_quality_gate_and_reselect(
    *,
    packet: LiteratureCatalogPacket,
    profile: AcquisitionProfile,
    upstream_assessments: list[CandidateAssessment],
    upstream_selected: list[SelectedCorpusWork],
    policy: CorpusQualityPolicy,
    quality_gate_id: str,
    progress_callback=None,
) -> tuple[
    list[CorpusQualityAssessment],
    list[SelectedCorpusWork],
    CorpusSelectionReport,
    CorpusQualityGateReport,
]:
    work_map = {row.work_id: row for row in packet.works}
    upstream_map = {
        row.work_id: row for row in upstream_assessments
    }
    if len(upstream_map) != len(upstream_assessments):
        raise ValueError("Duplicate upstream CandidateAssessment work_id")
    selected_ids = {
        row.work_id for row in upstream_selected
    }

    quality_rows: list[CorpusQualityAssessment] = []
    adjusted: list[CandidateAssessment] = []
    total = len(upstream_assessments)

    for index, upstream in enumerate(
        upstream_assessments,
        start=1,
    ):
        work = work_map.get(upstream.work_id)
        if work is None:
            raise ValueError(
                f"Candidate missing from catalog: {upstream.work_id}"
            )
        quality = assess_corpus_quality(
            work=work,
            upstream=upstream,
            policy=policy,
            originally_selected=(
                upstream.work_id in selected_ids
            ),
        )
        quality_rows.append(quality)

        auto_allowed = (
            quality.status == "pass"
            or (
                quality.status == "manual_review"
                and policy.allow_manual_review_for_auto_selection
            )
        )
        adjusted.append(
            upstream.model_copy(
                update={
                    "eligibility_status": (
                        "eligible"
                        if auto_allowed
                        else "excluded"
                    ),
                    "exclusion_reasons": (
                        []
                        if auto_allowed
                        else sorted(
                            set(
                                [
                                    *upstream.exclusion_reasons,
                                    *[
                                        "quality_gate:"
                                        + reason
                                        for reason in quality.reasons
                                    ],
                                ]
                            )
                        )
                    ),
                }
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "m2_1_quality",
                    "current": index,
                    "total": total,
                    "work_id": work.work_id,
                    "title": work.title,
                    "status": quality.status,
                    "originally_selected": quality.originally_selected,
                    "reasons": quality.reasons,
                }
            )

    final_selected, final_selection_report = select_candidates(
        packet=packet,
        profile=profile,
        assessments=adjusted,
        progress_callback=(
            (
                lambda event: progress_callback(
                    {
                        **event,
                        "stage": "m2_1_select",
                    }
                )
            )
            if progress_callback is not None
            else None
        ),
    )

    final_ids = {row.work_id for row in final_selected}
    retained = selected_ids & final_ids
    dropped = selected_ids - final_ids
    replacements = final_ids - selected_ids
    status_counts = Counter(row.status for row in quality_rows)
    reason_counts = Counter(
        reason
        for row in quality_rows
        for reason in row.reasons
    )

    report = CorpusQualityGateReport(
        quality_gate_id=quality_gate_id,
        policy_id=policy.policy_id,
        profile_id=profile.profile_id,
        source_catalog_id=packet.catalog_id,
        candidate_count=len(quality_rows),
        upstream_eligible_count=sum(
            row.eligibility_status == "eligible"
            for row in upstream_assessments
        ),
        quality_pass_count=status_counts["pass"],
        quality_manual_review_count=status_counts["manual_review"],
        quality_exclude_count=status_counts["exclude"],
        original_selected_count=len(selected_ids),
        retained_original_selected_count=len(retained),
        dropped_original_selected_count=len(dropped),
        replacement_selected_count=len(replacements),
        final_selected_count=len(final_selected),
        target_total=profile.selection.target_total,
        reason_counts={
            key: reason_counts[key]
            for key in sorted(reason_counts)
        },
        final_unfilled_axis_quotas=dict(
            final_selection_report.unfilled_axis_quotas
        ),
        final_selected_work_ids=[
            row.work_id for row in final_selected
        ],
        positive_evidence_promotion_performed=False,
    )
    return (
        quality_rows,
        final_selected,
        final_selection_report,
        report,
    )
