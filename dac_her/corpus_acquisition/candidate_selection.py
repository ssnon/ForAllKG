from __future__ import annotations

import re
from collections import defaultdict

from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    CandidateAssessment,
    CorpusSelectionReport,
    SelectedCorpusWork,
)
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def _norm(value: str | None) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def _term_match(
    normalized_text: str,
    term: str,
) -> bool:
    target = _norm(term)
    if not target:
        return False
    parts = target.split()
    if len(parts) == 1 and len(parts[0]) <= 3:
        return parts[0] in set(
            normalized_text.split()
        )
    return target in normalized_text


def _matching_terms(
    normalized_text: str,
    terms: list[str],
) -> list[str]:
    return sorted(
        {
            term
            for term in terms
            if _term_match(normalized_text, term)
        },
        key=lambda value: value.casefold(),
    )


def _axis_map(
    profile: AcquisitionProfile,
) -> dict[str, AcquisitionAxis]:
    return {
        axis.axis_id: axis
        for axis in profile.axes
    }


def assess_candidate(
    work: CatalogWork,
    profile: AcquisitionProfile,
) -> CandidateAssessment:
    title = _norm(work.title)
    abstract = _norm(work.abstract)
    combined = " ".join(
        value
        for value in (title, abstract)
        if value
    )
    exclusion_reasons: list[str] = []

    for group_index, group in enumerate(
        profile.scope.required_term_groups,
        start=1,
    ):
        if not _matching_terms(combined, group):
            exclusion_reasons.append(
                f"missing_required_term_group:{group_index}"
            )

    excluded_title = _matching_terms(
        title,
        profile.scope.excluded_title_terms,
    )
    if excluded_title:
        exclusion_reasons.append(
            "excluded_title_term:"
            + ",".join(excluded_title)
        )

    excluded_types = {
        _norm(value)
        for value in (
            profile.scope.excluded_publication_types
        )
    }
    observed_types = {
        _norm(value)
        for value in work.publication_types
    }
    type_overlap = sorted(
        value
        for value in (
            excluded_types & observed_types
        )
        if value
    )
    if type_overlap:
        exclusion_reasons.append(
            "excluded_publication_type:"
            + ",".join(type_overlap)
        )

    if (
        profile.scope.min_year is not None
        and work.year is not None
        and work.year < profile.scope.min_year
    ):
        exclusion_reasons.append(
            "before_min_year"
        )
    if (
        profile.scope.max_year is not None
        and work.year is not None
        and work.year > profile.scope.max_year
    ):
        exclusion_reasons.append(
            "after_max_year"
        )

    if (
        profile.scope.require_abstract
        and not work.abstract
    ):
        exclusion_reasons.append(
            "abstract_required"
        )

    axes = _axis_map(profile)
    matched_terms_by_axis: dict[
        str,
        list[str],
    ] = {}
    # Keep discovery provenance separate from content-grounded axis evidence.
    # A work being retrieved by an axis query is useful ranking provenance,
    # but it must not satisfy require_axis_match or consume that axis quota
    # unless the title/abstract actually contains an axis indicator.
    matched_axes: set[str] = set()
    for axis in profile.axes:
        hits = _matching_terms(
            combined,
            axis.indicators,
        )
        if hits:
            matched_axes.add(axis.axis_id)
            matched_terms_by_axis[
                axis.axis_id
            ] = hits

    if (
        profile.scope.require_axis_match
        and not matched_axes
    ):
        exclusion_reasons.append(
            "no_acquisition_axis_match"
        )

    if exclusion_reasons:
        status = "excluded"
    elif (
        not work.abstract
        and profile.scope.manual_review_if_no_abstract
    ):
        status = "manual_review"
    else:
        status = "eligible"

    score_components: dict[str, float] = {}

    if work.open_access_url:
        score_components["open_access"] = (
            profile.scoring.open_access_bonus
        )
    if work.abstract:
        score_components["abstract_available"] = (
            profile.scoring.abstract_available_bonus
        )

    retrieval_matches = sorted(
        set(work.retrieval_axis_ids) & set(axes)
    )
    if retrieval_matches:
        score_components["retrieval_axis"] = (
            profile.scoring.retrieval_axis_bonus
        )

    axis_bonus_total = 0.0
    for axis_id in sorted(matched_axes):
        axis = axes[axis_id]
        if matched_terms_by_axis.get(axis_id):
            axis_bonus_total += (
                profile.scoring.axis_indicator_bonus
                * axis.weight
            )
    if (
        profile.scoring.max_axis_bonus
        is not None
    ):
        axis_bonus_total = min(
            axis_bonus_total,
            profile.scoring.max_axis_bonus,
        )
    if axis_bonus_total:
        score_components["axis_indicators"] = (
            axis_bonus_total
        )

    for signal in profile.scoring.signals:
        hits = _matching_terms(
            combined,
            signal.terms,
        )
        matched = (
            bool(hits)
            if signal.match_mode == "any"
            else len(hits) == len(
                set(signal.terms)
            )
        )
        if matched:
            score_components[
                f"signal:{signal.signal_id}"
            ] = signal.weight

    if work.citation_count is not None:
        for rule in sorted(
            profile.scoring.citation_bonuses,
            key=lambda row: row.min_citations,
        ):
            if (
                work.citation_count
                >= rule.min_citations
            ):
                score_components[
                    (
                        "citation_bonus:"
                        f"{rule.min_citations}"
                    )
                ] = rule.bonus

    total_score = round(
        sum(score_components.values()),
        6,
    )

    return CandidateAssessment(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        year=work.year,
        eligibility_status=status,
        exclusion_reasons=sorted(
            set(exclusion_reasons)
        ),
        matched_axes=sorted(matched_axes),
        matched_terms_by_axis={
            key: value
            for key, value in sorted(
                matched_terms_by_axis.items()
            )
        },
        score_components={
            key: score_components[key]
            for key in sorted(score_components)
        },
        total_score=total_score,
        open_access_available=bool(
            work.open_access_url
        ),
        abstract_available=bool(work.abstract),
        scientific_result_direction_inferred=False,
    )


def assess_catalog(
    packet: LiteratureCatalogPacket,
    profile: AcquisitionProfile,
    *,
    progress_callback=None,
) -> list[CandidateAssessment]:
    if (
        packet.acquisition_profile_id
        != profile.profile_id
    ):
        raise ValueError(
            "Catalog/profile mismatch: "
            f"{packet.acquisition_profile_id!r} != "
            f"{profile.profile_id!r}"
        )
    total = len(packet.works)
    rows: list[CandidateAssessment] = []
    for index, work in enumerate(packet.works, start=1):
        row = assess_candidate(work, profile)
        rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "m2_assess",
                    "event": "complete",
                    "current": index,
                    "total": total,
                    "work_id": work.work_id,
                    "title": work.title,
                    "eligibility_status": row.eligibility_status,
                    "total_score": row.total_score,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -row.total_score,
            row.title.casefold(),
            row.work_id,
        ),
    )


def select_candidates(
    *,
    packet: LiteratureCatalogPacket,
    profile: AcquisitionProfile,
    assessments: list[CandidateAssessment],
    progress_callback=None,
) -> tuple[
    list[SelectedCorpusWork],
    CorpusSelectionReport,
]:
    works = {
        work.work_id: work
        for work in packet.works
    }
    assessment_map = {
        row.work_id: row
        for row in assessments
    }
    if len(assessment_map) != len(assessments):
        raise ValueError(
            "Duplicate CandidateAssessment work_id"
        )

    allowed_status = {"eligible"}
    if profile.selection.include_manual_review:
        allowed_status.add("manual_review")

    eligible = [
        row
        for row in assessments
        if row.eligibility_status
        in allowed_status
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row.total_score,
            -(works[row.work_id].citation_count or 0),
            row.title.casefold(),
            row.work_id,
        ),
    )

    axis_candidate_counts = {
        axis.axis_id: sum(
            axis.axis_id in row.matched_axes
            for row in eligible
        )
        for axis in profile.axes
    }
    axis_targets = {
        axis.axis_id: axis.target_selected
        for axis in profile.axes
    }

    # Scarce axes are filled first. Each selected work is charged to at most
    # one primary quota axis; matched_axes still records all coverage.
    axis_order = sorted(
        profile.axes,
        key=lambda axis: (
            (
                axis_candidate_counts[axis.axis_id]
                / axis.target_selected
            )
            if axis.target_selected > 0
            else float("inf"),
            axis_candidate_counts[axis.axis_id],
            axis.axis_id,
        ),
    )

    selected_ids: set[str] = set()
    primary_axis: dict[str, str | None] = {}
    selection_target = min(
        profile.selection.target_total,
        len(ranked),
    )

    def emit_selection_progress(
        *,
        work_id: str,
        axis_id: str | None,
        phase: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "stage": "m2_select",
                "event": "selected",
                "current": len(selected_ids),
                "total": selection_target,
                "work_id": work_id,
                "primary_quota_axis": axis_id,
                "phase": phase,
            }
        )

    primary_counts: dict[str, int] = {
        axis.axis_id: 0
        for axis in profile.axes
    }

    for axis in axis_order:
        target = axis.target_selected
        if target <= 0:
            continue
        for row in ranked:
            if (
                primary_counts[axis.axis_id]
                >= target
            ):
                break
            if row.work_id in selected_ids:
                continue
            if axis.axis_id not in row.matched_axes:
                continue
            selected_ids.add(row.work_id)
            primary_axis[row.work_id] = (
                axis.axis_id
            )
            primary_counts[axis.axis_id] += 1
            emit_selection_progress(
                work_id=row.work_id,
                axis_id=axis.axis_id,
                phase="quota",
            )
            if (
                len(selected_ids)
                >= profile.selection.target_total
            ):
                break
        if (
            len(selected_ids)
            >= profile.selection.target_total
        ):
            break

    # Fill any remaining global capacity from the highest scoring eligible
    # works, without pretending they satisfy an unfilled quota.
    for row in ranked:
        if (
            len(selected_ids)
            >= profile.selection.target_total
        ):
            break
        if row.work_id in selected_ids:
            continue
        selected_ids.add(row.work_id)
        primary_axis[row.work_id] = None
        emit_selection_progress(
            work_id=row.work_id,
            axis_id=None,
            phase="global_fill",
        )

    selected: list[SelectedCorpusWork] = []
    for row in ranked:
        if row.work_id not in selected_ids:
            continue
        work = works[row.work_id]
        selected.append(
            SelectedCorpusWork(
                work_id=work.work_id,
                title=work.title,
                doi=work.doi,
                year=work.year,
                venue=work.venue,
                open_access_url=(
                    work.open_access_url
                ),
                matched_axes=row.matched_axes,
                primary_quota_axis=(
                    primary_axis[row.work_id]
                ),
                total_score=row.total_score,
            )
        )

    unfilled = {
        axis_id: max(
            0,
            axis_targets[axis_id]
            - primary_counts[axis_id],
        )
        for axis_id in sorted(axis_targets)
        if (
            axis_targets[axis_id]
            - primary_counts[axis_id]
        ) > 0
    }

    report = CorpusSelectionReport(
        profile_id=profile.profile_id,
        source_catalog_id=packet.catalog_id,
        candidate_count=len(assessments),
        eligible_count=sum(
            row.eligibility_status == "eligible"
            for row in assessments
        ),
        manual_review_count=sum(
            row.eligibility_status
            == "manual_review"
            for row in assessments
        ),
        excluded_count=sum(
            row.eligibility_status == "excluded"
            for row in assessments
        ),
        selected_count=len(selected),
        target_total=profile.selection.target_total,
        axis_candidate_counts={
            key: axis_candidate_counts[key]
            for key in sorted(
                axis_candidate_counts
            )
        },
        axis_quota_targets={
            key: axis_targets[key]
            for key in sorted(axis_targets)
        },
        axis_primary_selected_counts={
            key: primary_counts[key]
            for key in sorted(primary_counts)
        },
        unfilled_axis_quotas=unfilled,
        selected_work_ids=[
            row.work_id
            for row in selected
        ],
        policy_notes=[
            (
                "Selection is metadata/abstract based and does not infer "
                "scientific result direction, causality, or effect sign."
            ),
            (
                "matched_axes records only title/abstract indicator matches; "
                "retrieval_axis_ids remains discovery provenance and may "
                "contribute only the configured retrieval ranking bonus."
            ),
            (
                "One work may match multiple evidence-grounded axes, but it "
                "is charged to at most one primary quota axis to preserve "
                "corpus diversity."
            ),
            (
                "Selected metadata is not positive KG evidence. Promotion "
                "requires later source acquisition, materialization, and "
                "existing extraction/provenance gates."
            ),
        ],
        positive_evidence_promotion_performed=False,
    )
    return selected, report
