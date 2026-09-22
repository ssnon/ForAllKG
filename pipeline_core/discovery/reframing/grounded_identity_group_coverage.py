from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotation,
    GroundedIdentityAnnotationReport,
    GroundedIdentityConstituentGroup,
    _group_matches_abstract,
    _material_work_ids,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroupSpanCoverage(StrictModel):
    group_label: str
    source_spans: list[str]
    matching_material_work_ids: list[str]
    matching_material_work_count: int = Field(ge=0)


class SlotGroupCoverage(StrictModel):
    slot: str
    material_abstract_work_ids: list[str]
    material_abstract_work_count: int = Field(ge=0)
    groups: list[GroupSpanCoverage]
    all_groups_matching_work_ids: list[str]
    all_groups_matching_work_count: int = Field(ge=0)
    group_match_signature_counts: dict[str, int]


class ClaimGroupCoverage(StrictModel):
    claim_id: str
    identity_term: str
    slots: list[SlotGroupCoverage]


class GroundedIdentityGroupCoverageReport(StrictModel):
    schema_version: Literal[
        "grounded-identity-group-coverage-report-v1"
    ] = "grounded-identity-group-coverage-report-v1"

    source_atomic_synthesis_report_id: str
    source_annotation_report_id: str | None = None
    claim_count: int = Field(ge=0)
    claims: list[ClaimGroupCoverage]

    current_claim_ids_only: Literal[True] = True
    stale_detail_directories_ignored: Literal[True] = True
    diagnostic_only: Literal[True] = True
    retrieval_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False


def _group_matching_ids(
    *,
    group: GroundedIdentityConstituentGroup,
    material_ids: set[str],
    works_by_id: Mapping[str, Mapping[str, object]],
) -> set[str]:
    return {
        work_id
        for work_id in material_ids
        if (
            work_id in works_by_id
            and _group_matches_abstract(
                abstract=str(
                    works_by_id[work_id].get("abstract") or ""
                ),
                group=group,
            )
        )
    }


def analyze_slot_group_coverage(
    *,
    slot_review: Mapping[str, object],
    annotation: GroundedIdentityAnnotation,
    works_by_id: Mapping[str, Mapping[str, object]],
) -> SlotGroupCoverage:
    material_ids = _material_work_ids(
        slot_review=slot_review,
    )

    group_rows: list[GroupSpanCoverage] = []
    group_sets: list[set[str]] = []

    for group in annotation.groups:
        matched = _group_matching_ids(
            group=group,
            material_ids=material_ids,
            works_by_id=works_by_id,
        )
        group_sets.append(matched)
        group_rows.append(
            GroupSpanCoverage(
                group_label=group.label,
                source_spans=[
                    span.exact_source_text
                    for span in group.spans
                ],
                matching_material_work_ids=sorted(matched),
                matching_material_work_count=len(matched),
            )
        )

    all_groups = (
        set.intersection(*group_sets)
        if group_sets
        else set()
    )

    signatures: Counter[str] = Counter()
    for work_id in sorted(material_ids):
        matched_labels = [
            group.label
            for group, matched
            in zip(annotation.groups, group_sets)
            if work_id in matched
        ]
        signature = (
            " + ".join(matched_labels)
            if matched_labels
            else "<none>"
        )
        signatures[signature] += 1

    return SlotGroupCoverage(
        slot=str(slot_review.get("slot") or ""),
        material_abstract_work_ids=sorted(material_ids),
        material_abstract_work_count=len(material_ids),
        groups=group_rows,
        all_groups_matching_work_ids=sorted(all_groups),
        all_groups_matching_work_count=len(all_groups),
        group_match_signature_counts=dict(
            sorted(signatures.items())
        ),
    )


def build_group_coverage_report(
    *,
    detail_root: Path,
    atomic_report: AtomicCrossLaneSynthesisReport,
    annotation_report: GroundedIdentityAnnotationReport,
) -> GroundedIdentityGroupCoverageReport:
    current_claim_ids = {
        spec.claim_id
        for hypothesis in atomic_report.hypotheses
        for spec in hypothesis.atomic_specifications
    }
    annotations = {
        row.claim_id: row
        for row in annotation_report.annotations
    }

    if set(annotations) != current_claim_ids:
        raise ValueError(
            "annotation/current atomic claim membership mismatch"
        )

    claims: list[ClaimGroupCoverage] = []

    for claim_id in sorted(current_claim_ids):
        claim_dir = detail_root / claim_id.replace(":", "_")
        if not claim_dir.is_dir():
            raise ValueError(
                f"missing current claim detail directory: {claim_dir}"
            )

        reviews = json.loads(
            (claim_dir / "slot_reviews.json").read_text(
                encoding="utf-8"
            )
        )
        prior_art = json.loads(
            (claim_dir / "prior_art.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(reviews, list):
            raise ValueError("slot_reviews must be a list")
        if not isinstance(prior_art, dict):
            raise ValueError("prior_art must be an object")

        works_by_id = {
            str(row.get("work_id") or ""): row
            for row in prior_art.get("works") or []
            if isinstance(row, Mapping)
            and str(row.get("work_id") or "").strip()
        }

        annotation = annotations[claim_id]
        slot_rows = [
            analyze_slot_group_coverage(
                slot_review=review,
                annotation=annotation,
                works_by_id=works_by_id,
            )
            for review in reviews
            if str(review.get("slot") or "") != "BASE_RELATION"
        ]

        claims.append(
            ClaimGroupCoverage(
                claim_id=claim_id,
                identity_term=annotation.identity_term,
                slots=slot_rows,
            )
        )

    return GroundedIdentityGroupCoverageReport(
        source_atomic_synthesis_report_id=atomic_report.report_id,
        claim_count=len(claims),
        claims=claims,
    )


__all__ = [
    "GroundedIdentityGroupCoverageReport",
    "analyze_slot_group_coverage",
    "build_group_coverage_report",
]
