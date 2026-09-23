from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyReport,
    HypothesisNoveltyDepthProfile,
    HypothesisSearchCoverage,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _project_claim_review(
    row: ClaimPriorArtReview,
    *,
    candidate_hypothesis_id: str,
    final_hypothesis_id: str,
) -> ClaimPriorArtReview:
    if row.hypothesis_id != candidate_hypothesis_id:
        raise ValueError(
            "source claim review hypothesis namespace does not match "
            "candidate lineage"
        )
    return row.model_copy(
        update={"hypothesis_id": final_hypothesis_id},
        deep=True,
    )


def _project_coverage(
    row: HypothesisSearchCoverage,
    *,
    candidate_hypothesis_id: str,
    final_hypothesis_id: str,
) -> HypothesisSearchCoverage:
    if row.hypothesis_id != candidate_hypothesis_id:
        raise ValueError(
            "source search coverage hypothesis namespace does not match "
            "candidate lineage"
        )
    return row.model_copy(
        update={"hypothesis_id": final_hypothesis_id},
        deep=True,
    )


def _project_depth_profile(
    row: HypothesisNoveltyDepthProfile | None,
    *,
    candidate_hypothesis_id: str,
    final_hypothesis_id: str,
) -> HypothesisNoveltyDepthProfile | None:
    if row is None:
        return None
    if row.hypothesis_id != candidate_hypothesis_id:
        raise ValueError(
            "source novelty-depth hypothesis namespace does not match "
            "candidate lineage"
        )
    return row.model_copy(
        update={"hypothesis_id": final_hypothesis_id},
        deep=True,
    )


def project_external_novelty_card_namespace(
    card: ExternalNoveltyCard,
    *,
    candidate_hypothesis_id: str,
    final_hypothesis_id: str,
) -> ExternalNoveltyCard:
    """Retarget only hypothesis namespace fields.

    Scientific content, prior-art relationships, coverage counts, status,
    policy-facing fields, evidence IDs, and interpretations are preserved.
    """
    if card.hypothesis_id != candidate_hypothesis_id:
        raise ValueError(
            "source external novelty card does not match candidate lineage"
        )

    return card.model_copy(
        update={
            "hypothesis_id": final_hypothesis_id,
            "claim_reviews": [
                _project_claim_review(
                    row,
                    candidate_hypothesis_id=candidate_hypothesis_id,
                    final_hypothesis_id=final_hypothesis_id,
                )
                for row in card.claim_reviews
            ],
            "coverage": _project_coverage(
                card.coverage,
                candidate_hypothesis_id=candidate_hypothesis_id,
                final_hypothesis_id=final_hypothesis_id,
            ),
            "novelty_depth_profile": _project_depth_profile(
                card.novelty_depth_profile,
                candidate_hypothesis_id=candidate_hypothesis_id,
                final_hypothesis_id=final_hypothesis_id,
            ),
        },
        deep=True,
    )


def _restore_candidate_namespace(
    card: ExternalNoveltyCard,
    *,
    candidate_hypothesis_id: str,
) -> dict:
    """Canonical comparison view used to prove namespace-only mutation."""
    payload = card.model_dump(mode="json")
    payload["hypothesis_id"] = candidate_hypothesis_id
    payload["coverage"]["hypothesis_id"] = candidate_hypothesis_id

    for row in payload.get("claim_reviews", []):
        row["hypothesis_id"] = candidate_hypothesis_id

    depth = payload.get("novelty_depth_profile")
    if depth is not None:
        depth["hypothesis_id"] = candidate_hypothesis_id

    return payload


class ExternalNoveltyLineageProjectionAudit(StrictModel):
    schema_version: Literal[
        "external-novelty-lineage-projection-audit-v1"
    ] = "external-novelty-lineage-projection-audit-v1"

    audit_id: str
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_external_novelty_report_id: str
    source_external_novelty_report_sha256: str

    candidate_hypothesis_id: str
    final_hypothesis_id: str

    projected_external_novelty_report_id: str
    projected_external_novelty_report_sha256: str

    source_status: str
    source_search_coverage_sufficient: bool

    candidate_final_authority_equivalence_verified: Literal[True] = True
    source_card_payload_unchanged_except_hypothesis_namespace: Literal[
        True
    ] = True
    external_novelty_reassessed: Literal[False] = False
    literature_search_rerun: Literal[False] = False
    scientific_content_added: Literal[False] = False
    evidence_relationships_changed: Literal[False] = False
    coverage_counts_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_hash(self) -> "ExternalNoveltyLineageProjectionAudit":
        body = self.model_dump(mode="json")
        observed_id = body.pop("audit_id")
        observed_sha = body.pop("audit_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("external novelty lineage audit SHA mismatch")
        if observed_id != (
            "external_novelty_lineage_projection_audit:"
            + expected_sha[:20]
        ):
            raise ValueError("external novelty lineage audit ID mismatch")
        return self


def project_external_novelty_report_to_final_hypothesis(
    *,
    plan: RelationalAtomicBindingPlan,
    source_report: ExternalNoveltyReport,
    final_hypothesis_id: str,
) -> tuple[
    ExternalNoveltyReport,
    ExternalNoveltyLineageProjectionAudit,
]:
    lineage_rows = [
        row
        for row in plan.hypotheses
        if row.final_hypothesis_id == final_hypothesis_id
    ]
    if len(lineage_rows) != 1:
        raise ValueError(
            "binding plan must resolve exactly one final hypothesis lineage"
        )
    lineage = lineage_rows[0]

    if lineage.candidate_final_authority_equivalent is not True:
        raise ValueError(
            "candidate/final authority equivalence was not verified"
        )

    candidate_hypothesis_id = lineage.candidate_hypothesis_id
    matching_cards = [
        row
        for row in source_report.cards
        if row.hypothesis_id == candidate_hypothesis_id
    ]
    if len(matching_cards) != 1:
        raise ValueError(
            "source external novelty report must contain exactly one card "
            "for the candidate hypothesis"
        )
    source_card = matching_cards[0]

    projected_card = project_external_novelty_card_namespace(
        source_card,
        candidate_hypothesis_id=candidate_hypothesis_id,
        final_hypothesis_id=final_hypothesis_id,
    )

    if _restore_candidate_namespace(
        projected_card,
        candidate_hypothesis_id=candidate_hypothesis_id,
    ) != source_card.model_dump(mode="json"):
        raise ValueError(
            "external novelty projection changed non-namespace card content"
        )

    counts = Counter([projected_card.status])

    report_body = {
        "schema_version": source_report.schema_version,
        # Preserve original acquisition provenance. The report is a
        # namespace compatibility view, not a new search run.
        "source_portfolio_id": source_report.source_portfolio_id,
        "source_prior_art_packet_id": source_report.source_prior_art_packet_id,
        "searched_at_utc": source_report.searched_at_utc,
        "cards": [projected_card.model_dump(mode="json")],
        "status_counts": dict(sorted(counts.items())),
        "policy": source_report.policy.model_dump(mode="json"),
        "external_novelty_claim_scope":
            source_report.external_novelty_claim_scope,
        "epistemic_usage": source_report.epistemic_usage,
    }
    projected_sha = _sha256_json(report_body)
    projected_report = ExternalNoveltyReport(
        **report_body,
        report_id="external_novelty_report:" + projected_sha[:20],
        report_sha256=projected_sha,
    )

    audit_body = {
        "schema_version":
            "external-novelty-lineage-projection-audit-v1",
        "source_binding_plan_id": plan.plan_id,
        "source_binding_plan_sha256": plan.plan_sha256,
        "source_external_novelty_report_id": source_report.report_id,
        "source_external_novelty_report_sha256":
            source_report.report_sha256,
        "candidate_hypothesis_id": candidate_hypothesis_id,
        "final_hypothesis_id": final_hypothesis_id,
        "projected_external_novelty_report_id":
            projected_report.report_id,
        "projected_external_novelty_report_sha256":
            projected_report.report_sha256,
        "source_status": source_card.status,
        "source_search_coverage_sufficient":
            source_card.coverage.sufficient_for_absence_based_novelty,
        "candidate_final_authority_equivalence_verified": True,
        "source_card_payload_unchanged_except_hypothesis_namespace": True,
        "external_novelty_reassessed": False,
        "literature_search_rerun": False,
        "scientific_content_added": False,
        "evidence_relationships_changed": False,
        "coverage_counts_changed": False,
        "production_selection_changed": False,
    }
    audit_sha = _sha256_json(audit_body)
    audit = ExternalNoveltyLineageProjectionAudit(
        **audit_body,
        audit_id=(
            "external_novelty_lineage_projection_audit:"
            + audit_sha[:20]
        ),
        audit_sha256=audit_sha,
    )

    return projected_report, audit


__all__ = [
    "ExternalNoveltyLineageProjectionAudit",
    "project_external_novelty_card_namespace",
    "project_external_novelty_report_to_final_hypothesis",
]
