from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.prior_art_coverage_probe import (
    PreReviewCoverageReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


DownstreamGateDecision = Literal[
    "CONTINUE_DOWNSTREAM",
    "HOLD_FOR_EVIDENCE",
]


class ExternalNoveltyDownstreamDisposition(StrictModel):
    schema_version: Literal[
        "external-novelty-downstream-disposition-v1"
    ] = "external-novelty-downstream-disposition-v1"

    hypothesis_id: str
    external_status: str
    pre_review_gate_decision: Literal[
        "KEEP_FOR_CLAIM_REVIEW",
        "EVIDENCE_REQUIRED",
    ]

    decision: DownstreamGateDecision
    reason_codes: list[str] = Field(default_factory=list)

    scientific_distinctiveness_authorized: bool
    semantic_distinctiveness_authorized: bool
    n10_review_authorized: bool

    evidence_acquisition_required: bool

    novelty_authority: Literal[False] = False
    selection_class_assigned: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_disposition(
        self,
    ) -> "ExternalNoveltyDownstreamDisposition":
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id is required")

        if self.decision == "HOLD_FOR_EVIDENCE":
            if self.scientific_distinctiveness_authorized:
                raise ValueError(
                    "held hypothesis cannot authorize scientific distinctiveness"
                )
            if self.semantic_distinctiveness_authorized:
                raise ValueError(
                    "held hypothesis cannot authorize semantic distinctiveness"
                )
            if self.n10_review_authorized:
                raise ValueError(
                    "held hypothesis cannot authorize N10 review"
                )
            if not self.evidence_acquisition_required:
                raise ValueError(
                    "held hypothesis must require evidence acquisition"
                )
        else:
            if not self.scientific_distinctiveness_authorized:
                raise ValueError(
                    "continued hypothesis must authorize scientific distinctiveness"
                )
            if not self.semantic_distinctiveness_authorized:
                raise ValueError(
                    "continued hypothesis must authorize semantic distinctiveness"
                )
            if not self.n10_review_authorized:
                raise ValueError(
                    "continued hypothesis must authorize N10 review"
                )
            if self.evidence_acquisition_required:
                raise ValueError(
                    "continued hypothesis cannot be marked evidence-acquisition-only"
                )

        return self


class ExternalNoveltyDownstreamGateReport(StrictModel):
    schema_version: Literal[
        "external-novelty-downstream-gate-report-v1"
    ] = "external-novelty-downstream-gate-report-v1"

    source_external_report_id: str
    source_external_report_sha256: str
    source_pre_review_query_plan_id: str
    source_pre_review_prior_art_packet_id: str

    dispositions: list[
        ExternalNoveltyDownstreamDisposition
    ] = Field(default_factory=list)

    continue_count: int = Field(ge=0)
    hold_for_evidence_count: int = Field(ge=0)

    novelty_authority: Literal[False] = False
    selection_class_assigned: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ExternalNoveltyDownstreamGateReport":
        ids = [
            row.hypothesis_id
            for row in self.dispositions
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate hypothesis disposition"
            )

        continue_count = sum(
            row.decision == "CONTINUE_DOWNSTREAM"
            for row in self.dispositions
        )
        hold_count = sum(
            row.decision == "HOLD_FOR_EVIDENCE"
            for row in self.dispositions
        )

        if continue_count != self.continue_count:
            raise ValueError(
                "continue_count mismatch"
            )
        if hold_count != self.hold_for_evidence_count:
            raise ValueError(
                "hold_for_evidence_count mismatch"
            )

        return self


class ExternalNoveltyDownstreamGate:
    """
    Deterministic orchestration gate after external prior-art review.

    This gate does not infer scientific quality, novelty, or N10 eligibility.
    It only prevents expensive downstream distinctiveness/N10 work when:

      1. the pre-review evidence probe was insufficient,
      2. the final external card remained INSUFFICIENT_SEARCH_EVIDENCE,
      3. the final coverage remains insufficient for absence-based novelty,
      4. the external reason explicitly records insufficient absence coverage.

    Strong positive/conflicting prior-art outcomes are never suppressed merely
    because the pre-review coverage probe was insufficient.
    """

    _COVERAGE_REASON = (
        "insufficient_coverage_for_absence_based_status"
    )

    def build(
        self,
        *,
        external_report: ExternalNoveltyReport,
        pre_review_report: PreReviewCoverageReport,
    ) -> ExternalNoveltyDownstreamGateReport:
        external_by_id = {
            row.hypothesis_id: row
            for row in external_report.cards
        }
        pre_by_id = {
            row.hypothesis_id: row
            for row in pre_review_report.hypotheses
        }

        if len(external_by_id) != len(external_report.cards):
            raise ValueError(
                "duplicate hypothesis_id in external report"
            )
        if len(pre_by_id) != len(pre_review_report.hypotheses):
            raise ValueError(
                "duplicate hypothesis_id in pre-review report"
            )

        if set(external_by_id) != set(pre_by_id):
            missing_pre = sorted(
                set(external_by_id) - set(pre_by_id)
            )
            missing_external = sorted(
                set(pre_by_id) - set(external_by_id)
            )
            raise ValueError(
                "external/pre-review hypothesis set mismatch: "
                f"missing_pre={missing_pre}, "
                f"missing_external={missing_external}"
            )

        dispositions: list[
            ExternalNoveltyDownstreamDisposition
        ] = []

        for hypothesis_id in [
            row.hypothesis_id
            for row in external_report.cards
        ]:
            card = external_by_id[hypothesis_id]
            probe = pre_by_id[hypothesis_id]

            if card.coverage != probe.coverage:
                raise ValueError(
                    "external/pre-review coverage mismatch for "
                    f"{hypothesis_id}"
                )

            coverage_insufficient = (
                not card.coverage
                .sufficient_for_absence_based_novelty
            )
            reason_present = (
                self._COVERAGE_REASON
                in set(card.reason_codes)
            )

            hold = bool(
                probe.gate_decision
                == "EVIDENCE_REQUIRED"
                and card.status
                == "INSUFFICIENT_SEARCH_EVIDENCE"
                and coverage_insufficient
                and reason_present
            )

            if hold:
                dispositions.append(
                    ExternalNoveltyDownstreamDisposition(
                        hypothesis_id=hypothesis_id,
                        external_status=card.status,
                        pre_review_gate_decision=(
                            probe.gate_decision
                        ),
                        decision="HOLD_FOR_EVIDENCE",
                        reason_codes=[
                            "pre_review_evidence_insufficient",
                            "external_status_insufficient_search_evidence",
                            self._COVERAGE_REASON,
                            "semantic_and_n10_deferred_pending_evidence",
                        ],
                        scientific_distinctiveness_authorized=False,
                        semantic_distinctiveness_authorized=False,
                        n10_review_authorized=False,
                        evidence_acquisition_required=True,
                    )
                )
                continue

            continue_reasons = [
                "preserve_existing_downstream_path"
            ]

            if (
                probe.gate_decision
                == "EVIDENCE_REQUIRED"
                and card.status
                != "INSUFFICIENT_SEARCH_EVIDENCE"
            ):
                continue_reasons.append(
                    "reviewer_found_actionable_positive_or_conflicting_prior_art"
                )
            elif (
                card.status
                == "INSUFFICIENT_SEARCH_EVIDENCE"
                and not reason_present
            ):
                continue_reasons.append(
                    "insufficient_status_not_caused_by_absence_coverage_gate"
                )

            dispositions.append(
                ExternalNoveltyDownstreamDisposition(
                    hypothesis_id=hypothesis_id,
                    external_status=card.status,
                    pre_review_gate_decision=(
                        probe.gate_decision
                    ),
                    decision="CONTINUE_DOWNSTREAM",
                    reason_codes=continue_reasons,
                    scientific_distinctiveness_authorized=True,
                    semantic_distinctiveness_authorized=True,
                    n10_review_authorized=True,
                    evidence_acquisition_required=False,
                )
            )

        return ExternalNoveltyDownstreamGateReport(
            source_external_report_id=(
                external_report.report_id
            ),
            source_external_report_sha256=(
                external_report.report_sha256
            ),
            source_pre_review_query_plan_id=(
                pre_review_report.source_query_plan_id
            ),
            source_pre_review_prior_art_packet_id=(
                pre_review_report.source_prior_art_packet_id
            ),
            dispositions=dispositions,
            continue_count=sum(
                row.decision == "CONTINUE_DOWNSTREAM"
                for row in dispositions
            ),
            hold_for_evidence_count=sum(
                row.decision == "HOLD_FOR_EVIDENCE"
                for row in dispositions
            ),
        )
