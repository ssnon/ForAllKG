from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    ExternalNoveltyReport,
)
from pipeline_core.discovery.positive_nonobviousness_adjudication import (
    PositiveNonObviousnessAdjudicationReport,
    PositiveNonObviousnessHypothesisGateDecision,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregation,
    ScientificHypothesisEvidenceAggregationReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CertificationDecision = Literal[
    "CERTIFIED",
    "UNRESOLVED",
    "REJECTED",
]

BoundedClosureState = Literal[
    "BOUNDED_REVIEW_CLOSED",
    "PARTIAL_REVIEW_COVERAGE",
    "INSUFFICIENT_METADATA_REMAINS",
]

BoundedExternalDistinctnessState = Literal[
    "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED",
    "RELATION_BACKED_EXTERNAL_PRIOR_ART",
    "CONFLICTING_EXTERNAL_PRIOR_ART",
    "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE",
    "MISSING_EXTERNAL_NOVELTY_CARD",
]

PositiveNonObviousnessAuthorityState = Literal[
    "AUTHORIZED",
    "NOT_AUTHORIZED",
]

FatalBlockerState = Literal[
    "NONE",
    "DIRECT_PRIOR_ART",
    "FATAL_CONTRADICTION",
    "DIRECT_PRIOR_ART_AND_FATAL_CONTRADICTION",
]


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


def _closure_state(
    aggregation: ScientificHypothesisEvidenceAggregation,
    report: ScientificHypothesisEvidenceAggregationReport,
) -> BoundedClosureState:
    claim_rows = [
        row
        for row in report.claim_impacts
        if row.hypothesis_id == aggregation.hypothesis_id
    ]

    if any(
        row.evidence_pressure_state == "INSUFFICIENT_METADATA"
        for row in claim_rows
    ):
        return "INSUFFICIENT_METADATA_REMAINS"

    if (
        not aggregation.all_claims_fully_classified
        or aggregation.any_unclassified_presented_work
    ):
        return "PARTIAL_REVIEW_COVERAGE"

    return "BOUNDED_REVIEW_CLOSED"


def _external_distinctness_state(
    card: ExternalNoveltyCard | None,
) -> BoundedExternalDistinctnessState:
    if card is None:
        return "MISSING_EXTERNAL_NOVELTY_CARD"

    if (
        not card.coverage.sufficient_for_absence_based_novelty
    ):
        return "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"

    if card.status in {
        "PLAUSIBLY_NOVEL",
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
    }:
        return "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"

    if card.status in {
        "WELL_ESTABLISHED",
        "LITERATURE_SUPPORTED_EXTENSION",
    }:
        return "RELATION_BACKED_EXTERNAL_PRIOR_ART"

    if card.status == "CONFLICTING_PRIOR_ART":
        return "CONFLICTING_EXTERNAL_PRIOR_ART"

    return "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"


def _fatal_blocker_state(
    decision: PositiveNonObviousnessHypothesisGateDecision,
) -> FatalBlockerState:
    direct = bool(
        decision.direct_prior_art_blocker_claim_ids
    )
    fatal = bool(
        decision.fatal_contradiction_blocker
        or decision.fatal_contradiction_claim_ids
    )

    if direct and fatal:
        return "DIRECT_PRIOR_ART_AND_FATAL_CONTRADICTION"
    if direct:
        return "DIRECT_PRIOR_ART"
    if fatal:
        return "FATAL_CONTRADICTION"
    return "NONE"


def _decision(
    *,
    closure_state: BoundedClosureState,
    external_distinctness_state: BoundedExternalDistinctnessState,
    positive_nonobviousness_authority_state: PositiveNonObviousnessAuthorityState,
    fatal_blocker_state: FatalBlockerState,
) -> tuple[CertificationDecision, list[str]]:
    reasons: list[str] = []

    # Positive blockers take precedence over all absence/coverage logic.
    if fatal_blocker_state != "NONE":
        reasons.append(
            "positive_scientific_or_prior_art_blocker_present"
        )
        if fatal_blocker_state in {
            "DIRECT_PRIOR_ART",
            "DIRECT_PRIOR_ART_AND_FATAL_CONTRADICTION",
        }:
            reasons.append(
                "direct_prior_art_blocker_present"
            )
        if fatal_blocker_state in {
            "FATAL_CONTRADICTION",
            "DIRECT_PRIOR_ART_AND_FATAL_CONTRADICTION",
        }:
            reasons.append(
                "fatal_contradiction_blocker_present"
            )
        return "REJECTED", reasons

    if closure_state != "BOUNDED_REVIEW_CLOSED":
        reasons.append(
            "bounded_evidence_review_not_closed"
        )

    if (
        external_distinctness_state
        != "BOUNDED_EXTERNAL_DISTINCTNESS_SUPPORTED"
    ):
        reasons.append(
            "bounded_external_distinctness_not_established"
        )

    if positive_nonobviousness_authority_state != "AUTHORIZED":
        reasons.append(
            "positive_nonobviousness_not_authorized"
        )

    if reasons:
        return "UNRESOLVED", reasons

    return (
        "CERTIFIED",
        [
            "bounded_review_closed",
            "bounded_external_distinctness_supported",
            "positive_nonobviousness_authorized",
            "no_fatal_or_direct_prior_art_blocker",
        ],
    )


class ScientificHypothesisCertificationDecision(StrictModel):
    hypothesis_id: str
    decision: CertificationDecision

    bounded_closure_state: BoundedClosureState
    bounded_external_distinctness_state: BoundedExternalDistinctnessState
    positive_nonobviousness_authority_state: PositiveNonObviousnessAuthorityState
    fatal_blocker_state: FatalBlockerState

    external_novelty_status: str | None = None
    external_search_coverage_sufficient: bool | None = None

    positive_nonobviousness_gate_state: str
    direct_prior_art_blocker_claim_ids: list[str]
    fatal_contradiction_claim_ids: list[str]

    reason_codes: list[str]

    # Certification here is explicitly bounded to the recorded search/review
    # universe. It is not a proof of literature-wide novelty or scientific truth.
    certification_scope: Literal[
        "SEARCH_BOUNDED_SCIENTIFIC_CERTIFICATION_NOT_LITERATURE_WIDE_PROOF"
    ] = (
        "SEARCH_BOUNDED_SCIENTIFIC_CERTIFICATION_NOT_LITERATURE_WIDE_PROOF"
    )

    literature_wide_novelty_proof: Literal[False] = False
    truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(
        self,
    ) -> "ScientificHypothesisCertificationDecision":
        recomputed, reasons = _decision(
            closure_state=self.bounded_closure_state,
            external_distinctness_state=(
                self.bounded_external_distinctness_state
            ),
            positive_nonobviousness_authority_state=(
                self.positive_nonobviousness_authority_state
            ),
            fatal_blocker_state=self.fatal_blocker_state,
        )
        if recomputed != self.decision:
            raise ValueError(
                "certification decision does not match gate state"
            )
        if reasons != self.reason_codes:
            raise ValueError(
                "certification reason codes do not match gate state"
            )
        return self


class ScientificCertificationGateReport(StrictModel):
    schema_version: Literal[
        "scientific-certification-gate-report-v1"
    ] = "scientific-certification-gate-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_hypothesis_evidence_aggregation_report_id: str
    source_positive_nonobviousness_adjudication_report_id: str
    source_external_novelty_report_id: str

    decisions: list[
        ScientificHypothesisCertificationDecision
    ]
    hypothesis_count: int = Field(ge=0)
    certified_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)

    decision_counts: dict[str, int]

    certification_semantics: Literal[
        "bounded_closure_and_external_distinctness_and_positive_nonobviousness_without_blocker_v1"
    ] = (
        "bounded_closure_and_external_distinctness_and_positive_nonobviousness_without_blocker_v1"
    )

    closure_is_bounded_review_closure_not_exhaustive_literature_closure: Literal[True] = True
    external_distinctness_is_search_bounded_not_literature_wide: Literal[True] = True
    positive_nonobviousness_requires_explicit_authority: Literal[True] = True
    absence_alone_can_never_certify: Literal[True] = True
    direct_prior_art_is_rejection_authority: Literal[True] = True
    fatal_contradiction_is_rejection_authority: Literal[True] = True

    production_authority_created: Literal[False] = False
    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ScientificCertificationGateReport":
        if self.hypothesis_count != len(self.decisions):
            raise ValueError(
                "certification hypothesis_count mismatch"
            )

        expected = defaultdict(int)
        for row in self.decisions:
            expected[row.decision] += 1

        if self.certified_count != expected["CERTIFIED"]:
            raise ValueError(
                "certification certified_count mismatch"
            )
        if self.unresolved_count != expected["UNRESOLVED"]:
            raise ValueError(
                "certification unresolved_count mismatch"
            )
        if self.rejected_count != expected["REJECTED"]:
            raise ValueError(
                "certification rejected_count mismatch"
            )
        if dict(sorted(expected.items())) != dict(
            sorted(self.decision_counts.items())
        ):
            raise ValueError(
                "certification decision_counts mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "scientific certification gate SHA mismatch"
            )
        if observed_id != (
            "scientific_certification_gate:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "scientific certification gate ID mismatch"
            )
        return self


def build_scientific_certification_gate_report(
    *,
    aggregation: ScientificHypothesisEvidenceAggregationReport,
    nonobviousness: PositiveNonObviousnessAdjudicationReport,
    external_novelty: ExternalNoveltyReport,
) -> ScientificCertificationGateReport:
    if (
        nonobviousness.source_claim_evidence_graph_id
        != aggregation.source_claim_evidence_graph_id
    ):
        raise ValueError(
            "aggregation/non-obviousness evidence-graph provenance mismatch"
        )

    agg_by_hypothesis = {
        row.hypothesis_id: row
        for row in aggregation.hypothesis_aggregations
    }
    nonobvious_by_hypothesis = {
        row.hypothesis_id: row
        for row in nonobviousness.hypothesis_decisions
    }
    external_by_hypothesis = {
        row.hypothesis_id: row
        for row in external_novelty.cards
    }

    if set(agg_by_hypothesis) != set(
        nonobvious_by_hypothesis
    ):
        raise ValueError(
            "aggregation/non-obviousness hypothesis sets differ"
        )

    decisions: list[
        ScientificHypothesisCertificationDecision
    ] = []

    for hypothesis_id in sorted(agg_by_hypothesis):
        agg = agg_by_hypothesis[hypothesis_id]
        nonob = nonobvious_by_hypothesis[hypothesis_id]
        external = external_by_hypothesis.get(hypothesis_id)

        closure = _closure_state(
            agg,
            aggregation,
        )
        distinctness = _external_distinctness_state(
            external
        )
        nonob_state: PositiveNonObviousnessAuthorityState = (
            "AUTHORIZED"
            if nonob.positive_nonobviousness_authority
            else "NOT_AUTHORIZED"
        )
        blocker = _fatal_blocker_state(nonob)

        decision, reasons = _decision(
            closure_state=closure,
            external_distinctness_state=distinctness,
            positive_nonobviousness_authority_state=nonob_state,
            fatal_blocker_state=blocker,
        )

        decisions.append(
            ScientificHypothesisCertificationDecision(
                hypothesis_id=hypothesis_id,
                decision=decision,
                bounded_closure_state=closure,
                bounded_external_distinctness_state=distinctness,
                positive_nonobviousness_authority_state=nonob_state,
                fatal_blocker_state=blocker,
                external_novelty_status=(
                    external.status
                    if external is not None
                    else None
                ),
                external_search_coverage_sufficient=(
                    external.coverage.sufficient_for_absence_based_novelty
                    if external is not None
                    else None
                ),
                positive_nonobviousness_gate_state=(
                    nonob.gate_state
                ),
                direct_prior_art_blocker_claim_ids=list(
                    nonob.direct_prior_art_blocker_claim_ids
                ),
                fatal_contradiction_claim_ids=list(
                    nonob.fatal_contradiction_claim_ids
                ),
                reason_codes=reasons,
            )
        )

    counts: dict[str, int] = defaultdict(int)
    for row in decisions:
        counts[row.decision] += 1

    body = {
        "schema_version":
            "scientific-certification-gate-report-v1",
        "source_hypothesis_evidence_aggregation_report_id":
            aggregation.report_id,
        "source_positive_nonobviousness_adjudication_report_id":
            nonobviousness.report_id,
        "source_external_novelty_report_id":
            external_novelty.report_id,
        "decisions": [
            row.model_dump(mode="json")
            for row in decisions
        ],
        "hypothesis_count":
            len(decisions),
        "certified_count":
            counts["CERTIFIED"],
        "unresolved_count":
            counts["UNRESOLVED"],
        "rejected_count":
            counts["REJECTED"],
        "decision_counts":
            dict(sorted(counts.items())),
        "certification_semantics":
            "bounded_closure_and_external_distinctness_and_positive_nonobviousness_without_blocker_v1",
        "closure_is_bounded_review_closure_not_exhaustive_literature_closure":
            True,
        "external_distinctness_is_search_bounded_not_literature_wide":
            True,
        "positive_nonobviousness_requires_explicit_authority":
            True,
        "absence_alone_can_never_certify":
            True,
        "direct_prior_art_is_rejection_authority":
            True,
        "fatal_contradiction_is_rejection_authority":
            True,
        "production_authority_created":
            False,
        "diagnostic_only":
            True,
        "n9_contract_changed":
            False,
        "n10_contract_changed":
            False,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)

    return ScientificCertificationGateReport(
        **body,
        report_id=(
            "scientific_certification_gate:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "ScientificCertificationGateReport",
    "ScientificHypothesisCertificationDecision",
    "build_scientific_certification_gate_report",
]
