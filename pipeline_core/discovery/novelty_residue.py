from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    NoveltyClaimImportance,
    NoveltyClaimScientificStructure,
)


NoveltyResidueDisposition = Literal[
    "SATURATED",
    "UNRESOLVED_PARTIAL",
    "RESIDUAL",
    "UNRESOLVED",
]


_SATURATED = {
    "DIRECT_PRIOR_ART",
}

_PARTIAL = {
    "PARTIAL_PRIOR_ART",
}

_RESIDUAL = {
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
}


def classify_prior_art_disposition(
    status: str,
) -> NoveltyResidueDisposition:
    """Map claim-level prior-art status to novelty-residue handling.

    DIRECT_PRIOR_ART:
        The atomic relation nucleus is already represented and is
        therefore removed from the novelty residue.

    PARTIAL_PRIOR_ART:
        The claim must not be silently removed. Partial overlap may
        mean that some scientific content remains unresolved, or that
        the claim itself still requires finer decomposition.

    COMPONENTS_ONLY / NO_DIRECT_MATCH_FOUND:
        The relation nucleus remains a candidate novelty residue.
        This does not imply scientific non-obviousness.

    Other statuses:
        Search/evidence resolution is insufficient for residue
        adjudication.
    """

    if status in _SATURATED:
        return "SATURATED"

    if status in _PARTIAL:
        return "UNRESOLVED_PARTIAL"

    if status in _RESIDUAL:
        return "RESIDUAL"

    return "UNRESOLVED"


@dataclass(frozen=True)
class NoveltyResidueClaim:
    hypothesis_id: str
    claim_id: str
    claim_text: str
    claim_kind: str
    prior_art_status: str

    disposition: NoveltyResidueDisposition
    is_residue: bool

    distinguishing_terms: tuple[str, ...]
    prior_art_identity_terms: tuple[str, ...]
    relation_nucleus_terms: tuple[str, ...]

    required_bridge: str
    predicted_observation: str
    falsification_condition: str

    direct_or_partial_work_ids: tuple[str, ...]
    lower_order_work_ids: tuple[str, ...]
    component_work_ids: tuple[str, ...]

    # Preserve the decomposition-time epistemic role through N9/N10.
    #
    # Legacy/manual constructors default fail-safe to core. Production
    # extraction below always supplies the upstream NoveltyClaim role
    # explicitly, so a supporting branch remains supporting while a
    # core branch can no longer silently degrade to supporting.
    importance: NoveltyClaimImportance = "core"

    # Alpha4 inference context preserved from the canonical
    # query-plan claim. Diagnostic/provenance only.
    inference_provenance: dict[str, object] | None = None

    # Diagnostic-only provenance inherited from the canonical
    # query-plan claim. It does not alter prior-art disposition.
    specification_sanitization_reason_codes: tuple[str, ...] = ()

    scientific_structure: NoveltyClaimScientificStructure = field(
        default_factory=NoveltyClaimScientificStructure
    )
    scientific_structure_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HypothesisNoveltyResidue:
    hypothesis_id: str
    external_status: str
    claims: tuple[NoveltyResidueClaim, ...]

    @property
    def residual_claims(
        self,
    ) -> tuple[NoveltyResidueClaim, ...]:
        return tuple(
            row
            for row in self.claims
            if row.disposition == "RESIDUAL"
        )

    @property
    def saturated_claims(
        self,
    ) -> tuple[NoveltyResidueClaim, ...]:
        return tuple(
            row
            for row in self.claims
            if row.disposition == "SATURATED"
        )

    @property
    def partial_claims(
        self,
    ) -> tuple[NoveltyResidueClaim, ...]:
        return tuple(
            row
            for row in self.claims
            if row.disposition
            == "UNRESOLVED_PARTIAL"
        )

    @property
    def unresolved_claims(
        self,
    ) -> tuple[NoveltyResidueClaim, ...]:
        return tuple(
            row
            for row in self.claims
            if row.disposition == "UNRESOLVED"
        )


ResidualSpecificationStatus = Literal[
    "NOT_APPLICABLE",
    "NEEDS_REFINEMENT",
    "READY_FOR_CLOSURE",
]


@dataclass(frozen=True)
class ResidualSpecificationAssessment:
    status: ResidualSpecificationStatus
    missing_fields: tuple[str, ...]
    reason_codes: tuple[str, ...]
    interpretation: str


def assess_residual_specification(
    claim: NoveltyResidueClaim,
) -> ResidualSpecificationAssessment:
    """Gate one prior-art residue before evidence-closure search.

    A search-bounded prior-art residue is not automatically eligible
    for scientific non-obviousness analysis. The original hypothesis
    must independently specify the residual branch's bridge,
    prediction, and falsifier.

    Empty fields intentionally fail closed. This function never
    invents missing scientific content.
    """

    if claim.disposition != "RESIDUAL":
        return ResidualSpecificationAssessment(
            status="NOT_APPLICABLE",
            missing_fields=(),
            reason_codes=(
                "claim_is_not_active_novelty_residue",
            ),
            interpretation=(
                "Only active novelty residues require branch-level "
                "specification before evidence-closure search."
            ),
        )

    required = {
        "required_bridge": claim.required_bridge,
        "predicted_observation": (
            claim.predicted_observation
        ),
        "falsification_condition": (
            claim.falsification_condition
        ),
    }

    missing = tuple(
        name
        for name, value in required.items()
        if not str(value or "").strip()
    )

    if missing:
        return ResidualSpecificationAssessment(
            status="NEEDS_REFINEMENT",
            missing_fields=missing,
            reason_codes=(
                "atomic_residue_under_specified",
                *tuple(
                    f"missing_{name}"
                    for name in missing
                ),
                *claim.specification_sanitization_reason_codes,
            ),
            interpretation=(
                "The atomic prior-art residue is not independently "
                "specified strongly enough for evidence-closure or "
                "non-obviousness review. Missing scientific content "
                "must not be invented by retrieval or adjudication."
            ),
        )

    return ResidualSpecificationAssessment(
        status="READY_FOR_CLOSURE",
        missing_fields=(),
        reason_codes=(
            "branch_bridge_prediction_falsifier_present",
        ),
        interpretation=(
            "The original hypothesis supplies a branch-specific "
            "bridge, prediction, and falsification condition. The "
            "residue may proceed to bounded evidence-closure search."
        ),
    )


def extract_novelty_residue(
    plan: LiteratureQueryPlan,
    report: ExternalNoveltyReport,
) -> list[HypothesisNoveltyResidue]:
    planned = {
        claim.claim_id: claim
        for group in plan.claims
        for claim in group.claims
    }

    results: list[
        HypothesisNoveltyResidue
    ] = []

    for card in report.cards:
        rows: list[
            NoveltyResidueClaim
        ] = []

        for review in card.claim_reviews:
            claim = planned.get(
                review.claim_id
            )

            if claim is None:
                raise ValueError(
                    "missing planned claim "
                    f"{review.claim_id}"
                )

            direct_or_partial = sorted({
                match.work_id
                for match in review.matches
                if match.relationship in {
                    "DIRECT_PRIOR_ART",
                    "PARTIAL_PRIOR_ART",
                }
            })

            lower_order = sorted({
                match.work_id
                for match in review.matches
                if (
                    match.relationship
                    == "LOWER_ORDER_RELATION_PRIOR_ART"
                )
            })

            components = sorted({
                match.work_id
                for match in review.matches
                if match.relationship in {
                    "COMPONENT_ONLY",
                    "CONTEXTUAL_CONFLICT",
                }
            })

            disposition = (
                classify_prior_art_disposition(
                    review.status
                )
            )

            rows.append(
                NoveltyResidueClaim(
                    hypothesis_id=(
                        card.hypothesis_id
                    ),
                    claim_id=review.claim_id,
                    claim_text=review.claim_text,
                    claim_kind=claim.kind,
                    prior_art_status=(
                        review.status
                    ),
                    disposition=disposition,
                    is_residue=(
                        disposition
                        == "RESIDUAL"
                    ),
                    distinguishing_terms=tuple(
                        claim.distinguishing_terms
                    ),
                    prior_art_identity_terms=tuple(
                        claim.prior_art_identity_terms
                    ),
                    relation_nucleus_terms=tuple(
                        claim.relation_nucleus_terms
                    ),
                    required_bridge=(
                        claim.required_bridge
                    ),
                    predicted_observation=(
                        claim.predicted_observation
                    ),
                    falsification_condition=(
                        claim.falsification_condition
                    ),
                    direct_or_partial_work_ids=(
                        tuple(
                            direct_or_partial
                        )
                    ),
                    lower_order_work_ids=tuple(
                        lower_order
                    ),
                    component_work_ids=tuple(
                        components
                    ),
                    importance=claim.importance,
                    inference_provenance=(
                        claim.inference_provenance.model_dump(
                            mode="json"
                        )
                        if claim.inference_provenance
                        is not None
                        else None
                    ),
                    specification_sanitization_reason_codes=tuple(
                        claim.specification_sanitization_reason_codes
                    ),
                    scientific_structure=(
                        claim.scientific_structure
                    ),
                    scientific_structure_reason_codes=tuple(
                        claim.scientific_structure_reason_codes
                    ),
                )
            )

        results.append(
            HypothesisNoveltyResidue(
                hypothesis_id=(
                    card.hypothesis_id
                ),
                external_status=card.status,
                claims=tuple(rows),
            )
        )

    return results
