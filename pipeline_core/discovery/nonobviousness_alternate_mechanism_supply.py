from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.nonobviousness_grounded_claim_attachment import (
    _as_bool,
    _claim_mentions_factor_family,
    _factor_node_matches,
    _factor_scope_features,
    _has_relation_language,
    _merged_jsonish,
    _merged_string_values,
    _node_text,
    _node_type,
    _stable_id,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.discovery.nonobviousness_operator_reeligibility import (
    N11RelativeContributionBranchDecision,
)
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11AlternateMechanismSupplyCandidate(
    StrictModel
):
    schema_version: Literal[
        "n11-alternate-mechanism-supply-candidate-v1"
    ] = (
        "n11-alternate-mechanism-supply-candidate-v1"
    )

    supply_candidate_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    source_d3_decision_id: str = Field(
        min_length=1
    )

    claim_node_id: str = Field(
        min_length=1
    )

    factor_node_id: str = Field(
        min_length=1
    )

    claim_node_type: Literal[
        "MechanismClaim"
    ] = "MechanismClaim"

    claim_text: str = Field(
        min_length=1
    )

    attachment_edge_id: str = Field(
        min_length=1
    )

    attachment_relation: Literal[
        "APPLIES_TO"
    ] = "APPLIES_TO"

    matched_factor_features: list[str] = Field(
        min_length=1
    )

    # Scopes found only in claim clauses that themselves
    # mention the task factor family. This is the semantic
    # scope used for alternate-mechanism supply.
    mechanism_scope_features: list[str]

    # Diagnostic only. Preserves scopes from the full
    # composite claim so scope leakage remains auditable.
    whole_claim_scope_features: list[str] = Field(
        default_factory=list
    )

    factor_local_text_segments: list[str] = Field(
        default_factory=list
    )

    source_paper_ids: list[str] = Field(
        min_length=1
    )

    evidence_pointer_count: int = Field(
        ge=1
    )

    factor_connected_grounded_claim: Literal[
        True
    ] = True

    # Important: this is supply for semantic review,
    # not a conclusion that the mechanism is actually
    # distinct from the baseline.
    distinct_from_baseline_assessed: Literal[
        False
    ] = False

    eligible_for_semantic_review: Literal[
        True
    ] = True

    eligible_as_positive_hypothesis_premise: Literal[
        False
    ] = False

    production_authority: Literal[
        False
    ] = False


class N11AlternateMechanismSupplyResult(
    StrictModel
):
    schema_version: Literal[
        "n11-alternate-mechanism-supply-result-v1"
    ] = (
        "n11-alternate-mechanism-supply-result-v1"
    )

    search_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    source_d3_decision_id: str = Field(
        min_length=1
    )

    status: Literal[
        "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY",
        "ABSTAIN_NO_FACTOR_GROUNDED_MECHANISM_SUPPLY",
        "NOT_ELIGIBLE_FROM_D3",
    ]

    reviewed_applies_to_edges: int = Field(
        ge=0
    )

    supply_candidate_count: int = Field(
        ge=0
    )

    candidates: list[
        N11AlternateMechanismSupplyCandidate
    ]

    rejection_reason_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    reason_codes: list[str] = Field(
        min_length=1
    )

    production_authority: Literal[
        False
    ] = False

    @model_validator(
        mode="after"
    )
    def _status_consistency(
        self,
    ) -> "N11AlternateMechanismSupplyResult":
        found = (
            self.status
            == "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY"
        )

        if found != bool(
            self.candidates
        ):
            raise ValueError(
                "FOUND status must agree with "
                "candidate presence"
            )

        if (
            self.supply_candidate_count
            != len(self.candidates)
        ):
            raise ValueError(
                "supply_candidate_count must equal "
                "candidate count"
            )

        if (
            self.status
            == "NOT_ELIGIBLE_FROM_D3"
            and self.reviewed_applies_to_edges != 0
        ):
            raise ValueError(
                "non-eligible D3 branch must not scan "
                "scientific supply"
            )

        return self


_CLAIM_CLAUSE_BOUNDARY_RE = re.compile(
    r"""
    (?<=[.!?;])\s+
    |
    \s+(?:
        while
        |whereas
        |however
        |but
        |in\s+contrast
    )\s+
    """,
    flags=re.I | re.X,
)


def _claim_label_text(
    claim_text: str,
) -> str:
    """Prefer the reported claim label over explanatory metadata."""

    match = re.search(
        r"(?:^|\n)label:\s*(.*?)"
        r"(?=\n(?:description|qualifiers|evidence\s+scope):|\Z)",
        claim_text,
        flags=re.I | re.S,
    )

    if match:
        return match.group(1).strip()

    return claim_text.strip()


def _segment_mentions_factor(
    *,
    segment: str,
    opportunity: N11MissingBridgeOpportunity,
    profile: ScientificDomainProfile,
    expected_features: set[str],
) -> bool:
    segment_features = (
        profile.novelty.scope_features(
            segment
        )
    )

    if expected_features:
        return bool(
            expected_features
            & segment_features
        )

    normalized = segment.lower()

    return any(
        term.lower() in normalized
        for term in opportunity.factor_identity_terms
        if term.strip()
    )


def _factor_local_mechanism_scopes(
    *,
    claim_text: str,
    opportunity: N11MissingBridgeOpportunity,
    profile: ScientificDomainProfile,
    expected_features: set[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """Return factor-local scopes, whole-claim scopes and local segments.

    A composite claim may report multiple independent mechanisms.
    Only clauses that themselves mention the task factor family may
    contribute mechanism scopes to alternate-mechanism supply.

    Example rejected leakage:

        nanogap -> EM hotspot,
        while charge transfer -> SERS

    The full claim contains charge-transfer vocabulary, but the
    factor-local nanogap clause does not.
    """

    label_text = _claim_label_text(
        claim_text
    )

    segments = [
        segment.strip(" ,")
        for segment in (
            _CLAIM_CLAUSE_BOUNDARY_RE
            .split(label_text)
        )
        if segment.strip(" ,")
    ]

    local_segments = [
        segment
        for segment in segments
        if _segment_mentions_factor(
            segment=segment,
            opportunity=opportunity,
            profile=profile,
            expected_features=(
                expected_features
            ),
        )
    ]

    local_scopes: set[str] = set()

    for segment in local_segments:
        local_scopes.update(
            profile.novelty.scope_features(
                segment
            )
        )

    whole_scopes = sorted(
        profile.novelty.scope_features(
            claim_text
        )
    )

    return (
        sorted(local_scopes),
        whole_scopes,
        local_segments,
    )


def scan_alternate_mechanism_supply(
    *,
    opportunity: N11MissingBridgeOpportunity,
    d3_decision: N11RelativeContributionBranchDecision,
    node_rows: list[
        dict[str, Any]
    ],
    edge_rows: list[
        dict[str, Any]
    ],
    profile: ScientificDomainProfile,
) -> N11AlternateMechanismSupplyResult:
    """Collect factor-grounded mechanism supply after D3 abstention.

    This stage does NOT decide whether a recovered mechanism is
    semantically distinct from the EM baseline.

    Eligibility here means only:
      - reported MechanismClaim;
      - pointer-grounded APPLIES_TO attachment;
      - factor identity is preserved;
      - claim text itself states the factor family;
      - claim text contains scientific relation language.

    Distinctness is delegated to the existing B1 semantic reviewer.
    """

    if (
        d3_decision
        .source_missing_bridge_opportunity_id
        != opportunity.opportunity_id
    ):
        raise ValueError(
            "D3 decision does not belong to "
            "the supplied opportunity"
        )

    search_id = _stable_id(
        "n11_alternate_mechanism_supply",
        opportunity.opportunity_id,
        d3_decision.decision_id,
        len(node_rows),
        len(edge_rows),
    )

    if (
        d3_decision.status
        != "ABSTAIN_MISSING_SUPPLEMENTAL_FACTOR_BRIDGE"
        or d3_decision.next_action
        != "SEARCH_ALTERNATE_SUPPLEMENTAL_MECHANISM_OR_GAP"
    ):
        return (
            N11AlternateMechanismSupplyResult(
                search_id=search_id,
                source_missing_bridge_opportunity_id=(
                    opportunity.opportunity_id
                ),
                source_d3_decision_id=(
                    d3_decision.decision_id
                ),
                status="NOT_ELIGIBLE_FROM_D3",
                reviewed_applies_to_edges=0,
                supply_candidate_count=0,
                candidates=[],
                rejection_reason_counts={},
                reason_codes=[
                    "D3_DOES_NOT_AUTHORIZE_ALTERNATE_SUPPLY_SEARCH"
                ],
            )
        )

    node_index = {
        str(
            row.get(
                "node_id",
                "",
            )
        ).strip(): row
        for row in node_rows
        if str(
            row.get(
                "node_id",
                "",
            )
        ).strip()
    }

    expected_factor_features = (
        _factor_scope_features(
            opportunity=opportunity,
            profile=profile,
        )
    )

    candidates: list[
        N11AlternateMechanismSupplyCandidate
    ] = []

    rejected: dict[
        str,
        int,
    ] = {}

    reviewed = 0

    seen: set[
        tuple[str, str]
    ] = set()

    def reject(
        reason: str,
    ) -> None:
        rejected[reason] = (
            rejected.get(
                reason,
                0,
            )
            + 1
        )

    for edge in edge_rows:
        relation = str(
            edge.get(
                "relation",
                "",
            )
        ).strip()

        if relation != "APPLIES_TO":
            continue

        reviewed += 1

        if (
            str(
                edge.get(
                    "graph_layer",
                    "",
                )
            )
            == "corpus_alignment"
            or str(
                edge.get(
                    "evidence_status",
                    "",
                )
            )
            == "derived_corpus_alignment"
        ):
            reject(
                "alignment_attachment"
            )
            continue

        if _as_bool(
            edge.get(
                "requires_verification",
                False,
            )
        ):
            reject(
                "attachment_requires_verification"
            )
            continue

        pointers = _merged_jsonish(
            edge.get(
                "evidence_pointers"
            ),
            edge.get(
                "evidence_pointers_json"
            ),
        )

        if not pointers:
            reject(
                "attachment_missing_pointer"
            )
            continue

        claim_id = str(
            edge.get(
                "source",
                "",
            )
        ).strip()

        factor_id = str(
            edge.get(
                "target",
                "",
            )
        ).strip()

        claim_row = node_index.get(
            claim_id
        )

        factor_row = node_index.get(
            factor_id
        )

        if (
            claim_row is None
            or factor_row is None
        ):
            reject(
                "attachment_endpoint_missing"
            )
            continue

        if (
            _node_type(
                claim_row
            )
            != "MechanismClaim"
        ):
            reject(
                "source_not_mechanism_claim"
            )
            continue

        if _as_bool(
            claim_row.get(
                "requires_verification",
                False,
            )
        ):
            reject(
                "claim_requires_verification"
            )
            continue

        (
            factor_matches,
            matched_factor_features,
        ) = _factor_node_matches(
            row=factor_row,
            opportunity=opportunity,
            profile=profile,
            expected_features=(
                expected_factor_features
            ),
        )

        if not factor_matches:
            reject(
                "factor_identity_not_matched"
            )
            continue

        claim_text = _node_text(
            claim_row
        )

        if not _claim_mentions_factor_family(
            claim_text=claim_text,
            factor_row=factor_row,
            opportunity=opportunity,
            profile=profile,
            expected_features=(
                expected_factor_features
            ),
        ):
            reject(
                "claim_does_not_state_factor"
            )
            continue

        if not _has_relation_language(
            claim_text=claim_text,
            profile=profile,
        ):
            reject(
                "claim_lacks_relation_language"
            )
            continue

        papers = set(
            _merged_string_values(
                edge.get(
                    "source_paper_ids"
                ),
                edge.get(
                    "source_paper_ids_json"
                ),
                claim_row.get(
                    "source_paper_ids"
                ),
                claim_row.get(
                    "source_paper_ids_json"
                ),
            )
        )

        direct_edge_paper = str(
            edge.get(
                "source_paper_id",
                "",
            )
        ).strip()

        direct_claim_paper = str(
            claim_row.get(
                "source_paper_id",
                "",
            )
        ).strip()

        if direct_edge_paper:
            papers.add(
                direct_edge_paper
            )

        if direct_claim_paper:
            papers.add(
                direct_claim_paper
            )

        if not papers:
            reject(
                "no_source_paper_provenance"
            )
            continue

        edge_id = str(
            edge.get(
                "edge_id",
                "",
            )
            or edge.get(
                "projection_edge_id",
                "",
            )
        ).strip()

        if not edge_id:
            reject(
                "attachment_edge_id_missing"
            )
            continue

        key = (
            claim_id,
            factor_id,
        )

        if key in seen:
            continue

        seen.add(key)

        (
            mechanism_scopes,
            whole_claim_scopes,
            factor_local_segments,
        ) = _factor_local_mechanism_scopes(
            claim_text=claim_text,
            opportunity=opportunity,
            profile=profile,
            expected_features=(
                expected_factor_features
            ),
        )

        candidates.append(
            N11AlternateMechanismSupplyCandidate(
                supply_candidate_id=_stable_id(
                    "n11_alt_mech_supply_candidate",
                    opportunity.opportunity_id,
                    d3_decision.decision_id,
                    claim_id,
                    factor_id,
                    edge_id,
                ),
                source_missing_bridge_opportunity_id=(
                    opportunity.opportunity_id
                ),
                source_d3_decision_id=(
                    d3_decision.decision_id
                ),
                claim_node_id=claim_id,
                factor_node_id=factor_id,
                claim_text=claim_text,
                attachment_edge_id=edge_id,
                matched_factor_features=(
                    matched_factor_features
                ),
                mechanism_scope_features=(
                    mechanism_scopes
                ),
                whole_claim_scope_features=(
                    whole_claim_scopes
                ),
                factor_local_text_segments=(
                    factor_local_segments
                ),
                source_paper_ids=sorted(
                    papers
                ),
                evidence_pointer_count=len(
                    pointers
                ),
            )
        )

    candidates.sort(
        key=lambda row: (
            row.claim_node_id,
            row.factor_node_id,
        )
    )

    if candidates:
        return (
            N11AlternateMechanismSupplyResult(
                search_id=search_id,
                source_missing_bridge_opportunity_id=(
                    opportunity.opportunity_id
                ),
                source_d3_decision_id=(
                    d3_decision.decision_id
                ),
                status=(
                    "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY"
                ),
                reviewed_applies_to_edges=(
                    reviewed
                ),
                supply_candidate_count=len(
                    candidates
                ),
                candidates=candidates,
                rejection_reason_counts=(
                    rejected
                ),
                reason_codes=[
                    "FACTOR_GROUNDED_MECHANISM_CLAIMS_FOUND",
                    "DISTINCTNESS_REQUIRES_SEPARATE_SEMANTIC_REVIEW",
                ],
            )
        )

    return (
        N11AlternateMechanismSupplyResult(
            search_id=search_id,
            source_missing_bridge_opportunity_id=(
                opportunity.opportunity_id
            ),
            source_d3_decision_id=(
                d3_decision.decision_id
            ),
            status=(
                "ABSTAIN_NO_FACTOR_GROUNDED_MECHANISM_SUPPLY"
            ),
            reviewed_applies_to_edges=(
                reviewed
            ),
            supply_candidate_count=0,
            candidates=[],
            rejection_reason_counts=(
                rejected
            ),
            reason_codes=[
                "NO_FACTOR_GROUNDED_MECHANISM_CLAIMS_FOUND"
            ],
        )
    )
