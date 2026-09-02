from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11GroundedClaimAttachment(
    StrictModel
):
    schema_version: Literal[
        "n11-grounded-claim-attachment-v1"
    ] = "n11-grounded-claim-attachment-v1"

    attachment_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    claim_node_id: str = Field(
        min_length=1
    )

    factor_node_id: str = Field(
        min_length=1
    )

    claim_node_type: Literal[
        "ObservationClaim",
        "MechanismClaim",
    ]

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

    matched_base_context_terms: list[str] = Field(
        min_length=1
    )

    source_paper_ids: list[str] = Field(
        min_length=1
    )

    evidence_pointer_count: int = Field(
        ge=1
    )

    relation_language_detected: Literal[
        True
    ] = True

    path_class: Literal[
        "GROUNDED_CLAIM_ATTACHMENT"
    ] = "GROUNDED_CLAIM_ATTACHMENT"

    # D2 only supplies evidence for later operator reconsideration.
    # It never approves a hypothesis or operator by itself.
    eligible_for_operator_reconsideration: Literal[
        True
    ] = True

    production_authority: Literal[
        False
    ] = False


class N11GroundedClaimAttachmentResult(
    StrictModel
):
    schema_version: Literal[
        "n11-grounded-claim-attachment-result-v1"
    ] = "n11-grounded-claim-attachment-result-v1"

    search_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    status: Literal[
        "FOUND_GROUNDED_CLAIM_ATTACHMENTS",
        "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM",
    ]

    reviewed_applies_to_edges: int = Field(
        ge=0
    )

    grounded_candidate_count: int = Field(
        ge=0
    )

    candidates: list[
        N11GroundedClaimAttachment
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
    ) -> "N11GroundedClaimAttachmentResult":
        if (
            self.status
            == "FOUND_GROUNDED_CLAIM_ATTACHMENTS"
            and not self.candidates
        ):
            raise ValueError(
                "found status requires candidates"
            )

        if (
            self.status
            == "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
            and self.candidates
        ):
            raise ValueError(
                "abstain status cannot contain candidates"
            )

        if (
            self.grounded_candidate_count
            != len(self.candidates)
        ):
            raise ValueError(
                "grounded_candidate_count must equal "
                "candidate count"
            )

        return self


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _jsonish(
    value: Any,
) -> list[Any]:
    if value in (
        None,
        "",
    ):
        return []

    if isinstance(
        value,
        list,
    ):
        return list(value)

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    try:
        parsed = json.loads(
            str(value)
        )
    except (
        TypeError,
        json.JSONDecodeError,
    ):
        return []

    return (
        parsed
        if isinstance(
            parsed,
            list,
        )
        else []
    )


def _jsonish_strings(
    value: Any,
) -> list[str]:
    return sorted({
        str(item).strip()
        for item in _jsonish(value)
        if str(item).strip()
    })


def _merged_jsonish(
    *values: Any,
) -> list[Any]:
    """Merge list and JSON-string representations without duplication."""
    merged: list[Any] = []
    seen: set[str] = set()

    for value in values:
        for item in _jsonish(value):
            try:
                key = json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except TypeError:
                key = repr(item)

            if key in seen:
                continue

            seen.add(key)
            merged.append(item)

    return merged


def _merged_string_values(
    *values: Any,
) -> list[str]:
    return sorted({
        str(item).strip()
        for value in values
        for item in _jsonish(value)
        if str(item).strip()
    })


def _as_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def _node_text(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("node_text")
        or row.get("label")
        or row.get("statement")
        or row.get("name")
        or ""
    ).strip()


def _node_type(
    row: dict[str, Any],
) -> str:
    return str(
        row.get("type")
        or row.get("node_type")
        or ""
    ).strip()


def _tokens(
    text: str,
) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if len(token) >= 3
        and token not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "into",
            "their",
            "measured",
            "mechanistically",
            "interpreted",
            "behavior",
            "response",
        }
    }


def _factor_scope_features(
    *,
    opportunity: N11MissingBridgeOpportunity,
    profile: ScientificDomainProfile,
) -> set[str]:
    features: set[str] = set()

    for term in (
        opportunity.factor_identity_terms
    ):
        features.update(
            profile.novelty.scope_features(
                term
            )
        )

    return features


def _factor_node_matches(
    *,
    row: dict[str, Any],
    opportunity: N11MissingBridgeOpportunity,
    profile: ScientificDomainProfile,
    expected_features: set[str],
) -> tuple[
    bool,
    list[str],
]:
    text = _node_text(row)

    actual_features = (
        profile.novelty.scope_features(
            text
        )
    )

    matched = sorted(
        expected_features
        & actual_features
    )

    if matched:
        return (
            True,
            matched,
        )

    # Fail-closed lexical fallback only when the
    # domain profile provides no factor-family feature.
    if expected_features:
        return (
            False,
            [],
        )

    node_tokens = _tokens(text)

    for term in (
        opportunity.factor_identity_terms
    ):
        term_tokens = _tokens(term)

        if (
            term_tokens
            and term_tokens
            <= node_tokens
        ):
            return (
                True,
                sorted(term_tokens),
            )

    return (
        False,
        [],
    )


def _claim_mentions_factor_family(
    *,
    claim_text: str,
    factor_row: dict[str, Any],
    opportunity: N11MissingBridgeOpportunity,
    profile: ScientificDomainProfile,
    expected_features: set[str],
) -> bool:
    """Require the reported claim itself to mention the task factor family.

    APPLIES_TO may retrieve a claim attached to a factor node, but that
    attachment alone cannot ground a factor -> outcome/mechanism relation.
    """

    claim_features = (
        profile.novelty.scope_features(
            claim_text
        )
    )

    if expected_features:
        return bool(
            expected_features
            & claim_features
        )

    # Conservative lexical fallback for domains without a registered
    # factor-family scope feature.
    claim_tokens = _tokens(
        claim_text
    )

    factor_texts = [
        _node_text(
            factor_row
        ),
        *opportunity.factor_identity_terms,
    ]

    for factor_text in factor_texts:
        factor_tokens = _tokens(
            factor_text
        )

        if (
            factor_tokens
            and factor_tokens
            <= claim_tokens
        ):
            return True

    return False


def _base_term_matches_claim(
    *,
    base_term: str,
    claim_text: str,
    profile: ScientificDomainProfile,
) -> bool:
    query_domains = (
        profile.novelty.domains(
            base_term
        )
    )

    query_scopes = (
        profile.novelty.scope_features(
            base_term
        )
    )

    claim_domains = (
        profile.novelty.domains(
            claim_text
        )
    )

    claim_scopes = (
        profile.novelty.scope_features(
            claim_text
        )
    )

    if (
        query_domains
        and query_domains
        & claim_domains
    ):
        return True

    if (
        query_scopes
        and query_scopes
        & claim_scopes
    ):
        return True

    if (
        query_domains
        or query_scopes
    ):
        return False

    query_tokens = _tokens(
        base_term
    )

    claim_tokens = _tokens(
        claim_text
    )

    return bool(
        query_tokens
        and query_tokens
        <= claim_tokens
    )


_COMPARATIVE_RELATION_PATTERNS = (
    r"\bvar(?:y|ies|ied|ying)\b",
    r"\bdepend(?:s|ed|ing)?\b",
    r"\bcorrelat(?:e|es|ed|ing|ion)\b",
    r"\bincreas(?:e|es|ed|ing)\b",
    r"\bdecreas(?:e|es|ed|ing)\b",
    r"\breduc(?:e|es|ed|ing)\b",
    r"\benhanc(?:e|es|ed|ing)\b",
    r"\bstronger\b",
    r"\bweaker\b",
    r"\bhigher\b",
    r"\blower\b",
    r"\bgreater\b",
    r"\bless\b",
    r"\bsmaller\b",
    r"\blarger\b",
    r"\bnarrower\b",
    r"\bwider\b",
    r"\bas\b.{0,80}\b(?:increase|decrease)",
    r"\bthan\b",
)


def _has_relation_language(
    *,
    claim_text: str,
    profile: ScientificDomainProfile,
) -> bool:
    patterns = (
        tuple(
            profile
            .discovery
            .strong_causal_text_patterns
        )
        + _COMPARATIVE_RELATION_PATTERNS
    )

    return any(
        re.search(
            pattern,
            claim_text,
            flags=re.I,
        )
        for pattern in patterns
    )


def scan_grounded_claim_attachments(
    *,
    opportunity: N11MissingBridgeOpportunity,
    node_rows: list[
        dict[str, Any]
    ],
    edge_rows: list[
        dict[str, Any]
    ],
    profile: ScientificDomainProfile,
) -> N11GroundedClaimAttachmentResult:
    """Recover explicit source-grounded factor-relation claims.

    APPLIES_TO is used only as a claim-attachment navigation edge.
    Scientific content comes from the attached ObservationClaim or
    MechanismClaim text itself.

    BridgeConcept nodes are never eligible.
    """

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
        N11GroundedClaimAttachment
    ] = []

    rejected: dict[
        str,
        int,
    ] = {}

    reviewed = 0

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

    seen: set[
        tuple[str, str]
    ] = set()

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

        # Corpus edge evidence may expose the same provenance
        # either as a materialized list or as its JSON serialization.
        # This is a schema-compatibility alias only; the epistemic
        # requirement for at least one pointer is unchanged.
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

        claim_type = _node_type(
            claim_row
        )

        if claim_type not in {
            "ObservationClaim",
            "MechanismClaim",
        }:
            # Explicitly excludes BridgeConcept.
            reject(
                "source_not_reported_claim"
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

        # Critical epistemic boundary:
        #
        # claim --APPLIES_TO--> factor
        #
        # is a retrieval attachment, not by itself evidence that the
        # factor changes the mechanism/outcome. The reported claim text
        # must independently mention the same factor family.
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

        matched_base_terms = [
            base_term
            for base_term
            in opportunity.base_relation_terms
            if _base_term_matches_claim(
                base_term=base_term,
                claim_text=claim_text,
                profile=profile,
            )
        ]

        if not matched_base_terms:
            reject(
                "base_context_not_matched"
            )
            continue

        # The claim must itself state a relation/trend.
        # Merely saying that a field is "in a nanogap"
        # is not enough.
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
            )
        )

        direct_edge_paper = str(
            edge.get(
                "source_paper_id",
                "",
            )
        ).strip()

        if direct_edge_paper:
            papers.add(
                direct_edge_paper
            )

        claim_paper = str(
            claim_row.get(
                "source_paper_id",
                "",
            )
        ).strip()

        if claim_paper:
            papers.add(
                claim_paper
            )

        papers.update(
            _merged_string_values(
                claim_row.get(
                    "source_paper_ids"
                ),
                claim_row.get(
                    "source_paper_ids_json"
                ),
            )
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

        candidates.append(
            N11GroundedClaimAttachment(
                attachment_id=_stable_id(
                    "n11_claim_attachment",
                    opportunity.opportunity_id,
                    claim_id,
                    factor_id,
                    edge_id,
                ),
                source_missing_bridge_opportunity_id=(
                    opportunity.opportunity_id
                ),
                claim_node_id=claim_id,
                factor_node_id=factor_id,
                claim_node_type=claim_type,
                claim_text=claim_text,
                attachment_edge_id=edge_id,
                matched_factor_features=(
                    matched_factor_features
                ),
                matched_base_context_terms=(
                    matched_base_terms
                ),
                source_paper_ids=sorted(
                    papers
                ),
                evidence_pointer_count=len(
                    pointers
                ),
            )
        )

    search_id = _stable_id(
        "n11_claim_attachment_search",
        opportunity.opportunity_id,
        len(node_rows),
        len(edge_rows),
    )

    if candidates:
        return (
            N11GroundedClaimAttachmentResult(
                search_id=search_id,
                source_missing_bridge_opportunity_id=(
                    opportunity.opportunity_id
                ),
                status=(
                    "FOUND_GROUNDED_CLAIM_ATTACHMENTS"
                ),
                reviewed_applies_to_edges=(
                    reviewed
                ),
                grounded_candidate_count=len(
                    candidates
                ),
                candidates=candidates,
                rejection_reason_counts=(
                    rejected
                ),
                reason_codes=[
                    "grounded_factor_relation_claim_found",
                    "applies_to_used_as_navigation_only",
                ],
            )
        )

    return (
        N11GroundedClaimAttachmentResult(
            search_id=search_id,
            source_missing_bridge_opportunity_id=(
                opportunity.opportunity_id
            ),
            status=(
                "ABSTAIN_NO_GROUNDED_FACTOR_RELATION_CLAIM"
            ),
            reviewed_applies_to_edges=(
                reviewed
            ),
            grounded_candidate_count=0,
            candidates=[],
            rejection_reason_counts=(
                rejected
            ),
            reason_codes=[
                "no_grounded_factor_relation_claim_found",
                "applies_to_not_promoted_to_scientific_relation",
            ],
        )
    )
