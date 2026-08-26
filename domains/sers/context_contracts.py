from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSContextScope = Literal[
    "grounded_premise",
    "axis_inspiration",
    "hypothesis",
    "question",
]


SERSContextDimension = Literal[
    "substrate",
    "material_identity",
    "material_state",
    "support",
    "morphology",
    "architecture",
    "structural_motif",
    "gap_regime",
    "optical_condition",
    "analyte",
    "reporter",
    "measurement_geometry",
    "environment",
]


SERSContextRole = Literal[
    "plasmonic_substrate",
    "component",
    "material_state",
    "support",
    "morphology",
    "architecture",
    "structural_motif",
    "gap_regime",
    "optical_condition",
    "analyte",
    "reporter",
    "measurement_geometry",
    "environment",
]


SERSContextKnowledgeState = Literal[
    "explicit",
    "unknown",
    "not_applicable",
]


SERSContextBindingBasis = Literal[
    "node",
    "direct_edge",
    "derived_material_state",
    "hypothesis_assertion",
]


SERSContextProvenanceKind = Literal[
    "axis_anchor",
    "axis_structural_edge",
    "axis_direct_claim",
    "grounded_support_node",
    "grounded_applies_to_target",
    "grounded_bridge_owner",
    "grounded_bridge_owner_applies_to",
    "grounded_structural_edge",
    "hypothesis_assertion",
    "question",
]


SERSContextCompatibilityStatus = Literal[
    "match",
    "intentional_variation",
    "compatible_extension",
    "role_mismatch",
    "context_conflation",
    "conflict",
    "unknown",
]


SERSContextFindingSeverity = Literal[
    "info",
    "warning",
    "actionable",
]


SERSContextReviewStatus = Literal[
    "pass",
    "pass_with_unknowns",
    "reframe_required",
]


_NON_ACTIONABLE_STATUSES = frozenset({
    "match",
    "intentional_variation",
    "compatible_extension",
})

_ACTIONABLE_STATUSES = frozenset({
    "role_mismatch",
    "context_conflation",
    "conflict",
})


def expected_context_finding_severity(
    status: SERSContextCompatibilityStatus,
) -> SERSContextFindingSeverity:
    if status in _NON_ACTIONABLE_STATUSES:
        return "info"

    if status == "unknown":
        return "warning"

    if status in _ACTIONABLE_STATUSES:
        return "actionable"

    raise ValueError(
        f"unsupported SERS context status: {status}"
    )


def expected_context_review_status(
    findings: list["SERSContextFinding"],
) -> SERSContextReviewStatus:
    if any(
        row.severity == "actionable"
        for row in findings
    ):
        return "reframe_required"

    if any(
        row.severity == "warning"
        for row in findings
    ):
        return "pass_with_unknowns"

    return "pass"


class SERSContextProvenance(StrictModel):
    kind: SERSContextProvenanceKind

    node_ids: list[str] = Field(
        default_factory=list
    )

    edge_ids: list[str] = Field(
        default_factory=list
    )

    paper_ids: list[str] = Field(
        default_factory=list
    )

    statement_ids: list[str] = Field(
        default_factory=list
    )

    candidate_unit_ids: list[str] = Field(
        default_factory=list
    )

    hypothesis_ids: list[str] = Field(
        default_factory=list
    )

    excerpt: str | None = None

    @model_validator(mode="after")
    def _requires_traceable_basis(
        self,
    ) -> "SERSContextProvenance":
        has_identifier = any((
            self.node_ids,
            self.edge_ids,
            self.paper_ids,
            self.statement_ids,
            self.candidate_unit_ids,
            self.hypothesis_ids,
        ))

        has_excerpt = bool(
            (self.excerpt or "").strip()
        )

        if not has_identifier and not has_excerpt:
            raise ValueError(
                "context provenance requires at least one "
                "traceable identifier or excerpt"
            )

        return self


class SERSContextBinding(StrictModel):
    """Attachment of one context fact to its scientific owner.

    This preserves distinctions such as:

      3D-Si substrate --HAS_MORPHOLOGY--> inserted pyramid

    versus:

      Au@Ag nanoparticle --HAS_STRUCTURAL_MOTIF--> nanoparticle gap

    which cannot be recovered safely from a dimension/value pair alone.
    """

    basis: SERSContextBindingBasis

    owner_ref_id: str | None = None

    owner_label: str = Field(
        min_length=1
    )

    owner_type: str = Field(
        min_length=1
    )

    relation: str | None = None

    @model_validator(mode="after")
    def _binding_consistency(
        self,
    ) -> "SERSContextBinding":
        if (
            self.basis == "direct_edge"
            and not (
                self.relation
                or ""
            ).strip()
        ):
            raise ValueError(
                "direct_edge context binding "
                "requires relation"
            )

        return self


class SERSContextFact(StrictModel):
    fact_id: str = Field(
        min_length=1
    )

    dimension: SERSContextDimension

    scientific_role: SERSContextRole

    knowledge_state: SERSContextKnowledgeState

    value: str | None = None

    normalized_value: str | None = None

    binding: SERSContextBinding | None = None

    provenance: list[
        SERSContextProvenance
    ] = Field(
        min_length=1
    )

    tags: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _knowledge_state_consistency(
        self,
    ) -> "SERSContextFact":
        value = (
            self.value.strip()
            if isinstance(
                self.value,
                str,
            )
            else ""
        )

        if self.knowledge_state == "explicit":
            if not value:
                raise ValueError(
                    "explicit context fact requires value"
                )

        else:
            if self.value is not None:
                raise ValueError(
                    "unknown/not_applicable context fact "
                    "must not invent a value"
                )

            if self.normalized_value is not None:
                raise ValueError(
                    "unknown/not_applicable context fact "
                    "must not have normalized_value"
                )

        return self


class SERSContextSignature(StrictModel):
    # v1 signatures remain parseable. v2 preserves scientific
    # attachment/binding for compiled context facts.
    schema_version: Literal[
        "sers-context-signature-v1",
        "sers-context-signature-v2",
    ] = "sers-context-signature-v2"

    signature_id: str = Field(
        min_length=1
    )

    domain_profile_id: str = Field(
        min_length=1
    )

    scope: SERSContextScope

    source_ref_id: str = Field(
        min_length=1
    )

    facts: list[
        SERSContextFact
    ] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _unique_fact_ids(
        self,
    ) -> "SERSContextSignature":
        ids = [
            row.fact_id
            for row in self.facts
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate context fact_id"
            )

        return self


class SERSContextFinding(StrictModel):
    finding_id: str = Field(
        min_length=1
    )

    dimension: SERSContextDimension

    status: SERSContextCompatibilityStatus

    severity: SERSContextFindingSeverity

    left_signature_id: str = Field(
        min_length=1
    )

    right_signature_id: str = Field(
        min_length=1
    )

    left_fact_ids: list[str] = Field(
        min_length=1
    )

    right_fact_ids: list[str] = Field(
        min_length=1
    )

    rationale: str = Field(
        min_length=1
    )

    tags: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _severity_is_deterministic(
        self,
    ) -> "SERSContextFinding":
        expected = (
            expected_context_finding_severity(
                self.status
            )
        )

        if self.severity != expected:
            raise ValueError(
                "context finding severity does not "
                f"match status: expected {expected}"
            )

        if (
            self.left_signature_id
            == self.right_signature_id
        ):
            raise ValueError(
                "context finding must compare "
                "distinct signatures"
            )

        return self


class SERSContextReview(StrictModel):
    schema_version: Literal[
        "sers-context-review-v1"
    ] = "sers-context-review-v1"

    policy_version: Literal[
        "sers-context-compatibility-v1"
    ] = "sers-context-compatibility-v1"

    review_id: str = Field(
        min_length=1
    )

    hypothesis_id: str = Field(
        min_length=1
    )

    signatures: list[
        SERSContextSignature
    ] = Field(
        min_length=2
    )

    findings: list[
        SERSContextFinding
    ] = Field(
        min_length=1
    )

    status: SERSContextReviewStatus

    @model_validator(mode="after")
    def _validate_review(
        self,
    ) -> "SERSContextReview":
        signature_by_id = {
            row.signature_id: row
            for row in self.signatures
        }

        if (
            len(signature_by_id)
            != len(self.signatures)
        ):
            raise ValueError(
                "duplicate context signature_id"
            )

        global_fact_ids: set[str] = set()

        for signature in self.signatures:
            for fact in signature.facts:
                if fact.fact_id in global_fact_ids:
                    raise ValueError(
                        "context fact_id must be globally "
                        "unique within review"
                    )

                global_fact_ids.add(
                    fact.fact_id
                )

        finding_ids = [
            row.finding_id
            for row in self.findings
        ]

        if (
            len(finding_ids)
            != len(set(finding_ids))
        ):
            raise ValueError(
                "duplicate context finding_id"
            )

        for finding in self.findings:
            left = signature_by_id.get(
                finding.left_signature_id
            )

            right = signature_by_id.get(
                finding.right_signature_id
            )

            if left is None or right is None:
                raise ValueError(
                    "context finding references "
                    "unknown signature"
                )

            left_ids = {
                row.fact_id
                for row in left.facts
            }

            right_ids = {
                row.fact_id
                for row in right.facts
            }

            if not set(
                finding.left_fact_ids
            ).issubset(
                left_ids
            ):
                raise ValueError(
                    "context finding references "
                    "foreign left fact"
                )

            if not set(
                finding.right_fact_ids
            ).issubset(
                right_ids
            ):
                raise ValueError(
                    "context finding references "
                    "foreign right fact"
                )

            referenced = (
                [
                    row
                    for row in left.facts
                    if row.fact_id
                    in finding.left_fact_ids
                ]
                + [
                    row
                    for row in right.facts
                    if row.fact_id
                    in finding.right_fact_ids
                ]
            )

            if not any(
                row.dimension
                == finding.dimension
                for row in referenced
            ):
                raise ValueError(
                    "context finding dimension is not "
                    "represented by referenced facts"
                )

        expected_status = (
            expected_context_review_status(
                self.findings
            )
        )

        if self.status != expected_status:
            raise ValueError(
                "context review status is not "
                f"deterministic: expected {expected_status}"
            )

        return self
