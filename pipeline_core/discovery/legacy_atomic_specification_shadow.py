from __future__ import annotations

import hashlib
import json
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
    compile_atomic_semantic_fidelity_taxonomy_shadow,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    AtomicClaimKind,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.atomic_scientific_source_binding import (
    resolve_atomic_source_reference,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceReferenceStatus = Literal["READY", "INCOMPLETE", "INVALID"]
FidelityStatus = Literal["PASS", "REVIEW", "INVALID"]
SpecificationStatus = Literal["COMPLETE", "INCOMPLETE"]
AtomicKindStatus = Literal["SUPPORTED", "UNSUPPORTED"]
ShadowCompilationStatus = Literal[
    "COMPILED_SHADOW",
    "ABSTAINED_SOURCE_REFERENCE",
    "ABSTAINED_UNSUPPORTED_KIND",
    "ABSTAINED_MISSING_NOVELTY_ROLE",
]


_SOURCE_ID_REASON_CODES = frozenset(
    {
        "atomic_prediction_source_id_missing",
        "atomic_prediction_source_id_unknown",
        "atomic_prediction_source_id_without_prediction",
        "atomic_falsifier_source_id_missing",
        "atomic_falsifier_source_id_unknown",
        "atomic_falsifier_source_id_without_falsifier",
    }
)

_PROPOSITION_INVALID_REASON_CODES = frozenset(
    {
        "atomic_claim_source_basis_empty",
        "atomic_claim_source_basis_not_exact",
        "atomic_claim_relation_endpoint_anchors_empty",
        "atomic_claim_relation_endpoint_missing_from_claim",
        "atomic_claim_relation_endpoint_missing_from_basis",
        "atomic_claim_scope_qualifier_not_preserved",
        "atomic_claim_scope_qualifier_not_source_backed",
        "atomic_claim_direction_qualifier_not_preserved",
        "atomic_claim_direction_qualifier_not_source_backed",
    }
)

_PROPOSITION_REVIEW_REASON_CODES = frozenset(
    {
        "atomic_claim_ordered_language_not_in_source_basis",
        "atomic_claim_comparative_scope_may_be_broadened",
        "atomic_claim_ordered_language_not_in_source_context",
        "atomic_claim_context_comparative_scope_may_be_broadened",
        "atomic_claim_condition_scope_may_be_dropped",
        "atomic_claim_context_condition_scope_may_be_dropped",
    }
)

_BRIDGE_INVALID_REASON_CODES = frozenset(
    {
        "required_bridge_anaphoric_relation_reference",
        "required_bridge_relation_endpoint_alignment_unresolved",
    }
)

_BRIDGE_REVIEW_REASON_CODES = frozenset(
    {
        "required_bridge_scope_qualifier_missing",
        "required_bridge_condition_scope_may_be_dropped",
        "required_bridge_comparative_scope_may_be_broadened",
    }
)

_CLASSIFIED_SEMANTIC_REASON_CODES = (
    _SOURCE_ID_REASON_CODES
    | _PROPOSITION_INVALID_REASON_CODES
    | _PROPOSITION_REVIEW_REASON_CODES
    | _BRIDGE_INVALID_REASON_CODES
    | _BRIDGE_REVIEW_REASON_CODES
)

_ATOMIC_KINDS = frozenset(get_args(AtomicClaimKind))


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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _surface(value: object) -> str:
    return " ".join(str(value or "").split())


def _normalized(value: object) -> str:
    return _surface(value).casefold()


def _status_from_reasons(
    reasons: list[str],
    *,
    invalid: frozenset[str],
    review: frozenset[str],
) -> FidelityStatus:
    if any(reason in invalid for reason in reasons):
        return "INVALID"
    if any(reason in review for reason in reasons):
        return "REVIEW"
    return "PASS"


def _combine_fidelity(
    proposition: FidelityStatus,
    bridge: FidelityStatus,
    unclassified: list[str],
) -> FidelityStatus:
    if unclassified or "INVALID" in {proposition, bridge}:
        return "INVALID"
    if "REVIEW" in {proposition, bridge}:
        return "REVIEW"
    return "PASS"


class LegacyAtomicSpecificationShadowRow(StrictModel):
    schema_version: Literal[
        "legacy-atomic-scientific-specification-shadow-row-v1"
    ] = "legacy-atomic-scientific-specification-shadow-row-v1"

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    claim_local_id: str
    kind: str
    novelty_selection_role: str | None = None

    source_reference_status: SourceReferenceStatus
    proposition_fidelity_status: FidelityStatus
    bridge_fidelity_status: FidelityStatus
    semantic_fidelity_status: FidelityStatus
    specification_status: SpecificationStatus
    atomic_kind_status: AtomicKindStatus
    compilation_status: ShadowCompilationStatus

    source_reference_reason_codes: list[str] = Field(default_factory=list)
    proposition_fidelity_reason_codes: list[str] = Field(default_factory=list)
    bridge_fidelity_reason_codes: list[str] = Field(default_factory=list)
    unclassified_semantic_reason_codes: list[str] = Field(default_factory=list)
    specification_reason_codes: list[str] = Field(default_factory=list)
    semantic_fidelity_reason_codes: list[str] = Field(default_factory=list)
    semantic_fidelity_taxonomy_shadow: dict[str, object]

    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None
    source_observable: str | None = None

    specification: CompiledAtomicSpecification | None = None

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    scientific_truth_assessed: Literal[False] = False
    scientific_equivalence_assessed: Literal[False] = False
    canonical_claim_mutated: Literal[False] = False
    query_plan_mutated: Literal[False] = False
    vpre_contract_changed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_shadow_boundary(self) -> "LegacyAtomicSpecificationShadowRow":
        compiled = self.compilation_status == "COMPILED_SHADOW"
        if compiled != (self.specification is not None):
            raise ValueError(
                "shadow compilation status/specification presence mismatch"
            )
        if compiled and self.source_reference_status != "READY":
            raise ValueError(
                "compiled shadow specification requires READY source references"
            )
        if compiled and self.atomic_kind_status != "SUPPORTED":
            raise ValueError(
                "compiled shadow specification requires supported atomic kind"
            )
        if compiled and self.novelty_selection_role is None:
            raise ValueError(
                "compiled shadow specification requires novelty selection role"
            )
        return self


class LegacyAtomicSpecificationShadowReport(StrictModel):
    schema_version: Literal[
        "legacy-atomic-scientific-specification-shadow-report-v1"
    ] = "legacy-atomic-scientific-specification-shadow-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_id: str
    row_count: int = Field(ge=0)
    compiled_count: int = Field(ge=0)
    source_ready_count: int = Field(ge=0)
    semantic_pass_count: int = Field(ge=0)
    semantic_review_count: int = Field(ge=0)
    semantic_invalid_count: int = Field(ge=0)
    specification_complete_count: int = Field(ge=0)
    rows: list[LegacyAtomicSpecificationShadowRow] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    canonical_claim_mutated: Literal[False] = False
    query_plan_mutated: Literal[False] = False
    vpre_contract_changed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "LegacyAtomicSpecificationShadowReport":
        if self.row_count != len(self.rows):
            raise ValueError("shadow row_count mismatch")
        if self.compiled_count != sum(
            row.compilation_status == "COMPILED_SHADOW"
            for row in self.rows
        ):
            raise ValueError("shadow compiled_count mismatch")
        if self.source_ready_count != sum(
            row.source_reference_status == "READY"
            for row in self.rows
        ):
            raise ValueError("shadow source_ready_count mismatch")
        observed_semantic = {
            "PASS": self.semantic_pass_count,
            "REVIEW": self.semantic_review_count,
            "INVALID": self.semantic_invalid_count,
        }
        for status, count in observed_semantic.items():
            if count != sum(
                row.semantic_fidelity_status == status
                for row in self.rows
            ):
                raise ValueError(
                    "shadow semantic status count mismatch: " + status
                )
        if self.specification_complete_count != sum(
            row.specification_status == "COMPLETE"
            for row in self.rows
        ):
            raise ValueError("shadow specification_complete_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("shadow report SHA mismatch")
        if observed_id != (
            "legacy_atomic_specification_shadow:" + expected_sha[:20]
        ):
            raise ValueError("shadow report ID mismatch")
        return self


def _source_reference_binding(
    *,
    hypothesis: HypothesisCard,
    draft_claim: NoveltyClaimDraft,
) -> tuple[
    SourceReferenceStatus,
    list[str],
    object | None,
    object | None,
]:
    binding = draft_claim.semantic_fidelity_binding
    return resolve_atomic_source_reference(
        hypothesis=hypothesis,
        prediction_observation_id=binding.prediction_observation_id,
        falsification_criterion_id=binding.falsification_criterion_id,
    )


def _specification_reasons(
    claim: NoveltyClaim,
) -> list[str]:
    reasons: list[str] = []
    if not _surface(claim.required_bridge):
        reasons.append("missing_required_bridge")
    if not _surface(claim.predicted_observation):
        reasons.append("missing_predicted_observation")
    if not _surface(claim.falsification_condition):
        reasons.append("missing_falsification_condition")
    if not claim.prior_art_identity_terms:
        reasons.append("missing_prior_art_identity_terms")
    if claim.novelty_selection_role is None:
        reasons.append("missing_novelty_selection_role")
    return reasons


def compile_legacy_atomic_specification_shadow_row(
    *,
    hypothesis: HypothesisCard,
    draft_claim: NoveltyClaimDraft,
    canonical_claim: NoveltyClaim,
) -> LegacyAtomicSpecificationShadowRow:
    if canonical_claim.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "canonical claim/hypothesis ID mismatch in atomic shadow compiler"
        )
    if canonical_claim.kind != draft_claim.kind:
        raise ValueError(
            "draft/canonical claim kind drift in atomic shadow compiler"
        )

    semantic = assess_atomic_semantic_fidelity(
        hypothesis,
        draft_claim,
    )
    taxonomy = compile_atomic_semantic_fidelity_taxonomy_shadow(
        draft_claim,
        semantic_fidelity_shadow=semantic,
        sanitized_required_bridge=canonical_claim.required_bridge,
        sanitized_predicted_observation=canonical_claim.predicted_observation,
        sanitized_falsification_condition=canonical_claim.falsification_condition,
    )
    semantic_reasons = list(semantic.get("reason_codes") or [])

    proposition_reasons = [
        reason
        for reason in semantic_reasons
        if reason in (
            _PROPOSITION_INVALID_REASON_CODES
            | _PROPOSITION_REVIEW_REASON_CODES
        )
    ]
    bridge_reasons = [
        reason
        for reason in semantic_reasons
        if reason in (
            _BRIDGE_INVALID_REASON_CODES
            | _BRIDGE_REVIEW_REASON_CODES
        )
    ]
    unclassified = [
        reason
        for reason in semantic_reasons
        if reason not in _CLASSIFIED_SEMANTIC_REASON_CODES
    ]

    proposition_status = _status_from_reasons(
        proposition_reasons,
        invalid=_PROPOSITION_INVALID_REASON_CODES,
        review=_PROPOSITION_REVIEW_REASON_CODES,
    )
    bridge_status = _status_from_reasons(
        bridge_reasons,
        invalid=_BRIDGE_INVALID_REASON_CODES,
        review=_BRIDGE_REVIEW_REASON_CODES,
    )
    semantic_status = _combine_fidelity(
        proposition_status,
        bridge_status,
        unclassified,
    )

    (
        source_status,
        source_reasons,
        prediction,
        falsifier,
    ) = _source_reference_binding(
        hypothesis=hypothesis,
        draft_claim=draft_claim,
    )

    specification_reasons = _specification_reasons(canonical_claim)
    specification_status: SpecificationStatus = (
        "INCOMPLETE" if specification_reasons else "COMPLETE"
    )
    atomic_kind_status: AtomicKindStatus = (
        "SUPPORTED"
        if canonical_claim.kind in _ATOMIC_KINDS
        else "UNSUPPORTED"
    )

    binding = draft_claim.semantic_fidelity_binding
    prediction_id = (
        _surface(binding.prediction_observation_id) or None
    )
    falsifier_id = (
        _surface(binding.falsification_criterion_id) or None
    )

    specification = None
    if source_status != "READY":
        compilation_status: ShadowCompilationStatus = (
            "ABSTAINED_SOURCE_REFERENCE"
        )
    elif atomic_kind_status != "SUPPORTED":
        compilation_status = "ABSTAINED_UNSUPPORTED_KIND"
    elif canonical_claim.novelty_selection_role is None:
        compilation_status = "ABSTAINED_MISSING_NOVELTY_ROLE"
    else:
        assert prediction is not None
        assert falsifier is not None
        specification = CompiledAtomicSpecification(
            local_id=draft_claim.local_id,
            claim_id=canonical_claim.claim_id,
            kind=canonical_claim.kind,
            importance=canonical_claim.importance,
            novelty_selection_role=canonical_claim.novelty_selection_role,
            text=canonical_claim.text,
            rationale=canonical_claim.rationale,
            source_candidate_ids=[],
            premise_statement_ids=list(
                hypothesis.premise_statement_ids
            ),
            gap_statement_ids=list(
                hypothesis.gap_statement_ids
            ),
            prior_art_identity_terms=list(
                canonical_claim.prior_art_identity_terms
            ),
            relation_endpoint_anchors=list(
                binding.relation_endpoint_anchors
            ),
            scope_qualifier_spans=list(
                binding.scope_qualifier_spans
            ),
            directional_qualifier_spans=list(
                binding.directional_qualifier_spans
            ),
            relation_nucleus_terms=list(
                canonical_claim.relation_nucleus_terms
            ),
            distinguishing_terms=list(
                canonical_claim.distinguishing_terms
            ),
            required_bridge=canonical_claim.required_bridge,
            observable=prediction.observable,
            predicted_observation=canonical_claim.predicted_observation,
            falsification_condition=canonical_claim.falsification_condition,
            prediction_observation_id=prediction.observation_id,
            falsification_criterion_id=falsifier.criterion_id,
            search_concepts=list(canonical_claim.search_concepts),
            search_queries=list(canonical_claim.search_queries),
            scientific_structure=canonical_claim.scientific_structure,
            scientific_structure_reason_codes=list(
                canonical_claim.scientific_structure_reason_codes
            ),
        )
        compilation_status = "COMPILED_SHADOW"

    return LegacyAtomicSpecificationShadowRow(
        hypothesis_id=hypothesis.hypothesis_id,
        claim_id=canonical_claim.claim_id,
        claim_rank=canonical_claim.claim_rank,
        claim_local_id=draft_claim.local_id,
        kind=canonical_claim.kind,
        novelty_selection_role=canonical_claim.novelty_selection_role,
        source_reference_status=source_status,
        proposition_fidelity_status=proposition_status,
        bridge_fidelity_status=bridge_status,
        semantic_fidelity_status=semantic_status,
        specification_status=specification_status,
        atomic_kind_status=atomic_kind_status,
        compilation_status=compilation_status,
        source_reference_reason_codes=source_reasons,
        proposition_fidelity_reason_codes=proposition_reasons,
        bridge_fidelity_reason_codes=bridge_reasons,
        unclassified_semantic_reason_codes=unclassified,
        specification_reason_codes=specification_reasons,
        semantic_fidelity_reason_codes=semantic_reasons,
        semantic_fidelity_taxonomy_shadow=taxonomy,
        prediction_observation_id=prediction_id,
        falsification_criterion_id=falsifier_id,
        source_observable=(
            prediction.observable
            if prediction is not None
            else None
        ),
        specification=specification,
    )


def compile_legacy_atomic_specification_shadow_report(
    *,
    hypothesis: HypothesisCard,
    decomposition_draft: NoveltyClaimDecompositionDraft,
    canonical_claims: HypothesisNoveltyClaims,
) -> LegacyAtomicSpecificationShadowReport:
    if canonical_claims.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError(
            "canonical claim group/hypothesis ID mismatch in atomic shadow compiler"
        )

    ordered_claims = sorted(
        canonical_claims.claims,
        key=lambda row: row.claim_rank,
    )
    expected_ranks = list(
        range(1, len(ordered_claims) + 1)
    )
    observed_ranks = [
        row.claim_rank for row in ordered_claims
    ]
    if observed_ranks != expected_ranks:
        raise ValueError(
            "canonical claim ranks must be contiguous for atomic shadow compiler"
        )
    if len(decomposition_draft.claims) != len(ordered_claims):
        raise ValueError(
            "draft/canonical claim count mismatch in atomic shadow compiler"
        )

    rows = [
        compile_legacy_atomic_specification_shadow_row(
            hypothesis=hypothesis,
            draft_claim=draft_claim,
            canonical_claim=canonical_claim,
        )
        for draft_claim, canonical_claim in zip(
            decomposition_draft.claims,
            ordered_claims,
            strict=True,
        )
    ]

    body = {
        "schema_version": (
            "legacy-atomic-scientific-specification-shadow-report-v1"
        ),
        "hypothesis_id": hypothesis.hypothesis_id,
        "row_count": len(rows),
        "compiled_count": sum(
            row.compilation_status == "COMPILED_SHADOW"
            for row in rows
        ),
        "source_ready_count": sum(
            row.source_reference_status == "READY"
            for row in rows
        ),
        "semantic_pass_count": sum(
            row.semantic_fidelity_status == "PASS"
            for row in rows
        ),
        "semantic_review_count": sum(
            row.semantic_fidelity_status == "REVIEW"
            for row in rows
        ),
        "semantic_invalid_count": sum(
            row.semantic_fidelity_status == "INVALID"
            for row in rows
        ),
        "specification_complete_count": sum(
            row.specification_status == "COMPLETE"
            for row in rows
        ),
        "rows": [row.model_dump(mode="json") for row in rows],
        "diagnostic_only": True,
        "production_authority": False,
        "canonical_claim_mutated": False,
        "query_plan_mutated": False,
        "vpre_contract_changed": False,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "production_selection_changed": False,
    }
    digest = _sha256_json(body)
    return LegacyAtomicSpecificationShadowReport(
        **body,
        report_id=(
            "legacy_atomic_specification_shadow:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "LegacyAtomicSpecificationShadowReport",
    "LegacyAtomicSpecificationShadowRow",
    "compile_legacy_atomic_specification_shadow_report",
    "compile_legacy_atomic_specification_shadow_row",
]
