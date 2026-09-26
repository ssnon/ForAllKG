from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_admissibility import (
    AtomicScientificAdmissibilityAssessment,
    AtomicScientificAdmissibilityAssessmentReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AtomicPreN10ClaimStatus = Literal[
    "READY_FOR_N10_SHADOW",
    "NOT_READY_FOR_N10_SHADOW",
]

AtomicPreN10Disposition = Literal[
    "READY_FOR_N10_SHADOW",
    "INTERVENTION_REQUIRED_SHADOW",
    "NO_CLAIMS_SHADOW",
]

BlockingDimension = Literal[
    "SOURCE_REFERENCE",
    "SEMANTIC_FIDELITY",
    "SPECIFICATION",
    "ATOMIC_KIND",
    "NOVELTY_ROLE",
    "STRUCTURAL_COMPILATION",
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


def _blocking_dimensions(
    source: AtomicScientificAdmissibilityAssessment,
) -> list[BlockingDimension]:
    blockers: list[BlockingDimension] = []

    if source.source_reference_status != "READY":
        blockers.append("SOURCE_REFERENCE")
    if source.semantic_fidelity_status != "PASS":
        blockers.append("SEMANTIC_FIDELITY")
    if source.specification_status != "COMPLETE":
        blockers.append("SPECIFICATION")
    if source.atomic_kind_status != "SUPPORTED":
        blockers.append("ATOMIC_KIND")
    if source.novelty_selection_role is None:
        blockers.append("NOVELTY_ROLE")
    if source.compilation_status != "COMPILED_SHADOW":
        blockers.append("STRUCTURAL_COMPILATION")

    return blockers


class AtomicPreN10ClaimShadowRow(StrictModel):
    schema_version: Literal[
        "atomic-pre-n10-admissibility-shadow-row-v1"
    ] = "atomic-pre-n10-admissibility-shadow-row-v1"

    hypothesis_id: str
    claim_id: str
    claim_rank: int = Field(ge=1)
    kind: str
    novelty_selection_role: str | None = None

    source_reference_status: str
    proposition_fidelity_status: str
    bridge_fidelity_status: str
    semantic_fidelity_status: str
    specification_status: str
    atomic_kind_status: str
    structural_compilation_status: str

    blocking_dimensions: list[BlockingDimension] = Field(
        default_factory=list
    )
    readiness_status: AtomicPreN10ClaimStatus

    source_reference_reason_codes: list[str] = Field(default_factory=list)
    proposition_fidelity_reason_codes: list[str] = Field(default_factory=list)
    bridge_fidelity_reason_codes: list[str] = Field(default_factory=list)
    specification_reason_codes: list[str] = Field(default_factory=list)
    semantic_fidelity_reason_codes: list[str] = Field(default_factory=list)

    prediction_observation_id: str | None = None
    falsification_criterion_id: str | None = None
    compiled_specification_present: bool

    source_reference_ready_required: Literal[True] = True
    semantic_fidelity_pass_required: Literal[True] = True
    specification_complete_required: Literal[True] = True
    supported_atomic_kind_required: Literal[True] = True
    novelty_selection_role_required: Literal[True] = True
    compiled_atomic_specification_required: Literal[True] = True

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    router_authority: Literal[False] = False
    scientific_truth_assessed: Literal[False] = False
    novelty_assessed: Literal[False] = False
    canonical_claim_mutated: Literal[False] = False
    vpre_contract_changed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_readiness(self) -> "AtomicPreN10ClaimShadowRow":
        ready = self.readiness_status == "READY_FOR_N10_SHADOW"
        if ready != (not self.blocking_dimensions):
            raise ValueError(
                "atomic pre-N10 shadow readiness must match blocker set"
            )
        if ready and not self.compiled_specification_present:
            raise ValueError(
                "ready atomic pre-N10 shadow row requires compiled specification"
            )
        if (
            self.structural_compilation_status == "COMPILED_SHADOW"
        ) != self.compiled_specification_present:
            raise ValueError(
                "structural compilation status/specification presence mismatch"
            )
        return self


class AtomicPreN10AdmissibilityShadowReport(StrictModel):
    schema_version: Literal[
        "atomic-pre-n10-admissibility-shadow-report-v1"
    ] = "atomic-pre-n10-admissibility-shadow-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_atomic_shadow_report_id: str
    source_atomic_shadow_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    hypothesis_id: str

    claim_count: int = Field(ge=0)
    ready_claim_count: int = Field(ge=0)
    not_ready_claim_count: int = Field(ge=0)
    novelty_bearing_ready_claim_count: int = Field(ge=0)
    blocker_dimension_counts: dict[str, int]

    disposition: AtomicPreN10Disposition
    rows: list[AtomicPreN10ClaimShadowRow] = Field(default_factory=list)

    all_claims_ready_required_for_n10: Literal[True] = True
    novelty_bearing_ready_claim_required_for_n10: Literal[True] = True

    source_reference_and_semantic_admissibility_separated: Literal[
        True
    ] = True
    exact_text_reconstruction_used_for_readiness: Literal[False] = False

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    router_authority: Literal[False] = False
    vpre_contract_changed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    novelty_assessed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "AtomicPreN10AdmissibilityShadowReport":
        if self.claim_count != len(self.rows):
            raise ValueError("atomic pre-N10 shadow claim_count mismatch")

        ready = sum(
            row.readiness_status == "READY_FOR_N10_SHADOW"
            for row in self.rows
        )
        if self.ready_claim_count != ready:
            raise ValueError(
                "atomic pre-N10 shadow ready_claim_count mismatch"
            )
        if self.not_ready_claim_count != self.claim_count - ready:
            raise ValueError(
                "atomic pre-N10 shadow not_ready_claim_count mismatch"
            )

        novelty_ready = sum(
            row.readiness_status == "READY_FOR_N10_SHADOW"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.rows
        )
        if self.novelty_bearing_ready_claim_count != novelty_ready:
            raise ValueError(
                "atomic pre-N10 shadow novelty-bearing ready count mismatch"
            )

        observed_blockers = Counter(
            blocker
            for row in self.rows
            for blocker in row.blocking_dimensions
        )
        if dict(sorted(observed_blockers.items())) != dict(
            sorted(self.blocker_dimension_counts.items())
        ):
            raise ValueError(
                "atomic pre-N10 shadow blocker count mismatch"
            )

        if self.claim_count == 0:
            expected_disposition: AtomicPreN10Disposition = (
                "NO_CLAIMS_SHADOW"
            )
        elif (
            self.ready_claim_count == self.claim_count
            and self.novelty_bearing_ready_claim_count > 0
        ):
            expected_disposition = "READY_FOR_N10_SHADOW"
        else:
            expected_disposition = "INTERVENTION_REQUIRED_SHADOW"

        if self.disposition != expected_disposition:
            raise ValueError(
                "atomic pre-N10 shadow disposition mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("atomic pre-N10 shadow SHA mismatch")
        if observed_id != (
            "atomic_pre_n10_admissibility_shadow:"
            + expected_sha[:20]
        ):
            raise ValueError("atomic pre-N10 shadow report ID mismatch")
        return self


def assess_atomic_pre_n10_shadow_row(
    source: AtomicScientificAdmissibilityAssessment,
) -> AtomicPreN10ClaimShadowRow:
    blockers = _blocking_dimensions(source)
    ready = not blockers

    return AtomicPreN10ClaimShadowRow(
        hypothesis_id=source.hypothesis_id,
        claim_id=source.claim_id,
        claim_rank=source.claim_rank,
        kind=source.kind,
        novelty_selection_role=source.novelty_selection_role,
        source_reference_status=source.source_reference_status,
        proposition_fidelity_status=source.proposition_fidelity_status,
        bridge_fidelity_status=source.bridge_fidelity_status,
        semantic_fidelity_status=source.semantic_fidelity_status,
        specification_status=source.specification_status,
        atomic_kind_status=source.atomic_kind_status,
        structural_compilation_status=source.compilation_status,
        blocking_dimensions=blockers,
        readiness_status=(
            "READY_FOR_N10_SHADOW"
            if ready
            else "NOT_READY_FOR_N10_SHADOW"
        ),
        source_reference_reason_codes=list(
            source.source_reference_reason_codes
        ),
        proposition_fidelity_reason_codes=list(
            source.proposition_fidelity_reason_codes
        ),
        bridge_fidelity_reason_codes=list(
            source.bridge_fidelity_reason_codes
        ),
        specification_reason_codes=list(
            source.specification_reason_codes
        ),
        semantic_fidelity_reason_codes=list(
            source.semantic_fidelity_reason_codes
        ),
        prediction_observation_id=source.prediction_observation_id,
        falsification_criterion_id=source.falsification_criterion_id,
        compiled_specification_present=source.specification is not None,
    )


def compile_atomic_pre_n10_admissibility_shadow(
    source: AtomicScientificAdmissibilityAssessmentReport,
) -> AtomicPreN10AdmissibilityShadowReport:
    rows = [
        assess_atomic_pre_n10_shadow_row(row)
        for row in source.rows
    ]

    ready_count = sum(
        row.readiness_status == "READY_FOR_N10_SHADOW"
        for row in rows
    )
    novelty_ready_count = sum(
        row.readiness_status == "READY_FOR_N10_SHADOW"
        and row.novelty_selection_role == "NOVELTY_BEARING"
        for row in rows
    )

    if not rows:
        disposition: AtomicPreN10Disposition = "NO_CLAIMS_SHADOW"
    elif ready_count == len(rows) and novelty_ready_count > 0:
        disposition = "READY_FOR_N10_SHADOW"
    else:
        disposition = "INTERVENTION_REQUIRED_SHADOW"

    blocker_counts = Counter(
        blocker
        for row in rows
        for blocker in row.blocking_dimensions
    )

    body = {
        "schema_version": (
            "atomic-pre-n10-admissibility-shadow-report-v1"
        ),
        "source_atomic_shadow_report_id": source.source_representation_id,
        "source_atomic_shadow_report_sha256": (
            source.source_representation_sha256
        ),
        "hypothesis_id": source.hypothesis_id,
        "claim_count": len(rows),
        "ready_claim_count": ready_count,
        "not_ready_claim_count": len(rows) - ready_count,
        "novelty_bearing_ready_claim_count": novelty_ready_count,
        "blocker_dimension_counts": dict(sorted(blocker_counts.items())),
        "disposition": disposition,
        "rows": [row.model_dump(mode="json") for row in rows],
        "all_claims_ready_required_for_n10": True,
        "novelty_bearing_ready_claim_required_for_n10": True,
        "source_reference_and_semantic_admissibility_separated": True,
        "exact_text_reconstruction_used_for_readiness": False,
        "diagnostic_only": True,
        "production_authority": False,
        "router_authority": False,
        "vpre_contract_changed": False,
        "endpoint_binding_performed": False,
        "retrieval_performed": False,
        "novelty_assessed": False,
        "n9_performed": False,
        "n10_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)

    return AtomicPreN10AdmissibilityShadowReport(
        **body,
        report_id=(
            "atomic_pre_n10_admissibility_shadow:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "AtomicPreN10AdmissibilityShadowReport",
    "AtomicPreN10ClaimShadowRow",
    "assess_atomic_pre_n10_shadow_row",
    "compile_atomic_pre_n10_admissibility_shadow",
]
