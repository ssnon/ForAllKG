from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SourceReferenceStatus = Literal["READY", "INCOMPLETE", "INVALID"]
FidelityStatus = Literal["PASS", "REVIEW", "INVALID"]
SpecificationStatus = Literal["COMPLETE", "INCOMPLETE"]
AtomicKindStatus = Literal["SUPPORTED", "UNSUPPORTED"]
AtomicCompilationStatus = Literal[
    "COMPILED_SHADOW",
    "ABSTAINED_SOURCE_REFERENCE",
    "ABSTAINED_UNSUPPORTED_KIND",
    "ABSTAINED_MISSING_NOVELTY_ROLE",
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


class AtomicScientificAdmissibilityAssessment(StrictModel):
    schema_version: Literal[
        "atomic-scientific-admissibility-assessment-v1"
    ] = "atomic-scientific-admissibility-assessment-v1"

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
    compilation_status: AtomicCompilationStatus

    source_reference_reason_codes: list[str] = Field(default_factory=list)
    proposition_fidelity_reason_codes: list[str] = Field(default_factory=list)
    bridge_fidelity_reason_codes: list[str] = Field(default_factory=list)
    unclassified_semantic_reason_codes: list[str] = Field(default_factory=list)
    specification_reason_codes: list[str] = Field(default_factory=list)
    semantic_fidelity_reason_codes: list[str] = Field(default_factory=list)

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
    def validate_compilation_state(
        self,
    ) -> "AtomicScientificAdmissibilityAssessment":
        compiled = self.compilation_status == "COMPILED_SHADOW"
        if compiled != (self.specification is not None):
            raise ValueError(
                "admissibility compilation status/specification presence mismatch"
            )
        if compiled and self.source_reference_status != "READY":
            raise ValueError(
                "compiled admissibility assessment requires READY source references"
            )
        if compiled and self.atomic_kind_status != "SUPPORTED":
            raise ValueError(
                "compiled admissibility assessment requires supported atomic kind"
            )
        if compiled and self.novelty_selection_role is None:
            raise ValueError(
                "compiled admissibility assessment requires novelty selection role"
            )
        return self


class AtomicScientificAdmissibilityAssessmentReport(StrictModel):
    schema_version: Literal[
        "atomic-scientific-admissibility-assessment-report-v1"
    ] = "atomic-scientific-admissibility-assessment-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_representation_id: str
    source_representation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_representation_schema: str

    hypothesis_id: str
    row_count: int = Field(ge=0)
    compiled_count: int = Field(ge=0)
    source_ready_count: int = Field(ge=0)
    semantic_pass_count: int = Field(ge=0)
    semantic_review_count: int = Field(ge=0)
    semantic_invalid_count: int = Field(ge=0)
    specification_complete_count: int = Field(ge=0)

    rows: list[AtomicScientificAdmissibilityAssessment] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    router_authority: Literal[False] = False
    source_reference_and_semantic_admissibility_separated: Literal[
        True
    ] = True
    exact_text_reconstruction_used_for_source_identity: Literal[
        False
    ] = False
    retrieval_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "AtomicScientificAdmissibilityAssessmentReport":
        if self.row_count != len(self.rows):
            raise ValueError("admissibility row_count mismatch")
        if any(row.hypothesis_id != self.hypothesis_id for row in self.rows):
            raise ValueError(
                "admissibility report contains cross-hypothesis rows"
            )
        if self.compiled_count != sum(
            row.compilation_status == "COMPILED_SHADOW"
            for row in self.rows
        ):
            raise ValueError("admissibility compiled_count mismatch")
        if self.source_ready_count != sum(
            row.source_reference_status == "READY"
            for row in self.rows
        ):
            raise ValueError("admissibility source_ready_count mismatch")

        observed_semantic = Counter(
            row.semantic_fidelity_status
            for row in self.rows
        )
        if self.semantic_pass_count != observed_semantic["PASS"]:
            raise ValueError("admissibility semantic_pass_count mismatch")
        if self.semantic_review_count != observed_semantic["REVIEW"]:
            raise ValueError("admissibility semantic_review_count mismatch")
        if self.semantic_invalid_count != observed_semantic["INVALID"]:
            raise ValueError("admissibility semantic_invalid_count mismatch")

        if self.specification_complete_count != sum(
            row.specification_status == "COMPLETE"
            for row in self.rows
        ):
            raise ValueError(
                "admissibility specification_complete_count mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("admissibility assessment report SHA mismatch")
        if observed_id != (
            "atomic_scientific_admissibility:" + expected_sha[:20]
        ):
            raise ValueError("admissibility assessment report ID mismatch")
        return self


def build_atomic_scientific_admissibility_assessment_report(
    *,
    source_representation_id: str,
    source_representation_sha256: str,
    source_representation_schema: str,
    hypothesis_id: str,
    rows: list[AtomicScientificAdmissibilityAssessment],
) -> AtomicScientificAdmissibilityAssessmentReport:
    body = {
        "schema_version": (
            "atomic-scientific-admissibility-assessment-report-v1"
        ),
        "source_representation_id": source_representation_id,
        "source_representation_sha256": source_representation_sha256,
        "source_representation_schema": source_representation_schema,
        "hypothesis_id": hypothesis_id,
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
        "router_authority": False,
        "source_reference_and_semantic_admissibility_separated": True,
        "exact_text_reconstruction_used_for_source_identity": False,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return AtomicScientificAdmissibilityAssessmentReport(
        **body,
        report_id="atomic_scientific_admissibility:" + digest[:20],
        report_sha256=digest,
    )


__all__ = [
    "AtomicCompilationStatus",
    "AtomicKindStatus",
    "AtomicScientificAdmissibilityAssessment",
    "AtomicScientificAdmissibilityAssessmentReport",
    "FidelityStatus",
    "SourceReferenceStatus",
    "SpecificationStatus",
    "build_atomic_scientific_admissibility_assessment_report",
]
