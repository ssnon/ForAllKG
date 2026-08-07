from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IssueStage(str, Enum):
    SCHEMA = "schema"
    PROVENANCE = "provenance"
    STRUCTURAL = "structural"
    MEASUREMENT = "measurement"
    CLAIM = "claim"
    RELATION = "relation"
    SCALARIZATION = "scalarization"
    VOCABULARY = "vocabulary"
    FINALIZATION = "finalization"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(str, Enum):
    DUPLICATE_NODE_ID = "DUPLICATE_NODE_ID"
    DUPLICATE_GRAPH_ASSET_ID = "DUPLICATE_GRAPH_ASSET_ID"
    CLAIM_LIKE_ENTITY = "CLAIM_LIKE_ENTITY"

    EDGE_MISSING_EVIDENCE_POINTER = "EDGE_MISSING_EVIDENCE_POINTER"
    POINTER_DOCUMENT_ID_MISMATCH = "POINTER_DOCUMENT_ID_MISMATCH"
    POINTER_DOCUMENT_ROLE_MISMATCH = "POINTER_DOCUMENT_ROLE_MISMATCH"
    POINTER_UNKNOWN_ASSET = "POINTER_UNKNOWN_ASSET"
    POINTER_UNKNOWN_PAGE = "POINTER_UNKNOWN_PAGE"

    UNDEFINED_EDGE_SOURCE = "UNDEFINED_EDGE_SOURCE"
    UNDEFINED_EDGE_TARGET = "UNDEFINED_EDGE_TARGET"
    ISOLATED_NODE = "ISOLATED_NODE"

    MISSING_MEASUREMENT_PRODUCER = "MISSING_MEASUREMENT_PRODUCER"
    INVALID_MEASURED_FOR_COUNT = "INVALID_MEASURED_FOR_COUNT"
    MEASURED_FOR_TARGET_MISMATCH = "MEASURED_FOR_TARGET_MISMATCH"
    MEASUREMENT_SUBJECT_NOT_ENTITY = "MEASUREMENT_SUBJECT_NOT_ENTITY"

    UNKNOWN_MEASUREMENT_GROUP = "UNKNOWN_MEASUREMENT_GROUP"
    MISSING_MEASUREMENT_GROUP_EDGE = "MISSING_MEASUREMENT_GROUP_EDGE"
    UNEXPECTED_MEASUREMENT_GROUP_EDGE = "UNEXPECTED_MEASUREMENT_GROUP_EDGE"
    MEASUREMENT_GROUP_UNKNOWN_MEMBER = "MEASUREMENT_GROUP_UNKNOWN_MEMBER"
    MEASUREMENT_GROUP_MEMBER_MISMATCH = "MEASUREMENT_GROUP_MEMBER_MISMATCH"
    SINGLETON_MEASUREMENT_GROUP = "SINGLETON_MEASUREMENT_GROUP"
    DUPLICATE_MEASUREMENT_GROUP_MEMBER = "DUPLICATE_MEASUREMENT_GROUP_MEMBER"

    OBSERVATION_MISSING_SUPPORT = "OBSERVATION_MISSING_SUPPORT"
    MECHANISM_MISSING_SUPPORT = "MECHANISM_MISSING_SUPPORT"
    CLAIM_MISSING_APPLICATION_TARGET = "CLAIM_MISSING_APPLICATION_TARGET"

    RELATION_SOURCE_TYPE_MISMATCH = "RELATION_SOURCE_TYPE_MISMATCH"
    RELATION_TARGET_TYPE_MISMATCH = "RELATION_TARGET_TYPE_MISMATCH"

    SCALARIZATION_FAILURE = "SCALARIZATION_FAILURE"
    FINAL_STRICT_VALIDATION_FAILURE = "FINAL_STRICT_VALIDATION_FAILURE"
    VOCABULARY_NORMALIZATION_FAILURE = "VOCABULARY_NORMALIZATION_FAILURE"
    EXTERNAL_PROVENANCE_FAILURE = "EXTERNAL_PROVENANCE_FAILURE"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    code: IssueCode
    stage: IssueStage
    severity: IssueSeverity = IssueSeverity.ERROR
    message: str

    node_id: str | None = None
    node_collection: str | None = None

    edge_index: int | None = None
    source_id: str | None = None
    target_id: str | None = None
    relation: str | None = None

    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> "ValidationReport":
        return cls(valid=not any(item.severity == IssueSeverity.ERROR for item in issues), issues=issues)

    def codes(self) -> set[IssueCode]:
        return {item.code for item in self.issues}

    def count(self, code: IssueCode) -> int:
        return sum(item.code == code for item in self.issues)

    def stage_counts(self) -> dict[str, int]:
        return dict(Counter(item.stage.value for item in self.issues))

    def code_counts(self) -> dict[str, int]:
        return dict(Counter(item.code.value for item in self.issues))

    def family_count(self) -> int:
        return len({item.stage for item in self.issues if item.severity == IssueSeverity.ERROR})


def make_issue_id(code: IssueCode, **parts: Any) -> str:
    canonical = json.dumps(
        {"code": code.value, **parts},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{code.value.lower()}:{digest}"


def issue(
    *,
    code: IssueCode,
    stage: IssueStage,
    message: str,
    severity: IssueSeverity = IssueSeverity.ERROR,
    **fields: Any,
) -> ValidationIssue:
    identity_fields = {
        key: value
        for key, value in fields.items()
        if key
        in {
            "node_id",
            "node_collection",
            "edge_index",
            "source_id",
            "target_id",
            "relation",
        }
    }
    return ValidationIssue(
        issue_id=make_issue_id(code, **identity_fields),
        code=code,
        stage=stage,
        severity=severity,
        message=message,
        **fields,
    )
