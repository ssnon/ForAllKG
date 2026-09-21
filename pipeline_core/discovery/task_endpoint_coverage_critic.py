from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.task_backbone_chain import (
    TaskEndpointCoverageLedger,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskEndpointCoverageIssueCode = Literal[
    "COMPOUND_ENDPOINT_COVERAGE_INCOMPLETE",
    "COMPOUND_ENDPOINT_FACET_UNRESOLVED",
    "COMPOUND_ENDPOINT_QUALIFIER_UNRESOLVED",
    "COMPOUND_ENDPOINT_QUALIFIER_LOCALITY_UNRESOLVED",
]


class TaskEndpointCoverageIssue(StrictModel):
    code: TaskEndpointCoverageIssueCode
    endpoint_side: Literal["source", "target"]
    endpoint: str
    facet_core_endpoint: str = ""
    detail: str = ""

    diagnostic_only: Literal[True] = True
    blocking: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False


class TaskEndpointCoverageCriticResult(StrictModel):
    schema_version: str = "task-endpoint-coverage-critic-v1"

    issues: list[TaskEndpointCoverageIssue] = Field(
        default_factory=list
    )
    issue_codes: list[TaskEndpointCoverageIssueCode] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    blocking: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    task_filter_relaxed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False


def critique_task_endpoint_coverage(
    *,
    source: TaskEndpointCoverageLedger,
    target: TaskEndpointCoverageLedger,
) -> TaskEndpointCoverageCriticResult:
    issues: list[TaskEndpointCoverageIssue] = []

    for side, ledger in (
        ("source", source),
        ("target", target),
    ):
        if not ledger.coordinated:
            continue

        if ledger.status in {
            "partial_or_qualifier_unresolved",
            "unresolved",
        }:
            issues.append(
                TaskEndpointCoverageIssue(
                    code="COMPOUND_ENDPOINT_COVERAGE_INCOMPLETE",
                    endpoint_side=side,
                    endpoint=ledger.original_endpoint,
                    detail=(
                        "compound endpoint coverage is not complete; "
                        "partial facet support must not be interpreted "
                        "as whole-endpoint support"
                    ),
                )
            )

        for facet in ledger.facets:
            if facet.status == "core_facet_unresolved":
                issues.append(
                    TaskEndpointCoverageIssue(
                        code="COMPOUND_ENDPOINT_FACET_UNRESOLVED",
                        endpoint_side=side,
                        endpoint=ledger.original_endpoint,
                        facet_core_endpoint=facet.core_endpoint,
                        detail=(
                            "facet core has no confirmed-known local "
                            "endpoint binding"
                        ),
                    )
                )
            elif (
                facet.status
                == "facet_core_supported_qualifier_unresolved"
            ):
                issues.append(
                    TaskEndpointCoverageIssue(
                        code="COMPOUND_ENDPOINT_QUALIFIER_UNRESOLVED",
                        endpoint_side=side,
                        endpoint=ledger.original_endpoint,
                        facet_core_endpoint=facet.core_endpoint,
                        detail=(
                            "facet core is locally supported but shared "
                            "qualifier obligation is unresolved"
                        ),
                    )
                )
            elif (
                facet.status
                == "facet_supported_qualifier_only_component_level"
            ):
                issues.append(
                    TaskEndpointCoverageIssue(
                        code=(
                            "COMPOUND_ENDPOINT_QUALIFIER_LOCALITY_UNRESOLVED"
                        ),
                        endpoint_side=side,
                        endpoint=ledger.original_endpoint,
                        facet_core_endpoint=facet.core_endpoint,
                        detail=(
                            "shared qualifier is present only elsewhere "
                            "in the relation component, not on the task slot"
                        ),
                    )
                )

    unique_codes = sorted({issue.code for issue in issues})

    return TaskEndpointCoverageCriticResult(
        issues=issues,
        issue_codes=unique_codes,
    )
