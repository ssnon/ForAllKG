from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relational_atomic_projection import (
    RelationalAtomicProjectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FinalDisposition = Literal[
    "SOURCE_BINDING_ABSTAINED_AFTER_R1",
    "ENDPOINT_BINDING_ABSTAINED_AFTER_R1",
    "ENDPOINT_BINDING_STAGE_FAILED_AFTER_R1",
    "EXCLUDED_UPSTREAM_FAILURE",
    "FULL_VERIFIER_READY_AFTER_R1",
]


class ProjectionReadinessCaseAccounting(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    final_disposition: FinalDisposition

    original_prospective_disposition: str
    repair_plan_status: str
    repair_materialized_claim_count: int = Field(ge=0)
    reentry_status: str

    endpoint_ready_after_r1: bool
    projection_preflight_observed: bool
    projected_claim_count: int = Field(ge=0)
    source_binding_abstention_count: int = Field(ge=0)
    endpoint_abstention_count: int = Field(ge=0)
    source_binding_reason_codes: list[str]

    full_verifier_ready_after_r1: bool
    novelty_verdict_observed: Literal[False] = False
    second_repair_attempt_performed: Literal[False] = False

    projection_report_id: str | None = None
    projection_report_sha256: str | None = None
    projection_report_file_sha256: str | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "ProjectionReadinessCaseAccounting":
        if self.full_verifier_ready_after_r1:
            if self.final_disposition != "FULL_VERIFIER_READY_AFTER_R1":
                raise ValueError(
                    "full verifier readiness/disposition mismatch"
                )
            if self.projected_claim_count < 1:
                raise ValueError(
                    "full verifier readiness requires projected claim"
                )
        if (
            self.final_disposition
            == "SOURCE_BINDING_ABSTAINED_AFTER_R1"
            and self.source_binding_abstention_count < 1
        ):
            raise ValueError(
                "source-binding abstention requires abstained source binding"
            )
        return self


class RepairedProjectionReadinessAccountingReport(StrictModel):
    schema_version: Literal[
        "repaired-projection-readiness-accounting-v1"
    ] = "repaired-projection-readiness-accounting-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_repair_plan_id: str
    source_repair_execution_report_id: str
    source_reentry_report_id: str
    source_provenance_retry_report_id: str

    cases: list[ProjectionReadinessCaseAccounting]
    case_ids: list[str]
    case_count: Literal[5] = 5
    disposition_counts: dict[str, int]

    endpoint_ready_case_ids: list[str]
    source_binding_abstained_case_ids: list[str]
    full_verifier_ready_case_ids: list[str]

    original_full_verifier_ready_count: Literal[0] = 0
    repaired_endpoint_ready_count: int = Field(ge=0)
    repaired_full_verifier_ready_count: int = Field(ge=0)

    novelty_verdict_count: Literal[0] = 0
    second_repair_attempt_performed: Literal[False] = False
    original_prospective_results_preserved: Literal[True] = True
    repair_results_preserved: Literal[True] = True
    projection_results_preserved: Literal[True] = True
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "RepairedProjectionReadinessAccountingReport":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("accounting cases must be P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError("accounting rows must be ordered P06-P10")

        counts = Counter(row.final_disposition for row in self.cases)
        if dict(sorted(counts.items())) != dict(
            sorted(self.disposition_counts.items())
        ):
            raise ValueError("disposition_counts mismatch")

        endpoint_ready = [
            row.case_id
            for row in self.cases
            if row.endpoint_ready_after_r1
        ]
        if endpoint_ready != self.endpoint_ready_case_ids:
            raise ValueError("endpoint_ready_case_ids mismatch")

        source_abstained = [
            row.case_id
            for row in self.cases
            if row.final_disposition
            == "SOURCE_BINDING_ABSTAINED_AFTER_R1"
        ]
        if source_abstained != self.source_binding_abstained_case_ids:
            raise ValueError("source_binding_abstained_case_ids mismatch")

        full_ready = [
            row.case_id
            for row in self.cases
            if row.full_verifier_ready_after_r1
        ]
        if full_ready != self.full_verifier_ready_case_ids:
            raise ValueError("full_verifier_ready_case_ids mismatch")
        if len(endpoint_ready) != self.repaired_endpoint_ready_count:
            raise ValueError("repaired_endpoint_ready_count mismatch")
        if len(full_ready) != self.repaired_full_verifier_ready_count:
            raise ValueError("repaired_full_verifier_ready_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("projection readiness accounting SHA mismatch")
        if observed_id != (
            "repaired_projection_readiness_accounting:"
            + expected_sha[:20]
        ):
            raise ValueError("projection readiness accounting ID mismatch")
        return self


def classify_projection_report(
    report: RelationalAtomicProjectionReport,
) -> tuple[FinalDisposition, list[str]]:
    reasons = list(
        dict.fromkeys(
            reason
            for row in report.rows
            if row.projection_status == "ABSTAINED_SOURCE_BINDING"
            for reason in row.reason_codes
        )
    )
    if report.projected_claim_count >= 1:
        return "FULL_VERIFIER_READY_AFTER_R1", reasons
    if report.source_binding_abstention_count >= 1:
        return "SOURCE_BINDING_ABSTAINED_AFTER_R1", reasons
    if report.skipped_endpoint_abstention_count >= 1:
        return "ENDPOINT_BINDING_ABSTAINED_AFTER_R1", reasons
    raise ValueError(
        "projection report has no projected or abstained claim accounting"
    )


def build_accounting_report(
    *,
    campaign_root: Path,
    repair_plan: dict,
    repair_execution: dict,
    reentry: dict,
    provenance_retry: dict,
) -> RepairedProjectionReadinessAccountingReport:
    root = campaign_root.expanduser().resolve()

    repair_plan_cases = {
        str(row["case_id"]): row
        for row in repair_plan["cases"]
    }
    repair_execution_cases = {
        str(row["case_id"]): row
        for row in repair_execution["cases"]
    }
    reentry_cases = {
        str(row["case_id"]): row
        for row in reentry["cases"]
    }

    original_dispositions = {
        str(row["case_id"]): str(row["source_original_disposition"])
        for row in repair_plan["cases"]
    }

    rows: list[ProjectionReadinessCaseAccounting] = []
    endpoint_ready_from_reentry = set(
        str(value)
        for value in reentry.get("verifier_ready_case_ids", [])
    )

    for case_id in ["P06", "P07", "P08", "P09", "P10"]:
        plan_case = repair_plan_cases[case_id]
        exec_case = repair_execution_cases[case_id]
        reentry_case = reentry_cases[case_id]

        endpoint_ready = case_id in endpoint_ready_from_reentry
        projection_path = (
            root
            / "specification_repair_r1"
            / case_id
            / "relational_scientific_verifier_shadow.r1_provenance_rebound"
            / "relational_atomic_projection.json"
        )

        if projection_path.is_file():
            projection = RelationalAtomicProjectionReport.model_validate_json(
                projection_path.read_text(encoding="utf-8")
            )
            disposition, reasons = classify_projection_report(projection)
            rows.append(
                ProjectionReadinessCaseAccounting(
                    case_id=case_id,
                    final_disposition=disposition,
                    original_prospective_disposition=(
                        original_dispositions[case_id]
                    ),
                    repair_plan_status=str(plan_case["status"]),
                    repair_materialized_claim_count=int(
                        exec_case["materialized_claim_count"]
                    ),
                    reentry_status=str(reentry_case["status"]),
                    endpoint_ready_after_r1=endpoint_ready,
                    projection_preflight_observed=True,
                    projected_claim_count=projection.projected_claim_count,
                    source_binding_abstention_count=(
                        projection.source_binding_abstention_count
                    ),
                    endpoint_abstention_count=(
                        projection.skipped_endpoint_abstention_count
                    ),
                    source_binding_reason_codes=reasons,
                    full_verifier_ready_after_r1=(
                        projection.projected_claim_count >= 1
                    ),
                    projection_report_id=projection.report_id,
                    projection_report_sha256=projection.report_sha256,
                    projection_report_file_sha256=_sha256_file(
                        projection_path
                    ),
                )
            )
            continue

        reentry_status = str(reentry_case["status"])
        if reentry_status == "ENDPOINT_BINDING_ABSTAINED_AFTER_R1":
            disposition: FinalDisposition = (
                "ENDPOINT_BINDING_ABSTAINED_AFTER_R1"
            )
        elif reentry_status == "ENDPOINT_BINDING_STAGE_FAILED_AFTER_R1":
            disposition = "ENDPOINT_BINDING_STAGE_FAILED_AFTER_R1"
        elif reentry_status in {
            "EXCLUDED_NO_AUTOMATIC_REPAIR",
            "NO_MATERIALIZED_R1_CLAIMS",
        }:
            disposition = "EXCLUDED_UPSTREAM_FAILURE"
        else:
            raise ValueError(
                case_id
                + ": missing projection artifact for unexpected re-entry "
                + reentry_status
            )

        rows.append(
            ProjectionReadinessCaseAccounting(
                case_id=case_id,
                final_disposition=disposition,
                original_prospective_disposition=(
                    original_dispositions[case_id]
                ),
                repair_plan_status=str(plan_case["status"]),
                repair_materialized_claim_count=int(
                    exec_case["materialized_claim_count"]
                ),
                reentry_status=reentry_status,
                endpoint_ready_after_r1=endpoint_ready,
                projection_preflight_observed=False,
                projected_claim_count=0,
                source_binding_abstention_count=0,
                endpoint_abstention_count=(
                    1
                    if disposition
                    == "ENDPOINT_BINDING_ABSTAINED_AFTER_R1"
                    else 0
                ),
                source_binding_reason_codes=[],
                full_verifier_ready_after_r1=False,
            )
        )

    counts = Counter(row.final_disposition for row in rows)
    endpoint_ready_ids = [
        row.case_id for row in rows if row.endpoint_ready_after_r1
    ]
    source_abstained_ids = [
        row.case_id
        for row in rows
        if row.final_disposition
        == "SOURCE_BINDING_ABSTAINED_AFTER_R1"
    ]
    full_ready_ids = [
        row.case_id
        for row in rows
        if row.full_verifier_ready_after_r1
    ]

    body = {
        "schema_version": "repaired-projection-readiness-accounting-v1",
        "source_repair_plan_id": str(repair_plan["plan_id"]),
        "source_repair_execution_report_id": str(
            repair_execution["report_id"]
        ),
        "source_reentry_report_id": str(reentry["report_id"]),
        "source_provenance_retry_report_id": str(
            provenance_retry["report_id"]
        ),
        "cases": [row.model_dump(mode="json") for row in rows],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "disposition_counts": dict(sorted(counts.items())),
        "endpoint_ready_case_ids": endpoint_ready_ids,
        "source_binding_abstained_case_ids": source_abstained_ids,
        "full_verifier_ready_case_ids": full_ready_ids,
        "original_full_verifier_ready_count": 0,
        "repaired_endpoint_ready_count": len(endpoint_ready_ids),
        "repaired_full_verifier_ready_count": len(full_ready_ids),
        "novelty_verdict_count": 0,
        "second_repair_attempt_performed": False,
        "original_prospective_results_preserved": True,
        "repair_results_preserved": True,
        "projection_results_preserved": True,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RepairedProjectionReadinessAccountingReport(
        **body,
        report_id=(
            "repaired_projection_readiness_accounting:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "FinalDisposition",
    "ProjectionReadinessCaseAccounting",
    "RepairedProjectionReadinessAccountingReport",
    "build_accounting_report",
    "classify_projection_report",
]
