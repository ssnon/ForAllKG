from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    PreN10SourceAlignmentPrimaryReportV1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
    ProspectiveRegenerationUnitV2Result,
    execute_regeneration_generation_unit_v2,
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


def _pretty_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_exact_or_validate(path: Path, payload: object) -> str:
    expected = _pretty_json_bytes(payload)
    path = path.expanduser().resolve()
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(
                "existing write-once pre-N10 regeneration artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


def _slug(value: str) -> str:
    slug = str(value).replace(":", "_").replace("/", "_")
    if not slug or slug in {".", ".."}:
        raise ValueError("invalid source hypothesis ID")
    return slug


class PreN10RegenerationLineageV1(StrictModel):
    source_hypothesis_id: str

    unit_result_path: str
    unit_result_id: str
    unit_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_status: str

    regenerated_portfolio_path: str | None = None
    regenerated_portfolio_id: str | None = None
    regenerated_hypothesis_count: int = Field(ge=0)

    generation_calls_attempted: Literal[1] = 1
    repair_calls_attempted: Literal[0] = 0

    previous_hypothesis_text_consumed: Literal[False] = False
    external_novelty_outcome_consumed: Literal[False] = False
    n10_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    semantic_critic_performed: Literal[False] = False
    claim_decomposition_performed: Literal[False] = False
    pre_n10_contract_reentry_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False


class PreN10RegenerationExecutionReportV1(StrictModel):
    schema_version: Literal[
        "pre-n10-regeneration-execution-report-v1"
    ] = "pre-n10-regeneration-execution-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_primary_report_id: str
    source_primary_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    source_context_id: str
    source_context_sha256: str

    lineages: list[PreN10RegenerationLineageV1]
    regeneration_required_count: int = Field(ge=0)
    generation_call_count: int = Field(ge=0)
    repair_call_count: Literal[0] = 0

    generated_and_compiled_count: int = Field(ge=0)
    generated_abstention_count: int = Field(ge=0)
    compile_rejected_count: int = Field(ge=0)
    generation_failed_count: int = Field(ge=0)

    regeneration_trigger: Literal[
        "PRE_N10_CONTRACT_PRIMARY_UNRECOVERED"
    ] = "PRE_N10_CONTRACT_PRIMARY_UNRECOVERED"

    full_e2e_rerun_performed: Literal[False] = False
    semantic_critic_performed: Literal[False] = False
    claim_decomposition_performed: Literal[False] = False
    pre_n10_contract_reentry_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False

    previous_hypothesis_text_consumed: Literal[False] = False
    external_novelty_outcome_consumed: Literal[False] = False
    n10_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10RegenerationExecutionReportV1":
        if self.regeneration_required_count != len(self.lineages):
            raise ValueError("regeneration_required_count mismatch")
        if self.generation_call_count != sum(
            row.generation_calls_attempted
            for row in self.lineages
        ):
            raise ValueError("generation_call_count mismatch")

        statuses = [row.unit_status for row in self.lineages]
        expected = {
            "GENERATED_AND_COMPILED": self.generated_and_compiled_count,
            "GENERATED_ABSTENTION_AND_COMPILED":
                self.generated_abstention_count,
            "DETERMINISTIC_COMPILE_REJECTED":
                self.compile_rejected_count,
            "GENERATION_CALL_FAILED":
                self.generation_failed_count,
        }
        for status, count in expected.items():
            if statuses.count(status) != count:
                raise ValueError(
                    "regeneration status count mismatch: " + status
                )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 regeneration execution SHA mismatch")
        if observed_id != (
            "pre_n10_regeneration_execution_v1:"
            + expected_sha[:20]
        ):
            raise ValueError("pre-N10 regeneration execution ID mismatch")
        return self


BackendFactory = Callable[[str, Path], HypothesisDraftBackend]


def execute_pre_n10_regeneration_v1(
    *,
    context: HypothesisContext,
    primary_report: PreN10SourceAlignmentPrimaryReportV1,
    unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    backend_factory: BackendFactory,
    output_root: Path,
) -> tuple[
    PreN10RegenerationExecutionReportV1,
    dict[str, ProspectiveRegenerationUnitV2Result],
]:
    root = output_root.expanduser().resolve()

    required = [
        row
        for row in primary_report.hypotheses
        if row.regeneration_fallback_required
    ]
    if len(required) != (
        primary_report.regeneration_fallback_required_count
    ):
        raise ValueError(
            "primary report regeneration-fallback accounting mismatch"
        )
    if not required:
        raise ValueError(
            "pre-N10 regeneration requires at least one fallback lineage"
        )

    lineages: list[PreN10RegenerationLineageV1] = []
    raw_results: dict[str, ProspectiveRegenerationUnitV2Result] = {}

    for source_row in required:
        source_id = source_row.hypothesis_id
        lineage_dir = root / "lineage" / _slug(source_id)
        backend = backend_factory(source_id, lineage_dir)

        result = execute_regeneration_generation_unit_v2(
            context=context,
            backend=backend,
            policy=unit_freeze.policy,
        )
        raw_results[source_id] = result

        result_path = lineage_dir / "regeneration_unit_v2.result.json"
        _write_exact_or_validate(result_path, result)

        portfolio_path: Path | None = None
        portfolio_id: str | None = None
        portfolio_count = 0
        if result.compiled_portfolio is not None:
            portfolio_path = lineage_dir / "regenerated.portfolio.json"
            _write_exact_or_validate(
                portfolio_path,
                result.compiled_portfolio,
            )
            portfolio_id = result.compiled_portfolio.portfolio_id
            portfolio_count = len(
                result.compiled_portfolio.hypotheses
            )

        lineages.append(
            PreN10RegenerationLineageV1(
                source_hypothesis_id=source_id,
                unit_result_path=str(result_path),
                unit_result_id=result.result_id,
                unit_result_sha256=result.result_sha256,
                unit_status=result.status,
                regenerated_portfolio_path=(
                    str(portfolio_path)
                    if portfolio_path is not None
                    else None
                ),
                regenerated_portfolio_id=portfolio_id,
                regenerated_hypothesis_count=portfolio_count,
            )
        )

    statuses = [row.unit_status for row in lineages]
    body = {
        "schema_version": "pre-n10-regeneration-execution-report-v1",
        "source_primary_report_id": primary_report.report_id,
        "source_primary_report_sha256": primary_report.report_sha256,
        "source_regeneration_unit_freeze_id": unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": unit_freeze.freeze_sha256,
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "lineages": [
            row.model_dump(mode="json")
            for row in lineages
        ],
        "regeneration_required_count": len(lineages),
        "generation_call_count": len(lineages),
        "repair_call_count": 0,
        "generated_and_compiled_count": statuses.count(
            "GENERATED_AND_COMPILED"
        ),
        "generated_abstention_count": statuses.count(
            "GENERATED_ABSTENTION_AND_COMPILED"
        ),
        "compile_rejected_count": statuses.count(
            "DETERMINISTIC_COMPILE_REJECTED"
        ),
        "generation_failed_count": statuses.count(
            "GENERATION_CALL_FAILED"
        ),
        "regeneration_trigger":
            "PRE_N10_CONTRACT_PRIMARY_UNRECOVERED",
        "full_e2e_rerun_performed": False,
        "semantic_critic_performed": False,
        "claim_decomposition_performed": False,
        "pre_n10_contract_reentry_performed": False,
        "retrieval_performed": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "previous_hypothesis_text_consumed": False,
        "external_novelty_outcome_consumed": False,
        "n10_outcome_consumed": False,
        "verifier_outcome_consumed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10RegenerationExecutionReportV1(
        **body,
        report_id=(
            "pre_n10_regeneration_execution_v1:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    _write_exact_or_validate(
        root / "regeneration.execution.json",
        report,
    )
    return report, raw_results


__all__ = [
    "PreN10RegenerationExecutionReportV1",
    "PreN10RegenerationLineageV1",
    "execute_pre_n10_regeneration_v1",
]
