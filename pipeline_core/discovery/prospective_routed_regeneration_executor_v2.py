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
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
    ProspectiveRegenerationUnitV2Result,
    execute_regeneration_generation_unit_v2,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    ProspectiveRoutedPrimaryMaterializationReportV2,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    ProspectiveRoutedDispatchReportV2,
    RegenerationFallbackLineageV2,
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


class RoutedRegenerationUnitCaseResultV2(StrictModel):
    final_hypothesis_id: str
    source_route_action: str

    regeneration_required: Literal[True] = True
    regeneration_lineage: RegenerationFallbackLineageV2

    unit_result_id: str
    unit_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_status: str

    regenerated_portfolio_written: bool
    regenerated_portfolio_id: str | None = None
    regenerated_hypothesis_count: int = Field(ge=0)

    generation_calls_attempted: Literal[1] = 1
    repair_calls_attempted: Literal[0] = 0

    previous_hypothesis_text_consumed: Literal[False] = False
    novelty_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    downstream_evaluation_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False


class ProspectiveRoutedRegenerationExecutionReportV2(StrictModel):
    schema_version: Literal[
        "prospective-routed-regeneration-execution-v2"
    ] = "prospective-routed-regeneration-execution-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P16", "P17", "P18", "P19", "P20"]

    source_dispatch_report_id: str
    source_dispatch_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_primary_report_id: str
    source_primary_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_regeneration_unit_freeze_id: str
    source_regeneration_unit_freeze_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    hypotheses: list[RoutedRegenerationUnitCaseResultV2]
    regeneration_required_count: int = Field(ge=0)
    generation_call_count: int = Field(ge=0)
    repair_call_count: Literal[0] = 0

    generated_and_compiled_count: int = Field(ge=0)
    generated_abstention_count: int = Field(ge=0)
    compile_rejected_count: int = Field(ge=0)
    generation_failed_count: int = Field(ge=0)

    full_e2e_rerun_performed: Literal[False] = False
    retrieval_performed: Literal[False] = False
    semantic_critic_performed: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False

    previous_hypothesis_text_consumed: Literal[False] = False
    novelty_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveRoutedRegenerationExecutionReportV2":
        if self.regeneration_required_count != len(self.hypotheses):
            raise ValueError("regeneration_required_count mismatch")
        if self.generation_call_count != sum(
            row.generation_calls_attempted for row in self.hypotheses
        ):
            raise ValueError("generation_call_count mismatch")

        status_counts = {
            "GENERATED_AND_COMPILED": self.generated_and_compiled_count,
            "GENERATED_ABSTENTION_AND_COMPILED":
                self.generated_abstention_count,
            "DETERMINISTIC_COMPILE_REJECTED":
                self.compile_rejected_count,
            "GENERATION_CALL_FAILED": self.generation_failed_count,
        }
        observed = {}
        for row in self.hypotheses:
            observed[row.unit_status] = observed.get(row.unit_status, 0) + 1
        for status, count in status_counts.items():
            if observed.get(status, 0) != count:
                raise ValueError("regeneration status count mismatch: " + status)

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("routed regeneration execution SHA mismatch")
        if observed_id != (
            "prospective_routed_regeneration_execution_v2:"
            + expected_sha[:20]
        ):
            raise ValueError("routed regeneration execution ID mismatch")
        return self


BackendFactory = Callable[
    [str, RegenerationFallbackLineageV2],
    HypothesisDraftBackend,
]


def execute_routed_regeneration_units_v2(
    *,
    context: HypothesisContext,
    dispatch: ProspectiveRoutedDispatchReportV2,
    primary: ProspectiveRoutedPrimaryMaterializationReportV2,
    unit_freeze: ProspectiveRegenerationUnitV2Freeze,
    backend_factory: BackendFactory,
) -> tuple[
    ProspectiveRoutedRegenerationExecutionReportV2,
    dict[str, ProspectiveRegenerationUnitV2Result],
]:
    if primary.source_dispatch_report_id != dispatch.report_id:
        raise ValueError("primary/dispatch report ID mismatch")
    if primary.source_dispatch_report_sha256 != dispatch.report_sha256:
        raise ValueError("primary/dispatch report SHA mismatch")

    dispatch_by_final = {
        row.final_hypothesis_id: row for row in dispatch.hypotheses
    }
    if len(dispatch_by_final) != len(dispatch.hypotheses):
        raise ValueError("duplicate dispatch final hypothesis IDs")

    required_rows = [
        row for row in primary.hypotheses
        if row.regeneration_fallback_required
    ]
    results: list[RoutedRegenerationUnitCaseResultV2] = []
    raw_results: dict[str, ProspectiveRegenerationUnitV2Result] = {}

    for primary_row in required_rows:
        decision = dispatch_by_final.get(primary_row.final_hypothesis_id)
        if decision is None:
            raise ValueError(
                "primary fallback hypothesis absent from dispatch"
            )
        lineage = decision.regeneration_fallback
        if lineage is None:
            raise ValueError(
                "primary fallback requires frozen regeneration lineage"
            )
        if (
            Path(lineage.input_hypothesis_context_path).expanduser().resolve()
            != Path(context_path := lineage.input_hypothesis_context_path)
            .expanduser()
            .resolve()
        ):
            raise AssertionError(context_path)

        backend = backend_factory(
            primary_row.final_hypothesis_id,
            lineage,
        )
        unit_result = execute_regeneration_generation_unit_v2(
            context=context,
            backend=backend,
            policy=unit_freeze.policy,
        )
        raw_results[primary_row.final_hypothesis_id] = unit_result

        portfolio = unit_result.compiled_portfolio
        portfolio_written = portfolio is not None
        portfolio_count = (
            len(portfolio.hypotheses) if portfolio is not None else 0
        )

        results.append(
            RoutedRegenerationUnitCaseResultV2(
                final_hypothesis_id=primary_row.final_hypothesis_id,
                source_route_action=primary_row.route_action,
                regeneration_lineage=lineage,
                unit_result_id=unit_result.result_id,
                unit_result_sha256=unit_result.result_sha256,
                unit_status=unit_result.status,
                regenerated_portfolio_written=portfolio_written,
                regenerated_portfolio_id=(
                    portfolio.portfolio_id if portfolio is not None else None
                ),
                regenerated_hypothesis_count=portfolio_count,
            )
        )

    statuses = [row.unit_status for row in results]
    body = {
        "schema_version": "prospective-routed-regeneration-execution-v2",
        "case_id": dispatch.case_id,
        "source_dispatch_report_id": dispatch.report_id,
        "source_dispatch_report_sha256": dispatch.report_sha256,
        "source_primary_report_id": primary.report_id,
        "source_primary_report_sha256": primary.report_sha256,
        "source_regeneration_unit_freeze_id": unit_freeze.freeze_id,
        "source_regeneration_unit_freeze_sha256": unit_freeze.freeze_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in results],
        "regeneration_required_count": len(results),
        "generation_call_count": len(results),
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
        "full_e2e_rerun_performed": False,
        "retrieval_performed": False,
        "semantic_critic_performed": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "previous_hypothesis_text_consumed": False,
        "novelty_outcome_consumed": False,
        "verifier_outcome_consumed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = ProspectiveRoutedRegenerationExecutionReportV2(
        **body,
        report_id=(
            "prospective_routed_regeneration_execution_v2:"
            + digest[:20]
        ),
        report_sha256=digest,
    )
    return report, raw_results


__all__ = [
    "ProspectiveRoutedRegenerationExecutionReportV2",
    "RoutedRegenerationUnitCaseResultV2",
    "execute_routed_regeneration_units_v2",
]
