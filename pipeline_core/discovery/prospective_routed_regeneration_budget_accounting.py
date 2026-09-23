from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    ProspectiveRoutedDecompositionPrimaryReport,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    ProspectiveRoutedDispatchReport,
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


BudgetDisposition = Literal[
    "NO_REGENERATION_FALLBACK_REQUIRED",
    "FROZEN_REGENERATION_BUDGET_CONFLICT",
    "FROZEN_REGENERATION_EXECUTABLE",
]


class RegenerationBudgetHypothesisAccounting(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    regeneration_fallback_required: bool

    frozen_fallback_operation: str | None
    frozen_fallback_max_attempts: int
    frozen_fallback_llm_calls_per_hypothesis_max: int

    compiled_regeneration_argv: list[str] | None = None
    compiled_regeneration_run_dir: str | None = None
    compiled_regeneration_is_full_e2e: bool = False

    disposition: BudgetDisposition
    reason_codes: list[str] = Field(default_factory=list)

    regeneration_executed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_mutation_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    verifier_result_observed: Literal[False] = False

    @model_validator(mode="after")
    def validate_row(self) -> "RegenerationBudgetHypothesisAccounting":
        if self.regeneration_fallback_required:
            if self.disposition == "NO_REGENERATION_FALLBACK_REQUIRED":
                raise ValueError(
                    "required fallback cannot be accounted as not required"
                )
        else:
            if self.disposition != "NO_REGENERATION_FALLBACK_REQUIRED":
                raise ValueError(
                    "non-required fallback must remain unexecuted/not required"
                )
        if self.disposition == "FROZEN_REGENERATION_BUDGET_CONFLICT":
            if not self.reason_codes:
                raise ValueError(
                    "budget conflict requires explicit reason codes"
                )
        return self


class ProspectiveRoutedRegenerationBudgetAccountingReport(StrictModel):
    schema_version: Literal[
        "prospective-routed-regeneration-budget-accounting-v1"
    ] = "prospective-routed-regeneration-budget-accounting-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P11", "P12", "P13", "P14", "P15"]

    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dispatch_report_id: str
    source_dispatch_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_decomposition_report_id: str
    source_decomposition_report_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    hypotheses: list[RegenerationBudgetHypothesisAccounting]
    hypothesis_count: int = Field(ge=0)
    disposition_counts: dict[str, int]
    regeneration_required_count: int = Field(ge=0)
    frozen_budget_conflict_count: int = Field(ge=0)
    frozen_executable_count: int = Field(ge=0)

    full_e2e_regeneration_forbidden_when_llm_budget_is_one: Literal[
        True
    ] = True
    posthoc_budget_reinterpretation_performed: Literal[False] = False
    execution_plan_modified: Literal[False] = False
    dispatch_modified: Literal[False] = False
    regeneration_executed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_mutation_performed: Literal[False] = False
    case_replacement_performed: Literal[False] = False
    later_case_settings_adapted: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveRoutedRegenerationBudgetAccountingReport":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")

        counts = Counter(row.disposition for row in self.hypotheses)
        if dict(sorted(counts.items())) != dict(
            sorted(self.disposition_counts.items())
        ):
            raise ValueError("disposition_counts mismatch")

        if self.regeneration_required_count != sum(
            row.regeneration_fallback_required
            for row in self.hypotheses
        ):
            raise ValueError("regeneration_required_count mismatch")
        if self.frozen_budget_conflict_count != sum(
            row.disposition == "FROZEN_REGENERATION_BUDGET_CONFLICT"
            for row in self.hypotheses
        ):
            raise ValueError("frozen_budget_conflict_count mismatch")
        if self.frozen_executable_count != sum(
            row.disposition == "FROZEN_REGENERATION_EXECUTABLE"
            for row in self.hypotheses
        ):
            raise ValueError("frozen_executable_count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "regeneration budget accounting SHA mismatch"
            )
        if observed_id != (
            "prospective_routed_regeneration_budget_accounting:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "regeneration budget accounting ID mismatch"
            )
        return self


def _is_full_e2e_argv(argv: list[str] | None) -> bool:
    if not argv:
        return False
    for index, value in enumerate(argv[:-1]):
        if value == "-m" and argv[index + 1] == (
            "scripts.discovery.run_dac_discovery_e2e"
        ):
            return True
    return False


def _regeneration_protocol(plan: ProspectiveRoutedExecutionPlan):
    matches = [
        row
        for row in plan.route_protocols
        if row.route_action
        == "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"
    ]
    if len(matches) != 1:
        raise ValueError(
            "execution plan must contain exactly one decomposition route "
            "protocol"
        )
    return matches[0]


def build_regeneration_budget_accounting(
    *,
    plan: ProspectiveRoutedExecutionPlan,
    dispatch: ProspectiveRoutedDispatchReport,
    decomposition: ProspectiveRoutedDecompositionPrimaryReport,
) -> ProspectiveRoutedRegenerationBudgetAccountingReport:
    if dispatch.case_id != decomposition.case_id:
        raise ValueError("dispatch/decomposition case mismatch")
    if decomposition.source_dispatch_report_id != dispatch.report_id:
        raise ValueError("decomposition/dispatch ID mismatch")
    if decomposition.source_dispatch_report_sha256 != dispatch.report_sha256:
        raise ValueError("decomposition/dispatch SHA mismatch")

    cases = [row for row in plan.cases if row.case_id == dispatch.case_id]
    if len(cases) != 1:
        raise ValueError(
            "case is not present exactly once in routed execution plan"
        )

    protocol = _regeneration_protocol(plan)
    if protocol.fallback_operation != "FRESH_REGENERATION":
        raise ValueError("frozen decomposition fallback is not regeneration")

    decisions = {
        row.final_hypothesis_id: row
        for row in dispatch.hypotheses
    }
    if len(decisions) != len(dispatch.hypotheses):
        raise ValueError("duplicate dispatch final hypothesis IDs")

    rows: list[RegenerationBudgetHypothesisAccounting] = []

    for source in decomposition.hypotheses:
        decision = decisions.get(source.final_hypothesis_id)
        if decision is None:
            raise ValueError(
                "decomposition hypothesis absent from dispatch: "
                + source.final_hypothesis_id
            )

        required = source.regeneration_fallback_required
        argv = decision.regeneration_argv
        run_dir = decision.regeneration_run_dir
        is_full_e2e = _is_full_e2e_argv(argv)
        reasons: list[str] = []

        if not required:
            disposition: BudgetDisposition = (
                "NO_REGENERATION_FALLBACK_REQUIRED"
            )
        else:
            if not decision.regeneration_fallback_allowed:
                reasons.append("dispatch_fallback_not_allowed")
            if protocol.fallback_max_attempts != 1:
                reasons.append(
                    "frozen_fallback_attempt_budget_not_one:"
                    + str(protocol.fallback_max_attempts)
                )
            if protocol.fallback_llm_calls_per_hypothesis_max != 1:
                reasons.append(
                    "frozen_fallback_llm_budget_not_one:"
                    + str(
                        protocol.fallback_llm_calls_per_hypothesis_max
                    )
                )
            if argv is None or run_dir is None:
                reasons.append("compiled_regeneration_command_missing")
            if (
                protocol.fallback_llm_calls_per_hypothesis_max == 1
                and is_full_e2e
            ):
                reasons.append(
                    "compiled_full_e2e_exceeds_single_llm_fallback_contract"
                )

            disposition = (
                "FROZEN_REGENERATION_BUDGET_CONFLICT"
                if reasons
                else "FROZEN_REGENERATION_EXECUTABLE"
            )

        rows.append(
            RegenerationBudgetHypothesisAccounting(
                candidate_hypothesis_id=source.candidate_hypothesis_id,
                final_hypothesis_id=source.final_hypothesis_id,
                regeneration_fallback_required=required,
                frozen_fallback_operation=protocol.fallback_operation,
                frozen_fallback_max_attempts=protocol.fallback_max_attempts,
                frozen_fallback_llm_calls_per_hypothesis_max=(
                    protocol.fallback_llm_calls_per_hypothesis_max
                ),
                compiled_regeneration_argv=argv,
                compiled_regeneration_run_dir=run_dir,
                compiled_regeneration_is_full_e2e=is_full_e2e,
                disposition=disposition,
                reason_codes=reasons,
            )
        )

    counts = Counter(row.disposition for row in rows)
    body = {
        "schema_version":
            "prospective-routed-regeneration-budget-accounting-v1",
        "case_id": dispatch.case_id,
        "source_execution_plan_id": plan.plan_id,
        "source_execution_plan_sha256": plan.plan_sha256,
        "source_dispatch_report_id": dispatch.report_id,
        "source_dispatch_report_sha256": dispatch.report_sha256,
        "source_decomposition_report_id": decomposition.report_id,
        "source_decomposition_report_sha256":
            decomposition.report_sha256,
        "hypotheses": [row.model_dump(mode="json") for row in rows],
        "hypothesis_count": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "regeneration_required_count": sum(
            row.regeneration_fallback_required for row in rows
        ),
        "frozen_budget_conflict_count": sum(
            row.disposition == "FROZEN_REGENERATION_BUDGET_CONFLICT"
            for row in rows
        ),
        "frozen_executable_count": sum(
            row.disposition == "FROZEN_REGENERATION_EXECUTABLE"
            for row in rows
        ),
        "full_e2e_regeneration_forbidden_when_llm_budget_is_one": True,
        "posthoc_budget_reinterpretation_performed": False,
        "execution_plan_modified": False,
        "dispatch_modified": False,
        "regeneration_executed": False,
        "llm_calls_performed": 0,
        "scientific_mutation_performed": False,
        "case_replacement_performed": False,
        "later_case_settings_adapted": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedRegenerationBudgetAccountingReport(
        **body,
        report_id=(
            "prospective_routed_regeneration_budget_accounting:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "BudgetDisposition",
    "ProspectiveRoutedRegenerationBudgetAccountingReport",
    "RegenerationBudgetHypothesisAccounting",
    "build_regeneration_budget_accounting",
]
