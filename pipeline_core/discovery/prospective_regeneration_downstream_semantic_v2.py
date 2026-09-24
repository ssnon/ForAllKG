from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_llm import HypothesisSemanticBackend
from pipeline_core.discovery.hypothesis_semantic_runtime import HypothesisSemanticCriticRuntime
from pipeline_core.discovery.prospective_regeneration_downstream_v2 import ProspectiveRegenerationDownstreamV2Freeze
from pipeline_core.discovery.prospective_routed_regeneration_executor_v2 import ProspectiveRoutedRegenerationExecutionReportV2


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SemanticCheckpointStatus = Literal[
    "SEMANTIC_ACCEPTED",
    "HARD_GATE_FAILED",
    "SEMANTIC_REVIEW_REJECTED",
    "SEMANTIC_STAGE_FAILED",
]


class RegenerationDownstreamSemanticLineageV2(StrictModel):
    source_final_hypothesis_id: str
    regenerated_portfolio_path: str
    regenerated_portfolio_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    regenerated_portfolio_id: str
    regenerated_hypothesis_count: int = Field(ge=0)

    downstream_dir: str
    semantic_prefix: str

    status: SemanticCheckpointStatus
    hard_gate_passed: bool | None = None
    semantic_generation_performed: bool
    semantic_review_accepted: bool

    evaluation_report_sha256: str | None = None
    semantic_run_id: str | None = None
    semantic_review_id: str | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    eligible_for_external_novelty: bool

    deterministic_benchmark_invocations: Literal[1] = 1
    semantic_critic_stage_invocations: Literal[1] = 1

    downstream_retry_performed: Literal[False] = False
    regenerated_portfolio_mutated: Literal[False] = False
    previous_hypothesis_text_consumed: Literal[False] = False
    novelty_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False


class ProspectiveRegenerationDownstreamSemanticReportV2(StrictModel):
    schema_version: Literal["prospective-regeneration-downstream-semantic-v2"] = (
        "prospective-regeneration-downstream-semantic-v2"
    )

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P16", "P17", "P18", "P19", "P20"]

    source_regeneration_execution_report_id: str
    source_regeneration_execution_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_downstream_freeze_id: str
    source_downstream_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_context_id: str
    source_context_sha256: str

    lineages: list[RegenerationDownstreamSemanticLineageV2]
    lineage_count: int = Field(ge=0)

    semantic_accepted_count: int = Field(ge=0)
    hard_gate_failed_count: int = Field(ge=0)
    semantic_review_rejected_count: int = Field(ge=0)
    semantic_stage_failed_count: int = Field(ge=0)
    eligible_for_external_novelty_count: int = Field(ge=0)

    deterministic_benchmark_stage_invocations: int = Field(ge=0)
    semantic_critic_stage_invocations: int = Field(ge=0)

    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    binding_plan_performed: Literal[False] = False
    preverifier_gate_v2_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False

    downstream_failure_retry_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    regenerated_portfolio_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self):
        if self.lineage_count != len(self.lineages):
            raise ValueError("lineage_count mismatch")
        if self.deterministic_benchmark_stage_invocations != len(self.lineages):
            raise ValueError("benchmark invocation count mismatch")
        if self.semantic_critic_stage_invocations != len(self.lineages):
            raise ValueError("semantic invocation count mismatch")

        statuses = [row.status for row in self.lineages]
        expected = {
            "SEMANTIC_ACCEPTED": self.semantic_accepted_count,
            "HARD_GATE_FAILED": self.hard_gate_failed_count,
            "SEMANTIC_REVIEW_REJECTED": self.semantic_review_rejected_count,
            "SEMANTIC_STAGE_FAILED": self.semantic_stage_failed_count,
        }
        for status, count in expected.items():
            if statuses.count(status) != count:
                raise ValueError("semantic status count mismatch: " + status)

        if self.eligible_for_external_novelty_count != sum(
            row.eligible_for_external_novelty for row in self.lineages
        ):
            raise ValueError("external-novelty eligibility count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("downstream semantic-v2 SHA mismatch")
        if observed_id != "prospective_regeneration_downstream_semantic_v2:" + expected_sha[:20]:
            raise ValueError("downstream semantic-v2 ID mismatch")
        return self


BackendFactory = Callable[[str, Path], HypothesisSemanticBackend]


def execute_regeneration_downstream_semantic_v2(
    *,
    context: HypothesisContext,
    regeneration_execution: ProspectiveRoutedRegenerationExecutionReportV2,
    downstream_freeze: ProspectiveRegenerationDownstreamV2Freeze,
    backend_factory: BackendFactory,
) -> tuple[ProspectiveRegenerationDownstreamSemanticReportV2, dict[str, Any]]:
    policy = downstream_freeze.policy
    budget = policy.budget

    if budget.deterministic_hypothesis_benchmark_max != 1:
        raise ValueError("frozen deterministic benchmark budget drift")
    if budget.semantic_critic_stage_max != 1:
        raise ValueError("frozen semantic critic budget drift")
    if policy.downstream_failure_retry_allowed:
        raise ValueError("downstream retry must remain forbidden")
    if policy.regenerated_portfolio_mutation_allowed:
        raise ValueError("regenerated portfolio mutation must remain forbidden")

    lineages = []
    outcomes: dict[str, Any] = {}

    for source_row in regeneration_execution.hypotheses:
        if source_row.unit_status not in {
            "GENERATED_AND_COMPILED",
            "GENERATED_ABSTENTION_AND_COMPILED",
        }:
            continue
        if not source_row.regenerated_portfolio_written:
            continue

        lineage = source_row.regeneration_lineage
        portfolio_path = Path(lineage.regenerated_portfolio_path).expanduser().resolve()
        if not portfolio_path.is_file():
            raise ValueError("regenerated portfolio missing: " + str(portfolio_path))

        before_sha = _sha256_file(portfolio_path)
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )

        downstream_dir = Path(lineage.downstream_dir).expanduser().resolve()
        semantic_prefix = downstream_dir / "semantic_v262"
        backend = backend_factory(source_row.final_hypothesis_id, semantic_prefix)

        try:
            outcome = HypothesisSemanticCriticRuntime(backend).run(context, portfolio)
            outcomes[source_row.final_hypothesis_id] = outcome

            if not outcome.evaluation.hard_gate_passed:
                status: SemanticCheckpointStatus = "HARD_GATE_FAILED"
            elif outcome.accepted:
                status = "SEMANTIC_ACCEPTED"
            else:
                status = "SEMANTIC_REVIEW_REJECTED"

            row = RegenerationDownstreamSemanticLineageV2(
                source_final_hypothesis_id=source_row.final_hypothesis_id,
                regenerated_portfolio_path=str(portfolio_path),
                regenerated_portfolio_file_sha256=before_sha,
                regenerated_portfolio_id=portfolio.portfolio_id,
                regenerated_hypothesis_count=len(portfolio.hypotheses),
                downstream_dir=str(downstream_dir),
                semantic_prefix=str(semantic_prefix),
                status=status,
                hard_gate_passed=outcome.evaluation.hard_gate_passed,
                semantic_generation_performed=(outcome.generation is not None),
                semantic_review_accepted=outcome.accepted,
                evaluation_report_sha256=_sha256_json(outcome.evaluation.model_dump(mode="json")),
                semantic_run_id=outcome.run_record.run_id,
                semantic_review_id=(
                    outcome.review.review_id if outcome.review is not None else None
                ),
                failure_stage=outcome.run_record.failure_stage,
                eligible_for_external_novelty=(status == "SEMANTIC_ACCEPTED"),
            )
        except Exception as exc:
            row = RegenerationDownstreamSemanticLineageV2(
                source_final_hypothesis_id=source_row.final_hypothesis_id,
                regenerated_portfolio_path=str(portfolio_path),
                regenerated_portfolio_file_sha256=before_sha,
                regenerated_portfolio_id=portfolio.portfolio_id,
                regenerated_hypothesis_count=len(portfolio.hypotheses),
                downstream_dir=str(downstream_dir),
                semantic_prefix=str(semantic_prefix),
                status="SEMANTIC_STAGE_FAILED",
                hard_gate_passed=None,
                semantic_generation_performed=False,
                semantic_review_accepted=False,
                failure_stage="semantic_stage_exception",
                failure_type=type(exc).__name__,
                failure_message=str(exc),
                eligible_for_external_novelty=False,
            )

        if _sha256_file(portfolio_path) != before_sha:
            raise ValueError("regenerated portfolio mutated during downstream semantic stage")
        lineages.append(row)

    statuses = [row.status for row in lineages]
    body = {
        "schema_version": "prospective-regeneration-downstream-semantic-v2",
        "case_id": regeneration_execution.case_id,
        "source_regeneration_execution_report_id": regeneration_execution.report_id,
        "source_regeneration_execution_report_sha256": regeneration_execution.report_sha256,
        "source_downstream_freeze_id": downstream_freeze.freeze_id,
        "source_downstream_freeze_sha256": downstream_freeze.freeze_sha256,
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "lineages": [row.model_dump(mode="json") for row in lineages],
        "lineage_count": len(lineages),
        "semantic_accepted_count": statuses.count("SEMANTIC_ACCEPTED"),
        "hard_gate_failed_count": statuses.count("HARD_GATE_FAILED"),
        "semantic_review_rejected_count": statuses.count("SEMANTIC_REVIEW_REJECTED"),
        "semantic_stage_failed_count": statuses.count("SEMANTIC_STAGE_FAILED"),
        "eligible_for_external_novelty_count": sum(
            row.eligible_for_external_novelty for row in lineages
        ),
        "deterministic_benchmark_stage_invocations": len(lineages),
        "semantic_critic_stage_invocations": len(lineages),
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "binding_plan_performed": False,
        "preverifier_gate_v2_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "downstream_failure_retry_performed": False,
        "second_regeneration_performed": False,
        "regenerated_portfolio_mutated": False,
    }
    digest = _sha256_json(body)
    report = ProspectiveRegenerationDownstreamSemanticReportV2(
        **body,
        report_id="prospective_regeneration_downstream_semantic_v2:" + digest[:20],
        report_sha256=digest,
    )
    return report, outcomes
