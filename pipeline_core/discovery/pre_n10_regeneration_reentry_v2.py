from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.hypothesis_semantic_disposition import (
    compile_hypothesis_semantic_disposition_v1,
)
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticOutcome,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    NoveltyClaimBackend,
    NoveltyClaimDecomposer,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    PreN10RegenerationExecutionReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    build_pre_n10_scientific_contract_v1,
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
                "existing write-once pre-N10 re-entry-v2 artifact differs: "
                + str(path)
            )
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(expected)
    return hashlib.sha256(expected).hexdigest()


class SemanticRunner(Protocol):
    def run(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> HypothesisSemanticOutcome: ...


SemanticRunnerFactory = Callable[[str, Path], SemanticRunner]
DecompositionBackendFactory = Callable[[str, Path], NoveltyClaimBackend]


ReentryStatusV2 = Literal[
    "NO_REGENERATED_HYPOTHESIS",
    "SEMANTIC_HARD_GATE_FAILED",
    "SEMANTIC_REVIEW_INVALID",
    "SEMANTIC_INTERVENTION_REQUIRED",
    "SEMANTIC_STAGE_FAILED",
    "PRE_N10_READY",
    "PRE_N10_INTERVENTION_REQUIRED",
]


class PreN10RegenerationReentryLineageV2(StrictModel):
    source_hypothesis_id: str
    regenerated_portfolio_path: str | None
    regenerated_portfolio_id: str | None
    regenerated_hypothesis_count: int = Field(ge=0)

    semantic_status: str
    semantic_hard_gate_passed: bool | None = None
    semantic_review_valid: bool = False
    semantic_disposition_path: str | None = None
    semantic_disposition_id: str | None = None
    semantic_disposition: str | None = None
    semantic_admissible_for_pre_n10: bool = False
    semantic_failed_dimensions: list[str] = Field(default_factory=list)
    semantic_warning_dimensions: list[str] = Field(default_factory=list)
    semantic_runtime_invoked: bool
    semantic_critic_llm_invoked: bool

    query_plan_path: str | None = None
    query_plan_id: str | None = None
    contract_report_path: str | None = None
    contract_report_id: str | None = None

    claim_decomposition_request_count: int = Field(ge=0)
    pre_n10_disposition: str | None = None
    final_status: ReentryStatusV2
    ready_for_n10: bool

    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    previous_hypothesis_text_consumed: Literal[False] = False
    external_novelty_outcome_consumed: Literal[False] = False
    n10_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    @model_validator(mode="after")
    def validate_lineage(self) -> "PreN10RegenerationReentryLineageV2":
        if self.ready_for_n10 != (self.final_status == "PRE_N10_READY"):
            raise ValueError("ready_for_n10 mismatch")

        if self.final_status in {
            "PRE_N10_READY",
            "PRE_N10_INTERVENTION_REQUIRED",
        }:
            if not self.semantic_review_valid:
                raise ValueError(
                    "pre-N10 contract re-entry requires valid semantic review"
                )
            if not self.semantic_admissible_for_pre_n10:
                raise ValueError(
                    "pre-N10 contract re-entry requires semantic admissibility"
                )
            if self.semantic_disposition != "PASS":
                raise ValueError(
                    "pre-N10 contract re-entry requires PASS disposition"
                )
            if (
                self.semantic_disposition_path is None
                or self.semantic_disposition_id is None
            ):
                raise ValueError(
                    "pre-N10 contract re-entry requires semantic disposition artifact"
                )
            if self.semantic_failed_dimensions:
                raise ValueError(
                    "pre-N10 contract re-entry cannot retain failed semantic dimensions"
                )
            if self.query_plan_path is None or self.contract_report_path is None:
                raise ValueError(
                    "pre-N10 contract re-entry requires query-plan artifacts"
                )
            if self.claim_decomposition_request_count < 1:
                raise ValueError(
                    "pre-N10 contract re-entry requires fresh decomposition"
                )
            return self

        if self.final_status == "SEMANTIC_INTERVENTION_REQUIRED":
            if not self.semantic_review_valid:
                raise ValueError(
                    "semantic intervention requires valid semantic review"
                )
            if self.semantic_admissible_for_pre_n10:
                raise ValueError(
                    "semantic intervention cannot be pre-N10 admissible"
                )
            if self.semantic_disposition != "REQUIRES_SEMANTIC_INTERVENTION":
                raise ValueError("semantic intervention disposition mismatch")
            if (
                self.semantic_disposition_path is None
                or self.semantic_disposition_id is None
            ):
                raise ValueError(
                    "semantic intervention requires semantic disposition artifact"
                )
            if not self.semantic_failed_dimensions:
                raise ValueError(
                    "semantic intervention requires at least one failed dimension"
                )
            if self.claim_decomposition_request_count != 0:
                raise ValueError(
                    "semantic intervention lineage cannot decompose claims"
                )
            return self

        if self.claim_decomposition_request_count != 0:
            raise ValueError(
                "semantic-terminal lineage cannot decompose claims"
            )
        if self.semantic_admissible_for_pre_n10:
            raise ValueError(
                "semantic-terminal lineage cannot be pre-N10 admissible"
            )
        return self


class PreN10RegenerationReentryReportV2(StrictModel):
    schema_version: Literal[
        "pre-n10-regeneration-reentry-report-v2"
    ] = "pre-n10-regeneration-reentry-report-v2"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_regeneration_report_id: str
    source_regeneration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_context_id: str
    source_context_sha256: str

    lineages: list[PreN10RegenerationReentryLineageV2]
    lineage_count: int = Field(ge=0)

    semantic_runtime_invocation_count: int = Field(ge=0)
    semantic_critic_llm_invocation_count: int = Field(ge=0)
    semantic_review_valid_count: int = Field(ge=0)
    semantic_admissible_count: int = Field(ge=0)
    semantic_intervention_required_count: int = Field(ge=0)
    semantic_terminal_count: int = Field(ge=0)

    claim_decomposition_request_count: int = Field(ge=0)
    ready_for_n10_count: int = Field(ge=0)
    contract_intervention_required_count: int = Field(ge=0)

    semantic_fail_blocks_pre_n10: Literal[True] = True
    semantic_warning_blocks_pre_n10: Literal[False] = False
    semantic_disposition_is_not_final_rejection_authority: Literal[True] = True
    semantic_disposition_is_not_novelty_authority: Literal[True] = True

    retrieval_performed: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    second_regeneration_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "PreN10RegenerationReentryReportV2":
        if self.lineage_count != len(self.lineages):
            raise ValueError("lineage_count mismatch")
        if self.semantic_runtime_invocation_count != sum(
            row.semantic_runtime_invoked for row in self.lineages
        ):
            raise ValueError("semantic runtime invocation count mismatch")
        if self.semantic_critic_llm_invocation_count != sum(
            row.semantic_critic_llm_invoked for row in self.lineages
        ):
            raise ValueError("semantic critic LLM count mismatch")
        if self.semantic_review_valid_count != sum(
            row.semantic_review_valid for row in self.lineages
        ):
            raise ValueError("semantic review-valid count mismatch")
        if self.semantic_admissible_count != sum(
            row.semantic_admissible_for_pre_n10 for row in self.lineages
        ):
            raise ValueError("semantic admissible count mismatch")
        if self.semantic_intervention_required_count != sum(
            row.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
            for row in self.lineages
        ):
            raise ValueError("semantic intervention count mismatch")
        if self.semantic_terminal_count != sum(
            row.final_status in {
                "NO_REGENERATED_HYPOTHESIS",
                "SEMANTIC_HARD_GATE_FAILED",
                "SEMANTIC_REVIEW_INVALID",
                "SEMANTIC_INTERVENTION_REQUIRED",
                "SEMANTIC_STAGE_FAILED",
            }
            for row in self.lineages
        ):
            raise ValueError("semantic terminal count mismatch")
        if self.claim_decomposition_request_count != sum(
            row.claim_decomposition_request_count for row in self.lineages
        ):
            raise ValueError("claim decomposition count mismatch")
        if self.ready_for_n10_count != sum(
            row.ready_for_n10 for row in self.lineages
        ):
            raise ValueError("ready-for-N10 count mismatch")
        if self.contract_intervention_required_count != sum(
            row.final_status == "PRE_N10_INTERVENTION_REQUIRED"
            for row in self.lineages
        ):
            raise ValueError("contract intervention count mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("pre-N10 regeneration re-entry-v2 SHA mismatch")
        if observed_id != (
            "pre_n10_regeneration_reentry_v2:" + expected_sha[:20]
        ):
            raise ValueError("pre-N10 regeneration re-entry-v2 ID mismatch")
        return self


def execute_pre_n10_regeneration_reentry_v2(
    *,
    context: HypothesisContext,
    regeneration_report: PreN10RegenerationExecutionReportV1,
    semantic_runner_factory: SemanticRunnerFactory,
    decomposition_backend_factory: DecompositionBackendFactory,
    output_root: Path,
    max_claims: int = 4,
    max_queries_per_claim: int = 2,
) -> tuple[
    PreN10RegenerationReentryReportV2,
    dict[str, HypothesisSemanticOutcome],
]:
    if context.context_id != regeneration_report.source_context_id:
        raise ValueError("context/regeneration report ID mismatch")
    if context.context_sha256 != regeneration_report.source_context_sha256:
        raise ValueError("context/regeneration report SHA mismatch")

    root = output_root.expanduser().resolve()
    rows: list[PreN10RegenerationReentryLineageV2] = []
    outcomes: dict[str, HypothesisSemanticOutcome] = {}

    for source in regeneration_report.lineages:
        lineage_dir = root / "lineage" / (
            source.source_hypothesis_id.replace(":", "_").replace("/", "_")
        )

        if (
            source.regenerated_portfolio_path is None
            or source.regenerated_hypothesis_count == 0
        ):
            rows.append(
                PreN10RegenerationReentryLineageV2(
                    source_hypothesis_id=source.source_hypothesis_id,
                    regenerated_portfolio_path=source.regenerated_portfolio_path,
                    regenerated_portfolio_id=source.regenerated_portfolio_id,
                    regenerated_hypothesis_count=source.regenerated_hypothesis_count,
                    semantic_status="NOT_RUN_NO_REGENERATED_HYPOTHESIS",
                    semantic_runtime_invoked=False,
                    semantic_critic_llm_invoked=False,
                    claim_decomposition_request_count=0,
                    final_status="NO_REGENERATED_HYPOTHESIS",
                    ready_for_n10=False,
                )
            )
            continue

        portfolio_path = Path(
            source.regenerated_portfolio_path
        ).expanduser().resolve()
        if not portfolio_path.is_file():
            raise ValueError(
                "regenerated portfolio missing: " + str(portfolio_path)
            )
        portfolio_sha_before = _sha256_file(portfolio_path)
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        if portfolio.portfolio_id != source.regenerated_portfolio_id:
            raise ValueError("regeneration report/portfolio ID mismatch")
        if len(portfolio.hypotheses) != source.regenerated_hypothesis_count:
            raise ValueError(
                "regeneration report/portfolio hypothesis count mismatch"
            )

        semantic_runner = semantic_runner_factory(
            source.source_hypothesis_id,
            lineage_dir,
        )
        try:
            outcome = semantic_runner.run(context, portfolio)
            outcomes[source.source_hypothesis_id] = outcome
        except Exception:
            rows.append(
                PreN10RegenerationReentryLineageV2(
                    source_hypothesis_id=source.source_hypothesis_id,
                    regenerated_portfolio_path=str(portfolio_path),
                    regenerated_portfolio_id=portfolio.portfolio_id,
                    regenerated_hypothesis_count=len(portfolio.hypotheses),
                    semantic_status="SEMANTIC_STAGE_EXCEPTION",
                    semantic_runtime_invoked=True,
                    semantic_critic_llm_invoked=True,
                    claim_decomposition_request_count=0,
                    final_status="SEMANTIC_STAGE_FAILED",
                    ready_for_n10=False,
                )
            )
            continue

        if _sha256_file(portfolio_path) != portfolio_sha_before:
            raise ValueError(
                "regenerated portfolio mutated during semantic re-entry-v2"
            )

        hard_gate = bool(outcome.evaluation.hard_gate_passed)
        critic_called = outcome.generation is not None

        if not hard_gate:
            rows.append(
                PreN10RegenerationReentryLineageV2(
                    source_hypothesis_id=source.source_hypothesis_id,
                    regenerated_portfolio_path=str(portfolio_path),
                    regenerated_portfolio_id=portfolio.portfolio_id,
                    regenerated_hypothesis_count=len(portfolio.hypotheses),
                    semantic_status="HARD_GATE_FAILED",
                    semantic_hard_gate_passed=False,
                    semantic_review_valid=False,
                    semantic_runtime_invoked=True,
                    semantic_critic_llm_invoked=False,
                    claim_decomposition_request_count=0,
                    final_status="SEMANTIC_HARD_GATE_FAILED",
                    ready_for_n10=False,
                )
            )
            continue

        if not outcome.accepted or outcome.review is None:
            rows.append(
                PreN10RegenerationReentryLineageV2(
                    source_hypothesis_id=source.source_hypothesis_id,
                    regenerated_portfolio_path=str(portfolio_path),
                    regenerated_portfolio_id=portfolio.portfolio_id,
                    regenerated_hypothesis_count=len(portfolio.hypotheses),
                    semantic_status="SEMANTIC_REVIEW_INVALID",
                    semantic_hard_gate_passed=True,
                    semantic_review_valid=False,
                    semantic_runtime_invoked=True,
                    semantic_critic_llm_invoked=critic_called,
                    claim_decomposition_request_count=0,
                    final_status="SEMANTIC_REVIEW_INVALID",
                    ready_for_n10=False,
                )
            )
            continue

        disposition = compile_hypothesis_semantic_disposition_v1(
            review=outcome.review,
            portfolio=portfolio,
        )
        disposition_path = lineage_dir / "semantic.disposition.json"
        _write_exact_or_validate(disposition_path, disposition)

        if not disposition.semantic_admissible_for_pre_n10:
            rows.append(
                PreN10RegenerationReentryLineageV2(
                    source_hypothesis_id=source.source_hypothesis_id,
                    regenerated_portfolio_path=str(portfolio_path),
                    regenerated_portfolio_id=portfolio.portfolio_id,
                    regenerated_hypothesis_count=len(portfolio.hypotheses),
                    semantic_status="SEMANTIC_INTERVENTION_REQUIRED",
                    semantic_hard_gate_passed=True,
                    semantic_review_valid=True,
                    semantic_disposition_path=str(disposition_path),
                    semantic_disposition_id=disposition.disposition_id,
                    semantic_disposition=disposition.disposition,
                    semantic_admissible_for_pre_n10=False,
                    semantic_failed_dimensions=list(disposition.failed_dimensions),
                    semantic_warning_dimensions=list(disposition.warning_dimensions),
                    semantic_runtime_invoked=True,
                    semantic_critic_llm_invoked=critic_called,
                    claim_decomposition_request_count=0,
                    final_status="SEMANTIC_INTERVENTION_REQUIRED",
                    ready_for_n10=False,
                )
            )
            continue

        decomposition_backend = decomposition_backend_factory(
            source.source_hypothesis_id,
            lineage_dir,
        )
        decomposer = NoveltyClaimDecomposer(
            decomposition_backend,
            max_claims_per_hypothesis=max_claims,
            max_queries_per_claim=max_queries_per_claim,
        )
        decompositions = [
            decomposer.decompose(card)
            for card in portfolio.hypotheses
        ]
        query_plan: LiteratureQueryPlan = LiteratureQueryPlanner().build(
            portfolio,
            decompositions,
        )
        query_plan_path = lineage_dir / "claims_queries.json"
        _write_exact_or_validate(query_plan_path, query_plan)

        contract = build_pre_n10_scientific_contract_v1(
            portfolio_path=portfolio_path,
            query_plan_path=query_plan_path,
            claim_decomposition_request_count=len(portfolio.hypotheses),
        )
        contract_path = lineage_dir / "contract.json"
        _write_exact_or_validate(contract_path, contract)

        ready = contract.disposition == "READY_FOR_N10"
        rows.append(
            PreN10RegenerationReentryLineageV2(
                source_hypothesis_id=source.source_hypothesis_id,
                regenerated_portfolio_path=str(portfolio_path),
                regenerated_portfolio_id=portfolio.portfolio_id,
                regenerated_hypothesis_count=len(portfolio.hypotheses),
                semantic_status="SEMANTIC_ADMISSIBLE",
                semantic_hard_gate_passed=True,
                semantic_review_valid=True,
                semantic_disposition_path=str(disposition_path),
                semantic_disposition_id=disposition.disposition_id,
                semantic_disposition=disposition.disposition,
                semantic_admissible_for_pre_n10=True,
                semantic_failed_dimensions=list(disposition.failed_dimensions),
                semantic_warning_dimensions=list(disposition.warning_dimensions),
                semantic_runtime_invoked=True,
                semantic_critic_llm_invoked=critic_called,
                query_plan_path=str(query_plan_path),
                query_plan_id=query_plan.plan_id,
                contract_report_path=str(contract_path),
                contract_report_id=contract.report_id,
                claim_decomposition_request_count=len(portfolio.hypotheses),
                pre_n10_disposition=contract.disposition,
                final_status=(
                    "PRE_N10_READY"
                    if ready
                    else "PRE_N10_INTERVENTION_REQUIRED"
                ),
                ready_for_n10=ready,
            )
        )

    body = {
        "schema_version": "pre-n10-regeneration-reentry-report-v2",
        "source_regeneration_report_id": regeneration_report.report_id,
        "source_regeneration_report_sha256": regeneration_report.report_sha256,
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "lineages": [row.model_dump(mode="json") for row in rows],
        "lineage_count": len(rows),
        "semantic_runtime_invocation_count": sum(
            row.semantic_runtime_invoked for row in rows
        ),
        "semantic_critic_llm_invocation_count": sum(
            row.semantic_critic_llm_invoked for row in rows
        ),
        "semantic_review_valid_count": sum(
            row.semantic_review_valid for row in rows
        ),
        "semantic_admissible_count": sum(
            row.semantic_admissible_for_pre_n10 for row in rows
        ),
        "semantic_intervention_required_count": sum(
            row.final_status == "SEMANTIC_INTERVENTION_REQUIRED"
            for row in rows
        ),
        "semantic_terminal_count": sum(
            row.final_status in {
                "NO_REGENERATED_HYPOTHESIS",
                "SEMANTIC_HARD_GATE_FAILED",
                "SEMANTIC_REVIEW_INVALID",
                "SEMANTIC_INTERVENTION_REQUIRED",
                "SEMANTIC_STAGE_FAILED",
            }
            for row in rows
        ),
        "claim_decomposition_request_count": sum(
            row.claim_decomposition_request_count for row in rows
        ),
        "ready_for_n10_count": sum(
            row.ready_for_n10 for row in rows
        ),
        "contract_intervention_required_count": sum(
            row.final_status == "PRE_N10_INTERVENTION_REQUIRED"
            for row in rows
        ),
        "semantic_fail_blocks_pre_n10": True,
        "semantic_warning_blocks_pre_n10": False,
        "semantic_disposition_is_not_final_rejection_authority": True,
        "semantic_disposition_is_not_novelty_authority": True,
        "retrieval_performed": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "second_regeneration_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    report = PreN10RegenerationReentryReportV2(
        **body,
        report_id=(
            "pre_n10_regeneration_reentry_v2:" + digest[:20]
        ),
        report_sha256=digest,
    )
    _write_exact_or_validate(root / "reentry_v2.report.json", report)
    return report, outcomes


__all__ = [
    "PreN10RegenerationReentryLineageV2",
    "PreN10RegenerationReentryReportV2",
    "ReentryStatusV2",
    "execute_pre_n10_regeneration_reentry_v2",
]
