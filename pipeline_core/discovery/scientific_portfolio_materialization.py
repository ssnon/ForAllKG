from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioSelectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _norm_observable(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[\u2010-\u2015\u2212-]+", " ", value)
    value = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _observable_matches(a: str, b: str) -> bool:
    na = _norm_observable(a)
    nb = _norm_observable(b)
    if not na or not nb:
        return False
    return na == nb or (len(na) >= 5 and na in nb) or (len(nb) >= 5 and nb in na)


MaterializationStatus = Literal[
    "MATERIALIZED",
    "ABSTAINED",
    "COMPILE_REJECTED",
    "HARD_GATE_REJECTED",
    "MISSING_DRAFT",
    "GENERATION_FAILED",
]


class ScientificPortfolioMaterializationItemDraft(StrictModel):
    candidate_id: str
    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    hypothesis_type: Literal[
        "mechanistic_extension",
        "cross_evidence_synthesis",
        "design_lever_interaction",
        "descriptor_mediation",
        "context_dependency",
    ]
    premise_statement_ids: list[str] = Field(min_length=1)
    gap_statement_ids: list[str] = Field(default_factory=list)
    inferential_bridge: str = Field(min_length=1)
    predicted_observations: list[PredictedObservationDraft] = Field(min_length=1)
    falsification_criteria: list[FalsificationCriterionDraft] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_testability_alignment(
        self,
    ) -> "ScientificPortfolioMaterializationItemDraft":
        for falsifier in self.falsification_criteria:
            if not any(
                _observable_matches(
                    falsifier.observable,
                    prediction.observable,
                )
                for prediction in self.predicted_observations
            ):
                raise ValueError(
                    "every falsifier observable must correspond to at least "
                    "one predicted observable"
                )
        return self


class ScientificPortfolioMaterializationAbstention(StrictModel):
    candidate_id: str
    reason: str = Field(min_length=1)


class ScientificPortfolioMaterializationBatchDraft(StrictModel):
    schema_version: Literal[
        "scientific-portfolio-materialization-batch-draft-v1"
    ] = "scientific-portfolio-materialization-batch-draft-v1"
    items: list[ScientificPortfolioMaterializationItemDraft] = Field(default_factory=list)
    abstentions: list[ScientificPortfolioMaterializationAbstention] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _unique(self) -> "ScientificPortfolioMaterializationBatchDraft":
        ids = [x.candidate_id for x in self.items] + [x.candidate_id for x in self.abstentions]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id may appear only once across items and abstentions")
        return self


class ScientificPortfolioMaterializationRecord(StrictModel):
    candidate_id: str
    source_object_id: str
    status: MaterializationStatus
    hypothesis_id: str | None = None
    issue_codes: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ScientificPortfolioMaterializationReport(StrictModel):
    schema_version: Literal[
        "scientific-portfolio-materialization-report-v1"
    ] = "scientific-portfolio-materialization-report-v1"
    report_id: str
    report_sha256: str
    source_selection_id: str
    source_selection_sha256: str
    source_context_id: str
    source_context_sha256: str
    output_portfolio_id: str
    output_portfolio_sha256: str
    selected_candidate_count: int = Field(ge=0)
    materialized_hypothesis_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    compile_rejected_count: int = Field(ge=0)
    hard_gate_rejected_count: int = Field(ge=0)
    missing_draft_count: int = Field(ge=0)
    generation_failed_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    records: list[ScientificPortfolioMaterializationRecord] = Field(default_factory=list)

    idea_used_as_inspiration_only: Literal[True] = True
    external_literature_as_positive_premise: Literal[False] = False
    candidate_lineage_as_automatic_positive_premise: Literal[False] = False
    standard_hypothesis_compiler_used: Literal[True] = True
    standard_hypothesis_validator_used: Literal[True] = True
    exploration_retention_independent_of_materialization: Literal[True] = True
    materialization_failure_is_not_exploration_rejection: Literal[True] = True
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _proposal(item: ScientificPortfolioMaterializationItemDraft) -> HypothesisProposalDraft:
    return HypothesisProposalDraft(
        local_id=item.candidate_id,
        title=item.title,
        hypothesis_statement=item.hypothesis_statement,
        hypothesis_type=item.hypothesis_type,
        premise_statement_ids=list(item.premise_statement_ids),
        gap_statement_ids=list(item.gap_statement_ids),
        inferential_bridge=item.inferential_bridge,
        predicted_observations=list(item.predicted_observations),
        falsification_criteria=list(item.falsification_criteria),
        assumptions=list(item.assumptions),
    )


def _issue_code(value: str) -> str:
    return str(value or "MATERIALIZATION_ERROR").strip() or "MATERIALIZATION_ERROR"


def compile_scientific_portfolio_materialization(
    *,
    context: HypothesisContext,
    pool: ScientificPortfolioCandidatePool,
    selection: ScientificPortfolioSelectionReport,
    draft: ScientificPortfolioMaterializationBatchDraft,
    generation_error: str | None = None,
) -> tuple[ScientificPortfolioMaterializationReport, HypothesisPortfolio]:
    if selection.source_pool_id != pool.pool_id:
        raise ValueError("selection/pool lineage mismatch")
    if pool.source_context_id != context.context_id:
        raise ValueError("pool/context ID mismatch")
    if pool.source_context_sha256 != context.context_sha256:
        raise ValueError("pool/context SHA mismatch")

    selected_ids = list(selection.retained_candidate_ids)
    selected_set = set(selected_ids)
    candidate_by_id = {x.candidate_id: x for x in pool.candidates}
    item_by_id = {x.candidate_id: x for x in draft.items}
    abstention_by_id = {x.candidate_id: x for x in draft.abstentions}
    supplied = set(item_by_id) | set(abstention_by_id)
    unknown = sorted(supplied - selected_set)
    if unknown:
        raise ValueError(f"materialization draft references non-selected candidates: {unknown}")

    compiler = HypothesisCompiler()
    validator = HypothesisValidator()
    cards = []
    records: list[ScientificPortfolioMaterializationRecord] = []

    if generation_error is not None:
        for candidate_id in selected_ids:
            candidate = candidate_by_id[candidate_id]
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="GENERATION_FAILED",
                    issue_codes=["MATERIALIZATION_GENERATION_FAILED"],
                    issues=[str(generation_error)],
                )
            )

    for candidate_id in ([] if generation_error is not None else selected_ids):
        candidate = candidate_by_id[candidate_id]
        if candidate_id in abstention_by_id:
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="ABSTAINED",
                    issue_codes=["MODEL_ABSTAINED"],
                    issues=[abstention_by_id[candidate_id].reason],
                )
            )
            continue
        item = item_by_id.get(candidate_id)
        if item is None:
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="MISSING_DRAFT",
                    issue_codes=["MISSING_MATERIALIZATION_DRAFT"],
                    issues=["Selected candidate was neither materialized nor explicitly abstained."],
                )
            )
            continue

        one_draft = HypothesisPortfolioDraft(hypotheses=[_proposal(item)], abstention_reason=None)
        try:
            one_portfolio = compiler.compile(context, one_draft)
        except HypothesisCompileError as exc:
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="COMPILE_REJECTED",
                    issue_codes=[_issue_code(x.code) for x in exc.issues],
                    issues=[f"{x.location}: {x.message}" for x in exc.issues],
                )
            )
            continue
        except Exception as exc:
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="COMPILE_REJECTED",
                    issue_codes=[type(exc).__name__],
                    issues=[str(exc)],
                )
            )
            continue

        validation = validator.validate(context, one_portfolio)
        if not validation.passes:
            records.append(
                ScientificPortfolioMaterializationRecord(
                    candidate_id=candidate_id,
                    source_object_id=candidate.source_object_id,
                    status="HARD_GATE_REJECTED",
                    issue_codes=[x.code for x in validation.issues if x.severity == "error"],
                    issues=[x.message for x in validation.issues if x.severity == "error"],
                )
            )
            continue

        card = one_portfolio.hypotheses[0]
        cards.append(card)
        records.append(
            ScientificPortfolioMaterializationRecord(
                candidate_id=candidate_id,
                source_object_id=candidate.source_object_id,
                status="MATERIALIZED",
                hypothesis_id=card.hypothesis_id,
                issue_codes=[x.code for x in validation.issues if x.severity == "warning"],
                issues=[x.message for x in validation.issues if x.severity == "warning"],
            )
        )

    portfolio_id = _stable_id(
        "scientific_portfolio_materialized_hypotheses",
        context.context_sha256,
        selection.selection_sha256,
        *(x.hypothesis_id for x in cards),
    )
    portfolio = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=(
            None
            if cards
            else "No retained scientific-portfolio candidate materialized under grounded hypothesis constraints."
        ),
    )
    portfolio_sha = _sha(portfolio)
    counts = Counter(x.status for x in records)

    provisional = ScientificPortfolioMaterializationReport(
        report_id="pending",
        report_sha256="pending",
        source_selection_id=selection.selection_id,
        source_selection_sha256=selection.selection_sha256,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        output_portfolio_id=portfolio.portfolio_id,
        output_portfolio_sha256=portfolio_sha,
        selected_candidate_count=len(selected_ids),
        materialized_hypothesis_count=len(cards),
        abstained_count=counts["ABSTAINED"],
        compile_rejected_count=counts["COMPILE_REJECTED"],
        hard_gate_rejected_count=counts["HARD_GATE_REJECTED"],
        missing_draft_count=counts["MISSING_DRAFT"],
        generation_failed_count=counts["GENERATION_FAILED"],
        status_counts=dict(sorted(counts.items())),
        records=records,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    report = provisional.model_copy(
        update={
            "report_id": f"scientific_portfolio_materialization:{digest[:20]}",
            "report_sha256": digest,
        }
    )
    return report, portfolio


__all__ = [
    "ScientificPortfolioMaterializationItemDraft",
    "ScientificPortfolioMaterializationAbstention",
    "ScientificPortfolioMaterializationBatchDraft",
    "ScientificPortfolioMaterializationRecord",
    "ScientificPortfolioMaterializationReport",
    "compile_scientific_portfolio_materialization",
]
