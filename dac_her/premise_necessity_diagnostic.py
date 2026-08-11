from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dac_her.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
)
from dac_her.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
)
from dac_her.node_mapping import SentenceTransformerEncoder


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PremiseSemanticScores(StrictModel):
    hypothesis_score: float
    bridge_score: float
    axis_score: float
    prediction_score: float
    core_score: float

    hypothesis_rank: int
    bridge_rank: int
    axis_rank: int
    prediction_rank: int
    core_rank: int


class PremiseProvenanceContribution(StrictModel):
    scientific_edge_count: int = 0
    unique_scientific_edge_count: int = 0
    unique_scientific_edge_fraction: float = 0.0

    scientific_node_count: int = 0
    unique_scientific_node_count: int = 0
    unique_scientific_node_fraction: float = 0.0

    paper_count: int = 0
    unique_paper_count: int = 0
    unique_paper_fraction: float = 0.0


class SelectedPremiseDiagnostic(StrictModel):
    statement_id: str
    text: str
    claim_kind: str
    epistemic_role: str
    paper_ids: list[str] = Field(default_factory=list)

    scores: PremiseSemanticScores
    provenance: PremiseProvenanceContribution

    pareto_dominated_by_unselected_statement_ids: list[str] = Field(
        default_factory=list
    )
    dominated_on_all_semantic_axes: bool = False

    selected_in_all_hypotheses: bool = False
    globally_shared_core: bool = False

    diagnostic_flags: list[
        Literal[
            "selected_in_all_hypotheses",
            "globally_shared_core",
            "pareto_dominated_by_unselected",
            "low_unique_provenance_contribution",
            "highly_axis_generic",
            "highly_hypothesis_generic",
        ]
    ] = Field(default_factory=list)


class UnselectedPremiseCandidate(StrictModel):
    statement_id: str
    text: str
    claim_kind: str
    epistemic_role: str
    paper_ids: list[str] = Field(default_factory=list)
    scores: PremiseSemanticScores


class HypothesisPremiseNecessityCard(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    premise_statement_ids: list[str] = Field(default_factory=list)
    unselected_eligible_statement_ids: list[str] = Field(default_factory=list)

    exact_shared_set_with_all_hypotheses: bool = False

    selected_premises: list[SelectedPremiseDiagnostic] = Field(
        default_factory=list
    )
    best_unselected_by_core_score: UnselectedPremiseCandidate | None = None
    best_unselected_by_axis_score: UnselectedPremiseCandidate | None = None

    selected_pareto_dominated_incidence_count: int = 0


class GlobalPremiseUsageDiagnostic(StrictModel):
    statement_id: str
    text: str
    usage_count: int
    hypothesis_count: int
    usage_fraction: float

    mean_core_score: float
    std_core_score: float
    core_score_range: float

    mean_axis_score: float
    std_axis_score: float
    axis_score_range: float

    axis_genericity: float
    hypothesis_genericity: float

    selected_in_all_hypotheses: bool = False
    selected_in_no_hypotheses: bool = False


class PremiseNecessityPolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False
    absolute_necessity_claims_allowed: Literal[False] = False
    semantic_similarity_is_scientific_entailment: Literal[False] = False
    pareto_domination_is_rejection_rule: Literal[False] = False
    unique_provenance_is_necessity_proof: Literal[False] = False

    pareto_epsilon: float = 0.01
    low_unique_provenance_fraction_threshold: float = 0.10
    genericity_range_threshold: float = 0.05


class PremiseNecessityDiagnosticReport(StrictModel):
    schema_version: Literal["premise-necessity-diagnostic-v1"] = (
        "premise-necessity-diagnostic-v1"
    )

    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str
    domain_profile_id: str
    corpus_id: str

    embedding_model: str

    hypothesis_count: int
    eligible_statement_count: int
    used_statement_count: int

    exact_same_premise_set_across_all_hypotheses: bool
    shared_core_statement_ids: list[str] = Field(default_factory=list)
    shared_core_statement_count: int = 0

    selected_pareto_dominated_incidence_count: int = 0
    hypotheses_with_pareto_dominated_selected_premise_count: int = 0

    cards: list[HypothesisPremiseNecessityCard] = Field(default_factory=list)
    global_premise_diagnostics: list[GlobalPremiseUsageDiagnostic] = Field(
        default_factory=list
    )

    policy: PremiseNecessityPolicy = Field(
        default_factory=PremiseNecessityPolicy
    )


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _cosine(
    document_vector: np.ndarray,
    query_vector: np.ndarray,
) -> float:
    return float(
        np.asarray(document_vector, dtype=np.float32)
        @ np.asarray(query_vector, dtype=np.float32)
    )


def _axis_text(axis: DiscoveryAxis) -> str:
    parts = [
        axis.label,
        axis.proposed_subject,
        axis.proposed_relation,
        axis.proposed_object,
        axis.entry_anchor_label,
        axis.exit_anchor_label,
    ]
    return " | ".join(
        part.strip()
        for part in parts
        if str(part).strip()
    )


def _hypothesis_text(card: HypothesisCard) -> str:
    return f"{card.title}. {card.hypothesis_statement}"


def _prediction_text(card: HypothesisCard) -> str:
    parts: list[str] = []
    for row in card.predicted_observations:
        parts.extend(
            [
                row.observable,
                row.rationale,
                row.expected_direction,
            ]
        )
    return " | ".join(
        part.strip()
        for part in parts
        if str(part).strip()
    )


def _rank_descending(
    scores_by_id: dict[str, float],
) -> dict[str, int]:
    ordered = sorted(
        scores_by_id,
        key=lambda sid: (
            -scores_by_id[sid],
            sid,
        ),
    )
    return {
        sid: rank
        for rank, sid in enumerate(
            ordered,
            start=1,
        )
    }


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _provenance_contribution(
    statement: HypothesisEvidenceStatement,
    selected_statements: list[HypothesisEvidenceStatement],
) -> PremiseProvenanceContribution:
    other_edges: set[str] = set()
    other_nodes: set[str] = set()
    other_papers: set[str] = set()

    for other in selected_statements:
        if other.statement_id == statement.statement_id:
            continue
        other_edges.update(other.scientific_support_edge_ids)
        other_nodes.update(other.scientific_support_node_ids)
        other_papers.update(other.paper_ids)

    edges = set(statement.scientific_support_edge_ids)
    nodes = set(statement.scientific_support_node_ids)
    papers = set(statement.paper_ids)

    unique_edges = edges - other_edges
    unique_nodes = nodes - other_nodes
    unique_papers = papers - other_papers

    return PremiseProvenanceContribution(
        scientific_edge_count=len(edges),
        unique_scientific_edge_count=len(unique_edges),
        unique_scientific_edge_fraction=_safe_fraction(
            len(unique_edges),
            len(edges),
        ),
        scientific_node_count=len(nodes),
        unique_scientific_node_count=len(unique_nodes),
        unique_scientific_node_fraction=_safe_fraction(
            len(unique_nodes),
            len(nodes),
        ),
        paper_count=len(papers),
        unique_paper_count=len(unique_papers),
        unique_paper_fraction=_safe_fraction(
            len(unique_papers),
            len(papers),
        ),
    )


def _pareto_dominators(
    selected_scores: PremiseSemanticScores,
    unselected_scores_by_id: dict[str, PremiseSemanticScores],
    *,
    epsilon: float,
) -> list[str]:
    selected = (
        selected_scores.hypothesis_score,
        selected_scores.bridge_score,
        selected_scores.axis_score,
        selected_scores.prediction_score,
    )

    dominators: list[str] = []
    for sid, row in unselected_scores_by_id.items():
        candidate = (
            row.hypothesis_score,
            row.bridge_score,
            row.axis_score,
            row.prediction_score,
        )
        no_worse = all(
            candidate[i] >= selected[i] - epsilon
            for i in range(4)
        )
        strictly_better = any(
            candidate[i] > selected[i] + epsilon
            for i in range(4)
        )
        if no_worse and strictly_better:
            dominators.append(sid)

    return sorted(dominators)


def _build_scores(
    *,
    statement_ids: list[str],
    statement_vectors: dict[str, np.ndarray],
    hypothesis_query: np.ndarray,
    bridge_query: np.ndarray,
    axis_query: np.ndarray,
    prediction_query: np.ndarray,
) -> dict[str, PremiseSemanticScores]:
    hypothesis_scores = {
        sid: _cosine(
            statement_vectors[sid],
            hypothesis_query,
        )
        for sid in statement_ids
    }
    bridge_scores = {
        sid: _cosine(
            statement_vectors[sid],
            bridge_query,
        )
        for sid in statement_ids
    }
    axis_scores = {
        sid: _cosine(
            statement_vectors[sid],
            axis_query,
        )
        for sid in statement_ids
    }
    prediction_scores = {
        sid: _cosine(
            statement_vectors[sid],
            prediction_query,
        )
        for sid in statement_ids
    }
    core_scores = {
        sid: (
            hypothesis_scores[sid]
            + bridge_scores[sid]
        )
        / 2.0
        for sid in statement_ids
    }

    hypothesis_ranks = _rank_descending(hypothesis_scores)
    bridge_ranks = _rank_descending(bridge_scores)
    axis_ranks = _rank_descending(axis_scores)
    prediction_ranks = _rank_descending(prediction_scores)
    core_ranks = _rank_descending(core_scores)

    return {
        sid: PremiseSemanticScores(
            hypothesis_score=hypothesis_scores[sid],
            bridge_score=bridge_scores[sid],
            axis_score=axis_scores[sid],
            prediction_score=prediction_scores[sid],
            core_score=core_scores[sid],
            hypothesis_rank=hypothesis_ranks[sid],
            bridge_rank=bridge_ranks[sid],
            axis_rank=axis_ranks[sid],
            prediction_rank=prediction_ranks[sid],
            core_rank=core_ranks[sid],
        )
        for sid in statement_ids
    }


class PremiseNecessityDiagnosticAssessor:
    def __init__(
        self,
        *,
        index_dir: str | Path,
        device: str | None = None,
        policy: PremiseNecessityPolicy | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        manifest = json.loads(
            (self.index_dir / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.embedding_model = str(manifest["model_name"])
        self.encoder = SentenceTransformerEncoder(
            self.embedding_model,
            device=device,
        )
        self.policy = policy or PremiseNecessityPolicy()

    def assess(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        axis_plan: DiscoveryAxisPlan,
        axis_report: DiscoveryAxisSynthesisReport,
    ) -> PremiseNecessityDiagnosticReport:
        if portfolio.source_context_id != context.context_id:
            raise ValueError(
                "PS1 portfolio/context ID mismatch"
            )
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError(
                "PS1 portfolio/context SHA mismatch"
            )
        if axis_report.final_portfolio_id != portfolio.portfolio_id:
            raise ValueError(
                "PS1 axis report/portfolio ID mismatch"
            )
        if axis_report.axis_plan_id != axis_plan.plan_id:
            raise ValueError(
                "PS1 axis report/plan ID mismatch"
            )

        axis_by_id = {
            axis.axis_id: axis
            for axis in axis_plan.axes
        }
        lineage_by_hypothesis = {
            row.hypothesis_id: row
            for row in axis_report.lineages
        }

        eligible = [
            row
            for row in context.evidence_statements
            if row.eligible_as_premise
        ]
        statement_by_id = {
            row.statement_id: row
            for row in eligible
        }
        eligible_ids = sorted(statement_by_id)

        if not eligible_ids:
            raise ValueError(
                "PS1 requires at least one eligible premise statement"
            )

        missing_selected = sorted(
            {
                sid
                for card in portfolio.hypotheses
                for sid in card.premise_statement_ids
                if sid not in statement_by_id
            }
        )
        if missing_selected:
            raise ValueError(
                "PS1 portfolio uses premise IDs not present as eligible "
                f"context statements: {missing_selected}"
            )

        statement_texts = [
            statement_by_id[sid].text
            for sid in eligible_ids
        ]
        statement_matrix = self.encoder.encode_documents(
            statement_texts,
            batch_size=min(32, max(1, len(statement_texts))),
        )
        statement_vectors = {
            sid: statement_matrix[index]
            for index, sid in enumerate(eligible_ids)
        }

        usage = Counter(
            sid
            for card in portfolio.hypotheses
            for sid in card.premise_statement_ids
        )
        hypothesis_count = len(portfolio.hypotheses)

        premise_sets = [
            tuple(sorted(card.premise_statement_ids))
            for card in portfolio.hypotheses
        ]
        exact_same_set = (
            bool(premise_sets)
            and len(set(premise_sets)) == 1
        )

        shared_core = set(eligible_ids)
        if portfolio.hypotheses:
            for card in portfolio.hypotheses:
                shared_core &= set(card.premise_statement_ids)
        else:
            shared_core = set()

        card_scores_by_hypothesis: dict[
            str,
            dict[str, PremiseSemanticScores],
        ] = {}
        cards: list[HypothesisPremiseNecessityCard] = []

        for card in portfolio.hypotheses:
            lineage = lineage_by_hypothesis.get(
                card.hypothesis_id
            )
            if lineage is None:
                raise ValueError(
                    "PS1 missing discovery lineage for hypothesis: "
                    f"{card.hypothesis_id}"
                )
            axis = axis_by_id.get(lineage.axis_id)
            if axis is None:
                raise ValueError(
                    "PS1 lineage refers to unknown axis: "
                    f"{lineage.axis_id}"
                )

            hypothesis_query = self.encoder.encode_query(
                _hypothesis_text(card)
            )
            bridge_query = self.encoder.encode_query(
                card.inferential_bridge
            )
            axis_query = self.encoder.encode_query(
                _axis_text(axis)
            )
            prediction_query = self.encoder.encode_query(
                _prediction_text(card)
            )

            scores_by_id = _build_scores(
                statement_ids=eligible_ids,
                statement_vectors=statement_vectors,
                hypothesis_query=hypothesis_query,
                bridge_query=bridge_query,
                axis_query=axis_query,
                prediction_query=prediction_query,
            )
            card_scores_by_hypothesis[
                card.hypothesis_id
            ] = scores_by_id

            selected_ids = list(
                card.premise_statement_ids
            )
            unselected_ids = [
                sid
                for sid in eligible_ids
                if sid not in set(selected_ids)
            ]
            selected_statements = [
                statement_by_id[sid]
                for sid in selected_ids
            ]
            unselected_scores = {
                sid: scores_by_id[sid]
                for sid in unselected_ids
            }

            selected_rows: list[
                SelectedPremiseDiagnostic
            ] = []

            for sid in selected_ids:
                statement = statement_by_id[sid]
                scores = scores_by_id[sid]
                provenance = _provenance_contribution(
                    statement,
                    selected_statements,
                )
                dominators = _pareto_dominators(
                    scores,
                    unselected_scores,
                    epsilon=self.policy.pareto_epsilon,
                )

                flags: list[str] = []
                selected_all = (
                    hypothesis_count > 0
                    and usage[sid] == hypothesis_count
                )
                if selected_all:
                    flags.append(
                        "selected_in_all_hypotheses"
                    )
                if sid in shared_core:
                    flags.append(
                        "globally_shared_core"
                    )
                if dominators:
                    flags.append(
                        "pareto_dominated_by_unselected"
                    )

                max_unique_fraction = max(
                    provenance.unique_scientific_edge_fraction,
                    provenance.unique_scientific_node_fraction,
                    provenance.unique_paper_fraction,
                )
                if (
                    max_unique_fraction
                    < self.policy.low_unique_provenance_fraction_threshold
                ):
                    flags.append(
                        "low_unique_provenance_contribution"
                    )

                selected_rows.append(
                    SelectedPremiseDiagnostic(
                        statement_id=sid,
                        text=statement.text,
                        claim_kind=statement.claim_kind,
                        epistemic_role=statement.epistemic_role,
                        paper_ids=list(statement.paper_ids),
                        scores=scores,
                        provenance=provenance,
                        pareto_dominated_by_unselected_statement_ids=dominators,
                        dominated_on_all_semantic_axes=bool(dominators),
                        selected_in_all_hypotheses=selected_all,
                        globally_shared_core=(
                            sid in shared_core
                        ),
                        diagnostic_flags=flags,
                    )
                )

            def candidate(
                sid: str,
            ) -> UnselectedPremiseCandidate:
                statement = statement_by_id[sid]
                return UnselectedPremiseCandidate(
                    statement_id=sid,
                    text=statement.text,
                    claim_kind=statement.claim_kind,
                    epistemic_role=statement.epistemic_role,
                    paper_ids=list(statement.paper_ids),
                    scores=scores_by_id[sid],
                )

            best_core = (
                max(
                    unselected_ids,
                    key=lambda sid: (
                        scores_by_id[sid].core_score,
                        sid,
                    ),
                )
                if unselected_ids
                else None
            )
            best_axis = (
                max(
                    unselected_ids,
                    key=lambda sid: (
                        scores_by_id[sid].axis_score,
                        sid,
                    ),
                )
                if unselected_ids
                else None
            )

            cards.append(
                HypothesisPremiseNecessityCard(
                    hypothesis_id=card.hypothesis_id,
                    title=card.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    premise_statement_ids=selected_ids,
                    unselected_eligible_statement_ids=unselected_ids,
                    exact_shared_set_with_all_hypotheses=exact_same_set,
                    selected_premises=selected_rows,
                    best_unselected_by_core_score=(
                        candidate(best_core)
                        if best_core is not None
                        else None
                    ),
                    best_unselected_by_axis_score=(
                        candidate(best_axis)
                        if best_axis is not None
                        else None
                    ),
                    selected_pareto_dominated_incidence_count=sum(
                        row.dominated_on_all_semantic_axes
                        for row in selected_rows
                    ),
                )
            )

        global_rows: list[
            GlobalPremiseUsageDiagnostic
        ] = []

        for sid in eligible_ids:
            core_scores = [
                card_scores_by_hypothesis[
                    card.hypothesis_id
                ][sid].core_score
                for card in portfolio.hypotheses
            ]
            axis_scores = [
                card_scores_by_hypothesis[
                    card.hypothesis_id
                ][sid].axis_score
                for card in portfolio.hypotheses
            ]

            if core_scores:
                core_mean = float(np.mean(core_scores))
                core_std = float(np.std(core_scores))
                core_range = float(
                    max(core_scores) - min(core_scores)
                )
            else:
                core_mean = core_std = core_range = 0.0

            if axis_scores:
                axis_mean = float(np.mean(axis_scores))
                axis_std = float(np.std(axis_scores))
                axis_range = float(
                    max(axis_scores) - min(axis_scores)
                )
            else:
                axis_mean = axis_std = axis_range = 0.0

            statement = statement_by_id[sid]
            global_rows.append(
                GlobalPremiseUsageDiagnostic(
                    statement_id=sid,
                    text=statement.text,
                    usage_count=usage[sid],
                    hypothesis_count=hypothesis_count,
                    usage_fraction=_safe_fraction(
                        usage[sid],
                        hypothesis_count,
                    ),
                    mean_core_score=core_mean,
                    std_core_score=core_std,
                    core_score_range=core_range,
                    mean_axis_score=axis_mean,
                    std_axis_score=axis_std,
                    axis_score_range=axis_range,
                    axis_genericity=1.0 - min(
                        1.0,
                        max(0.0, axis_range),
                    ),
                    hypothesis_genericity=1.0 - min(
                        1.0,
                        max(0.0, core_range),
                    ),
                    selected_in_all_hypotheses=(
                        hypothesis_count > 0
                        and usage[sid] == hypothesis_count
                    ),
                    selected_in_no_hypotheses=(
                        usage[sid] == 0
                    ),
                )
            )

        # Add conservative genericity flags after portfolio-wide score ranges
        # are known.
        global_by_id = {
            row.statement_id: row
            for row in global_rows
        }
        for card in cards:
            for row in card.selected_premises:
                global_row = global_by_id[row.statement_id]
                flags = list(row.diagnostic_flags)
                if (
                    global_row.axis_score_range
                    <= self.policy.genericity_range_threshold
                    and row.selected_in_all_hypotheses
                ):
                    flags.append("highly_axis_generic")
                if (
                    global_row.core_score_range
                    <= self.policy.genericity_range_threshold
                    and row.selected_in_all_hypotheses
                ):
                    flags.append("highly_hypothesis_generic")
                row.diagnostic_flags = sorted(set(flags))

        dominated_count = sum(
            card.selected_pareto_dominated_incidence_count
            for card in cards
        )
        dominated_hypothesis_count = sum(
            card.selected_pareto_dominated_incidence_count > 0
            for card in cards
        )

        payload = {
            "schema_version": "premise-necessity-diagnostic-v1",
            "report_id": _stable_id(
                "premise_necessity_diagnostic",
                context.context_sha256,
                portfolio.portfolio_id,
                axis_plan.plan_id,
                axis_report.report_id,
                self.embedding_model,
            ),
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_axis_plan_id": axis_plan.plan_id,
            "source_axis_report_id": axis_report.report_id,
            "domain_profile_id": context.domain_profile_id,
            "corpus_id": context.corpus_id,
            "embedding_model": self.embedding_model,
            "hypothesis_count": hypothesis_count,
            "eligible_statement_count": len(eligible_ids),
            "used_statement_count": len(
                {
                    sid
                    for card in portfolio.hypotheses
                    for sid in card.premise_statement_ids
                }
            ),
            "exact_same_premise_set_across_all_hypotheses": (
                exact_same_set
            ),
            "shared_core_statement_ids": sorted(shared_core),
            "shared_core_statement_count": len(shared_core),
            "selected_pareto_dominated_incidence_count": dominated_count,
            "hypotheses_with_pareto_dominated_selected_premise_count": (
                dominated_hypothesis_count
            ),
            "cards": [
                row.model_dump(mode="json")
                for row in cards
            ],
            "global_premise_diagnostics": [
                row.model_dump(mode="json")
                for row in global_rows
            ],
            "policy": self.policy.model_dump(mode="json"),
        }

        return PremiseNecessityDiagnosticReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
