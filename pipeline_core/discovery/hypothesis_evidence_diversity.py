from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairwisePremiseOverlap(StrictModel):
    left_hypothesis_id: str
    right_hypothesis_id: str
    intersection_statement_ids: list[str] = Field(default_factory=list)
    union_statement_count: int = 0
    statement_jaccard: float = 0.0


class ExactPremiseSetGroup(StrictModel):
    premise_statement_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)


class EvidenceStatementUsage(StrictModel):
    statement_id: str
    epistemic_role: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    paper_count: int = 0
    hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_usage_count: int = 0
    requires_verification: bool = False


class HypothesisEvidenceDiversityCard(StrictModel):
    hypothesis_id: str
    title: str
    premise_statement_ids: list[str] = Field(default_factory=list)
    premise_paper_ids: list[str] = Field(default_factory=list)
    premise_count: int = 0
    premise_paper_count: int = 0
    portfolio_unique_premise_statement_ids: list[str] = Field(default_factory=list)
    portfolio_unique_premise_count: int = 0
    shared_core_premise_count: int = 0
    max_statement_jaccard: float = 0.0
    most_overlapping_hypothesis_ids: list[str] = Field(default_factory=list)
    exact_premise_set_duplicate: bool = False


class HypothesisEvidenceDiversityReport(StrictModel):
    schema_version: Literal["hypothesis-evidence-diversity-report-v1"] = (
        "hypothesis-evidence-diversity-report-v1"
    )
    report_id: str
    report_sha256: str
    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_portfolio_sha256: str

    hypothesis_count: int = 0
    eligible_statement_count: int = 0
    used_statement_count: int = 0
    eligible_statement_coverage: float = 0.0

    used_statement_ids: list[str] = Field(default_factory=list)
    unused_eligible_statement_ids: list[str] = Field(default_factory=list)
    shared_core_statement_ids: list[str] = Field(default_factory=list)
    shared_core_statement_count: int = 0
    statement_usage_counts: dict[str, int] = Field(default_factory=dict)

    distinct_premise_set_count: int = 0
    exact_premise_set_duplicate_group_count: int = 0
    mean_pairwise_statement_jaccard: float = 0.0
    max_pairwise_statement_jaccard: float = 0.0

    multi_paper_used_statement_count: int = 0
    mean_papers_per_used_statement: float = 0.0

    pairwise_overlaps: list[PairwisePremiseOverlap] = Field(default_factory=list)
    exact_premise_set_groups: list[ExactPremiseSetGroup] = Field(default_factory=list)
    statement_usage: list[EvidenceStatementUsage] = Field(default_factory=list)
    cards: list[HypothesisEvidenceDiversityCard] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False


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


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class HypothesisEvidenceDiversityAssessor:
    """Deterministic, diagnostic-only premise-diversity analysis."""

    def assess(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> HypothesisEvidenceDiversityReport:
        if portfolio.source_context_id != context.context_id:
            raise ValueError("portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError("portfolio/context SHA mismatch")

        eligible_rows = [
            row for row in context.evidence_statements
            if row.eligible_as_premise
        ]
        eligible_by_id = {row.statement_id: row for row in eligible_rows}
        eligible_ids = [row.statement_id for row in eligible_rows]
        eligible_id_set = set(eligible_ids)

        hypothesis_ids = [card.hypothesis_id for card in portfolio.hypotheses]
        premise_sets: dict[str, set[str]] = {}
        premise_lists: dict[str, list[str]] = {}
        usage_by_statement: dict[str, list[str]] = defaultdict(list)

        for card in portfolio.hypotheses:
            premise_list = list(dict.fromkeys(card.premise_statement_ids))
            premise_set = set(premise_list)
            invalid = sorted(premise_set - eligible_id_set)
            if invalid:
                raise ValueError(
                    f"{card.hypothesis_id}: premise IDs are not eligible in "
                    f"the supplied context: {invalid}"
                )
            premise_sets[card.hypothesis_id] = premise_set
            premise_lists[card.hypothesis_id] = premise_list
            for statement_id in premise_set:
                usage_by_statement[statement_id].append(card.hypothesis_id)

        usage_counts = {
            sid: len(usage_by_statement.get(sid, []))
            for sid in eligible_ids
        }
        used_ids = [sid for sid in eligible_ids if usage_counts[sid] > 0]
        unused_ids = [sid for sid in eligible_ids if usage_counts[sid] == 0]

        if len(portfolio.hypotheses) >= 2:
            shared_core = set.intersection(
                *[
                    premise_sets[card.hypothesis_id]
                    for card in portfolio.hypotheses
                ]
            )
        else:
            shared_core = set()
        shared_core_ids = [sid for sid in eligible_ids if sid in shared_core]

        pairwise: list[PairwisePremiseOverlap] = []
        overlaps_by_hypothesis: dict[str, list[tuple[str, float]]] = defaultdict(list)
        cards_in_order = list(portfolio.hypotheses)

        for i, left in enumerate(cards_in_order):
            left_set = premise_sets[left.hypothesis_id]
            for right in cards_in_order[i + 1:]:
                right_set = premise_sets[right.hypothesis_id]
                intersection = left_set & right_set
                union = left_set | right_set
                jaccard = len(intersection) / len(union) if union else 0.0
                row = PairwisePremiseOverlap(
                    left_hypothesis_id=left.hypothesis_id,
                    right_hypothesis_id=right.hypothesis_id,
                    intersection_statement_ids=[
                        sid for sid in eligible_ids if sid in intersection
                    ],
                    union_statement_count=len(union),
                    statement_jaccard=float(jaccard),
                )
                pairwise.append(row)
                overlaps_by_hypothesis[left.hypothesis_id].append(
                    (right.hypothesis_id, float(jaccard))
                )
                overlaps_by_hypothesis[right.hypothesis_id].append(
                    (left.hypothesis_id, float(jaccard))
                )

        signature_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for card in cards_in_order:
            signature = tuple(sorted(premise_sets[card.hypothesis_id]))
            signature_groups[signature].append(card.hypothesis_id)

        exact_groups = [
            ExactPremiseSetGroup(
                premise_statement_ids=list(signature),
                hypothesis_ids=list(ids),
            )
            for signature, ids in sorted(
                signature_groups.items(),
                key=lambda item: (item[0], tuple(item[1])),
            )
            if len(ids) >= 2
        ]
        duplicate_ids = {
            hid for group in exact_groups for hid in group.hypothesis_ids
        }

        statement_usage: list[EvidenceStatementUsage] = []
        for row in eligible_rows:
            users = [
                hid for hid in hypothesis_ids
                if hid in usage_by_statement.get(row.statement_id, [])
            ]
            paper_ids = list(dict.fromkeys(row.paper_ids))
            statement_usage.append(
                EvidenceStatementUsage(
                    statement_id=row.statement_id,
                    epistemic_role=row.epistemic_role,
                    claim_kind=row.claim_kind,
                    paper_ids=paper_ids,
                    paper_count=len(paper_ids),
                    hypothesis_ids=users,
                    hypothesis_usage_count=len(users),
                    requires_verification=row.requires_verification,
                )
            )

        diversity_cards: list[HypothesisEvidenceDiversityCard] = []
        for card in cards_in_order:
            premise_ids = premise_lists[card.hypothesis_id]
            premise_papers: set[str] = set()
            for sid in premise_ids:
                premise_papers.update(eligible_by_id[sid].paper_ids)

            unique_ids = [
                sid for sid in premise_ids if usage_counts[sid] == 1
            ]
            overlap_rows = overlaps_by_hypothesis.get(card.hypothesis_id, [])
            max_overlap = (
                max(value for _, value in overlap_rows)
                if overlap_rows else 0.0
            )
            most_overlapping = sorted(
                hid for hid, value in overlap_rows
                if abs(value - max_overlap) <= 1e-12
            )
            diversity_cards.append(
                HypothesisEvidenceDiversityCard(
                    hypothesis_id=card.hypothesis_id,
                    title=card.title,
                    premise_statement_ids=premise_ids,
                    premise_paper_ids=sorted(premise_papers),
                    premise_count=len(premise_ids),
                    premise_paper_count=len(premise_papers),
                    portfolio_unique_premise_statement_ids=unique_ids,
                    portfolio_unique_premise_count=len(unique_ids),
                    shared_core_premise_count=len(
                        premise_sets[card.hypothesis_id] & shared_core
                    ),
                    max_statement_jaccard=float(max_overlap),
                    most_overlapping_hypothesis_ids=most_overlapping,
                    exact_premise_set_duplicate=card.hypothesis_id in duplicate_ids,
                )
            )

        pairwise_values = [row.statement_jaccard for row in pairwise]
        mean_pairwise = (
            sum(pairwise_values) / len(pairwise_values)
            if pairwise_values else 0.0
        )
        max_pairwise = max(pairwise_values) if pairwise_values else 0.0

        used_statement_cards = [
            row for row in statement_usage
            if row.hypothesis_usage_count > 0
        ]
        mean_papers_per_used = (
            sum(row.paper_count for row in used_statement_cards)
            / len(used_statement_cards)
            if used_statement_cards else 0.0
        )
        multi_paper_used_count = sum(
            1 for row in used_statement_cards if row.paper_count > 1
        )

        portfolio_sha = _sha256_json(portfolio)
        report_id = _stable_id(
            "hypothesis_evidence_diversity_report",
            context.context_sha256,
            portfolio.portfolio_id,
            portfolio_sha,
        )

        payload = {
            "schema_version": "hypothesis-evidence-diversity-report-v1",
            "report_id": report_id,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_portfolio_sha256": portfolio_sha,
            "hypothesis_count": len(cards_in_order),
            "eligible_statement_count": len(eligible_ids),
            "used_statement_count": len(used_ids),
            "eligible_statement_coverage": (
                len(used_ids) / len(eligible_ids) if eligible_ids else 0.0
            ),
            "used_statement_ids": used_ids,
            "unused_eligible_statement_ids": unused_ids,
            "shared_core_statement_ids": shared_core_ids,
            "shared_core_statement_count": len(shared_core_ids),
            "statement_usage_counts": usage_counts,
            "distinct_premise_set_count": len(signature_groups),
            "exact_premise_set_duplicate_group_count": len(exact_groups),
            "mean_pairwise_statement_jaccard": float(mean_pairwise),
            "max_pairwise_statement_jaccard": float(max_pairwise),
            "multi_paper_used_statement_count": multi_paper_used_count,
            "mean_papers_per_used_statement": float(mean_papers_per_used),
            "pairwise_overlaps": [
                row.model_dump(mode="json") for row in pairwise
            ],
            "exact_premise_set_groups": [
                row.model_dump(mode="json") for row in exact_groups
            ],
            "statement_usage": [
                row.model_dump(mode="json") for row in statement_usage
            ],
            "cards": [
                row.model_dump(mode="json") for row in diversity_cards
            ],
            "diagnostic_only": True,
            "scientific_selection_changed": False,
        }
        return HypothesisEvidenceDiversityReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
