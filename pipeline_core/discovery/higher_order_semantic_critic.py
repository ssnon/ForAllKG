from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_hypothesis_batch import (
    HigherOrderShadowBatchOutcome,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


HigherOrderSemanticIssueCode = Literal[
    "MODIFIER_UTILIZATION_UNRESOLVED",
    "MODIFIER_ANCHOR_BRIDGE_OMITTED",
    "BACKBONE_RESTATEMENT_RISK",
    "KNOWN_CONTEXT_DUPLICATE",
]


class HigherOrderSemanticCriticIssue(StrictModel):
    code: HigherOrderSemanticIssueCode
    message: str
    related_arm_index: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class HigherOrderArmSemanticCritique(StrictModel):
    arm_index: int = Field(ge=1)
    higher_order_context_id: str
    modifier_component_id: str
    modifier_authority: str
    candidate_backed: bool
    requested_source: str
    requested_target: str
    modifier_text: str
    modifier_anchor_role: str
    modifier_anchor_text: str
    hypothesis_statement: str
    inferential_bridge: str
    modifier_statement_coverage: float
    modifier_full_card_coverage: float
    modifier_anchor_full_card_coverage: float
    requested_source_statement_coverage: float
    requested_target_statement_coverage: float
    backbone_restatement_overlap: float
    issues: list[HigherOrderSemanticCriticIssue] = Field(default_factory=list)
    shadow_only: Literal[True] = True
    diagnostic_only: Literal[True] = True
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class HigherOrderBatchSemanticCritique(StrictModel):
    schema_version: Literal["higher-order-semantic-critic-v1"] = (
        "higher-order-semantic-critic-v1"
    )
    arm_count: int = Field(ge=0)
    candidate_backed_arm_count: int = Field(ge=0)
    known_backed_arm_count: int = Field(ge=0)
    flagged_arm_count: int = Field(ge=0)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    candidate_issue_counts: dict[str, int] = Field(default_factory=dict)
    known_context_duplicate_pair_count: int = Field(ge=0)
    arms: list[HigherOrderArmSemanticCritique] = Field(default_factory=list)
    shadow_only: Literal[True] = True
    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


_DASHES = "‐-‒–—−"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "do", "does", "for", "from", "has", "have", "how", "in", "into", "is",
    "may", "of", "on", "or", "such", "than", "that", "the", "their", "to",
    "under", "with",
}
_GENERIC_MODIFIER_TOKENS = {
    "behavior", "behaviour", "characteristic", "characteristics", "effect",
    "effects", "feature", "features", "parameter", "parameters", "property",
    "properties", "response", "responses",
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    for dash in _DASHES:
        value = value.replace(dash, "-")
    return value


def _tokens(text: str) -> set[str]:
    value = _normalize(text)
    raw = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", value)
    out: set[str] = set()
    for token in raw:
        parts = [part for part in token.split("-") if part]
        candidates = list(parts)
        if len(parts) > 1:
            candidates.append("".join(parts))
        for candidate in candidates:
            if len(candidate) > 1 and candidate not in _STOPWORDS:
                out.add(candidate)
    return out


def _coverage(phrase: str, text: str) -> float:
    required = _tokens(phrase)
    if not required:
        return 0.0
    return float(len(required & _tokens(text)) / len(required))


def _containment(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return float(len(a & b) / min(len(a), len(b)))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _authority_value(value: object) -> str:
    return str(getattr(value, "value", value)).strip()


def _modifier_premise(context: object) -> object | None:
    for premise in getattr(context, "premises", ()):
        if getattr(premise, "premise_role", None) == "modifier_relation":
            return premise
    return None


def _card_text(card: object) -> str:
    parts = [
        str(getattr(card, "title", "")),
        str(getattr(card, "hypothesis_statement", "")),
        str(getattr(card, "inferential_bridge", "")),
    ]
    for row in getattr(card, "predicted_observations", ()):
        parts.extend([
            str(getattr(row, "observable", "")),
            str(getattr(row, "rationale", "")),
        ])
    for row in getattr(card, "falsification_criteria", ()):
        parts.extend([
            str(getattr(row, "observable", "")),
            str(getattr(row, "falsifying_outcome", "")),
        ])
    parts.extend(str(value) for value in getattr(card, "assumptions", ()))
    return "\n".join(parts)


def _accepted_card(arm: object) -> object | None:
    run = getattr(arm, "run_outcome", None)
    canonical = getattr(run, "canonical_outcome", None)
    portfolio = getattr(canonical, "accepted_portfolio", None)
    hypotheses = tuple(getattr(portfolio, "hypotheses", ()))
    if len(hypotheses) != 1:
        return None
    return hypotheses[0]


def _base_arm_critique(
    *,
    arm_index: int,
    arm: object,
) -> HigherOrderArmSemanticCritique:
    context = getattr(arm, "higher_order_context")
    opportunity = getattr(context, "structural_opportunity")
    premise = _modifier_premise(context)
    modifier_authority = (
        _authority_value(getattr(premise, "authority", ""))
        if premise is not None
        else ""
    )
    card = _accepted_card(arm)
    hypothesis_statement = (
        str(getattr(card, "hypothesis_statement", ""))
        if card is not None else ""
    )
    inferential_bridge = (
        str(getattr(card, "inferential_bridge", ""))
        if card is not None else ""
    )
    full_card_text = _card_text(card) if card is not None else ""

    modifier_text = str(getattr(opportunity, "modifier_text", ""))
    anchor_text = str(getattr(opportunity, "modifier_anchor_text", ""))
    anchor_role = str(getattr(opportunity, "modifier_anchor_role", ""))
    requested_source = str(getattr(context, "requested_source", ""))
    requested_target = str(getattr(context, "requested_target", ""))
    source_mediator = str(
        getattr(opportunity, "source_side_mediator_text", "")
    )
    target_mediator = str(
        getattr(opportunity, "target_side_mediator_text", "")
    )

    modifier_statement_coverage = _coverage(
        modifier_text, hypothesis_statement
    )
    modifier_full_card_coverage = _coverage(
        modifier_text, full_card_text
    )
    anchor_full_card_coverage = _coverage(
        anchor_text, full_card_text
    )
    source_statement_coverage = _coverage(
        requested_source, hypothesis_statement
    )
    target_statement_coverage = _coverage(
        requested_target, hypothesis_statement
    )
    backbone_restatement_overlap = max(
        _containment(modifier_text, requested_source),
        _containment(modifier_text, requested_target),
        _containment(modifier_text, source_mediator),
        _containment(modifier_text, target_mediator),
    )

    issues: list[HigherOrderSemanticCriticIssue] = []

    if card is not None and modifier_full_card_coverage < 0.50:
        issues.append(
            HigherOrderSemanticCriticIssue(
                code="MODIFIER_UTILIZATION_UNRESOLVED",
                message=(
                    "The generated card does not lexically preserve enough of "
                    "the supplied modifier to establish that it was used. "
                    "This is a diagnostic risk, not a semantic rejection."
                ),
                metrics={
                    "modifier_full_card_coverage":
                        modifier_full_card_coverage,
                    "modifier_statement_coverage":
                        modifier_statement_coverage,
                },
            )
        )

    task_statement_coverage = min(
        source_statement_coverage,
        target_statement_coverage,
    )
    if (
        card is not None
        and anchor_role == "mediator"
        and modifier_full_card_coverage >= 0.50
        and anchor_full_card_coverage < 0.50
        and task_statement_coverage >= 0.40
    ):
        issues.append(
            HigherOrderSemanticCriticIssue(
                code="MODIFIER_ANCHOR_BRIDGE_OMITTED",
                message=(
                    "The card uses the modifier in a source-target claim while "
                    "its validated mediator anchor is not lexically preserved. "
                    "This is a conservative proxy for a possible intervention/"
                    "measurement-level jump, not proof of an invalid bridge."
                ),
                metrics={
                    "modifier_full_card_coverage":
                        modifier_full_card_coverage,
                    "modifier_anchor_full_card_coverage":
                        anchor_full_card_coverage,
                    "task_statement_coverage":
                        task_statement_coverage,
                },
            )
        )

    modifier_tokens = _tokens(modifier_text)
    generic_source_or_target_adjacent = bool(
        anchor_role in {"source", "target"}
        and len(modifier_tokens) <= 3
        and (modifier_tokens & _GENERIC_MODIFIER_TOKENS)
    )
    if (
        card is not None
        and (
            backbone_restatement_overlap >= 0.80
            or generic_source_or_target_adjacent
        )
    ):
        issues.append(
            HigherOrderSemanticCriticIssue(
                code="BACKBONE_RESTATEMENT_RISK",
                message=(
                    "The modifier is lexically close to an existing backbone "
                    "role, or is a short generic source/target-adjacent label. "
                    "It may relabel the task relation rather than introduce a "
                    "distinct higher-order condition."
                ),
                metrics={
                    "backbone_restatement_overlap":
                        backbone_restatement_overlap,
                    "generic_source_or_target_adjacent":
                        1.0 if generic_source_or_target_adjacent else 0.0,
                },
            )
        )

    return HigherOrderArmSemanticCritique(
        arm_index=arm_index,
        higher_order_context_id=str(getattr(context, "context_id", "")),
        modifier_component_id=str(
            getattr(
                getattr(context, "lineage", None),
                "modifier_component_id",
                "",
            )
        ),
        modifier_authority=modifier_authority,
        candidate_backed=(
            modifier_authority == "candidate_inspiration"
        ),
        requested_source=requested_source,
        requested_target=requested_target,
        modifier_text=modifier_text,
        modifier_anchor_role=anchor_role,
        modifier_anchor_text=anchor_text,
        hypothesis_statement=hypothesis_statement,
        inferential_bridge=inferential_bridge,
        modifier_statement_coverage=modifier_statement_coverage,
        modifier_full_card_coverage=modifier_full_card_coverage,
        modifier_anchor_full_card_coverage=anchor_full_card_coverage,
        requested_source_statement_coverage=source_statement_coverage,
        requested_target_statement_coverage=target_statement_coverage,
        backbone_restatement_overlap=backbone_restatement_overlap,
        issues=issues,
    )


def _append_known_context_duplicate_issues(
    rows: list[HigherOrderArmSemanticCritique],
) -> int:
    candidates = [
        row for row in rows
        if row.candidate_backed and row.hypothesis_statement
    ]
    known = [
        row for row in rows
        if not row.candidate_backed and row.hypothesis_statement
    ]
    pair_count = 0

    for candidate in candidates:
        for baseline in known:
            modifier_containment = _containment(
                candidate.modifier_text,
                baseline.modifier_text,
            )
            statement_jaccard = _jaccard(
                candidate.hypothesis_statement,
                baseline.hypothesis_statement,
            )
            duplicate_risk = bool(
                (
                    modifier_containment >= 0.50
                    and statement_jaccard >= 0.22
                )
                or statement_jaccard >= 0.55
            )
            if not duplicate_risk:
                continue

            candidate.issues.append(
                HigherOrderSemanticCriticIssue(
                    code="KNOWN_CONTEXT_DUPLICATE",
                    related_arm_index=baseline.arm_index,
                    message=(
                        "This candidate-backed hypothesis is lexically close "
                        "to a confirmed-known-modifier arm. The pair may "
                        "represent the same higher-order idea under different "
                        "wording. This is a diagnostic duplication risk, not "
                        "a novelty decision."
                    ),
                    metrics={
                        "modifier_containment": modifier_containment,
                        "statement_jaccard": statement_jaccard,
                    },
                )
            )
            pair_count += 1

    return pair_count


def critique_higher_order_shadow_batch(
    outcome: HigherOrderShadowBatchOutcome,
) -> HigherOrderBatchSemanticCritique:
    rows = [
        _base_arm_critique(arm_index=index, arm=arm)
        for index, arm in enumerate(outcome.arms, start=1)
    ]
    duplicate_pair_count = _append_known_context_duplicate_issues(rows)

    issue_counts = Counter(
        issue.code
        for row in rows
        for issue in row.issues
    )
    candidate_issue_counts = Counter(
        issue.code
        for row in rows
        if row.candidate_backed
        for issue in row.issues
    )

    return HigherOrderBatchSemanticCritique(
        arm_count=len(rows),
        candidate_backed_arm_count=sum(
            row.candidate_backed for row in rows
        ),
        known_backed_arm_count=sum(
            not row.candidate_backed for row in rows
        ),
        flagged_arm_count=sum(bool(row.issues) for row in rows),
        issue_counts=dict(sorted(issue_counts.items())),
        candidate_issue_counts=dict(
            sorted(candidate_issue_counts.items())
        ),
        known_context_duplicate_pair_count=duplicate_pair_count,
        arms=rows,
    )
