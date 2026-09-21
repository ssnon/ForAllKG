from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    CrossLaneScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AblationCondition = Literal[
    "RELATIONAL_ONLY",
    "REFRAMING_ONLY",
    "COMBINED",
]

AblationComparisonKind = Literal[
    "RELATIONAL_VS_COMBINED",
    "RELATIONAL_VS_REFRAMING",
    "REFRAMING_VS_COMBINED",
]

AblationDimensionId = Literal[
    "task_relevance_and_coverage",
    "evidence_discipline",
    "explanatory_gain",
    "differential_prediction_quality",
    "falsifiability",
    "discriminating_experiment_quality",
    "portfolio_complementarity",
]


class AblationDimensionSpec(StrictModel):
    dimension_id: AblationDimensionId
    question: str = Field(min_length=1)
    candidate_count_must_not_determine_judgment: Literal[True] = True
    no_single_composite_score: Literal[True] = True


class BlindEvidenceStatement(StrictModel):
    evidence_alias: str = Field(pattern=r"^E\d{2,}$")
    text: str = Field(min_length=1)
    epistemic_role: str = Field(min_length=1)
    claim_kind: str = Field(min_length=1)
    paper_count: int = Field(ge=0)
    requires_verification: bool
    eligible_as_positive_premise: bool
    eligible_as_gap: bool


class BlindPrediction(StrictModel):
    observable: str = Field(min_length=1)
    expected_direction: str | None = None
    rationale: str | None = None
    baseline_expectation: str | None = None
    alternative_expectation: str | None = None
    discriminating_outcome: str | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> "BlindPrediction":
        relational_shape = self.expected_direction is not None and self.rationale is not None
        contrast_shape = (
            self.baseline_expectation is not None
            and self.alternative_expectation is not None
            and self.discriminating_outcome is not None
        )
        if not (relational_shape or contrast_shape):
            raise ValueError("blind prediction must preserve a supported prediction shape")
        return self


class BlindFalsifier(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)


class BlindDiscriminatingTest(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    baseline_favoring_outcome: str = Field(min_length=1)
    alternative_favoring_outcome: str = Field(min_length=1)


class BlindScientificCandidate(StrictModel):
    candidate_alias: str = Field(pattern=r"^C\d{2,}$")
    title: str = Field(min_length=1)
    scientific_proposal: str = Field(min_length=1)
    baseline_model_summary: str | None = None
    alternative_or_resolution_summary: str | None = None
    reasoning_rationale: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    premise_evidence_aliases: list[str] = Field(default_factory=list)
    gap_evidence_aliases: list[str] = Field(default_factory=list)
    predictions: list[BlindPrediction] = Field(default_factory=list)
    falsifiers: list[BlindFalsifier] = Field(default_factory=list)
    discriminating_test: BlindDiscriminatingTest | None = None
    unresolved_questions: list[str] = Field(default_factory=list)

    source_candidate_identity_hidden: Literal[True] = True
    source_lane_identity_hidden: Literal[True] = True
    source_operator_identity_hidden: Literal[True] = True
    source_schema_identity_hidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_candidate(self) -> "BlindScientificCandidate":
        if not self.premise_evidence_aliases:
            raise ValueError("blind candidate requires grounded premise evidence aliases")
        if len(self.premise_evidence_aliases) != len(set(self.premise_evidence_aliases)):
            raise ValueError("blind premise evidence aliases must be unique")
        if len(self.gap_evidence_aliases) != len(set(self.gap_evidence_aliases)):
            raise ValueError("blind gap evidence aliases must be unique")
        if not self.predictions:
            raise ValueError("blind candidate requires predictions")
        if not self.falsifiers:
            raise ValueError("blind candidate requires falsifiers")
        return self


class BlindAblationArm(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    candidates: list[BlindScientificCandidate]
    candidate_count: int = Field(ge=0)

    source_condition_hidden: Literal[True] = True
    source_lane_labels_hidden: Literal[True] = True
    source_operator_labels_hidden: Literal[True] = True
    source_candidate_ids_hidden: Literal[True] = True

    @model_validator(mode="after")
    def validate_count(self) -> "BlindAblationArm":
        if self.candidate_count != len(self.candidates):
            raise ValueError("arm candidate_count must match candidates")
        aliases = [row.candidate_alias for row in self.candidates]
        if len(aliases) != len(set(aliases)):
            raise ValueError("arm candidate aliases must be unique")
        return self


class BlindAblationComparison(StrictModel):
    comparison_id: str = Field(min_length=1)
    comparison_alias: str = Field(pattern=r"^COMPARISON_\d{2,}$")
    arm_a: BlindAblationArm
    arm_b: BlindAblationArm
    candidate_count_equal: bool
    absolute_candidate_count_difference: int = Field(ge=0)
    interpretation_cautions: list[str] = Field(min_length=1)

    source_conditions_hidden: Literal[True] = True
    candidate_count_is_not_quality_signal: Literal[True] = True
    evaluator_must_not_infer_quality_from_portfolio_size: Literal[True] = True


class ScientificReasoningAblationPacket(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-evaluation-packet-v1"
    ] = "scientific-reasoning-ablation-evaluation-packet-v1"

    packet_id: str = Field(min_length=1)
    task_alias: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence_statements: list[BlindEvidenceStatement]
    evaluation_dimensions: list[AblationDimensionSpec]
    comparisons: list[BlindAblationComparison]

    comparison_count: int = Field(ge=0)
    evidence_statement_count: int = Field(ge=0)
    source_candidate_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    blind_evaluation_packet: Literal[True] = True
    source_task_identity_hidden_from_evaluator: Literal[True] = True
    source_condition_labels_hidden_from_evaluator: Literal[True] = True
    source_candidate_ids_hidden_from_evaluator: Literal[True] = True
    source_lane_labels_hidden_from_evaluator: Literal[True] = True
    source_operator_labels_hidden_from_evaluator: Literal[True] = True
    identical_task_evidence_supplied_to_all_arms: Literal[True] = True
    candidate_count_asymmetry_recorded: Literal[True] = True
    candidate_count_is_not_quality_signal: Literal[True] = True
    evaluation_judgment_performed: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    winner_selected: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    scientific_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_packet(self) -> "ScientificReasoningAblationPacket":
        if self.comparison_count != len(self.comparisons):
            raise ValueError("comparison_count must match comparisons")
        if self.evidence_statement_count != len(self.evidence_statements):
            raise ValueError("evidence_statement_count must match evidence_statements")
        dimension_ids = [row.dimension_id for row in self.evaluation_dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("evaluation dimensions must be unique")
        comparison_ids = [row.comparison_id for row in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("comparison IDs must be unique")
        return self


class AblationArmKey(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    condition: AblationCondition
    candidate_alias_to_source_id: dict[str, str]


class AblationComparisonKey(StrictModel):
    comparison_id: str = Field(min_length=1)
    comparison_alias: str = Field(min_length=1)
    comparison_kind: AblationComparisonKind
    arms: list[AblationArmKey] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_arms(self) -> "AblationComparisonKey":
        aliases = [row.arm_alias for row in self.arms]
        if set(aliases) != {"ARM_A", "ARM_B"}:
            raise ValueError("comparison key must map ARM_A and ARM_B exactly once")
        return self


class ScientificReasoningAblationBlindKey(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-blind-key-v1"
    ] = "scientific-reasoning-ablation-blind-key-v1"

    packet_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_cross_lane_portfolio_id: str = Field(min_length=1)
    source_cross_lane_portfolio_file_sha256: str = Field(min_length=64, max_length=64)
    source_context_file_sha256: str = Field(min_length=64, max_length=64)
    source_relational_portfolio_file_sha256: str = Field(min_length=64, max_length=64)
    source_reframe_shadow_file_sha256: str = Field(min_length=64, max_length=64)
    source_proxy_shadow_file_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    source_contradiction_shadow_file_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )

    evidence_alias_to_statement_id: dict[str, str]
    condition_candidate_source_ids: dict[AblationCondition, list[str]]
    comparisons: list[AblationComparisonKey]

    blind_key_must_not_be_supplied_to_evaluator: Literal[True] = True
    unblinding_only: Literal[True] = True
    evaluation_judgment_performed: Literal[False] = False
    scientific_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False


_DIMENSIONS: tuple[tuple[AblationDimensionId, str], ...] = (
    (
        "task_relevance_and_coverage",
        "How directly and comprehensively does the portfolio address the stated scientific task without drifting to adjacent questions?",
    ),
    (
        "evidence_discipline",
        "Are claims, mechanisms, and uncertainty proportionate to the supplied evidence, with unresolved points remaining explicitly hypothetical?",
    ),
    (
        "explanatory_gain",
        "Does the portfolio add a non-trivial explanatory representation rather than merely restating, combining, or renaming the supplied observations?",
    ),
    (
        "differential_prediction_quality",
        "Does the portfolio produce concrete observations whose outcomes would differ between plausible explanations or models?",
    ),
    (
        "falsifiability",
        "Are there credible observable outcomes that would count against the proposed scientific explanations?",
    ),
    (
        "discriminating_experiment_quality",
        "Where experiments or tests are proposed, do they distinguish competing explanations rather than merely collect more measurements?",
    ),
    (
        "portfolio_complementarity",
        "Do the candidates contribute complementary scientific lines of reasoning rather than surface paraphrases of the same proposal?",
    ),
)


_COMPARISONS: tuple[
    tuple[AblationComparisonKind, AblationCondition, AblationCondition], ...
] = (
    ("RELATIONAL_VS_COMBINED", "RELATIONAL_ONLY", "COMBINED"),
    ("RELATIONAL_VS_REFRAMING", "RELATIONAL_ONLY", "REFRAMING_ONLY"),
    ("REFRAMING_VS_COMBINED", "REFRAMING_ONLY", "COMBINED"),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _evaluation_dimensions() -> list[AblationDimensionSpec]:
    return [
        AblationDimensionSpec(dimension_id=dimension_id, question=question)
        for dimension_id, question in _DIMENSIONS
    ]


def _source_candidate_map(
    *,
    relational_portfolio: HypothesisPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for card in relational_portfolio.hypotheses:
        rows[card.hypothesis_id] = card
    for candidate in reframe_shadow.candidates:
        rows[candidate.reframe_id] = candidate
    if proxy_shadow is not None:
        for candidate in proxy_shadow.candidates:
            rows[candidate.candidate_id] = candidate
    if contradiction_shadow is not None:
        for candidate in contradiction_shadow.candidates:
            rows[candidate.candidate_id] = candidate
    return rows


def _validate_lineage(
    *,
    context: HypothesisContext,
    cross_lane_portfolio: CrossLaneScientificReasoningShadowPortfolio,
    relational_portfolio: HypothesisPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
    source_candidates: dict[str, object],
) -> None:
    if cross_lane_portfolio.source_task_id != context.task_id:
        raise ValueError("cross-lane portfolio task_id mismatch")
    if cross_lane_portfolio.source_context_id != context.context_id:
        raise ValueError("cross-lane portfolio source_context_id mismatch")
    if cross_lane_portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("cross-lane portfolio source_context_sha256 mismatch")
    if relational_portfolio.source_context_id != context.context_id:
        raise ValueError("relational portfolio source_context_id mismatch")
    if relational_portfolio.source_context_sha256 != context.context_sha256:
        raise ValueError("relational portfolio source_context_sha256 mismatch")
    if relational_portfolio.portfolio_id != cross_lane_portfolio.relational_lane.source_portfolio_id:
        raise ValueError("cross-lane relational source_portfolio_id mismatch")

    reports: list[tuple[str, object]] = [("reframe", reframe_shadow)]
    if proxy_shadow is not None:
        reports.append(("proxy", proxy_shadow))
    if contradiction_shadow is not None:
        reports.append(("contradiction", contradiction_shadow))
    for name, report in reports:
        if report.source_task_id != context.task_id:
            raise ValueError(f"{name} shadow task_id mismatch")
        if report.source_context_id != context.context_id:
            raise ValueError(f"{name} shadow source_context_id mismatch")
        if report.source_context_sha256 != context.context_sha256:
            raise ValueError(f"{name} shadow source_context_sha256 mismatch")

    expected_ids = {row.source_object_id for row in cross_lane_portfolio.entries}
    actual_ids = set(source_candidates)
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "cross-lane/source candidate lineage mismatch; "
            f"missing={missing}, extra={extra}"
        )

    relational_ids = {row.hypothesis_id for row in relational_portfolio.hypotheses}
    if relational_ids != set(cross_lane_portfolio.relational_lane.hypothesis_ids):
        raise ValueError("cross-lane relational hypothesis lineage mismatch")
    reframing_ids = {row.reframe_id for row in reframe_shadow.candidates}
    if proxy_shadow is not None:
        reframing_ids.update(row.candidate_id for row in proxy_shadow.candidates)
    if contradiction_shadow is not None:
        reframing_ids.update(
            row.candidate_id for row in contradiction_shadow.candidates
        )
    if reframing_ids != set(cross_lane_portfolio.reframing_lane.candidate_ids):
        raise ValueError("cross-lane reframing candidate lineage mismatch")


def _evidence_aliases(
    context: HypothesisContext,
) -> tuple[list[BlindEvidenceStatement], dict[str, str], dict[str, str]]:
    eligible = [
        row
        for row in context.evidence_statements
        if row.eligible_as_premise or row.eligible_as_gap
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{context.task_id}|{row.statement_id}".encode("utf-8")
        ).hexdigest()
    )
    statement_to_alias: dict[str, str] = {}
    alias_to_statement: dict[str, str] = {}
    blind_rows: list[BlindEvidenceStatement] = []
    for index, row in enumerate(eligible, start=1):
        alias = f"E{index:02d}"
        statement_to_alias[row.statement_id] = alias
        alias_to_statement[alias] = row.statement_id
        blind_rows.append(
            BlindEvidenceStatement(
                evidence_alias=alias,
                text=row.text,
                epistemic_role=row.epistemic_role,
                claim_kind=row.claim_kind,
                paper_count=len(set(row.paper_ids)),
                requires_verification=row.requires_verification,
                eligible_as_positive_premise=row.eligible_as_premise,
                eligible_as_gap=row.eligible_as_gap,
            )
        )
    return blind_rows, statement_to_alias, alias_to_statement


def _aliases_for_statement_ids(
    statement_ids: list[str],
    statement_to_alias: dict[str, str],
    *,
    role: Literal["premise", "gap"],
    evidence_by_id: dict[str, object],
) -> list[str]:
    aliases: list[str] = []
    for statement_id in statement_ids:
        if statement_id not in statement_to_alias:
            raise ValueError(
                f"candidate references {role} statement outside eligible evaluation evidence: {statement_id}"
            )
        evidence = evidence_by_id[statement_id]
        if role == "premise" and not getattr(evidence, "eligible_as_premise"):
            raise ValueError(f"candidate premise is not eligible_as_premise: {statement_id}")
        if role == "gap" and not getattr(evidence, "eligible_as_gap"):
            raise ValueError(f"candidate gap is not eligible_as_gap: {statement_id}")
        aliases.append(statement_to_alias[statement_id])
    return aliases


def _blind_test(test) -> BlindDiscriminatingTest:
    return BlindDiscriminatingTest(
        test_design=test.test_design,
        primary_observables=list(test.primary_observables),
        baseline_favoring_outcome=test.baseline_favoring_outcome,
        alternative_favoring_outcome=test.alternative_favoring_outcome,
    )


def _blind_relational_candidate(
    *,
    card: HypothesisCard,
    alias: str,
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> BlindScientificCandidate:
    premises = _aliases_for_statement_ids(
        card.premise_statement_ids,
        statement_to_alias,
        role="premise",
        evidence_by_id=evidence_by_id,
    )
    gaps = _aliases_for_statement_ids(
        card.gap_statement_ids,
        statement_to_alias,
        role="gap",
        evidence_by_id=evidence_by_id,
    )
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=card.title,
        scientific_proposal=card.hypothesis_statement,
        baseline_model_summary=None,
        alternative_or_resolution_summary=None,
        reasoning_rationale=card.inferential_bridge,
        assumptions=list(card.assumptions),
        premise_evidence_aliases=premises,
        gap_evidence_aliases=gaps,
        predictions=[
            BlindPrediction(
                observable=row.observable,
                expected_direction=row.expected_direction,
                rationale=row.rationale,
            )
            for row in card.predicted_observations
        ],
        falsifiers=[
            BlindFalsifier(
                observable=row.observable,
                falsifying_outcome=row.falsifying_outcome,
            )
            for row in card.falsification_criteria
        ],
        discriminating_test=None,
        unresolved_questions=[],
    )


def _blind_reframe_candidate(
    *,
    candidate: ScientificReframeCandidate,
    alias: str,
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> BlindScientificCandidate:
    premises = _aliases_for_statement_ids(
        candidate.premise_statement_ids,
        statement_to_alias,
        role="premise",
        evidence_by_id=evidence_by_id,
    )
    gaps = _aliases_for_statement_ids(
        candidate.gap_statement_ids,
        statement_to_alias,
        role="gap",
        evidence_by_id=evidence_by_id,
    )
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=candidate.title,
        scientific_proposal=candidate.alternative_model.summary,
        baseline_model_summary=candidate.baseline_model.summary,
        alternative_or_resolution_summary=candidate.alternative_model.summary,
        reasoning_rationale=candidate.challenged_assumption,
        assumptions=list(candidate.alternative_model.assumptions),
        premise_evidence_aliases=premises,
        gap_evidence_aliases=gaps,
        predictions=[
            BlindPrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            BlindFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_blind_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _blind_proxy_candidate(
    *,
    candidate: ScientificProxyChallengeCandidate,
    alias: str,
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> BlindScientificCandidate:
    premises = _aliases_for_statement_ids(
        candidate.premise_statement_ids,
        statement_to_alias,
        role="premise",
        evidence_by_id=evidence_by_id,
    )
    gaps = _aliases_for_statement_ids(
        candidate.gap_statement_ids,
        statement_to_alias,
        role="gap",
        evidence_by_id=evidence_by_id,
    )
    rationale = (
        f"Challenge the assumption that {candidate.challenged_proxy_assumption} "
        f"The observable under challenge is {candidate.challenged_observable}; "
        f"the intended construct is {candidate.target_construct}."
    )
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=candidate.title,
        scientific_proposal=candidate.alternative_model.summary,
        baseline_model_summary=candidate.baseline_model.summary,
        alternative_or_resolution_summary=candidate.alternative_model.summary,
        reasoning_rationale=rationale,
        assumptions=list(candidate.alternative_model.assumptions),
        premise_evidence_aliases=premises,
        gap_evidence_aliases=gaps,
        predictions=[
            BlindPrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            BlindFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_blind_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _blind_contradiction_candidate(
    *,
    candidate: ScientificContradictionResolutionCandidate,
    alias: str,
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> BlindScientificCandidate:
    premises = _aliases_for_statement_ids(
        candidate.premise_statement_ids,
        statement_to_alias,
        role="premise",
        evidence_by_id=evidence_by_id,
    )
    gaps = _aliases_for_statement_ids(
        candidate.gap_statement_ids,
        statement_to_alias,
        role="gap",
        evidence_by_id=evidence_by_id,
    )
    rationale = (
        f"Apparent tension: {candidate.apparent_contradiction} "
        f"Proposed reconciliation principle: {candidate.resolution_principle}"
    )
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=candidate.title,
        scientific_proposal=candidate.resolution_model.summary,
        baseline_model_summary=candidate.baseline_model.summary,
        alternative_or_resolution_summary=candidate.resolution_model.summary,
        reasoning_rationale=rationale,
        assumptions=list(candidate.resolution_model.assumptions),
        premise_evidence_aliases=premises,
        gap_evidence_aliases=gaps,
        predictions=[
            BlindPrediction(
                observable=row.observable,
                baseline_expectation=row.baseline_expectation,
                alternative_expectation=row.alternative_expectation,
                discriminating_outcome=row.discriminating_outcome,
            )
            for row in candidate.differential_predictions
        ],
        falsifiers=[
            BlindFalsifier(falsifying_outcome=row.falsifying_outcome)
            for row in candidate.falsifiers
        ],
        discriminating_test=_blind_test(candidate.discriminating_test),
        unresolved_questions=list(candidate.unresolved_questions),
    )


def _blind_candidate(
    *,
    source: object,
    alias: str,
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> BlindScientificCandidate:
    if isinstance(source, HypothesisCard):
        return _blind_relational_candidate(
            card=source,
            alias=alias,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
    if isinstance(source, ScientificReframeCandidate):
        return _blind_reframe_candidate(
            candidate=source,
            alias=alias,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
    if isinstance(source, ScientificProxyChallengeCandidate):
        return _blind_proxy_candidate(
            candidate=source,
            alias=alias,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
    if isinstance(source, ScientificContradictionResolutionCandidate):
        return _blind_contradiction_candidate(
            candidate=source,
            alias=alias,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
    raise TypeError(f"unsupported source candidate type: {type(source).__name__}")


def _condition_ids(
    cross_lane_portfolio: CrossLaneScientificReasoningShadowPortfolio,
) -> dict[AblationCondition, list[str]]:
    relational = list(cross_lane_portfolio.relational_lane.hypothesis_ids)
    reframing = list(cross_lane_portfolio.reframing_lane.candidate_ids)
    return {
        "RELATIONAL_ONLY": relational,
        "REFRAMING_ONLY": reframing,
        "COMBINED": [*relational, *reframing],
    }


def _arm_order(
    *,
    task_id: str,
    comparison_kind: AblationComparisonKind,
    condition_left: AblationCondition,
    condition_right: AblationCondition,
) -> tuple[AblationCondition, AblationCondition]:
    digest = hashlib.sha256(
        f"{task_id}|{comparison_kind}|blind-arm-order-v1".encode("utf-8")
    ).digest()
    if digest[0] % 2:
        return condition_right, condition_left
    return condition_left, condition_right


def _ordered_source_ids(
    *,
    comparison_id: str,
    arm_alias: str,
    source_ids: list[str],
) -> list[str]:
    return sorted(
        source_ids,
        key=lambda source_id: hashlib.sha256(
            f"{comparison_id}|{arm_alias}|{source_id}|candidate-order-v1".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def _arm(
    *,
    arm_alias: Literal["ARM_A", "ARM_B"],
    comparison_id: str,
    condition: AblationCondition,
    source_ids: list[str],
    source_candidates: dict[str, object],
    statement_to_alias: dict[str, str],
    evidence_by_id: dict[str, object],
) -> tuple[BlindAblationArm, AblationArmKey]:
    ordered = _ordered_source_ids(
        comparison_id=comparison_id,
        arm_alias=arm_alias,
        source_ids=source_ids,
    )
    blind_candidates: list[BlindScientificCandidate] = []
    alias_to_source: dict[str, str] = {}
    for index, source_id in enumerate(ordered, start=1):
        alias = f"C{index:02d}"
        blind_candidates.append(
            _blind_candidate(
                source=source_candidates[source_id],
                alias=alias,
                statement_to_alias=statement_to_alias,
                evidence_by_id=evidence_by_id,
            )
        )
        alias_to_source[alias] = source_id
    return (
        BlindAblationArm(
            arm_alias=arm_alias,
            candidates=blind_candidates,
            candidate_count=len(blind_candidates),
        ),
        AblationArmKey(
            arm_alias=arm_alias,
            condition=condition,
            candidate_alias_to_source_id=alias_to_source,
        ),
    )


def build_scientific_reasoning_ablation_packet(
    *,
    context: HypothesisContext,
    cross_lane_portfolio: CrossLaneScientificReasoningShadowPortfolio,
    relational_portfolio: HypothesisPortfolio,
    reframe_shadow: ScientificReframingShadowReport,
    proxy_shadow: ProxyChallengeRunReport | None,
    contradiction_shadow: ContradictionResolutionRunReport | None,
    source_cross_lane_portfolio_file_sha256: str,
    source_context_file_sha256: str,
    source_relational_portfolio_file_sha256: str,
    source_reframe_shadow_file_sha256: str,
    source_proxy_shadow_file_sha256: str | None,
    source_contradiction_shadow_file_sha256: str | None,
) -> tuple[ScientificReasoningAblationPacket, ScientificReasoningAblationBlindKey]:
    source_candidates = _source_candidate_map(
        relational_portfolio=relational_portfolio,
        reframe_shadow=reframe_shadow,
        proxy_shadow=proxy_shadow,
        contradiction_shadow=contradiction_shadow,
    )
    _validate_lineage(
        context=context,
        cross_lane_portfolio=cross_lane_portfolio,
        relational_portfolio=relational_portfolio,
        reframe_shadow=reframe_shadow,
        proxy_shadow=proxy_shadow,
        contradiction_shadow=contradiction_shadow,
        source_candidates=source_candidates,
    )

    evidence_rows, statement_to_alias, alias_to_statement = _evidence_aliases(context)
    evidence_by_id = {row.statement_id: row for row in context.evidence_statements}
    if not evidence_rows:
        raise ValueError("ablation evaluation requires eligible task evidence")

    condition_ids = _condition_ids(cross_lane_portfolio)
    if not condition_ids["RELATIONAL_ONLY"]:
        raise ValueError("ablation evaluation requires at least one relational candidate")
    if not condition_ids["REFRAMING_ONLY"]:
        raise ValueError("ablation evaluation requires at least one reframing candidate")

    packet_id = _stable_id(
        "scientific_reasoning_ablation_packet",
        context.task_id,
        context.context_sha256,
        cross_lane_portfolio.portfolio_id,
        *sorted(source_candidates),
    )
    task_alias = _stable_id("task_alias", context.task_id)[:31]

    comparisons: list[BlindAblationComparison] = []
    comparison_keys: list[AblationComparisonKey] = []
    for index, (kind, left, right) in enumerate(_COMPARISONS, start=1):
        comparison_id = _stable_id("ablation_comparison", packet_id, kind)
        comparison_alias = f"COMPARISON_{index:02d}"
        condition_a, condition_b = _arm_order(
            task_id=context.task_id,
            comparison_kind=kind,
            condition_left=left,
            condition_right=right,
        )
        arm_a, key_a = _arm(
            arm_alias="ARM_A",
            comparison_id=comparison_id,
            condition=condition_a,
            source_ids=condition_ids[condition_a],
            source_candidates=source_candidates,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
        arm_b, key_b = _arm(
            arm_alias="ARM_B",
            comparison_id=comparison_id,
            condition=condition_b,
            source_ids=condition_ids[condition_b],
            source_candidates=source_candidates,
            statement_to_alias=statement_to_alias,
            evidence_by_id=evidence_by_id,
        )
        count_diff = abs(arm_a.candidate_count - arm_b.candidate_count)
        cautions = [
            "Do not reward an arm merely because it contains more candidates or more text.",
            "Judge scientific content dimension by dimension; ties are valid and no single composite score is requested.",
            "All arms receive the identical task question and eligible evidence view.",
        ]
        if count_diff:
            cautions.append(
                "The arms have different candidate counts; interpret breadth and complementarity separately from per-candidate scientific quality."
            )
        comparisons.append(
            BlindAblationComparison(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                arm_a=arm_a,
                arm_b=arm_b,
                candidate_count_equal=(count_diff == 0),
                absolute_candidate_count_difference=count_diff,
                interpretation_cautions=cautions,
            )
        )
        comparison_keys.append(
            AblationComparisonKey(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                comparison_kind=kind,
                arms=[key_a, key_b],
            )
        )

    packet = ScientificReasoningAblationPacket(
        packet_id=packet_id,
        task_alias=task_alias,
        question=context.question,
        evidence_statements=evidence_rows,
        evaluation_dimensions=_evaluation_dimensions(),
        comparisons=comparisons,
        comparison_count=len(comparisons),
        evidence_statement_count=len(evidence_rows),
        source_candidate_count=len(source_candidates),
    )
    key = ScientificReasoningAblationBlindKey(
        packet_id=packet.packet_id,
        source_task_id=context.task_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_cross_lane_portfolio_id=cross_lane_portfolio.portfolio_id,
        source_cross_lane_portfolio_file_sha256=source_cross_lane_portfolio_file_sha256,
        source_context_file_sha256=source_context_file_sha256,
        source_relational_portfolio_file_sha256=source_relational_portfolio_file_sha256,
        source_reframe_shadow_file_sha256=source_reframe_shadow_file_sha256,
        source_proxy_shadow_file_sha256=source_proxy_shadow_file_sha256,
        source_contradiction_shadow_file_sha256=source_contradiction_shadow_file_sha256,
        evidence_alias_to_statement_id=alias_to_statement,
        condition_candidate_source_ids={
            condition: list(ids) for condition, ids in condition_ids.items()
        },
        comparisons=comparison_keys,
    )
    return packet, key


def ablation_condition_counts(
    key: ScientificReasoningAblationBlindKey,
) -> dict[str, int]:
    return {
        condition: len(ids)
        for condition, ids in key.condition_candidate_source_ids.items()
    }


def ablation_comparison_count_differences(
    packet: ScientificReasoningAblationPacket,
) -> dict[str, int]:
    return {
        row.comparison_alias: row.absolute_candidate_count_difference
        for row in packet.comparisons
    }
