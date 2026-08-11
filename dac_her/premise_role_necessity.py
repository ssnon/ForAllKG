from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

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
from dac_her.premise_necessity_diagnostic import (
    PremiseNecessityDiagnosticReport,
)


ROLE_PROMPT_VERSION = "premise-role-critic-v2.9.1-ps2"
ABLATION_PROMPT_VERSION = "premise-ablation-critic-v2.9.1-ps2"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PremiseRole = Literal[
    "direct_clause_support",
    "inferential_bridge_support",
    "contextual_support",
    "tangential_support",
    "scope_mismatch",
    "counterevidence_or_limitation",
]

GroundingStatus = Literal[
    "sufficiently_grounded",
    "partially_grounded",
    "insufficiently_grounded",
    "scope_conflicted",
    "uncertain",
]

Confidence = Literal["low", "medium", "high"]

NecessityVerdict = Literal[
    "critical_for_current_grounded_chain",
    "material_support_loss",
    "redundant_or_replaceable_for_current_chain",
    "contextual_or_nonessential_for_current_chain",
    "scope_problem_or_counterevidence",
    "uncertain",
]


class CriticPrompt(StrictModel):
    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


class PremiseRoleReviewDraft(StrictModel):
    hypothesis_id: str
    premise_statement_id: str
    role: PremiseRole

    supported_hypothesis_clause: str | None = None
    supported_bridge_clause: str | None = None

    support_explanation: str = Field(min_length=1)
    limiting_or_scope_caveat: str | None = None
    topical_overlap_without_support: bool = False
    confidence: Confidence


class PremiseAblationReviewDraft(StrictModel):
    hypothesis_id: str
    omitted_premise_statement_id: str

    remaining_grounding_status: GroundingStatus
    inferential_bridge_grounded: bool

    unsupported_or_weak_hypothesis_clauses: list[str] = Field(
        default_factory=list
    )
    critical_missing_link: str | None = None
    remaining_support_summary: str = Field(min_length=1)
    confidence: Confidence


class QuotedClauseAudit(StrictModel):
    text: str
    exact_substring_match: bool
    source_field: Literal[
        "hypothesis_statement",
        "inferential_bridge",
        "unknown",
    ]


class PremiseRoleReview(StrictModel):
    hypothesis_id: str
    premise_statement_id: str
    role: PremiseRole

    supported_hypothesis_clause: str | None = None
    supported_hypothesis_clause_audit: QuotedClauseAudit | None = None

    supported_bridge_clause: str | None = None
    supported_bridge_clause_audit: QuotedClauseAudit | None = None

    support_explanation: str
    limiting_or_scope_caveat: str | None = None
    topical_overlap_without_support: bool = False
    confidence: Confidence


class PremiseAblationReview(StrictModel):
    hypothesis_id: str
    omitted_premise_statement_id: str

    remaining_grounding_status: GroundingStatus
    inferential_bridge_grounded: bool

    unsupported_or_weak_hypothesis_clauses: list[str] = Field(
        default_factory=list
    )
    unsupported_clause_audits: list[QuotedClauseAudit] = Field(
        default_factory=list
    )

    critical_missing_link: str | None = None
    remaining_support_summary: str
    confidence: Confidence


class PS1Snapshot(StrictModel):
    core_score: float | None = None
    core_rank: int | None = None
    axis_score: float | None = None
    axis_rank: int | None = None
    prediction_score: float | None = None
    prediction_rank: int | None = None
    diagnostic_flags: list[str] = Field(default_factory=list)


class PremiseRoleNecessityCard(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    premise_statement_id: str
    premise_text: str
    premise_claim_kind: str
    premise_epistemic_role: str
    premise_paper_ids: list[str] = Field(default_factory=list)

    remaining_premise_statement_ids: list[str] = Field(default_factory=list)

    role_review: PremiseRoleReview
    ablation_review: PremiseAblationReview
    necessity_verdict: NecessityVerdict
    verdict_reason: str

    ps1_snapshot: PS1Snapshot | None = None

    role_prompt_sha256: str
    ablation_prompt_sha256: str


class HypothesisPS2Summary(StrictModel):
    hypothesis_id: str
    title: str
    axis_id: str
    axis_label: str

    premise_count: int
    critical_premise_statement_ids: list[str] = Field(default_factory=list)
    material_premise_statement_ids: list[str] = Field(default_factory=list)
    replaceable_premise_statement_ids: list[str] = Field(default_factory=list)
    contextual_or_nonessential_statement_ids: list[str] = Field(
        default_factory=list
    )
    scope_problem_statement_ids: list[str] = Field(default_factory=list)
    uncertain_statement_ids: list[str] = Field(default_factory=list)


class PremiseRoleNecessityPolicy(StrictModel):
    diagnostic_only: Literal[True] = True
    scientific_selection_changed: Literal[False] = False

    role_review_is_target_premise_isolated: Literal[True] = True
    ablation_review_hides_omitted_premise_text: Literal[True] = True
    external_knowledge_allowed: Literal[False] = False

    critic_verdict_is_scientific_truth: Literal[False] = False
    quote_must_be_audited: Literal[True] = True
    low_confidence_is_accepted_as_decisive: Literal[False] = False


class PremiseRoleNecessityReport(StrictModel):
    schema_version: Literal["premise-role-necessity-report-v1"] = (
        "premise-role-necessity-report-v1"
    )

    report_id: str
    report_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_axis_plan_id: str
    source_axis_report_id: str
    source_ps1_report_id: str | None = None

    domain_profile_id: str
    corpus_id: str

    critic_model: str
    role_prompt_version: str
    ablation_prompt_version: str

    hypothesis_count: int
    selected_premise_incidence_count: int
    role_call_count: int
    ablation_call_count: int

    necessity_verdict_counts: dict[str, int] = Field(default_factory=dict)
    role_counts: dict[str, int] = Field(default_factory=dict)
    ablation_status_counts: dict[str, int] = Field(default_factory=dict)

    invalid_quoted_clause_count: int = 0

    cards: list[PremiseRoleNecessityCard] = Field(default_factory=list)
    hypothesis_summaries: list[HypothesisPS2Summary] = Field(
        default_factory=list
    )

    policy: PremiseRoleNecessityPolicy = Field(
        default_factory=PremiseRoleNecessityPolicy
    )


@dataclass(frozen=True)
class StructuredGeneration:
    value: BaseModel
    elapsed_seconds: float


class InstructorPremiseCriticBackend:
    backend_name = "instructor_openai_compatible_premise_critic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 2,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} "
                "or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "PS2 requires installed 'openai' and 'instructor'."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )
        if mode is None:
            available = sorted(
                name
                for name in dir(instructor.Mode)
                if name.isupper()
            )
            raise ValueError(
                f"Unknown Instructor mode {self.instructor_mode!r}. "
                f"Available modes include: {available}"
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        raw_client = OpenAI(**kwargs)
        self._client = instructor.from_openai(
            raw_client,
            mode=mode,
        )
        return self._client

    def call(
        self,
        prompt: CriticPrompt,
        response_model: type[BaseModel],
    ) -> StructuredGeneration:
        client = self._get_client()
        started = time.perf_counter()
        value = client.chat.completions.create(
            model=self.model_name,
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt.user_prompt,
                },
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
        )
        elapsed = time.perf_counter() - started
        if not isinstance(value, response_model):
            value = response_model.model_validate(value)
        return StructuredGeneration(
            value=value,
            elapsed_seconds=elapsed,
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
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")
    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _make_prompt(
    version: str,
    system_prompt: str,
    user_prompt: str,
) -> CriticPrompt:
    payload = {
        "prompt_version": version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    return CriticPrompt(
        **payload,
        prompt_sha256=_sha256_json(payload),
    )


def build_role_prompt(
    *,
    hypothesis: HypothesisCard,
    axis: DiscoveryAxis,
    premise: HypothesisEvidenceStatement,
) -> CriticPrompt:
    system = """You are a scientific premise-role critic.

Evaluate ONLY the relationship between the ONE supplied premise and the
accepted hypothesis. Do not use outside knowledge. Do not assume that topical
similarity is scientific support. Respect explicit caveats and negative
statements in the premise.

Role definitions:
- direct_clause_support: the premise directly supports a material clause of the
  hypothesis as written.
- inferential_bridge_support: the premise supports the stated inferential bridge
  but not a material hypothesis clause directly.
- contextual_support: scientifically relevant background/context, but the claim
  can be formulated without relying on it as a grounding link.
- tangential_support: related topic, but does not materially ground the claim or
  bridge.
- scope_mismatch: the premise concerns a materially different system,
  relation, condition, or outcome such that using it as support risks scope
  transfer.
- counterevidence_or_limitation: the premise explicitly limits, weakens, or
  cautions against the relationship asserted by the hypothesis.

If you provide supported_hypothesis_clause or supported_bridge_clause, copy a
short contiguous fragment VERBATIM from the supplied hypothesis_statement or
inferential_bridge respectively. Use null if no exact clause is supported.

The discovery axis is inspiration/context only; it is NOT scientific evidence.
"""
    user = f"""HYPOTHESIS
ID: {hypothesis.hypothesis_id}
Title: {hypothesis.title}
Hypothesis statement:
{hypothesis.hypothesis_statement}

Inferential bridge:
{hypothesis.inferential_bridge}

Predicted observations:
{json.dumps([
    {
        "observable": row.observable,
        "expected_direction": row.expected_direction,
        "rationale": row.rationale,
    }
    for row in hypothesis.predicted_observations
], ensure_ascii=False, indent=2)}

ASSIGNED DISCOVERY AXIS (INSPIRATION ONLY)
ID: {axis.axis_id}
Label: {axis.label}
Subject: {axis.proposed_subject}
Relation: {axis.proposed_relation}
Object: {axis.proposed_object}

TARGET PREMISE — THIS IS THE ONLY PREMISE YOU MAY EVALUATE
ID: {premise.statement_id}
Epistemic role: {premise.epistemic_role}
Claim kind: {premise.claim_kind}
Papers: {", ".join(premise.paper_ids) or "-"}
Text:
{premise.text}

Return the structured role review. Do not speculate about what other premises
might contain; they are intentionally hidden from you.
"""
    return _make_prompt(
        ROLE_PROMPT_VERSION,
        system,
        user,
    )


def build_ablation_prompt(
    *,
    hypothesis: HypothesisCard,
    axis: DiscoveryAxis,
    omitted_premise_statement_id: str,
    remaining_premises: list[HypothesisEvidenceStatement],
) -> CriticPrompt:
    system = """You are a blinded leave-one-premise-out grounding critic.

One selected premise has been withheld. You are NOT given its text. Judge only
whether the REMAINING supplied premises are sufficient to ground the accepted
hypothesis as written.

Do not use outside knowledge. Do not reconstruct or guess the omitted premise.
Do not treat the discovery axis as scientific evidence. Distinguish:
- sufficiently_grounded: the remaining premises still provide a coherent
  grounding chain for the material hypothesis claim and inferential bridge.
- partially_grounded: important parts remain grounded but at least one material
  clause or link becomes weak/unsupported.
- insufficiently_grounded: a material claim or inferential link no longer has
  adequate support in the remaining premises.
- scope_conflicted: the remaining evidence requires a material scope transfer or
  conflicts with the claim.
- uncertain: the supplied texts do not permit a reliable judgment.

If listing unsupported_or_weak_hypothesis_clauses, copy short contiguous
fragments VERBATIM from hypothesis_statement. Use an empty list if none.
"""
    remaining_payload = [
        {
            "statement_id": row.statement_id,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "text": row.text,
        }
        for row in remaining_premises
    ]
    user = f"""HYPOTHESIS
ID: {hypothesis.hypothesis_id}
Title: {hypothesis.title}
Hypothesis statement:
{hypothesis.hypothesis_statement}

Inferential bridge:
{hypothesis.inferential_bridge}

Predicted observations:
{json.dumps([
    {
        "observable": row.observable,
        "expected_direction": row.expected_direction,
        "rationale": row.rationale,
    }
    for row in hypothesis.predicted_observations
], ensure_ascii=False, indent=2)}

ASSIGNED DISCOVERY AXIS (INSPIRATION ONLY)
ID: {axis.axis_id}
Label: {axis.label}

ABLATION
Omitted premise ID: {omitted_premise_statement_id}
The omitted premise TEXT is intentionally hidden.

REMAINING PREMISES
{json.dumps(remaining_payload, ensure_ascii=False, indent=2)}

Judge only the grounding capability of this remaining set.
"""
    return _make_prompt(
        ABLATION_PROMPT_VERSION,
        system,
        user,
    )


def _audit_quote(
    text: str | None,
    *,
    hypothesis_statement: str,
    inferential_bridge: str,
    expected_field: Literal[
        "hypothesis_statement",
        "inferential_bridge",
    ],
) -> QuotedClauseAudit | None:
    if text is None:
        return None

    if expected_field == "hypothesis_statement":
        valid = text in hypothesis_statement
    else:
        valid = text in inferential_bridge

    return QuotedClauseAudit(
        text=text,
        exact_substring_match=valid,
        source_field=(
            expected_field
            if valid
            else "unknown"
        ),
    )


def _sanitize_role_review(
    draft: PremiseRoleReviewDraft,
    hypothesis: HypothesisCard,
) -> PremiseRoleReview:
    return PremiseRoleReview(
        hypothesis_id=draft.hypothesis_id,
        premise_statement_id=draft.premise_statement_id,
        role=draft.role,
        supported_hypothesis_clause=draft.supported_hypothesis_clause,
        supported_hypothesis_clause_audit=_audit_quote(
            draft.supported_hypothesis_clause,
            hypothesis_statement=hypothesis.hypothesis_statement,
            inferential_bridge=hypothesis.inferential_bridge,
            expected_field="hypothesis_statement",
        ),
        supported_bridge_clause=draft.supported_bridge_clause,
        supported_bridge_clause_audit=_audit_quote(
            draft.supported_bridge_clause,
            hypothesis_statement=hypothesis.hypothesis_statement,
            inferential_bridge=hypothesis.inferential_bridge,
            expected_field="inferential_bridge",
        ),
        support_explanation=draft.support_explanation,
        limiting_or_scope_caveat=draft.limiting_or_scope_caveat,
        topical_overlap_without_support=(
            draft.topical_overlap_without_support
        ),
        confidence=draft.confidence,
    )


def _sanitize_ablation_review(
    draft: PremiseAblationReviewDraft,
    hypothesis: HypothesisCard,
) -> PremiseAblationReview:
    audits = [
        QuotedClauseAudit(
            text=text,
            exact_substring_match=(
                text in hypothesis.hypothesis_statement
            ),
            source_field=(
                "hypothesis_statement"
                if text in hypothesis.hypothesis_statement
                else "unknown"
            ),
        )
        for text in draft.unsupported_or_weak_hypothesis_clauses
    ]
    return PremiseAblationReview(
        hypothesis_id=draft.hypothesis_id,
        omitted_premise_statement_id=(
            draft.omitted_premise_statement_id
        ),
        remaining_grounding_status=(
            draft.remaining_grounding_status
        ),
        inferential_bridge_grounded=(
            draft.inferential_bridge_grounded
        ),
        unsupported_or_weak_hypothesis_clauses=list(
            draft.unsupported_or_weak_hypothesis_clauses
        ),
        unsupported_clause_audits=audits,
        critical_missing_link=draft.critical_missing_link,
        remaining_support_summary=draft.remaining_support_summary,
        confidence=draft.confidence,
    )


def _combine_verdict(
    role: PremiseRoleReview,
    ablation: PremiseAblationReview,
) -> tuple[NecessityVerdict, str]:
    if (
        role.confidence == "low"
        or ablation.confidence == "low"
        or ablation.remaining_grounding_status == "uncertain"
    ):
        return (
            "uncertain",
            "At least one critic judgment is low-confidence or uncertain.",
        )

    if role.role in {
        "scope_mismatch",
        "counterevidence_or_limitation",
    }:
        return (
            "scope_problem_or_counterevidence",
            "The isolated premise is judged to create a scope problem or "
            "explicitly limit the asserted relationship.",
        )

    if (
        role.role
        in {
            "direct_clause_support",
            "inferential_bridge_support",
        }
        and ablation.remaining_grounding_status
        in {
            "insufficiently_grounded",
            "scope_conflicted",
        }
    ):
        return (
            "critical_for_current_grounded_chain",
            "The premise materially supports the claim/bridge, and the "
            "remaining evidence set is not sufficient after ablation.",
        )

    if (
        ablation.remaining_grounding_status
        == "partially_grounded"
    ):
        return (
            "material_support_loss",
            "Removing the premise leaves only partial grounding for the "
            "accepted hypothesis.",
        )

    if (
        role.role
        in {
            "direct_clause_support",
            "inferential_bridge_support",
        }
        and ablation.remaining_grounding_status
        == "sufficiently_grounded"
    ):
        return (
            "redundant_or_replaceable_for_current_chain",
            "The premise provides material support, but the remaining selected "
            "premises still ground the claim as written.",
        )

    if (
        role.role
        in {
            "contextual_support",
            "tangential_support",
        }
        and ablation.remaining_grounding_status
        == "sufficiently_grounded"
    ):
        return (
            "contextual_or_nonessential_for_current_chain",
            "The isolated role is contextual/tangential and the ablated "
            "evidence set remains sufficiently grounded.",
        )

    return (
        "uncertain",
        "The isolated role and ablation result do not support a stronger "
        "necessity classification.",
    )


def _ps1_snapshot(
    ps1: PremiseNecessityDiagnosticReport | None,
    *,
    hypothesis_id: str,
    premise_statement_id: str,
) -> PS1Snapshot | None:
    if ps1 is None:
        return None

    for card in ps1.cards:
        if card.hypothesis_id != hypothesis_id:
            continue
        for row in card.selected_premises:
            if row.statement_id != premise_statement_id:
                continue
            return PS1Snapshot(
                core_score=row.scores.core_score,
                core_rank=row.scores.core_rank,
                axis_score=row.scores.axis_score,
                axis_rank=row.scores.axis_rank,
                prediction_score=row.scores.prediction_score,
                prediction_rank=row.scores.prediction_rank,
                diagnostic_flags=list(row.diagnostic_flags),
            )
    return None


class PremiseRoleNecessityRuntime:
    def __init__(
        self,
        backend: InstructorPremiseCriticBackend,
    ) -> None:
        self.backend = backend

    def run(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        axis_plan: DiscoveryAxisPlan,
        axis_report: DiscoveryAxisSynthesisReport,
        *,
        ps1_report: PremiseNecessityDiagnosticReport | None = None,
        hypothesis_ids: set[str] | None = None,
        premise_ids: set[str] | None = None,
    ) -> tuple[
        PremiseRoleNecessityReport,
        list[tuple[str, str, CriticPrompt]],
    ]:
        if portfolio.source_context_id != context.context_id:
            raise ValueError("PS2 portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError("PS2 portfolio/context SHA mismatch")
        if axis_report.final_portfolio_id != portfolio.portfolio_id:
            raise ValueError("PS2 axis report/portfolio ID mismatch")
        if axis_report.axis_plan_id != axis_plan.plan_id:
            raise ValueError("PS2 axis report/plan ID mismatch")

        if ps1_report is not None:
            if ps1_report.source_context_id != context.context_id:
                raise ValueError("PS2 PS1/context ID mismatch")
            if ps1_report.source_context_sha256 != context.context_sha256:
                raise ValueError("PS2 PS1/context SHA mismatch")
            if ps1_report.source_portfolio_id != portfolio.portfolio_id:
                raise ValueError("PS2 PS1/portfolio ID mismatch")

        eligible_by_id = {
            row.statement_id: row
            for row in context.evidence_statements
            if row.eligible_as_premise
        }
        axis_by_id = {
            row.axis_id: row
            for row in axis_plan.axes
        }
        lineage_by_hypothesis = {
            row.hypothesis_id: row
            for row in axis_report.lineages
        }

        cards: list[PremiseRoleNecessityCard] = []
        prompts: list[tuple[str, str, CriticPrompt]] = []
        role_calls = 0
        ablation_calls = 0

        for hypothesis in portfolio.hypotheses:
            if (
                hypothesis_ids is not None
                and hypothesis.hypothesis_id not in hypothesis_ids
            ):
                continue

            lineage = lineage_by_hypothesis.get(
                hypothesis.hypothesis_id
            )
            if lineage is None:
                raise ValueError(
                    "PS2 missing discovery lineage for hypothesis "
                    f"{hypothesis.hypothesis_id}"
                )
            axis = axis_by_id.get(lineage.axis_id)
            if axis is None:
                raise ValueError(
                    f"PS2 missing axis {lineage.axis_id}"
                )

            selected = list(hypothesis.premise_statement_ids)
            for target_id in selected:
                if (
                    premise_ids is not None
                    and target_id not in premise_ids
                ):
                    continue

                target = eligible_by_id.get(target_id)
                if target is None:
                    raise ValueError(
                        "PS2 target premise is not eligible in context: "
                        f"{target_id}"
                    )

                remaining_ids = [
                    sid
                    for sid in selected
                    if sid != target_id
                ]
                remaining = []
                for sid in remaining_ids:
                    statement = eligible_by_id.get(sid)
                    if statement is None:
                        raise ValueError(
                            "PS2 remaining premise is not eligible: "
                            f"{sid}"
                        )
                    remaining.append(statement)

                role_prompt = build_role_prompt(
                    hypothesis=hypothesis,
                    axis=axis,
                    premise=target,
                )
                prompts.append(
                    (
                        hypothesis.hypothesis_id,
                        f"role__{target_id}",
                        role_prompt,
                    )
                )
                role_generation = self.backend.call(
                    role_prompt,
                    PremiseRoleReviewDraft,
                )
                role_calls += 1
                role_draft = role_generation.value
                assert isinstance(
                    role_draft,
                    PremiseRoleReviewDraft,
                )
                if role_draft.hypothesis_id != hypothesis.hypothesis_id:
                    raise ValueError(
                        "PS2 role critic returned wrong hypothesis ID"
                    )
                if role_draft.premise_statement_id != target_id:
                    raise ValueError(
                        "PS2 role critic returned wrong premise ID"
                    )
                role_review = _sanitize_role_review(
                    role_draft,
                    hypothesis,
                )

                ablation_prompt = build_ablation_prompt(
                    hypothesis=hypothesis,
                    axis=axis,
                    omitted_premise_statement_id=target_id,
                    remaining_premises=remaining,
                )
                prompts.append(
                    (
                        hypothesis.hypothesis_id,
                        f"ablation__{target_id}",
                        ablation_prompt,
                    )
                )
                ablation_generation = self.backend.call(
                    ablation_prompt,
                    PremiseAblationReviewDraft,
                )
                ablation_calls += 1
                ablation_draft = ablation_generation.value
                assert isinstance(
                    ablation_draft,
                    PremiseAblationReviewDraft,
                )
                if (
                    ablation_draft.hypothesis_id
                    != hypothesis.hypothesis_id
                ):
                    raise ValueError(
                        "PS2 ablation critic returned wrong hypothesis ID"
                    )
                if (
                    ablation_draft.omitted_premise_statement_id
                    != target_id
                ):
                    raise ValueError(
                        "PS2 ablation critic returned wrong omitted premise ID"
                    )
                ablation_review = _sanitize_ablation_review(
                    ablation_draft,
                    hypothesis,
                )

                verdict, reason = _combine_verdict(
                    role_review,
                    ablation_review,
                )

                cards.append(
                    PremiseRoleNecessityCard(
                        hypothesis_id=hypothesis.hypothesis_id,
                        title=hypothesis.title,
                        axis_id=axis.axis_id,
                        axis_label=axis.label,
                        premise_statement_id=target.statement_id,
                        premise_text=target.text,
                        premise_claim_kind=target.claim_kind,
                        premise_epistemic_role=target.epistemic_role,
                        premise_paper_ids=list(target.paper_ids),
                        remaining_premise_statement_ids=remaining_ids,
                        role_review=role_review,
                        ablation_review=ablation_review,
                        necessity_verdict=verdict,
                        verdict_reason=reason,
                        ps1_snapshot=_ps1_snapshot(
                            ps1_report,
                            hypothesis_id=hypothesis.hypothesis_id,
                            premise_statement_id=target_id,
                        ),
                        role_prompt_sha256=role_prompt.prompt_sha256,
                        ablation_prompt_sha256=(
                            ablation_prompt.prompt_sha256
                        ),
                    )
                )

        selected_hypothesis_ids = sorted(
            {
                row.hypothesis_id
                for row in cards
            }
        )
        hypothesis_by_id = {
            row.hypothesis_id: row
            for row in portfolio.hypotheses
        }

        summaries: list[HypothesisPS2Summary] = []
        for hypothesis_id in selected_hypothesis_ids:
            hypothesis = hypothesis_by_id[hypothesis_id]
            lineage = lineage_by_hypothesis[hypothesis_id]
            axis = axis_by_id[lineage.axis_id]
            rows = [
                row
                for row in cards
                if row.hypothesis_id == hypothesis_id
            ]

            def ids(verdict: str) -> list[str]:
                return [
                    row.premise_statement_id
                    for row in rows
                    if row.necessity_verdict == verdict
                ]

            summaries.append(
                HypothesisPS2Summary(
                    hypothesis_id=hypothesis_id,
                    title=hypothesis.title,
                    axis_id=axis.axis_id,
                    axis_label=axis.label,
                    premise_count=len(rows),
                    critical_premise_statement_ids=ids(
                        "critical_for_current_grounded_chain"
                    ),
                    material_premise_statement_ids=ids(
                        "material_support_loss"
                    ),
                    replaceable_premise_statement_ids=ids(
                        "redundant_or_replaceable_for_current_chain"
                    ),
                    contextual_or_nonessential_statement_ids=ids(
                        "contextual_or_nonessential_for_current_chain"
                    ),
                    scope_problem_statement_ids=ids(
                        "scope_problem_or_counterevidence"
                    ),
                    uncertain_statement_ids=ids("uncertain"),
                )
            )

        verdict_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        ablation_counts: dict[str, int] = {}
        invalid_quotes = 0

        for row in cards:
            verdict_counts[row.necessity_verdict] = (
                verdict_counts.get(row.necessity_verdict, 0) + 1
            )
            role = row.role_review.role
            role_counts[role] = role_counts.get(role, 0) + 1
            status = row.ablation_review.remaining_grounding_status
            ablation_counts[status] = (
                ablation_counts.get(status, 0) + 1
            )

            audits = [
                row.role_review.supported_hypothesis_clause_audit,
                row.role_review.supported_bridge_clause_audit,
                *row.ablation_review.unsupported_clause_audits,
            ]
            invalid_quotes += sum(
                audit is not None
                and not audit.exact_substring_match
                for audit in audits
            )

        payload = {
            "schema_version": "premise-role-necessity-report-v1",
            "report_id": _stable_id(
                "premise_role_necessity_report",
                context.context_sha256,
                portfolio.portfolio_id,
                axis_plan.plan_id,
                axis_report.report_id,
                (
                    ps1_report.report_id
                    if ps1_report is not None
                    else "-"
                ),
                self.backend.model_name,
            ),
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_axis_plan_id": axis_plan.plan_id,
            "source_axis_report_id": axis_report.report_id,
            "source_ps1_report_id": (
                ps1_report.report_id
                if ps1_report is not None
                else None
            ),
            "domain_profile_id": context.domain_profile_id,
            "corpus_id": context.corpus_id,
            "critic_model": self.backend.model_name,
            "role_prompt_version": ROLE_PROMPT_VERSION,
            "ablation_prompt_version": ABLATION_PROMPT_VERSION,
            "hypothesis_count": len(selected_hypothesis_ids),
            "selected_premise_incidence_count": len(cards),
            "role_call_count": role_calls,
            "ablation_call_count": ablation_calls,
            "necessity_verdict_counts": verdict_counts,
            "role_counts": role_counts,
            "ablation_status_counts": ablation_counts,
            "invalid_quoted_clause_count": invalid_quotes,
            "cards": [
                row.model_dump(mode="json")
                for row in cards
            ],
            "hypothesis_summaries": [
                row.model_dump(mode="json")
                for row in summaries
            ],
            "policy": PremiseRoleNecessityPolicy().model_dump(
                mode="json"
            ),
        }

        report = PremiseRoleNecessityReport(
            **payload,
            report_sha256=_sha256_json(payload),
        )
        return report, prompts
