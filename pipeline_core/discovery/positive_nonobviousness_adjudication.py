from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.positive_nonobviousness_basis import (
    ClaimPositiveNonObviousnessBasisRecord,
    HypothesisPositiveNonObviousnessBasisSummary,
    PositiveNonObviousnessBasisEvidence,
    PositiveNonObviousnessBasisReport,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PositiveBasisDisposition = Literal[
    "PRODUCTIVE_EXPECTATION_TENSION",
    "ROUTINE_OR_EXPECTED_EXTENSION",
    "FATAL_OR_DIRECT_CONTRADICTION",
    "CONTEXT_MISMATCH_OR_NONDIAGNOSTIC",
    "INSUFFICIENT_BASIS",
]

CompiledPositiveNonObviousnessClaimState = Literal[
    "POSITIVE_NONOBVIOUSNESS_SUPPORTED",
    "POSITIVE_BASIS_NOT_ESTABLISHED",
    "FATAL_CONTRADICTION_FOUND",
    "INSUFFICIENT_FOR_ADJUDICATION",
]

PositiveNonObviousnessGateState = Literal[
    "POSITIVE_NONOBVIOUSNESS_AUTHORIZED",
    "NO_POSITIVE_BASIS_UNRESOLVED",
    "NONQUALIFYING_TENSION_ONLY",
    "EPISTEMICALLY_PARTIAL_UNRESOLVED",
    "DIRECT_PRIOR_ART_BLOCKED",
    "FATAL_CONTRADICTION_BLOCKED",
    "POSITIVE_BASIS_NOT_ESTABLISHED",
    "UNRESOLVED_AFTER_ADJUDICATION",
]


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


class PositiveNonObviousnessClaimDraft(StrictModel):
    claim_id: str
    disposition: PositiveBasisDisposition
    confidence: float = Field(ge=0.0, le=1.0)

    supporting_basis_work_ids: list[str] = Field(
        default_factory=list
    )
    fatal_basis_work_ids: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_work_lists(
        self,
    ) -> "PositiveNonObviousnessClaimDraft":
        if len(self.supporting_basis_work_ids) != len(
            set(self.supporting_basis_work_ids)
        ):
            raise ValueError(
                "supporting_basis_work_ids must be unique"
            )
        if len(self.fatal_basis_work_ids) != len(
            set(self.fatal_basis_work_ids)
        ):
            raise ValueError(
                "fatal_basis_work_ids must be unique"
            )

        if (
            self.disposition
            == "PRODUCTIVE_EXPECTATION_TENSION"
            and not self.supporting_basis_work_ids
        ):
            raise ValueError(
                "productive tension requires supporting basis work"
            )
        if (
            self.disposition
            == "FATAL_OR_DIRECT_CONTRADICTION"
            and not self.fatal_basis_work_ids
        ):
            raise ValueError(
                "fatal contradiction requires fatal basis work"
            )
        return self


class PositiveNonObviousnessHypothesisDraft(StrictModel):
    hypothesis_id: str
    claim_assessments: list[
        PositiveNonObviousnessClaimDraft
    ] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_claim_ids(
        self,
    ) -> "PositiveNonObviousnessHypothesisDraft":
        claim_ids = [
            row.claim_id
            for row in self.claim_assessments
        ]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                "positive non-obviousness draft contains duplicate claim IDs"
            )
        return self


class CompiledPositiveNonObviousnessClaimReview(StrictModel):
    hypothesis_id: str
    claim_id: str

    original_disposition: PositiveBasisDisposition
    compiled_disposition: PositiveBasisDisposition
    compiled_state: CompiledPositiveNonObviousnessClaimState
    confidence: float = Field(ge=0.0, le=1.0)

    valid_supporting_basis_work_ids: list[str]
    valid_fatal_basis_work_ids: list[str]

    rationale: str
    deterministic_reason_codes: list[str] = Field(
        default_factory=list
    )

    positive_nonobviousness_authority: bool
    fatal_contradiction_authority: bool

    novelty_authority: Literal[False] = False
    truth_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_authority(
        self,
    ) -> "CompiledPositiveNonObviousnessClaimReview":
        if self.positive_nonobviousness_authority != (
            self.compiled_state
            == "POSITIVE_NONOBVIOUSNESS_SUPPORTED"
        ):
            raise ValueError(
                "claim positive non-obviousness authority mismatch"
            )
        if self.fatal_contradiction_authority != (
            self.compiled_state
            == "FATAL_CONTRADICTION_FOUND"
        ):
            raise ValueError(
                "claim fatal contradiction authority mismatch"
            )
        return self


class PositiveNonObviousnessHypothesisGateDecision(StrictModel):
    hypothesis_id: str

    basis_readiness: str
    gate_state: PositiveNonObviousnessGateState

    eligible_claim_ids: list[str]
    reviewed_claim_ids: list[str]
    positive_support_claim_ids: list[str]
    fatal_contradiction_claim_ids: list[str]

    direct_prior_art_blocker_claim_ids: list[str]
    unclassified_presented_work_remains: bool

    llm_call_performed: bool
    deterministic_skip_reason: str | None = None

    positive_nonobviousness_authority: bool
    fatal_contradiction_blocker: bool

    # This gate establishes at most a bounded positive non-obviousness
    # authority. It never certifies novelty or the scientific truth of the
    # proposed hypothesis.
    novelty_authority: Literal[False] = False
    truth_authority: Literal[False] = False
    certification_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_gate(
        self,
    ) -> "PositiveNonObviousnessHypothesisGateDecision":
        if self.positive_nonobviousness_authority != (
            self.gate_state
            == "POSITIVE_NONOBVIOUSNESS_AUTHORIZED"
        ):
            raise ValueError(
                "hypothesis positive non-obviousness authority mismatch"
            )
        if self.fatal_contradiction_blocker != (
            self.gate_state
            == "FATAL_CONTRADICTION_BLOCKED"
        ):
            raise ValueError(
                "hypothesis fatal contradiction blocker mismatch"
            )
        if (
            not self.llm_call_performed
            and self.deterministic_skip_reason is None
        ):
            raise ValueError(
                "skipped hypothesis requires deterministic reason"
            )
        return self


class PositiveNonObviousnessAdjudicationReport(StrictModel):
    schema_version: Literal[
        "positive-nonobviousness-adjudication-report-v1"
    ] = "positive-nonobviousness-adjudication-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_claim_evidence_graph_id: str
    source_positive_nonobviousness_basis_report_id: str

    backend_name: str
    model_name: str

    claim_reviews: list[
        CompiledPositiveNonObviousnessClaimReview
    ]
    hypothesis_decisions: list[
        PositiveNonObviousnessHypothesisGateDecision
    ]

    hypothesis_count: int = Field(ge=0)
    reviewed_hypothesis_count: int = Field(ge=0)
    skipped_hypothesis_count: int = Field(ge=0)
    llm_calls_performed: int = Field(ge=0)

    gate_state_counts: dict[str, int]
    positive_nonobviousness_authorized_count: int = Field(ge=0)
    fatal_contradiction_blocked_count: int = Field(ge=0)

    adjudication_semantics: Literal[
        "productive_tension_vs_fatal_contradiction_fail_closed_v1"
    ] = "productive_tension_vs_fatal_contradiction_fail_closed_v1"

    absence_based_nonobviousness_forbidden: Literal[True] = True
    basis_candidate_is_not_automatically_supportive: Literal[True] = True
    counterevidence_can_be_fatal_not_supportive: Literal[True] = True
    partial_coverage_blocks_authority: Literal[True] = True
    direct_prior_art_blocks_authority: Literal[True] = True

    novelty_verdict_created: Literal[False] = False
    certification_performed: Literal[False] = False
    production_authority_created: Literal[False] = False
    diagnostic_only: Literal[True] = True
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "PositiveNonObviousnessAdjudicationReport":
        if self.hypothesis_count != len(
            self.hypothesis_decisions
        ):
            raise ValueError(
                "positive non-obviousness hypothesis count mismatch"
            )
        if self.reviewed_hypothesis_count != sum(
            row.llm_call_performed
            for row in self.hypothesis_decisions
        ):
            raise ValueError(
                "reviewed hypothesis count mismatch"
            )
        if self.skipped_hypothesis_count != (
            self.hypothesis_count
            - self.reviewed_hypothesis_count
        ):
            raise ValueError(
                "skipped hypothesis count mismatch"
            )
        if self.llm_calls_performed != self.reviewed_hypothesis_count:
            raise ValueError(
                "positive non-obviousness LLM call count mismatch"
            )
        if self.positive_nonobviousness_authorized_count != sum(
            row.positive_nonobviousness_authority
            for row in self.hypothesis_decisions
        ):
            raise ValueError(
                "positive non-obviousness authority count mismatch"
            )
        if self.fatal_contradiction_blocked_count != sum(
            row.fatal_contradiction_blocker
            for row in self.hypothesis_decisions
        ):
            raise ValueError(
                "fatal contradiction blocker count mismatch"
            )

        expected_counts: dict[str, int] = defaultdict(int)
        for row in self.hypothesis_decisions:
            expected_counts[row.gate_state] += 1
        if dict(sorted(expected_counts.items())) != dict(
            sorted(self.gate_state_counts.items())
        ):
            raise ValueError(
                "positive non-obviousness gate state counts mismatch"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError(
                "positive non-obviousness adjudication SHA mismatch"
            )
        if observed_id != (
            "positive_nonobviousness_adjudication:"
            + expected_sha[:20]
        ):
            raise ValueError(
                "positive non-obviousness adjudication ID mismatch"
            )
        return self


class PositiveNonObviousnessBackend(Protocol):
    backend_name: str
    model_name: str

    def review(
        self,
        *,
        hypothesis_id: str,
        claim_records: list[
            ClaimPositiveNonObviousnessBasisRecord
        ],
        claim_text_by_id: dict[str, str],
    ) -> PositiveNonObviousnessHypothesisDraft: ...


_ADJUDICATION_SYSTEM = """You adjudicate POSITIVE NON-OBVIOUSNESS BASIS for a bounded scientific hypothesis.

You do NOT judge novelty, publication priority, truth, or literature-wide absence.

You receive only claims that already have exact-span adjudicated tension/conflict evidence. Decide whether that supplied tension is:

1. PRODUCTIVE_EXPECTATION_TENSION
   The cited prior evidence establishes a scientifically relevant expectation, decoupling, or conflict such that the proposed relation would not be a routine/straightforward extension. The tension must be relevant to the actual claim and not simply a different field or unrelated scope.

2. ROUTINE_OR_EXPECTED_EXTENSION
   The cited evidence is relevant but the proposed relation still appears to be a routine contextualization, parameter change, measurement substitution, or expected extension. Do not call it productive merely because terminology differs.

3. FATAL_OR_DIRECT_CONTRADICTION
   The cited evidence materially contradicts the proposed relation in substantially overlapping scope and the proposed claim contains no explicit condition/modifier that resolves the contradiction. This is a blocker, not positive support.

4. CONTEXT_MISMATCH_OR_NONDIAGNOSTIC
   The cited tension is from materially different scope/system or does not diagnose whether the proposed relation is non-routine.

5. INSUFFICIENT_BASIS
   The supplied exact spans are insufficient to make one of the judgments above.

Hard rules:
- Use ONLY supplied basis evidence. Never infer from retrieval absence or uncited literature.
- A DIRECTIONAL_COUNTEREVIDENCE or CONFLICTING_PRIOR_ART label is not automatically productive; it may be fatal.
- Do not reward surprising wording, rarity, complexity, or the number of qualifiers.
- Do not claim the proposed hypothesis is true or novel.
- For PRODUCTIVE_EXPECTATION_TENSION, list at least one supplied basis work ID in supporting_basis_work_ids.
- For FATAL_OR_DIRECT_CONTRADICTION, list at least one supplied basis work ID in fatal_basis_work_ids.
- Copy claim IDs and work IDs exactly.
"""


def _claim_texts(
    graph: ScientificClaimEvidenceGraphReport,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_kind != "CLAIM":
            continue
        for claim_id in node.claim_ids:
            output[claim_id] = node.label
    return output


def _prompt_for_hypothesis(
    *,
    hypothesis_id: str,
    claim_records: list[
        ClaimPositiveNonObviousnessBasisRecord
    ],
    claim_text_by_id: dict[str, str],
) -> str:
    lines = [
        "HYPOTHESIS",
        "==========",
        hypothesis_id,
        "",
        "QUALIFYING CLAIM BASIS",
        "======================",
    ]

    for row in claim_records:
        lines.extend(
            [
                f"claim_id: {row.claim_id}",
                f"claim_text: {claim_text_by_id.get(row.claim_id, '')}",
                f"role: {row.novelty_selection_role}",
                f"role_relevance: {row.role_relevance}",
                f"structural_centrality: {row.structural_centrality_score}",
                f"classification_fraction: {row.classification_fraction}",
                "basis_evidence:",
            ]
        )
        for evidence in row.basis_evidence:
            lines.extend(
                [
                    f"  work_id: {evidence.work_id}",
                    f"  relationship: {evidence.relationship}",
                    f"  basis_kind: {evidence.basis_kind}",
                    f"  evidence_span: {evidence.evidence_span}",
                    f"  prior_rationale: {evidence.rationale}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Return one claim_assessment for every supplied claim.",
            "Do not assess any claim not supplied above.",
            "Copy the HYPOTHESIS ID exactly into the top-level "
            "hypothesis_id field.",
        ]
    )
    return "\n".join(lines)


class InstructorPositiveNonObviousnessBackend:
    backend_name = (
        "instructor_positive_nonobviousness_adjudication"
    )

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        capture_prompts: bool = False,
    ) -> None:
        self.model_name = model
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.capture_prompts = bool(capture_prompts)
        self.prompt_records: list[dict[str, str]] = []
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Positive non-obviousness adjudication requires "
                "openai + instructor."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=instructor.Mode.JSON,
        )
        return self._client

    def review(
        self,
        *,
        hypothesis_id: str,
        claim_records: list[
            ClaimPositiveNonObviousnessBasisRecord
        ],
        claim_text_by_id: dict[str, str],
    ) -> PositiveNonObviousnessHypothesisDraft:
        user = _prompt_for_hypothesis(
            hypothesis_id=hypothesis_id,
            claim_records=claim_records,
            claim_text_by_id=claim_text_by_id,
        )

        if self.capture_prompts:
            self.prompt_records.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "system_prompt": _ADJUDICATION_SYSTEM,
                    "user_prompt": user,
                }
            )

        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=PositiveNonObviousnessHypothesisDraft,
            messages=[
                {
                    "role": "system",
                    "content": _ADJUDICATION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=0.0,
            max_retries=self.parse_retries,
            telemetry_context={
                "pipeline":
                    "positive_nonobviousness_adjudication_shadow",
                "stage":
                    "positive_nonobviousness_adjudication",
                "call_kind":
                    "basis_interpretation",
                "hypothesis_id":
                    hypothesis_id,
            },
        )

        if not isinstance(
            result,
            PositiveNonObviousnessHypothesisDraft,
        ):
            result = (
                PositiveNonObviousnessHypothesisDraft.model_validate(
                    result
                )
            )
        return result


def compile_positive_nonobviousness_claim_review(
    *,
    basis_record: ClaimPositiveNonObviousnessBasisRecord,
    draft: PositiveNonObviousnessClaimDraft | None,
) -> CompiledPositiveNonObviousnessClaimReview:
    allowed = set(basis_record.basis_work_ids)
    reasons: list[str] = []

    if draft is None:
        return CompiledPositiveNonObviousnessClaimReview(
            hypothesis_id=basis_record.hypothesis_id,
            claim_id=basis_record.claim_id,
            original_disposition="INSUFFICIENT_BASIS",
            compiled_disposition="INSUFFICIENT_BASIS",
            compiled_state="INSUFFICIENT_FOR_ADJUDICATION",
            confidence=0.0,
            valid_supporting_basis_work_ids=[],
            valid_fatal_basis_work_ids=[],
            rationale=(
                "No model assessment was returned for this eligible claim."
            ),
            deterministic_reason_codes=[
                "eligible_claim_missing_from_model_output"
            ],
            positive_nonobviousness_authority=False,
            fatal_contradiction_authority=False,
        )

    if draft.claim_id != basis_record.claim_id:
        raise ValueError(
            "draft claim ID does not match basis record"
        )

    valid_support = sorted(
        set(draft.supporting_basis_work_ids) & allowed
    )
    valid_fatal = sorted(
        set(draft.fatal_basis_work_ids) & allowed
    )

    invalid_support = sorted(
        set(draft.supporting_basis_work_ids) - allowed
    )
    invalid_fatal = sorted(
        set(draft.fatal_basis_work_ids) - allowed
    )
    if invalid_support:
        reasons.append(
            "unsupported_supporting_basis_work_ids_dropped"
        )
    if invalid_fatal:
        reasons.append(
            "unsupported_fatal_basis_work_ids_dropped"
        )

    compiled = draft.disposition

    if (
        compiled == "PRODUCTIVE_EXPECTATION_TENSION"
        and not valid_support
    ):
        compiled = "INSUFFICIENT_BASIS"
        reasons.append(
            "productive_tension_requires_valid_basis_work"
        )
    if (
        compiled == "FATAL_OR_DIRECT_CONTRADICTION"
        and not valid_fatal
    ):
        compiled = "INSUFFICIENT_BASIS"
        reasons.append(
            "fatal_contradiction_requires_valid_basis_work"
        )

    if compiled == "PRODUCTIVE_EXPECTATION_TENSION":
        state = "POSITIVE_NONOBVIOUSNESS_SUPPORTED"
    elif compiled == "FATAL_OR_DIRECT_CONTRADICTION":
        state = "FATAL_CONTRADICTION_FOUND"
    elif compiled == "INSUFFICIENT_BASIS":
        state = "INSUFFICIENT_FOR_ADJUDICATION"
    else:
        state = "POSITIVE_BASIS_NOT_ESTABLISHED"

    return CompiledPositiveNonObviousnessClaimReview(
        hypothesis_id=basis_record.hypothesis_id,
        claim_id=basis_record.claim_id,
        original_disposition=draft.disposition,
        compiled_disposition=compiled,
        compiled_state=state,
        confidence=draft.confidence,
        valid_supporting_basis_work_ids=valid_support,
        valid_fatal_basis_work_ids=valid_fatal,
        rationale=draft.rationale,
        deterministic_reason_codes=reasons,
        positive_nonobviousness_authority=(
            state == "POSITIVE_NONOBVIOUSNESS_SUPPORTED"
        ),
        fatal_contradiction_authority=(
            state == "FATAL_CONTRADICTION_FOUND"
        ),
    )


def _deterministic_skip_gate(
    summary: HypothesisPositiveNonObviousnessBasisSummary,
) -> PositiveNonObviousnessGateState | None:
    if summary.direct_prior_art_blocker_claim_ids:
        return "DIRECT_PRIOR_ART_BLOCKED"

    if (
        summary.positive_basis_state
        == "NO_POSITIVE_BASIS_CANDIDATE"
    ):
        return "NO_POSITIVE_BASIS_UNRESOLVED"

    if (
        summary.positive_basis_state
        == "NONQUALIFYING_TENSION_ONLY"
    ):
        return "NONQUALIFYING_TENSION_ONLY"

    if not summary.all_claims_fully_classified:
        return "EPISTEMICALLY_PARTIAL_UNRESOLVED"

    if (
        summary.future_adjudication_readiness
        != "READY_FOR_POSITIVE_NONOBVIOUSNESS_ADJUDICATION"
    ):
        return "UNRESOLVED_AFTER_ADJUDICATION"

    return None


def _gate_from_reviews(
    reviews: list[
        CompiledPositiveNonObviousnessClaimReview
    ],
) -> PositiveNonObviousnessGateState:
    if any(
        row.fatal_contradiction_authority
        for row in reviews
    ):
        return "FATAL_CONTRADICTION_BLOCKED"

    if any(
        row.positive_nonobviousness_authority
        for row in reviews
    ):
        return "POSITIVE_NONOBVIOUSNESS_AUTHORIZED"

    if any(
        row.compiled_state
        == "POSITIVE_BASIS_NOT_ESTABLISHED"
        for row in reviews
    ):
        return "POSITIVE_BASIS_NOT_ESTABLISHED"

    return "UNRESOLVED_AFTER_ADJUDICATION"


def _scope_model_draft_to_call(
    *,
    draft: PositiveNonObviousnessHypothesisDraft,
    expected_hypothesis_id: str,
    eligible_claim_ids: list[str],
) -> tuple[
    dict[str, PositiveNonObviousnessClaimDraft],
    list[str],
]:
    """Bind untrusted model output to the deterministic call scope.

    The backend is invoked once for an already-selected hypothesis and only
    receives that hypothesis's qualifying claim records. Therefore the model's
    echoed hypothesis_id is audit metadata, not routing authority.

    Claim IDs remain fail-closed: unknown IDs are dropped and any missing
    eligible claim is later compiled as INSUFFICIENT_FOR_ADJUDICATION.
    """
    reason_codes: list[str] = []

    if draft.hypothesis_id != expected_hypothesis_id:
        reason_codes.append(
            "model_hypothesis_id_mismatch_bound_to_call_scope"
        )

    eligible = set(eligible_claim_ids)
    unknown_claim_ids = sorted(
        {
            row.claim_id
            for row in draft.claim_assessments
        }
        - eligible
    )
    if unknown_claim_ids:
        reason_codes.append(
            "model_unknown_claim_ids_dropped"
        )

    return (
        {
            row.claim_id: row
            for row in draft.claim_assessments
            if row.claim_id in eligible
        },
        reason_codes,
    )


def build_positive_nonobviousness_adjudication_report(
    *,
    graph: ScientificClaimEvidenceGraphReport,
    basis: PositiveNonObviousnessBasisReport,
    backend: PositiveNonObviousnessBackend,
) -> PositiveNonObviousnessAdjudicationReport:
    if (
        basis.source_claim_evidence_graph_id
        != graph.graph_id
    ):
        raise ValueError(
            "basis report does not derive from supplied evidence graph"
        )

    claim_text_by_id = _claim_texts(graph)
    basis_by_claim = {
        row.claim_id: row
        for row in basis.claim_records
    }

    claim_reviews: list[
        CompiledPositiveNonObviousnessClaimReview
    ] = []
    decisions: list[
        PositiveNonObviousnessHypothesisGateDecision
    ] = []
    llm_calls = 0

    for summary in basis.hypothesis_summaries:
        eligible_claim_ids = list(
            summary.qualifying_basis_claim_ids
        )
        skip_state = _deterministic_skip_gate(
            summary
        )

        if skip_state is not None:
            decisions.append(
                PositiveNonObviousnessHypothesisGateDecision(
                    hypothesis_id=summary.hypothesis_id,
                    basis_readiness=(
                        summary.future_adjudication_readiness
                    ),
                    gate_state=skip_state,
                    eligible_claim_ids=eligible_claim_ids,
                    reviewed_claim_ids=[],
                    positive_support_claim_ids=[],
                    fatal_contradiction_claim_ids=[],
                    direct_prior_art_blocker_claim_ids=list(
                        summary.direct_prior_art_blocker_claim_ids
                    ),
                    unclassified_presented_work_remains=(
                        summary.any_unclassified_presented_work
                    ),
                    llm_call_performed=False,
                    deterministic_skip_reason=skip_state,
                    positive_nonobviousness_authority=False,
                    fatal_contradiction_blocker=False,
                )
            )
            continue

        records = [
            basis_by_claim[claim_id]
            for claim_id in eligible_claim_ids
        ]
        draft = backend.review(
            hypothesis_id=summary.hypothesis_id,
            claim_records=records,
            claim_text_by_id=claim_text_by_id,
        )
        llm_calls += 1

        draft_by_claim, scope_reason_codes = (
            _scope_model_draft_to_call(
                draft=draft,
                expected_hypothesis_id=summary.hypothesis_id,
                eligible_claim_ids=eligible_claim_ids,
            )
        )

        compiled_for_hypothesis = [
            compile_positive_nonobviousness_claim_review(
                basis_record=record,
                draft=draft_by_claim.get(record.claim_id),
            )
            for record in records
        ]
        if scope_reason_codes:
            compiled_for_hypothesis = [
                row.model_copy(
                    update={
                        "deterministic_reason_codes": list(
                            dict.fromkeys(
                                [
                                    *row.deterministic_reason_codes,
                                    *scope_reason_codes,
                                ]
                            )
                        )
                    }
                )
                for row in compiled_for_hypothesis
            ]

        claim_reviews.extend(
            compiled_for_hypothesis
        )

        gate_state = _gate_from_reviews(
            compiled_for_hypothesis
        )
        decisions.append(
            PositiveNonObviousnessHypothesisGateDecision(
                hypothesis_id=summary.hypothesis_id,
                basis_readiness=(
                    summary.future_adjudication_readiness
                ),
                gate_state=gate_state,
                eligible_claim_ids=eligible_claim_ids,
                reviewed_claim_ids=[
                    row.claim_id
                    for row in compiled_for_hypothesis
                ],
                positive_support_claim_ids=[
                    row.claim_id
                    for row in compiled_for_hypothesis
                    if row.positive_nonobviousness_authority
                ],
                fatal_contradiction_claim_ids=[
                    row.claim_id
                    for row in compiled_for_hypothesis
                    if row.fatal_contradiction_authority
                ],
                direct_prior_art_blocker_claim_ids=[],
                unclassified_presented_work_remains=False,
                llm_call_performed=True,
                deterministic_skip_reason=None,
                positive_nonobviousness_authority=(
                    gate_state
                    == "POSITIVE_NONOBVIOUSNESS_AUTHORIZED"
                ),
                fatal_contradiction_blocker=(
                    gate_state
                    == "FATAL_CONTRADICTION_BLOCKED"
                ),
            )
        )

    decisions.sort(
        key=lambda row: row.hypothesis_id
    )
    claim_reviews.sort(
        key=lambda row: (
            row.hypothesis_id,
            row.claim_id,
        )
    )

    gate_counts: dict[str, int] = defaultdict(int)
    for row in decisions:
        gate_counts[row.gate_state] += 1

    body = {
        "schema_version":
            "positive-nonobviousness-adjudication-report-v1",
        "source_claim_evidence_graph_id":
            graph.graph_id,
        "source_positive_nonobviousness_basis_report_id":
            basis.report_id,
        "backend_name":
            backend.backend_name,
        "model_name":
            backend.model_name,
        "claim_reviews": [
            row.model_dump(mode="json")
            for row in claim_reviews
        ],
        "hypothesis_decisions": [
            row.model_dump(mode="json")
            for row in decisions
        ],
        "hypothesis_count":
            len(decisions),
        "reviewed_hypothesis_count":
            sum(
                row.llm_call_performed
                for row in decisions
            ),
        "skipped_hypothesis_count":
            sum(
                not row.llm_call_performed
                for row in decisions
            ),
        "llm_calls_performed":
            llm_calls,
        "gate_state_counts":
            dict(sorted(gate_counts.items())),
        "positive_nonobviousness_authorized_count":
            sum(
                row.positive_nonobviousness_authority
                for row in decisions
            ),
        "fatal_contradiction_blocked_count":
            sum(
                row.fatal_contradiction_blocker
                for row in decisions
            ),
        "adjudication_semantics":
            "productive_tension_vs_fatal_contradiction_fail_closed_v1",
        "absence_based_nonobviousness_forbidden":
            True,
        "basis_candidate_is_not_automatically_supportive":
            True,
        "counterevidence_can_be_fatal_not_supportive":
            True,
        "partial_coverage_blocks_authority":
            True,
        "direct_prior_art_blocks_authority":
            True,
        "novelty_verdict_created":
            False,
        "certification_performed":
            False,
        "production_authority_created":
            False,
        "diagnostic_only":
            True,
        "n9_contract_changed":
            False,
        "n10_contract_changed":
            False,
        "production_selection_changed":
            False,
    }
    digest = _sha256_json(body)

    return PositiveNonObviousnessAdjudicationReport(
        **body,
        report_id=(
            "positive_nonobviousness_adjudication:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "CompiledPositiveNonObviousnessClaimReview",
    "InstructorPositiveNonObviousnessBackend",
    "PositiveNonObviousnessAdjudicationReport",
    "PositiveNonObviousnessClaimDraft",
    "PositiveNonObviousnessHypothesisDraft",
    "PositiveNonObviousnessHypothesisGateDecision",
    "build_positive_nonobviousness_adjudication_report",
    "compile_positive_nonobviousness_claim_review",
]
