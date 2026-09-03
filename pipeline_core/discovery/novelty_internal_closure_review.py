from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.novelty_closure_execution import (
    ExecutableClosureTarget,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


InternalClosureRelationship = Literal[
    "ESTABLISHES_SLOT",
    "PARTIAL_SLOT_RELATION",
    "COMPONENT_ONLY",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class InternalClosureMatchDraft(StrictModel):
    statement_id: str
    relationship: InternalClosureRelationship
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        min_length=1,
    )


class InternalClosureReviewDraft(StrictModel):
    matches: list[
        InternalClosureMatchDraft
    ] = Field(
        default_factory=list
    )

    interpretation: str = Field(
        min_length=1,
    )


class InternalClosurePositiveReview(StrictModel):
    target_id: str
    slot: str
    source_claim_id: str

    considered_statement_ids: list[str] = Field(
        default_factory=list
    )

    positive_statement_ids: list[str] = Field(
        default_factory=list
    )

    positive_paper_ids: list[str] = Field(
        default_factory=list
    )

    positive_support_node_ids: list[str] = Field(
        default_factory=list
    )

    positive_support_edge_ids: list[str] = Field(
        default_factory=list
    )

    reason_codes: list[str] = Field(
        default_factory=list
    )

    reviewer_unknown_statement_ids: list[str] = Field(
        default_factory=list
    )

    interpretation: str


@dataclass(frozen=True)
class InternalClosurePromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


_INTERNAL_BASE_REVIEW_SYSTEM = """
You are reviewing source-grounded scientific premise statements
for ONE lower-order BASE_RELATION closure target.

This is a POSITIVE-ONLY evidence review.

You may establish a BASE_RELATION only when ONE supplied premise
statement itself explicitly states, reports, tests, compares, or
otherwise establishes the scientific relation required by the
BASE_RELATION target.

Do NOT combine separate statements into a new relation.
Do NOT infer a relation from co-mention.
Do NOT infer causality from association.
Do NOT generalize beyond the scope of the statement.
Do NOT use outside knowledge.
Do NOT infer literature-wide absence.

The generated hypothesis is NOT evidence.

RELATIONSHIP LABELS

ESTABLISHES_SLOT:
One supplied statement itself explicitly establishes the
lower-order BASE_RELATION.

PARTIAL_SLOT_RELATION:
The statement supports a substantial neighboring or incomplete
part of the BASE_RELATION but does not establish the full
lower-order relation.

COMPONENT_ONLY:
The statement reports relevant variables, materials, descriptors,
mechanisms, or separate effects, but does not establish the
BASE_RELATION.

UNRELATED:
The statement does not materially bear on the BASE_RELATION.

INSUFFICIENT_METADATA:
The statement is too ambiguous to classify.

IMPORTANT:
- Review each statement independently.
- Never combine two statements to produce ESTABLISHES_SLOT.
- Internal evidence can establish positive lower-order evidence.
- Internal evidence can NEVER establish that a relation is absent.
- Copy every statement_id exactly from ALLOWED_STATEMENT_IDS.
""".strip()


def _sha256(
    system: str,
    user: str,
) -> str:
    return hashlib.sha256(
        (
            system
            + "\n---\n"
            + user
        ).encode("utf-8")
    ).hexdigest()


def authoritative_premise_statements(
    *,
    hypothesis: HypothesisCard,
    context: HypothesisContext,
) -> list[HypothesisEvidenceStatement]:
    """Return only canonical, positive-authority hypothesis premises.

    This function does not perform semantic matching.

    A statement is eligible only when:
    - selected by the canonical hypothesis;
    - source-reported;
    - premise eligible;
    - not marked as requiring verification;
    - backed by paper provenance; and
    - backed by at least one scientific support node or edge.
    """

    if (
        hypothesis.source_context_id
        != context.context_id
    ):
        return []

    if (
        hypothesis.source_context_sha256
        != context.context_sha256
    ):
        return []

    by_id = {
        row.statement_id: row
        for row
        in context.evidence_statements
    }

    output: list[
        HypothesisEvidenceStatement
    ] = []

    seen: set[str] = set()

    for statement_id in (
        hypothesis.premise_statement_ids
    ):
        if statement_id in seen:
            continue

        seen.add(statement_id)

        row = by_id.get(
            statement_id
        )

        if row is None:
            continue

        if row.epistemic_role != "reported":
            continue

        if not row.eligible_as_premise:
            continue

        if row.requires_verification:
            continue

        if not row.paper_ids:
            continue

        if not (
            row.scientific_support_node_ids
            or row.scientific_support_edge_ids
        ):
            continue

        output.append(row)

    return output


def build_internal_base_review_prompt(
    *,
    target: ExecutableClosureTarget,
    statements: list[
        HypothesisEvidenceStatement
    ],
) -> str:
    lines = [
        "INTERNAL GROUNDING CLOSURE TARGET",
        "=================================",
        f"target_id: {target.target_id}",
        f"slot: {target.slot}",
        f"source_claim_id: {target.source_claim_id}",
        "",
        "BASE RELATION TARGET",
        "====================",
        target.source_text,
        "",
        "SEARCH TERMS",
        "============",
        " | ".join(
            target.search_terms
        ),
        "",
        "IMPORTANT",
        "=========",
        (
            "The target is a review target only. "
            "It is not evidence."
        ),
        (
            "Judge only whether ONE supplied canonical "
            "premise statement itself establishes "
            "the BASE_RELATION."
        ),
        "",
        "CANONICAL SELECTED PREMISES",
        "===========================",
    ]

    allowed: list[str] = []

    for index, row in enumerate(
        statements,
        start=1,
    ):
        allowed.append(
            row.statement_id
        )

        lines.extend(
            [
                (
                    f"[{index}] "
                    f"statement_id={row.statement_id}"
                ),
                (
                    "epistemic_role: "
                    f"{row.epistemic_role}"
                ),
                (
                    "claim_kind: "
                    f"{row.claim_kind}"
                ),
                (
                    "paper_ids: "
                    + " | ".join(
                        row.paper_ids
                    )
                ),
                (
                    "statement: "
                    + row.text
                ),
                "",
            ]
        )

    if not allowed:
        lines.extend(
            [
                "- NONE",
                "",
            ]
        )

    lines.extend(
        [
            "ALLOWED_STATEMENT_IDS",
            "=====================",
            *(
                allowed
                if allowed
                else ["NONE"]
            ),
            "",
            "OUTPUT REQUIREMENTS",
            "===================",
            (
                "Every returned statement_id must be "
                "copied exactly from "
                "ALLOWED_STATEMENT_IDS."
            ),
            (
                "Do not combine separate statements "
                "into a relation."
            ),
            (
                "Do not produce any negative-closure "
                "or absence judgment."
            ),
        ]
    )

    return "\n".join(lines)


class InstructorOpenAICompatibleInternalClosureBackend:
    backend_name = (
        "instructor_openai_compatible_"
        "internal_closure_positive_review"
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        capture_prompts: bool = False,
        telemetry_path: (
            str
            | os.PathLike[str]
            | None
        ) = None,
        telemetry_context: (
            dict[str, Any]
            | None
        ) = None,
    ) -> None:
        self.model_name = str(
            model
        )

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(
                api_key_env
            )
        )

        self.api_key_env = (
            api_key_env
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OPENAI_BASE_URL"
            )
            or None
        )

        self.instructor_mode = str(
            instructor_mode
        ).upper()

        self.temperature = float(
            temperature
        )

        self.parse_retries = int(
            parse_retries
        )

        self.timeout = timeout

        self.capture_prompts = bool(
            capture_prompts
        )

        self.telemetry_path = (
            telemetry_path
        )

        self.telemetry_context = dict(
            telemetry_context
            or {}
        )

        self.prompt_records: list[
            InternalClosurePromptRecord
        ] = []

        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError(
                "No API key available. "
                f"Set {self.api_key_env} "
                "or pass api_key explicitly."
            )

        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Internal closure reviewer requires "
                "'openai' and 'instructor'."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )

        if mode is None:
            raise ValueError(
                "Unknown Instructor mode: "
                f"{self.instructor_mode}"
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }

        if self.base_url:
            kwargs[
                "base_url"
            ] = self.base_url

        if self.timeout is not None:
            kwargs[
                "timeout"
            ] = self.timeout

        self._client = (
            instructor.from_openai(
                OpenAI(**kwargs),
                mode=mode,
            )
        )

        return self._client

    def review_target(
        self,
        *,
        target: ExecutableClosureTarget,
        statements: list[
            HypothesisEvidenceStatement
        ],
    ) -> InternalClosureReviewDraft:
        if target.slot != "BASE_RELATION":
            raise ValueError(
                "Internal positive closure v1 "
                "supports BASE_RELATION only."
            )

        user = (
            build_internal_base_review_prompt(
                target=target,
                statements=statements,
            )
        )

        if self.capture_prompts:
            self.prompt_records.append(
                InternalClosurePromptRecord(
                    name=(
                        "internal_closure_"
                        + target.target_id
                    ),
                    system_prompt=(
                        _INTERNAL_BASE_REVIEW_SYSTEM
                    ),
                    user_prompt=user,
                    prompt_sha256=_sha256(
                        _INTERNAL_BASE_REVIEW_SYSTEM,
                        user,
                    ),
                )
            )

        result, _event = (
            run_instructor_structured_call(
                self._get_client()
                .chat.completions,
                model=self.model_name,
                response_model=(
                    InternalClosureReviewDraft
                ),
                messages=[
                    {
                        "role": "system",
                        "content":
                            _INTERNAL_BASE_REVIEW_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                temperature=self.temperature,
                max_retries=self.parse_retries,
                telemetry_path=(
                    self.telemetry_path
                ),
                telemetry_context={
                    **self.telemetry_context,
                    "pipeline":
                        "nonobviousness_closure",
                    "stage":
                        "internal_grounding_"
                        "positive_review",
                    "call_kind":
                        "structured",
                    "target_id":
                        target.target_id,
                    "slot":
                        target.slot,
                    "source_claim_id":
                        target.source_claim_id,
                },
            )
        )

        if not isinstance(
            result,
            InternalClosureReviewDraft,
        ):
            result = (
                InternalClosureReviewDraft
                .model_validate(result)
            )

        return result


def review_and_compile_internal_base_target(
    *,
    backend:
        InstructorOpenAICompatibleInternalClosureBackend,
    target: ExecutableClosureTarget,
    hypothesis: HypothesisCard,
    context: HypothesisContext,
    min_positive_confidence: float = 0.70,
) -> InternalClosurePositiveReview:
    if target.slot != "BASE_RELATION":
        raise ValueError(
            "Internal positive closure v1 "
            "supports BASE_RELATION only."
        )

    statements = (
        authoritative_premise_statements(
            hypothesis=hypothesis,
            context=context,
        )
    )

    considered_ids = [
        row.statement_id
        for row in statements
    ]

    if not statements:
        return InternalClosurePositiveReview(
            target_id=target.target_id,
            slot=target.slot,
            source_claim_id=(
                target.source_claim_id
            ),
            considered_statement_ids=[],
            reason_codes=[
                "no_authoritative_internal_premises",
            ],
            interpretation=(
                "No canonical source-reported, "
                "premise-eligible internal evidence "
                "was available for positive BASE "
                "closure review."
            ),
        )

    draft = backend.review_target(
        target=target,
        statements=statements,
    )

    allowed = {
        row.statement_id: row
        for row in statements
    }

    unknown = sorted(
        {
            row.statement_id
            for row in draft.matches
        }
        - set(allowed)
    )

    positive_statement_ids: list[str] = []

    for match in draft.matches:
        if match.statement_id not in allowed:
            continue

        if (
            match.relationship
            != "ESTABLISHES_SLOT"
        ):
            continue

        if (
            match.confidence
            < min_positive_confidence
        ):
            continue

        if (
            match.statement_id
            not in positive_statement_ids
        ):
            positive_statement_ids.append(
                match.statement_id
            )

    positive_rows = [
        allowed[statement_id]
        for statement_id
        in positive_statement_ids
    ]

    positive_paper_ids = list(
        dict.fromkeys(
            paper_id
            for row in positive_rows
            for paper_id in row.paper_ids
        )
    )

    positive_support_node_ids = list(
        dict.fromkeys(
            node_id
            for row in positive_rows
            for node_id
            in row.scientific_support_node_ids
        )
    )

    positive_support_edge_ids = list(
        dict.fromkeys(
            edge_id
            for row in positive_rows
            for edge_id
            in row.scientific_support_edge_ids
        )
    )

    reason_codes: list[str] = []

    if unknown:
        reason_codes.append(
            "reviewer_unknown_statement_id_dropped"
        )

    if positive_statement_ids:
        reason_codes.append(
            "canonical_internal_base_relation_established"
        )
    else:
        reason_codes.append(
            "no_direct_internal_base_relation_established"
        )

    return InternalClosurePositiveReview(
        target_id=target.target_id,
        slot=target.slot,
        source_claim_id=(
            target.source_claim_id
        ),
        considered_statement_ids=(
            considered_ids
        ),
        positive_statement_ids=(
            positive_statement_ids
        ),
        positive_paper_ids=(
            positive_paper_ids
        ),
        positive_support_node_ids=(
            positive_support_node_ids
        ),
        positive_support_edge_ids=(
            positive_support_edge_ids
        ),
        reason_codes=reason_codes,
        reviewer_unknown_statement_ids=(
            unknown
        ),
        interpretation=(
            draft.interpretation
        ),
    )
