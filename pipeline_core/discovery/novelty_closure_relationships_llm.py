from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
)
from pipeline_core.discovery.novelty_closure_relationships import (
    ClosureRelationshipAssessmentDraft,
    CompiledClosureRelationships,
    compile_closure_relationship_assessment,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


_LOWER_ORDER_SLOTS = (
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
)


@dataclass(frozen=True)
class ClosureRelationshipPromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class ClosureRelationshipReviewOutcome:
    review_performed: bool
    draft: ClosureRelationshipAssessmentDraft
    compiled: CompiledClosureRelationships


class ClosureRelationshipBackendProtocol(
    Protocol
):
    def review_relationships(
        self,
        *,
        reviews: Sequence[Any],
        packet: PriorArtPacket,
        targets_by_slot: Mapping[str, Any],
    ) -> ClosureRelationshipAssessmentDraft: ...


_RELATIONSHIP_REVIEW_SYSTEM = """You review ONLY already-ESTABLISHED positive prior-art evidence
from three lower-order evidence-closure slots.

This is NOT a novelty judgment.
This is NOT a literature-wide judgment.
This is NOT a retrieval relevance judgment.

You are given only:
- BASE_RELATION positive evidence,
- DISTINGUISHING_FACTOR_EFFECT positive evidence,
- BRIDGE_RELATION positive evidence,
- the scientific source text associated with those slots.

You MUST NOT use outside knowledge.
You MUST NOT infer scientific relations from search queries.
You MUST NOT combine separate effects merely because they are plausible.
You MUST NOT use any work that is not listed in ALLOWED_POSITIVE_WORK_IDS.

Your task has exactly two outputs:

1. bridge_kind

UNASSESSED:
The supplied positive abstracts do not support a reliable bridge-type classification.

NONE:
The supplied positive evidence is scientifically relevant but does not establish
a recognizable lower-order bridge structure.

MAIN_EFFECTS_ONLY:
The positive evidence supports only ordinary lower-order/main-effect relations.
It does NOT establish a mediation chain and does NOT establish an
interaction-compatible relation.

MEDIATION_CHAIN:
The positive abstracts explicitly support a sequential/mediating connection
linking the distinguishing factor into the variables, state, mechanism, or
outcome surrounding the base relation.
This does NOT itself mean the distinguishing factor moderates the base relation.

INTERACTION_COMPATIBLE:
The positive abstracts explicitly support a relation in which the distinguishing
factor conditions, modifies, or changes a relation sufficiently close to the
base relation that the residual interaction may be routine composition.
Do NOT use this label when the evidence only states separate effects,
co-mentions variables, or supplies a mediation chain.

2. scope_compatibility

COMPATIBLE:
The positive evidence in BASE_RELATION, DISTINGUISHING_FACTOR_EFFECT, and
BRIDGE_RELATION is in sufficiently aligned scientific scope for a cross-slot
structural obviousness comparison.

The relevant scope includes, as applicable:
- material/system class,
- architecture or site class,
- scientific variables and observable,
- physical/chemical regime,
- experimental or theoretical context when it materially changes the relation.

The same paper is NOT required across slots, but the supplied abstracts must
make compatibility scientifically defensible.

INCOMPATIBLE:
The positive slot evidence has a material scientific-scope mismatch that prevents
cross-slot composition.

UNASSESSED:
The supplied positive metadata is insufficient to establish either compatibility
or incompatibility.

BASIS CONTRACT:
- bridge_basis_work_ids may contain ONLY positive ESTABLISHED work IDs supplied here.
- A non-NONE bridge_kind must cite the work(s) that explicitly support that bridge classification.
- scope_basis_work_ids must cite positive evidence supporting the scope judgment.
- For COMPATIBLE, cite at least one positive work contributing to EACH of
  BASE_RELATION, DISTINGUISHING_FACTOR_EFFECT, and BRIDGE_RELATION.
- Copy work IDs byte-for-byte from ALLOWED_POSITIVE_WORK_IDS.
- Do not cite retrieved, partial, negative, title-only, or unrelated works.
- Do not convert absence of evidence into a positive bridge or compatibility judgment.

Return only the structured ClosureRelationshipAssessmentDraft.
"""


def _field(
    row: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _sha256(
    system: str,
    user: str,
) -> str:
    raw = (
        system
        + "\n---\n"
        + user
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


def _positive_ids(
    row: Any,
) -> tuple[str, ...]:
    if (
        str(
            _field(
                row,
                "evidence_state",
                "UNASSESSED",
            )
        )
        != "ESTABLISHED"
    ):
        return ()

    result: list[str] = []

    for value in (
        _field(
            row,
            "positive_work_ids",
            (),
        )
        or ()
    ):
        work_id = str(
            value or ""
        ).strip()

        if (
            work_id
            and work_id not in result
        ):
            result.append(work_id)

    return tuple(result)


def relationship_review_needed(
    reviews: Sequence[Any],
) -> bool:
    """Only complete positive lower-order closure can justify review."""

    by_slot = {
        str(
            _field(
                row,
                "slot",
                "",
            )
        ): row
        for row in reviews
    }

    # Full relation already known makes cross-slot obviousness
    # analysis unnecessary.
    full = by_slot.get(
        "FULL_RELATION"
    )

    if (
        full is not None
        and str(
            _field(
                full,
                "evidence_state",
                "UNASSESSED",
            )
        )
        == "ESTABLISHED"
    ):
        return False

    return all(
        slot in by_slot
        and bool(
            _positive_ids(
                by_slot[slot]
            )
        )
        for slot
        in _LOWER_ORDER_SLOTS
    )


def build_closure_relationship_user_prompt(
    *,
    reviews: Sequence[Any],
    packet: PriorArtPacket,
    targets_by_slot: Mapping[str, Any],
    max_abstract_chars: int = 1800,
) -> str:
    """Expose only positive ESTABLISHED lower-order evidence."""

    by_slot = {
        str(
            _field(
                row,
                "slot",
                "",
            )
        ): row
        for row in reviews
    }

    works = {
        work.work_id: work
        for work in packet.works
    }

    allowed: list[str] = []

    lines: list[str] = [
        "CROSS-SLOT POSITIVE EVIDENCE REVIEW",
        "===================================",
        "",
        (
            "Only the positive ESTABLISHED evidence below "
            "may be used."
        ),
        (
            "Search queries, negative results, partial matches, "
            "and unrelated retrieved records are intentionally omitted."
        ),
        "",
    ]

    for slot in _LOWER_ORDER_SLOTS:
        row = by_slot.get(slot)
        target = targets_by_slot.get(
            slot
        )

        positive = (
            _positive_ids(row)
            if row is not None
            else ()
        )

        source_text = str(
            _field(
                target,
                "source_text",
                "",
            )
            if target is not None
            else ""
        )

        lines.extend(
            [
                slot,
                "=" * len(slot),
                (
                    "source_text: "
                    + source_text
                ),
                "",
            ]
        )

        if not positive:
            lines.extend(
                [
                    "- NO ESTABLISHED POSITIVE EVIDENCE",
                    "",
                ]
            )
            continue

        for work_id in positive:
            work = works.get(
                work_id
            )

            if work is None:
                continue

            if work_id not in allowed:
                allowed.append(
                    work_id
                )

            abstract = str(
                work.abstract or ""
            )

            if (
                len(abstract)
                > max_abstract_chars
            ):
                abstract = (
                    abstract[
                        : max_abstract_chars - 1
                    ].rstrip()
                    + "…"
                )

            lines.extend(
                [
                    f"work_id: {work.work_id}",
                    f"title: {work.title}",
                    f"year: {work.year}",
                    f"doi: {work.doi}",
                    (
                        "abstract: "
                        + (
                            abstract
                            if abstract
                            else "[NO ABSTRACT]"
                        )
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "ALLOWED_POSITIVE_WORK_IDS",
            "=========================",
            *(
                allowed
                if allowed
                else ["NONE"]
            ),
            "",
            "OUTPUT DISCIPLINE",
            "=================",
            (
                "Do not use or cite any work ID outside "
                "ALLOWED_POSITIVE_WORK_IDS."
            ),
            (
                "Do not infer an interaction from separate "
                "main effects."
            ),
            (
                "Do not infer scope compatibility merely "
                "because papers share vocabulary."
            ),
            (
                "Use UNASSESSED whenever the supplied "
                "positive abstracts are insufficient."
            ),
        ]
    )

    return "\n".join(
        lines
    )


class InstructorOpenAICompatibleClosureRelationshipBackend:
    backend_name = (
        "instructor_openai_compatible_"
        "closure_relationship_review"
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
        max_abstract_chars: int = 1800,
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

        self.max_abstract_chars = int(
            max_abstract_chars
        )

        self.telemetry_path = (
            telemetry_path
        )

        self.telemetry_context = dict(
            telemetry_context or {}
        )

        self.prompt_records: list[
            ClosureRelationshipPromptRecord
        ] = []

        self._client: Any | None = None

    def _get_client(
        self,
    ) -> Any:
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
                "Closure relationship reviewer requires "
                "installed 'openai' and 'instructor'."
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

    def review_relationships(
        self,
        *,
        reviews: Sequence[Any],
        packet: PriorArtPacket,
        targets_by_slot: Mapping[str, Any],
    ) -> ClosureRelationshipAssessmentDraft:
        user = (
            build_closure_relationship_user_prompt(
                reviews=reviews,
                packet=packet,
                targets_by_slot=targets_by_slot,
                max_abstract_chars=(
                    self.max_abstract_chars
                ),
            )
        )

        if self.capture_prompts:
            self.prompt_records.append(
                ClosureRelationshipPromptRecord(
                    name=(
                        "closure_relationship_review"
                    ),
                    system_prompt=(
                        _RELATIONSHIP_REVIEW_SYSTEM
                    ),
                    user_prompt=user,
                    prompt_sha256=_sha256(
                        _RELATIONSHIP_REVIEW_SYSTEM,
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
                    ClosureRelationshipAssessmentDraft
                ),
                messages=[
                    {
                        "role": "system",
                        "content":
                            _RELATIONSHIP_REVIEW_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                temperature=(
                    self.temperature
                ),
                max_retries=(
                    self.parse_retries
                ),
                telemetry_path=(
                    self.telemetry_path
                ),
                telemetry_context={
                    **self.telemetry_context,
                    "pipeline":
                        "nonobviousness_closure",
                    "stage":
                        "closure_relationship_review",
                    "call_kind":
                        "structured",
                },
            )
        )

        if not isinstance(
            result,
            ClosureRelationshipAssessmentDraft,
        ):
            result = (
                ClosureRelationshipAssessmentDraft
                .model_validate(
                    result
                )
            )

        return result


def review_and_compile_closure_relationships(
    *,
    backend: ClosureRelationshipBackendProtocol,
    reviews: Sequence[Any],
    packet: PriorArtPacket,
    targets_by_slot: Mapping[str, Any],
) -> ClosureRelationshipReviewOutcome:
    """Run one evidence-only relationship review when scientifically useful."""

    if not relationship_review_needed(
        reviews
    ):
        draft = (
            ClosureRelationshipAssessmentDraft(
                bridge_kind="UNASSESSED",
                scope_compatibility="UNASSESSED",
                interpretation=(
                    "Cross-slot relationship review was not run "
                    "because complete positive lower-order closure "
                    "was not available or the full relation was "
                    "already established."
                ),
            )
        )

        compiled = (
            compile_closure_relationship_assessment(
                reviews=reviews,
                draft=draft,
            )
        )

        return (
            ClosureRelationshipReviewOutcome(
                review_performed=False,
                draft=draft,
                compiled=compiled,
            )
        )

    draft = backend.review_relationships(
        reviews=reviews,
        packet=packet,
        targets_by_slot=targets_by_slot,
    )

    compiled = (
        compile_closure_relationship_assessment(
            reviews=reviews,
            draft=draft,
        )
    )

    return ClosureRelationshipReviewOutcome(
        review_performed=True,
        draft=draft,
        compiled=compiled,
    )
