from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateRelationView(StrictModel):
    unit_id: str
    label: str = ""

    proposed_subject: str
    proposed_relation: str
    proposed_object: str

    occurrences: int = 0


class TaskBridgeCompositeCandidate(StrictModel):
    schema_version: str = (
        "task-bridge-composite-candidate-v1"
    )

    composite_id: str

    source_unit_id: str
    target_unit_id: str

    source_overlap_tokens: list[str]
    target_overlap_tokens: list[str]

    source_mediator_tokens: list[str]
    target_mediator_tokens: list[str]
    shared_mediator_tokens: list[str]

    source_relation: CandidateRelationView
    target_relation: CandidateRelationView

    compatibility_score: float

    epistemic_status: str = (
        "inspiration_only"
    )

    requires_verification: bool = True

    reason_codes: list[str] = Field(
        default_factory=lambda: [
            "task_conditioned_candidate_composition",
            "source_relation_inspiration_only",
            "target_relation_inspiration_only",
            "shared_mediator_required",
            "composite_requires_verification",
        ]
    )


_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+"
)

_STOP = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "within",

    # Generic relation-language tokens.
    "affect",
    "affects",
    "alter",
    "alters",
    "depend",
    "depends",
    "modulate",
    "modulates",
    "promote",
    "promotes",
    "relate",
    "relates",
    "relationship",
    "suggest",
    "suggests",
    "vary",
    "varies",
}


def _stem_token(
    token: str,
) -> str:
    token = token.lower().strip()

    if len(token) <= 3:
        return token

    if (
        token.endswith("ies")
        and len(token) > 4
    ):
        return (
            token[:-3]
            + "y"
        )

    if (
        token.endswith("ses")
        and len(token) > 4
    ):
        # e.g. "responses" is handled below;
        # avoid aggressive linguistic stemming.
        return token

    if (
        token.endswith("s")
        and not token.endswith("ss")
        and len(token) > 4
    ):
        return token[:-1]

    return token


def lexical_tokens(
    text: str,
) -> frozenset[str]:
    normalized = str(text).replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    ).replace(
        "/",
        " ",
    )

    out = set()

    for token in _TOKEN_RE.findall(
        normalized
    ):
        token = _stem_token(
            token
        )

        if (
            len(token) >= 2
            and token not in _STOP
        ):
            out.add(token)

    return frozenset(
        out
    )


def relation_tokens(
    relation: CandidateRelationView,
) -> frozenset[str]:
    return lexical_tokens(
        " ".join(
            [
                relation.proposed_subject,
                relation.proposed_relation,
                relation.proposed_object,
            ]
        )
    )


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(
            raw
        ).hexdigest()[:20]
    )


def candidate_relation_from_mapping(
    row: Mapping[str, Any],
) -> CandidateRelationView:
    return CandidateRelationView(
        unit_id=str(
            row.get(
                "unit_id",
                "",
            )
        ),
        label=str(
            row.get(
                "label",
                "",
            )
            or ""
        ),
        proposed_subject=str(
            row.get(
                "proposed_subject",
                "",
            )
            or ""
        ),
        proposed_relation=str(
            row.get(
                "proposed_relation",
                "",
            )
            or ""
        ),
        proposed_object=str(
            row.get(
                "proposed_object",
                "",
            )
            or ""
        ),
        occurrences=int(
            row.get(
                "occurrences",
                0,
            )
            or 0
        ),
    )


def compose_task_bridge_candidates(
    *,
    candidates: Sequence[
        CandidateRelationView
    ],
    requested_source: str,
    requested_target: str,
    max_composites: int = 12,
) -> tuple[
    TaskBridgeCompositeCandidate,
    ...,
]:
    """
    Compose source-side and target-side unverified candidate relations
    through shared mediator vocabulary.

    This function does NOT infer that either component relation applies
    to the user's requested system. Component relations remain
    inspiration-only and every composite requires verification.

    Selection contract:
      1. source-side candidate must lexically overlap requested source;
      2. target-side candidate must lexically overlap requested target;
      3. after removing task-nucleus tokens, both candidate relations
         must share at least one mediator token;
      4. rank by mediator overlap/coverage, never by downstream novelty
         or semantic-distinctiveness outcomes.
    """

    if max_composites < 1:
        raise ValueError(
            "max_composites must be >= 1"
        )

    source_task_tokens = lexical_tokens(
        requested_source
    )

    target_task_tokens = lexical_tokens(
        requested_target
    )

    if not source_task_tokens:
        raise ValueError(
            "requested_source produced no tokens"
        )

    if not target_task_tokens:
        raise ValueError(
            "requested_target produced no tokens"
        )

    rows = []

    source_candidates = []
    target_candidates = []

    for candidate in candidates:
        tokens = relation_tokens(
            candidate
        )

        source_overlap = (
            tokens
            & source_task_tokens
        )

        target_overlap = (
            tokens
            & target_task_tokens
        )

        if source_overlap:
            source_candidates.append(
                (
                    candidate,
                    tokens,
                    source_overlap,
                )
            )

        if target_overlap:
            target_candidates.append(
                (
                    candidate,
                    tokens,
                    target_overlap,
                )
            )

    for (
        source_candidate,
        source_tokens,
        source_overlap,
    ) in source_candidates:

        source_mediator = (
            source_tokens
            - source_task_tokens
            - target_task_tokens
        )

        if not source_mediator:
            continue

        for (
            target_candidate,
            target_tokens,
            target_overlap,
        ) in target_candidates:

            if (
                source_candidate.unit_id
                == target_candidate.unit_id
            ):
                continue

            target_mediator = (
                target_tokens
                - source_task_tokens
                - target_task_tokens
            )

            if not target_mediator:
                continue

            shared = (
                source_mediator
                & target_mediator
            )

            if not shared:
                continue

            union = (
                source_mediator
                | target_mediator
            )

            jaccard = (
                len(shared)
                / len(union)
                if union
                else 0.0
            )

            source_coverage = (
                len(shared)
                / len(source_mediator)
            )

            target_coverage = (
                len(shared)
                / len(target_mediator)
            )

            # Shared-token count is primary so two independently
            # matching mediator concepts outrank a one-token accident.
            # Coverage then prefers tighter bridge relations.
            score = (
                2.0 * len(shared)
                + 0.75 * jaccard
                + 0.25 * min(
                    source_coverage,
                    target_coverage,
                )
            )

            composite_id = _stable_id(
                "task_bridge_composite",
                source_candidate.unit_id,
                target_candidate.unit_id,
                *sorted(shared),
            )

            rows.append(
                TaskBridgeCompositeCandidate(
                    composite_id=composite_id,

                    source_unit_id=(
                        source_candidate.unit_id
                    ),

                    target_unit_id=(
                        target_candidate.unit_id
                    ),

                    source_overlap_tokens=sorted(
                        source_overlap
                    ),

                    target_overlap_tokens=sorted(
                        target_overlap
                    ),

                    source_mediator_tokens=sorted(
                        source_mediator
                    ),

                    target_mediator_tokens=sorted(
                        target_mediator
                    ),

                    shared_mediator_tokens=sorted(
                        shared
                    ),

                    source_relation=(
                        source_candidate
                    ),

                    target_relation=(
                        target_candidate
                    ),

                    compatibility_score=float(
                        score
                    ),
                )
            )

    rows.sort(
        key=lambda row: (
            -len(
                row.shared_mediator_tokens
            ),
            -row.compatibility_score,
            row.source_unit_id,
            row.target_unit_id,
        )
    )

    # Stable pair-level de-duplication.
    seen = set()
    selected = []

    for row in rows:
        key = (
            row.source_unit_id,
            row.target_unit_id,
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(
            row
        )

        if len(selected) >= max_composites:
            break

    return tuple(
        selected
    )
