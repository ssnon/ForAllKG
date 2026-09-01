from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


MediatorIdentityStatus = Literal[
    "LEXICALLY_SUPPORTED",
    "IDENTITY_AMBIGUOUS",
    "ROLE_INCOMPATIBLE",
]


@dataclass(frozen=True)
class CompositeMediatorIdentityShadowReview:
    """
    Shadow-only lexical identity diagnostic.

    IMPORTANT:
    LEXICALLY_SUPPORTED does NOT mean scientifically valid.
    This review only asks whether a lexical shared mediator has
    sufficient role/context support to be treated as a plausible
    composition key before independent scientific verification.
    """

    status: MediatorIdentityStatus

    mediator_tokens: tuple[str, ...]

    source_mediator_fields: tuple[str, ...]
    target_mediator_fields: tuple[str, ...]
    common_mediator_fields: tuple[str, ...]

    source_context_tokens: tuple[str, ...]
    target_context_tokens: tuple[str, ...]
    context_overlap_tokens: tuple[str, ...]

    multi_token_mediator: bool

    reason_codes: tuple[str, ...]


_STOP = {
    "and",
    "or",
    "of",
    "the",
    "a",
    "an",
    "in",
    "on",
    "with",
    "to",
    "for",
    "by",
    "via",
}


def _tokens(
    value: object,
) -> set[str]:

    return {
        token
        for token in re.findall(
            r"[a-z0-9α-ω]+",
            str(
                value
                or ""
            ).lower(),
        )
        if (
            len(token) >= 2
            and
            token not in _STOP
        )
    }


def _field_hits(
    *,
    subject: str,
    relation: str,
    object_: str,
    mediator_tokens: set[str],
) -> dict[str, set[str]]:

    values = {
        "subject":
            subject,

        "relation":
            relation,

        "object":
            object_,
    }

    result = {}

    for field, value in values.items():

        hits = (
            _tokens(
                value
            )
            &
            mediator_tokens
        )

        if hits:
            result[
                field
            ] = hits

    return result


def _context_for_fields(
    *,
    subject: str,
    relation: str,
    object_: str,
    fields: set[str],
    mediator_tokens: set[str],
) -> set[str]:

    values = {
        "subject":
            subject,

        "relation":
            relation,

        "object":
            object_,
    }

    result = set()

    for field in fields:
        result |= (
            _tokens(
                values[
                    field
                ]
            )
            -
            mediator_tokens
        )

    return result


class CompositeMediatorIdentityShadowCritic:
    """
    Frozen N8-A16B lexical-role mediator identity policy.

    Decision contract
    -----------------
    ROLE_INCOMPATIBLE:
        Shared mediator vocabulary does not occupy any common SRO
        field between source and target candidate relations.

    LEXICALLY_SUPPORTED:
        Either
          - at least two mediator tokens occupy a common SRO role, or
          - one mediator token occupies a common role and the
            surrounding role-local contexts overlap.

    IDENTITY_AMBIGUOUS:
        Exactly one mediator token occupies a common SRO role, but
        the source/target role-local contexts share no additional
        vocabulary.

    This critic has no scientific-validity authority.
    """

    def review(
        self,
        *,
        mediator_tokens,
        source_subject: str,
        source_relation: str,
        source_object: str,
        target_subject: str,
        target_relation: str,
        target_object: str,
    ) -> CompositeMediatorIdentityShadowReview:

        mediator = {
            token
            for item in (
                mediator_tokens
                or ()
            )
            for token in _tokens(
                item
            )
        }

        if not mediator:
            return (
                CompositeMediatorIdentityShadowReview(
                    status="ROLE_INCOMPATIBLE",
                    mediator_tokens=(),
                    source_mediator_fields=(),
                    target_mediator_fields=(),
                    common_mediator_fields=(),
                    source_context_tokens=(),
                    target_context_tokens=(),
                    context_overlap_tokens=(),
                    multi_token_mediator=False,
                    reason_codes=(
                        "NO_MEDIATOR_TOKENS",
                    ),
                )
            )

        source_hits = _field_hits(
            subject=source_subject,
            relation=source_relation,
            object_=source_object,
            mediator_tokens=mediator,
        )

        target_hits = _field_hits(
            subject=target_subject,
            relation=target_relation,
            object_=target_object,
            mediator_tokens=mediator,
        )

        source_fields = set(
            source_hits
        )

        target_fields = set(
            target_hits
        )

        common_fields = (
            source_fields
            &
            target_fields
        )

        source_context = (
            _context_for_fields(
                subject=source_subject,
                relation=source_relation,
                object_=source_object,
                fields=common_fields,
                mediator_tokens=mediator,
            )
        )

        target_context = (
            _context_for_fields(
                subject=target_subject,
                relation=target_relation,
                object_=target_object,
                fields=common_fields,
                mediator_tokens=mediator,
            )
        )

        overlap = (
            source_context
            &
            target_context
        )

        multi_token = (
            len(
                mediator
            )
            >= 2
        )

        if not common_fields:

            status = (
                "ROLE_INCOMPATIBLE"
            )

            reasons = (
                "NO_COMMON_MEDIATOR_SRO_ROLE",
            )

        elif multi_token:

            status = (
                "LEXICALLY_SUPPORTED"
            )

            reasons = (
                "MULTI_TOKEN_MEDIATOR",
                "COMMON_MEDIATOR_SRO_ROLE",
            )

        elif overlap:

            status = (
                "LEXICALLY_SUPPORTED"
            )

            reasons = (
                "SINGLE_TOKEN_MEDIATOR",
                "COMMON_MEDIATOR_SRO_ROLE",
                "ROLE_LOCAL_CONTEXT_OVERLAP",
            )

        else:

            status = (
                "IDENTITY_AMBIGUOUS"
            )

            reasons = (
                "SINGLE_TOKEN_MEDIATOR",
                "COMMON_MEDIATOR_SRO_ROLE",
                "NO_ROLE_LOCAL_CONTEXT_OVERLAP",
            )

        return CompositeMediatorIdentityShadowReview(
            status=status,

            mediator_tokens=tuple(
                sorted(
                    mediator
                )
            ),

            source_mediator_fields=tuple(
                sorted(
                    source_fields
                )
            ),

            target_mediator_fields=tuple(
                sorted(
                    target_fields
                )
            ),

            common_mediator_fields=tuple(
                sorted(
                    common_fields
                )
            ),

            source_context_tokens=tuple(
                sorted(
                    source_context
                )
            ),

            target_context_tokens=tuple(
                sorted(
                    target_context
                )
            ),

            context_overlap_tokens=tuple(
                sorted(
                    overlap
                )
            ),

            multi_token_mediator=(
                multi_token
            ),

            reason_codes=reasons,
        )
