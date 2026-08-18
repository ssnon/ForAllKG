from __future__ import annotations

from typing import Final


RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY: Final = (
    "relation_semantics_already_validated"
)


def relation_semantics_already_validated(
    context: object,
) -> bool:
    """Return whether relation semantics were explicitly validated upstream."""

    return (
        isinstance(context, dict)
        and context.get(
            RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY
        )
        is True
    )
