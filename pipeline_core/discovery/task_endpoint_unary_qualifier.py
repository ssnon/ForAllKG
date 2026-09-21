from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Sequence

from pydantic import Field

from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
)
from pipeline_core.discovery.task_backbone_chain import (
    StrictModel,
    _coordinated_head_facet_spec,
    _structured_task_binding,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    lexical_tokens,
)


class UnaryEndpointQualifierStatus(str, Enum):
    NOT_APPLICABLE_COORDINATED_ENDPOINT = (
        "not_applicable_coordinated_endpoint"
    )
    FULL_ENDPOINT_BINDS = "full_endpoint_binds"
    SINGLE_TOKEN_OBSTRUCTION_CANDIDATE = (
        "single_token_obstruction_candidate"
    )
    MULTIPLE_SINGLE_TOKEN_OBSTRUCTIONS = (
        "multiple_single_token_obstructions"
    )
    NO_SINGLE_TOKEN_UNLOCK = "no_single_token_unlock"


class UnaryEndpointTokenRemovalView(StrictModel):
    removed_surface_token: str
    removed_lexical_tokens: list[str] = Field(
        default_factory=list
    )
    candidate_core_endpoint: str
    binding_count_after_removal: int = 0
    removed_token_on_unlocked_task_slot_count: int = 0
    removed_token_elsewhere_in_unlocked_component_count: int = 0
    diagnostic_only: Literal[True] = True
    endpoint_substitution_authority: Literal[False] = False
    qualifier_authority: Literal[False] = False


class UnaryEndpointQualifierObligationView(StrictModel):
    endpoint: str
    coordinated_endpoint: bool
    full_binding_count: int = 0
    status: UnaryEndpointQualifierStatus
    candidate_qualifier_surface: str | None = None
    candidate_qualifier_tokens: list[str] = Field(
        default_factory=list
    )
    candidate_core_endpoint: str | None = None
    candidate_core_binding_count: int = 0
    token_removals: list[
        UnaryEndpointTokenRemovalView
    ] = Field(default_factory=list)
    diagnostic_only: Literal[True] = True
    blocking: Literal[False] = False
    eligibility_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    endpoint_substitution_authority: Literal[False] = False
    task_filter_relaxed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class TaskEndpointUnaryQualifierObligations(StrictModel):
    requested_source: str
    requested_target: str
    source: UnaryEndpointQualifierObligationView
    target: UnaryEndpointQualifierObligationView
    diagnostic_only: Literal[True] = True
    blocking: Literal[False] = False
    eligibility_changed: Literal[False] = False
    threshold_changed: Literal[False] = False
    task_filter_relaxed: Literal[False] = False
    endpoint_substitution_performed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False
    rejection_authority_created: Literal[False] = False


_SURFACE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_./+\-]*"
)


def _known_components(
    components: Sequence[RelationComponentView],
) -> tuple[RelationComponentView, ...]:
    return tuple(
        component
        for component in components
        if (
            component.authority
            == RelationComponentAuthority.CONFIRMED_KNOWN
        )
    )


def _binding_rows(
    *,
    components: Sequence[RelationComponentView],
    endpoint: str,
):
    rows = []
    for component in _known_components(components):
        resolved = _structured_task_binding(
            component=component,
            task_endpoint=endpoint,
            endpoint_equivalences=(),
        )
        if resolved is None:
            continue
        binding, mode = resolved
        rows.append((component, binding, mode))
    return tuple(rows)


def _surface_token_removals(
    endpoint: str,
) -> tuple[tuple[str, frozenset[str], str], ...]:
    matches = list(
        _SURFACE_TOKEN_RE.finditer(str(endpoint))
    )
    rows = []
    seen = set()

    for match in matches:
        surface = match.group(0)
        removed_tokens = frozenset(
            lexical_tokens(surface)
        )
        if not removed_tokens:
            continue

        reduced = (
            str(endpoint)[: match.start()]
            + " "
            + str(endpoint)[match.end() :]
        )
        reduced = " ".join(reduced.split()).strip()
        if not reduced:
            continue

        key = (
            tuple(sorted(removed_tokens)),
            tuple(sorted(lexical_tokens(reduced))),
        )
        if key in seen:
            continue

        seen.add(key)
        rows.append(
            (
                surface,
                removed_tokens,
                reduced,
            )
        )

    return tuple(rows)


def build_unary_endpoint_qualifier_obligation_view(
    *,
    components: Sequence[RelationComponentView],
    endpoint: str,
) -> UnaryEndpointQualifierObligationView:
    endpoint = str(endpoint).strip()
    if not endpoint:
        raise ValueError("endpoint must be non-empty")

    if _coordinated_head_facet_spec(endpoint) is not None:
        return UnaryEndpointQualifierObligationView(
            endpoint=endpoint,
            coordinated_endpoint=True,
            full_binding_count=0,
            status=(
                UnaryEndpointQualifierStatus
                .NOT_APPLICABLE_COORDINATED_ENDPOINT
            ),
        )

    full_rows = _binding_rows(
        components=components,
        endpoint=endpoint,
    )
    if full_rows:
        return UnaryEndpointQualifierObligationView(
            endpoint=endpoint,
            coordinated_endpoint=False,
            full_binding_count=len(full_rows),
            status=(
                UnaryEndpointQualifierStatus
                .FULL_ENDPOINT_BINDS
            ),
        )

    removals: list[
        UnaryEndpointTokenRemovalView
    ] = []

    for (
        surface,
        removed_tokens,
        reduced,
    ) in _surface_token_removals(endpoint):
        unlocked = _binding_rows(
            components=components,
            endpoint=reduced,
        )

        task_slot_count = 0
        elsewhere_count = 0

        for component, binding, _ in unlocked:
            task_slot_text = (
                component.subject
                if binding.task_slot == "subject"
                else component.object
            )
            other_slot_text = (
                component.object
                if binding.task_slot == "subject"
                else component.subject
            )

            if (
                removed_tokens
                <= frozenset(
                    lexical_tokens(task_slot_text)
                )
            ):
                task_slot_count += 1

            if (
                removed_tokens
                <= frozenset(
                    lexical_tokens(other_slot_text)
                )
            ):
                elsewhere_count += 1

        removals.append(
            UnaryEndpointTokenRemovalView(
                removed_surface_token=surface,
                removed_lexical_tokens=sorted(
                    removed_tokens
                ),
                candidate_core_endpoint=reduced,
                binding_count_after_removal=len(unlocked),
                removed_token_on_unlocked_task_slot_count=(
                    task_slot_count
                ),
                removed_token_elsewhere_in_unlocked_component_count=(
                    elsewhere_count
                ),
            )
        )

    unlocking = [
        row
        for row in removals
        if row.binding_count_after_removal > 0
    ]

    if len(unlocking) == 1:
        candidate = unlocking[0]
        return UnaryEndpointQualifierObligationView(
            endpoint=endpoint,
            coordinated_endpoint=False,
            full_binding_count=0,
            status=(
                UnaryEndpointQualifierStatus
                .SINGLE_TOKEN_OBSTRUCTION_CANDIDATE
            ),
            candidate_qualifier_surface=(
                candidate.removed_surface_token
            ),
            candidate_qualifier_tokens=list(
                candidate.removed_lexical_tokens
            ),
            candidate_core_endpoint=(
                candidate.candidate_core_endpoint
            ),
            candidate_core_binding_count=(
                candidate.binding_count_after_removal
            ),
            token_removals=removals,
        )

    if len(unlocking) > 1:
        status = (
            UnaryEndpointQualifierStatus
            .MULTIPLE_SINGLE_TOKEN_OBSTRUCTIONS
        )
    else:
        status = (
            UnaryEndpointQualifierStatus
            .NO_SINGLE_TOKEN_UNLOCK
        )

    return UnaryEndpointQualifierObligationView(
        endpoint=endpoint,
        coordinated_endpoint=False,
        full_binding_count=0,
        status=status,
        token_removals=removals,
    )


def build_task_endpoint_unary_qualifier_obligations(
    *,
    components: Sequence[RelationComponentView],
    requested_source: str,
    requested_target: str,
) -> TaskEndpointUnaryQualifierObligations:
    return TaskEndpointUnaryQualifierObligations(
        requested_source=requested_source,
        requested_target=requested_target,
        source=build_unary_endpoint_qualifier_obligation_view(
            components=components,
            endpoint=requested_source,
        ),
        target=build_unary_endpoint_qualifier_obligation_view(
            components=components,
            endpoint=requested_target,
        ),
    )
