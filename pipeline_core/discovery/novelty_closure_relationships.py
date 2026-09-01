from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


EvidenceBridgeKind = Literal[
    "UNASSESSED",
    "NONE",
    "MAIN_EFFECTS_ONLY",
    "MEDIATION_CHAIN",
    "INTERACTION_COMPATIBLE",
]

EvidenceScopeCompatibility = Literal[
    "UNASSESSED",
    "COMPATIBLE",
    "INCOMPATIBLE",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ClosureRelationshipAssessmentDraft(
    StrictModel
):
    """Reviewer proposal over ESTABLISHED positive closure evidence only."""

    bridge_kind: EvidenceBridgeKind = (
        "UNASSESSED"
    )

    scope_compatibility: (
        EvidenceScopeCompatibility
    ) = "UNASSESSED"

    bridge_basis_work_ids: list[str] = (
        Field(default_factory=list)
    )

    scope_basis_work_ids: list[str] = (
        Field(default_factory=list)
    )

    interpretation: str = Field(
        min_length=1
    )


@dataclass(frozen=True)
class CompiledClosureRelationships:
    """Fail-closed evidence-side inputs for structural adjudication."""

    bridge_kind: Literal[
        "NONE",
        "MAIN_EFFECTS_ONLY",
        "MEDIATION_CHAIN",
        "INTERACTION_COMPATIBLE",
    ]

    scope_compatible: bool

    bridge_basis_work_ids: tuple[str, ...]
    scope_basis_work_ids: tuple[str, ...]

    reason_codes: tuple[str, ...]
    interpretation: str


_EXPECTED_LOWER_ORDER_SLOTS = (
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
)


def _field(
    row: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(row, Mapping):
        return row.get(
            name,
            default,
        )

    return getattr(
        row,
        name,
        default,
    )


def _ids(
    values: Any,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values or ():
        work_id = str(
            value or ""
        ).strip()

        if (
            not work_id
            or work_id in seen
        ):
            continue

        seen.add(work_id)
        result.append(work_id)

    return tuple(result)


def _established_positive_by_slot(
    reviews: Sequence[Any],
) -> dict[str, tuple[str, ...]]:
    result: dict[
        str,
        tuple[str, ...],
    ] = {}

    for row in reviews:
        slot = str(
            _field(
                row,
                "slot",
                "",
            )
        ).strip()

        if not slot:
            continue

        state = str(
            _field(
                row,
                "evidence_state",
                "UNASSESSED",
            )
        ).strip()

        positive = _ids(
            _field(
                row,
                "positive_work_ids",
                (),
            )
        )

        result[slot] = (
            positive
            if (
                state == "ESTABLISHED"
                and positive
            )
            else ()
        )

    return result


def compile_closure_relationship_assessment(
    *,
    reviews: Sequence[Any],
    draft: ClosureRelationshipAssessmentDraft,
) -> CompiledClosureRelationships:
    """Compile bridge-kind and cross-slot scope from positive evidence.

    This compiler does NOT derive scientific relations itself.

    A reviewer may propose bridge/scope semantics, but:
    - every cited work must already be positive ESTABLISHED evidence;
    - a non-NONE bridge kind requires positive BRIDGE_RELATION evidence;
    - scope compatibility requires positive evidence in BASE, FACTOR,
      and BRIDGE, with at least one cited work from every slot;
    - unknown, partial, negative, or merely retrieved works cannot
      support bridge or scope classification.

    Invalid or incomplete proposals fail closed to NONE / False.
    """

    positive = (
        _established_positive_by_slot(
            reviews
        )
    )

    reasons: list[str] = []

    lower_order_positive = {
        work_id
        for slot
        in _EXPECTED_LOWER_ORDER_SLOTS
        for work_id
        in positive.get(
            slot,
            (),
        )
    }

    bridge_positive = set(
        positive.get(
            "BRIDGE_RELATION",
            (),
        )
    )

    bridge_basis = _ids(
        draft.bridge_basis_work_ids
    )

    scope_basis = _ids(
        draft.scope_basis_work_ids
    )

    valid_bridge_basis = tuple(
        work_id
        for work_id in bridge_basis
        if work_id
        in lower_order_positive
    )

    valid_scope_basis = tuple(
        work_id
        for work_id in scope_basis
        if work_id
        in lower_order_positive
    )

    unknown_bridge = sorted(
        set(bridge_basis)
        - lower_order_positive
    )

    unknown_scope = sorted(
        set(scope_basis)
        - lower_order_positive
    )

    if unknown_bridge:
        reasons.append(
            "bridge_basis_contains_nonpositive_work"
        )

    if unknown_scope:
        reasons.append(
            "scope_basis_contains_nonpositive_work"
        )

    # --------------------------------------------------------------
    # Bridge kind
    # --------------------------------------------------------------

    requested_bridge = (
        draft.bridge_kind
    )

    bridge_kind: Literal[
        "NONE",
        "MAIN_EFFECTS_ONLY",
        "MEDIATION_CHAIN",
        "INTERACTION_COMPATIBLE",
    ] = "NONE"

    if requested_bridge in {
        "MAIN_EFFECTS_ONLY",
        "MEDIATION_CHAIN",
        "INTERACTION_COMPATIBLE",
    }:
        if not bridge_positive:
            reasons.append(
                "bridge_kind_requires_established_bridge_relation"
            )

        elif not valid_bridge_basis:
            reasons.append(
                "bridge_kind_missing_positive_basis"
            )

        elif not (
            set(valid_bridge_basis)
            & bridge_positive
        ):
            reasons.append(
                "bridge_kind_basis_not_tied_to_bridge_slot"
            )

        elif unknown_bridge:
            reasons.append(
                "bridge_kind_basis_not_fully_traceable"
            )

        else:
            bridge_kind = (
                requested_bridge
            )

    elif requested_bridge == "NONE":
        bridge_kind = "NONE"

    else:
        reasons.append(
            "bridge_kind_unassessed_fail_closed"
        )

    # --------------------------------------------------------------
    # Scope compatibility
    # --------------------------------------------------------------

    scope_compatible = False

    if (
        draft.scope_compatibility
        == "COMPATIBLE"
    ):
        missing_slots = [
            slot
            for slot
            in _EXPECTED_LOWER_ORDER_SLOTS
            if not positive.get(
                slot,
                (),
            )
        ]

        if missing_slots:
            reasons.append(
                "scope_compatibility_requires_all_lower_order_slots_established"
            )

        elif unknown_scope:
            reasons.append(
                "scope_basis_not_fully_traceable"
            )

        else:
            basis_set = set(
                valid_scope_basis
            )

            uncovered = [
                slot
                for slot
                in _EXPECTED_LOWER_ORDER_SLOTS
                if not (
                    basis_set
                    & set(
                        positive[
                            slot
                        ]
                    )
                )
            ]

            if uncovered:
                reasons.append(
                    "scope_basis_does_not_cover_all_lower_order_slots"
                )
            else:
                scope_compatible = True

    elif (
        draft.scope_compatibility
        == "INCOMPATIBLE"
    ):
        scope_compatible = False
        reasons.append(
            "reviewed_lower_order_scope_incompatible"
        )

    else:
        scope_compatible = False
        reasons.append(
            "scope_compatibility_unassessed_fail_closed"
        )

    return CompiledClosureRelationships(
        bridge_kind=bridge_kind,
        scope_compatible=(
            scope_compatible
        ),
        bridge_basis_work_ids=(
            valid_bridge_basis
        ),
        scope_basis_work_ids=(
            valid_scope_basis
        ),
        reason_codes=tuple(
            dict.fromkeys(
                reasons
            )
        ),
        interpretation=(
            draft.interpretation
        ),
    )
