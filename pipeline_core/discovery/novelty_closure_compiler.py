from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.novelty_nonobviousness import (
    NonObviousnessEvidenceClosure,
)


_EXPECTED_SLOTS = (
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
    "FULL_RELATION",
)

_VALID_STATES = {
    "ESTABLISHED",
    "NOT_FOUND",
    "UNASSESSED",
}

_VALID_BRIDGE_KINDS = {
    "NONE",
    "MAIN_EFFECTS_ONLY",
    "MEDIATION_CHAIN",
    "INTERACTION_COMPATIBLE",
}


@dataclass(frozen=True)
class NonObviousnessClosureCompilation:
    """Auditable conversion from slot reviews into structural closure."""

    closure: NonObviousnessEvidenceClosure
    reason_codes: tuple[str, ...] = ()


def _field(
    row: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)

    return getattr(
        row,
        name,
        default,
    )


def _unique_ids(
    values: Any,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values or ():
        work_id = str(value).strip()

        if (
            not work_id
            or work_id in seen
        ):
            continue

        seen.add(work_id)
        output.append(work_id)

    return tuple(output)


def compile_nonobviousness_evidence_closure(
    *,
    reviews: Sequence[Any],
    internal_reviews: Sequence[Any] = (),
    bridge_kind: str = "NONE",
    scope_compatible: bool = True,
) -> NonObviousnessClosureCompilation:
    """Compile reviewed closure slots into the structural evaluator contract.

    Fail-closed rules:
    - missing/duplicate slots are invalid artifacts;
    - unsupported evidence states become UNASSESSED;
    - NOT_FOUND without sufficient negative coverage becomes UNASSESSED;
    - ESTABLISHED requires traceable positive external work IDs or
      positive internal statement IDs;
    - internal grounding is positive-only and can never create NOT_FOUND.

    This compiler never turns absence-like evidence into novelty evidence.
    """

    if bridge_kind not in _VALID_BRIDGE_KINDS:
        raise ValueError(
            f"Unsupported bridge_kind: {bridge_kind!r}"
        )

    by_slot: dict[str, Any] = {}

    for row in reviews:
        slot = str(
            _field(row, "slot", "")
        ).strip()

        if slot not in _EXPECTED_SLOTS:
            raise ValueError(
                f"Unexpected closure slot: {slot!r}"
            )

        if slot in by_slot:
            raise ValueError(
                f"Duplicate closure slot: {slot}"
            )

        by_slot[slot] = row

    missing = [
        slot
        for slot in _EXPECTED_SLOTS
        if slot not in by_slot
    ]

    if missing:
        raise ValueError(
            "Missing closure slots: "
            + ", ".join(missing)
        )

    internal_by_slot: dict[str, Any] = {}

    for row in internal_reviews:
        slot = str(
            _field(row, "slot", "")
        ).strip()

        if slot not in _EXPECTED_SLOTS:
            raise ValueError(
                "Unexpected internal closure slot: "
                f"{slot!r}"
            )

        if slot in internal_by_slot:
            raise ValueError(
                "Duplicate internal closure slot: "
                + slot
            )

        internal_by_slot[slot] = row

    reason_codes: list[str] = []
    states: dict[str, str] = {}
    work_ids: dict[str, tuple[str, ...]] = {}
    internal_statement_ids: dict[
        str,
        tuple[str, ...],
    ] = {}

    for slot in _EXPECTED_SLOTS:
        row = by_slot[slot]

        raw_state = str(
            _field(
                row,
                "evidence_state",
                "UNASSESSED",
            )
        ).strip()

        positive_ids = _unique_ids(
            _field(
                row,
                "positive_work_ids",
                (),
            )
        )

        internal_row = (
            internal_by_slot.get(slot)
        )

        positive_internal_ids = _unique_ids(
            _field(
                internal_row,
                "positive_statement_ids",
                (),
            )
        )

        negative_sufficient = bool(
            _field(
                row,
                "negative_coverage_sufficient",
                False,
            )
        )

        state = raw_state

        if state not in _VALID_STATES:
            state = "UNASSESSED"

            reason_codes.append(
                f"{slot.lower()}:unsupported_state_fail_closed"
            )

        if (
            state == "NOT_FOUND"
            and not negative_sufficient
        ):
            state = "UNASSESSED"

            reason_codes.append(
                f"{slot.lower()}:illegal_negative_closure_fail_closed"
            )

        if (
            state == "ESTABLISHED"
            and not positive_ids
            and not positive_internal_ids
        ):
            state = "UNASSESSED"

            reason_codes.append(
                f"{slot.lower()}:established_without_positive_provenance"
            )

        if positive_internal_ids:
            if state == "NOT_FOUND":
                reason_codes.append(
                    f"{slot.lower()}:"
                    "internal_positive_overrides_"
                    "search_bounded_negative"
                )

            if (
                state != "ESTABLISHED"
                or not positive_ids
            ):
                reason_codes.append(
                    f"{slot.lower()}:"
                    "established_from_internal_grounding"
                )

            state = "ESTABLISHED"

        states[slot] = state

        work_ids[slot] = (
            positive_ids
            if state == "ESTABLISHED"
            else ()
        )

        internal_statement_ids[slot] = (
            positive_internal_ids
            if state == "ESTABLISHED"
            else ()
        )

    closure = NonObviousnessEvidenceClosure(
        base_relation=states[
            "BASE_RELATION"
        ],
        distinguishing_factor_effect=states[
            "DISTINGUISHING_FACTOR_EFFECT"
        ],
        bridge_relation=states[
            "BRIDGE_RELATION"
        ],
        full_relation=states[
            "FULL_RELATION"
        ],
        bridge_kind=bridge_kind,
        scope_compatible=bool(
            scope_compatible
        ),
        base_work_ids=work_ids[
            "BASE_RELATION"
        ],
        factor_work_ids=work_ids[
            "DISTINGUISHING_FACTOR_EFFECT"
        ],
        bridge_work_ids=work_ids[
            "BRIDGE_RELATION"
        ],
        full_relation_work_ids=work_ids[
            "FULL_RELATION"
        ],
        base_internal_statement_ids=(
            internal_statement_ids[
                "BASE_RELATION"
            ]
        ),
        factor_internal_statement_ids=(
            internal_statement_ids[
                "DISTINGUISHING_FACTOR_EFFECT"
            ]
        ),
        bridge_internal_statement_ids=(
            internal_statement_ids[
                "BRIDGE_RELATION"
            ]
        ),
        full_relation_internal_statement_ids=(
            internal_statement_ids[
                "FULL_RELATION"
            ]
        ),
    )

    return NonObviousnessClosureCompilation(
        closure=closure,
        reason_codes=tuple(
            reason_codes
        ),
    )
