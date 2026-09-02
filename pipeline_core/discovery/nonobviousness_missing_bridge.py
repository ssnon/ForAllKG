from __future__ import annotations

import hashlib
from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeCompilation,
    N11MissingBridgeOpportunity,
)


_REQUIRED_SLOTS = (
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
    "FULL_RELATION",
)


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


def _text(
    value: object,
) -> str:
    return str(
        value or ""
    ).strip()


def _unique_strings(
    values: object,
) -> list[str]:
    if not isinstance(
        values,
        (list, tuple),
    ):
        return []

    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = _text(value)

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(text)

    return output


def _slot_map(
    rows: Sequence[
        Mapping[str, Any]
    ],
    *,
    label: str,
) -> dict[
    str,
    Mapping[str, Any],
]:
    result: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for row in rows:
        slot = _text(
            row.get("slot")
        )

        if not slot:
            raise ValueError(
                f"{label} row missing slot"
            )

        if slot in result:
            raise ValueError(
                f"duplicate {label} slot: {slot}"
            )

        result[slot] = row

    missing = [
        slot
        for slot in _REQUIRED_SLOTS
        if slot not in result
    ]

    if missing:
        raise ValueError(
            f"{label} missing required slots: "
            + ", ".join(missing)
        )

    return result


def _positive_work_ids(
    review: Mapping[str, Any],
) -> list[str]:
    return _unique_strings(
        review.get(
            "positive_work_ids",
            [],
        )
    )


def _validate_state_support(
    *,
    slot: str,
    state: str,
    positive_work_ids: list[str],
) -> None:
    if (
        state == "ESTABLISHED"
        and not positive_work_ids
    ):
        raise ValueError(
            f"{slot} is ESTABLISHED without "
            "positive work IDs"
        )

    if (
        state == "NOT_FOUND"
        and positive_work_ids
    ):
        raise ValueError(
            f"{slot} is NOT_FOUND but contains "
            "positive work IDs"
        )


def _not_eligible(
    *reason_codes: str,
) -> N11MissingBridgeCompilation:
    return N11MissingBridgeCompilation(
        status="NOT_ELIGIBLE",
        opportunity=None,
        reason_codes=list(
            dict.fromkeys(
                reason_codes
            )
        ),
    )


def compile_missing_bridge_opportunity(
    *,
    execution_plan: Mapping[
        str,
        Any,
    ],
    slot_reviews: Sequence[
        Mapping[str, Any]
    ],
    closure_relationships: Mapping[
        str,
        Any,
    ],
) -> N11MissingBridgeCompilation:
    """Compile an N10 missing-bridge state into an N11 search task.

    This function never infers a new scientific relation.

    Eligibility means only:
      BASE   = ESTABLISHED
      FACTOR = ESTABLISHED
      BRIDGE = NOT_FOUND
      FULL   = NOT_FOUND

    and cross-slot scope remains unassessed because no positive bridge
    relation exists.

    The resulting opportunity is therefore a request to search for a
    grounded lower-order relation, not evidence that the relation exists.
    """

    targets_raw = execution_plan.get(
        "targets",
        [],
    )

    if not isinstance(
        targets_raw,
        list,
    ):
        raise ValueError(
            "execution plan targets must be a list"
        )

    reviews_raw = list(
        slot_reviews
    )

    targets = _slot_map(
        targets_raw,
        label="execution target",
    )

    reviews = _slot_map(
        reviews_raw,
        label="slot review",
    )

    states: dict[str, str] = {}

    positives: dict[
        str,
        list[str],
    ] = {}

    for slot in _REQUIRED_SLOTS:
        state = _text(
            reviews[slot].get(
                "evidence_state"
            )
        )

        if state not in {
            "ESTABLISHED",
            "NOT_FOUND",
            "UNASSESSED",
        }:
            raise ValueError(
                f"unsupported evidence state "
                f"for {slot}: {state}"
            )

        work_ids = _positive_work_ids(
            reviews[slot]
        )

        _validate_state_support(
            slot=slot,
            state=state,
            positive_work_ids=work_ids,
        )

        states[slot] = state
        positives[slot] = work_ids

    compiled = closure_relationships.get(
        "compiled",
        {},
    )

    draft = closure_relationships.get(
        "draft",
        {},
    )

    if not isinstance(
        compiled,
        Mapping,
    ):
        raise ValueError(
            "closure relationship compiled record "
            "must be a mapping"
        )

    if not isinstance(
        draft,
        Mapping,
    ):
        raise ValueError(
            "closure relationship draft record "
            "must be a mapping"
        )

    review_performed = bool(
        closure_relationships.get(
            "review_performed",
            False,
        )
    )

    scope_status = _text(
        draft.get(
            "scope_compatibility",
            "UNASSESSED",
        )
    )

    bridge_kind_status = _text(
        draft.get(
            "bridge_kind",
            "UNASSESSED",
        )
    )

    compiled_scope = bool(
        compiled.get(
            "scope_compatible",
            False,
        )
    )

    compiled_bridge_kind = _text(
        compiled.get(
            "bridge_kind",
            "NONE",
        )
    )

    # A NOT_FOUND bridge cannot simultaneously have compiled positive
    # bridge/scope semantics.
    if (
        states["BRIDGE_RELATION"]
        == "NOT_FOUND"
        and (
            compiled_scope
            or compiled_bridge_kind
            != "NONE"
        )
    ):
        raise ValueError(
            "NOT_FOUND bridge is inconsistent with "
            "compiled positive relationship semantics"
        )

    if (
        states["BASE_RELATION"]
        != "ESTABLISHED"
    ):
        return _not_eligible(
            "base_relation_not_established"
        )

    if (
        states[
            "DISTINGUISHING_FACTOR_EFFECT"
        ]
        != "ESTABLISHED"
    ):
        return _not_eligible(
            "factor_relation_not_established"
        )

    if (
        states["BRIDGE_RELATION"]
        == "UNASSESSED"
    ):
        return _not_eligible(
            "bridge_relation_requires_more_closure"
        )

    if (
        states["BRIDGE_RELATION"]
        == "ESTABLISHED"
    ):
        return _not_eligible(
            "bridge_relation_already_established"
        )

    if (
        states["FULL_RELATION"]
        == "UNASSESSED"
    ):
        return _not_eligible(
            "full_relation_requires_more_closure"
        )

    if (
        states["FULL_RELATION"]
        == "ESTABLISHED"
    ):
        return _not_eligible(
            "full_relation_already_established"
        )

    if scope_status == "INCOMPATIBLE":
        return _not_eligible(
            "lower_order_scope_explicitly_incompatible",
            "bridge_search_alone_cannot_repair_scope",
        )

    if scope_status != "UNASSESSED":
        return _not_eligible(
            "unexpected_scope_state_for_missing_bridge"
        )

    if bridge_kind_status != "UNASSESSED":
        return _not_eligible(
            "unexpected_bridge_kind_state_for_missing_bridge"
        )

    if review_performed:
        raise ValueError(
            "missing-bridge trigger is inconsistent "
            "with performed cross-slot relationship review"
        )

    base_target = targets[
        "BASE_RELATION"
    ]

    factor_target = targets[
        "DISTINGUISHING_FACTOR_EFFECT"
    ]

    bridge_target = targets[
        "BRIDGE_RELATION"
    ]

    full_target = targets[
        "FULL_RELATION"
    ]

    factor_identity_terms = (
        _unique_strings(
            factor_target.get(
                "identity_anchor_terms",
                [],
            )
        )
    )

    base_relation_terms = (
        _unique_strings(
            base_target.get(
                "search_terms",
                [],
            )
        )
    )

    bridge_text = _text(
        bridge_target.get(
            "source_text"
        )
    )

    full_text = _text(
        full_target.get(
            "source_text"
        )
    )

    bridge_search_terms = (
        _unique_strings(
            bridge_target.get(
                "search_terms",
                [],
            )
        )
    )

    if not factor_identity_terms:
        raise ValueError(
            "missing-bridge opportunity requires "
            "factor identity terms"
        )

    if not base_relation_terms:
        raise ValueError(
            "missing-bridge opportunity requires "
            "base relation terms"
        )

    if not bridge_text:
        raise ValueError(
            "missing-bridge opportunity requires "
            "exact bridge target text for audit"
        )

    if not full_text:
        raise ValueError(
            "missing-bridge opportunity requires "
            "exact full relation text for audit"
        )

    source_portfolio_id = _text(
        execution_plan.get(
            "source_portfolio_id"
        )
    )

    source_hypothesis_id = _text(
        execution_plan.get(
            "source_hypothesis_id"
        )
    )

    source_claim_id = _text(
        execution_plan.get(
            "source_claim_id"
        )
    )

    source_plan_id = _text(
        execution_plan.get(
            "plan_id"
        )
    )

    for name, value in (
        (
            "source_portfolio_id",
            source_portfolio_id,
        ),
        (
            "source_hypothesis_id",
            source_hypothesis_id,
        ),
        (
            "source_claim_id",
            source_claim_id,
        ),
        (
            "source_execution_plan_id",
            source_plan_id,
        ),
    ):
        if not value:
            raise ValueError(
                f"missing {name}"
            )

    opportunity_id = _stable_id(
        "n11_missing_bridge",
        source_portfolio_id,
        source_hypothesis_id,
        source_claim_id,
        source_plan_id,
        ",".join(
            factor_identity_terms
        ),
        ",".join(
            base_relation_terms
        ),
        bridge_text,
        full_text,
    )

    opportunity = (
        N11MissingBridgeOpportunity(
            opportunity_id=opportunity_id,

            source_portfolio_id=(
                source_portfolio_id
            ),

            source_hypothesis_id=(
                source_hypothesis_id
            ),

            source_claim_id=(
                source_claim_id
            ),

            source_execution_plan_id=(
                source_plan_id
            ),

            factor_identity_terms=(
                factor_identity_terms
            ),

            base_relation_terms=(
                base_relation_terms
            ),

            bridge_target_text_for_audit=(
                bridge_text
            ),

            full_relation_text_for_audit=(
                full_text
            ),

            bridge_retrieval_terms_for_audit=(
                bridge_search_terms
            ),

            established_base_work_ids=(
                positives[
                    "BASE_RELATION"
                ]
            ),

            established_factor_work_ids=(
                positives[
                    "DISTINGUISHING_FACTOR_EFFECT"
                ]
            ),

            relationship_review_performed=False,

            relationship_scope_status=(
                "UNASSESSED"
            ),

            relationship_bridge_kind_status=(
                "UNASSESSED"
            ),
        )
    )

    return N11MissingBridgeCompilation(
        status=(
            "ELIGIBLE_FOR_GROUNDED_BRIDGE_SEARCH"
        ),
        opportunity=opportunity,
        reason_codes=[
            (
                "base_and_factor_established_"
                "bridge_and_full_not_found"
            ),
            (
                "explicit_task_connected_"
                "lower_order_bridge_required"
            ),
            (
                "common_anchor_context_"
                "insufficient"
            ),
        ],
    )
