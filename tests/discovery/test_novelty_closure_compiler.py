from dataclasses import dataclass

import pytest

from pipeline_core.discovery.novelty_closure_compiler import (
    compile_nonobviousness_evidence_closure,
)


@dataclass
class Review:
    slot: str
    evidence_state: str
    positive_work_ids: tuple[str, ...] = ()
    negative_coverage_sufficient: bool = False


def actual_like_reviews():
    return [
        Review(
            slot="BASE_RELATION",
            evidence_state="ESTABLISHED",
            positive_work_ids=(
                "work:base:1",
                "work:base:2",
            ),
            negative_coverage_sufficient=True,
        ),
        Review(
            slot="DISTINGUISHING_FACTOR_EFFECT",
            evidence_state="UNASSESSED",
        ),
        Review(
            slot="BRIDGE_RELATION",
            evidence_state="UNASSESSED",
        ),
        Review(
            slot="FULL_RELATION",
            evidence_state="UNASSESSED",
        ),
    ]


def test_actual_like_four_slot_closure_compiles():
    result = compile_nonobviousness_evidence_closure(
        reviews=actual_like_reviews(),
        bridge_kind="NONE",
        scope_compatible=True,
    )

    closure = result.closure

    assert closure.base_relation == "ESTABLISHED"
    assert closure.distinguishing_factor_effect == "UNASSESSED"
    assert closure.bridge_relation == "UNASSESSED"
    assert closure.full_relation == "UNASSESSED"

    assert closure.base_work_ids == (
        "work:base:1",
        "work:base:2",
    )

    assert closure.factor_work_ids == ()
    assert closure.bridge_work_ids == ()
    assert closure.full_relation_work_ids == ()

    assert result.reason_codes == ()


def test_illegal_negative_closure_fails_closed():
    reviews = actual_like_reviews()

    reviews[-1] = Review(
        slot="FULL_RELATION",
        evidence_state="NOT_FOUND",
        negative_coverage_sufficient=False,
    )

    result = compile_nonobviousness_evidence_closure(
        reviews=reviews,
    )

    assert result.closure.full_relation == "UNASSESSED"

    assert (
        "full_relation:illegal_negative_closure_fail_closed"
        in result.reason_codes
    )


def test_established_without_positive_provenance_fails_closed():
    reviews = actual_like_reviews()

    reviews[0] = Review(
        slot="BASE_RELATION",
        evidence_state="ESTABLISHED",
        positive_work_ids=(),
    )

    result = compile_nonobviousness_evidence_closure(
        reviews=reviews,
    )

    assert result.closure.base_relation == "UNASSESSED"

    assert (
        "base_relation:established_without_positive_provenance"
        in result.reason_codes
    )


def test_duplicate_slot_is_rejected():
    reviews = actual_like_reviews()
    reviews.append(
        Review(
            slot="FULL_RELATION",
            evidence_state="UNASSESSED",
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate closure slot",
    ):
        compile_nonobviousness_evidence_closure(
            reviews=reviews
        )


def test_missing_slot_is_rejected():
    reviews = actual_like_reviews()[:-1]

    with pytest.raises(
        ValueError,
        match="Missing closure slots",
    ):
        compile_nonobviousness_evidence_closure(
            reviews=reviews
        )
